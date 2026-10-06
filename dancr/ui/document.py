"""Document = pipeline + executor + undo stack + Qt signals. All edits go through here.

Threading: the GUI thread owns ``pipeline`` and ``executor``. Everything that works off the GUI
thread (a run, previews, schemas, state polls, the data model) gets its own Executor over a
*snapshot* of the project from ``snapshot_executor()`` (same cache dir), so an edit made
meanwhile never changes what a worker is reading. Threads are always joined before the objects
they use are replaced.

Autosave, recovery copies and earlier versions all live here. Autosave is off unless the person
turns it on (File → Autosave); unsaved edits are copied to a recovery file either way. Autosave
pauses whenever what is in memory should not silently overwrite what is on disk (the file changed
under us, an earlier version was restored, a recovered project was brought back) until the person
saves, saves as, or reverts.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import time
import re
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from PySide6.QtCore import QObject, Signal, QTimer, QFileSystemWatcher, QEventLoop
from PySide6.QtGui import QUndoStack
from PySide6.QtWidgets import QApplication

from ..core import Pipeline, PipelineError, registry
from ..core.model import Edge, Answer, Input, rebase_params
from ..core.executor import Executor, NodeState, _pid_alive
from ..headless import ProjectBusy, project_lock, read_project, unsafe_outputs, output_files
from .workers import RunThread, Task, view_pool
from . import commands as cmd
from .doc_edits import _DocEdits
from .doc_recovery import _saved_paths, dead_recovery_files, recovery_path

from .doc_common import (  # noqa: F401,E402
    AUTOSAVE_SECS, AUTO_RUN_DELAY_MS, POLL_MS, SOURCE_SETTLE_MS, UNDO_LIMIT, ChangedOnDisk, log,
)
from .doc_watch import DocWatch
from .doc_autosave import DocAutosave
from .doc_state import DocState












class Document(DocWatch, DocAutosave, DocState, QObject):
    nodeAdded = Signal(str)
    nodeRemoved = Signal(str)
    nodeChanged = Signal(str)          # params or title
    nodeMoved = Signal(str)
    edgeAdded = Signal(object)         # Edge
    edgeRemoved = Signal(object)
    noteAdded = Signal(str)
    noteRemoved = Signal(str)
    noteChanged = Signal(str)
    reloaded = Signal()                # whole pipeline replaced
    statesChanged = Signal()           # any node state may have changed
    nodeState = Signal(str, object)    # node id, NodeState (during a run)
    runStarted = Signal()
    runProgress = Signal(int, int, str)   # index, total, node title
    runFinished = Signal(bool, object)    # ok, {node_id: NodeState} of this run
    dirtyChanged = Signal(bool)
    pathChanged = Signal(object)
    message = Signal(str)              # status bar text
    inputsChanged = Signal()
    columnsChanged = Signal()
    answerAdded = Signal(str)
    answerRemoved = Signal(str)
    answerChanged = Signal(str)
    autoRunChanged = Signal(bool)
    autosaveChanged = Signal(object)   # autosave turned on or off, or paused (the reason) or resumed (None)
    autosaved = Signal()               # a quiet save just happened
    flushRequested = Signal()          # settings typed but not yet committed (editors wait ~350 ms) must go in now
    busy = Signal(object)              # why the window must wait (a run is stopping), or None once it may go on

    AUTO_RUN_BYTES = 200_000_000       # sources smaller than this in total run automatically after every change

    def __init__(self, pipeline: Pipeline | None = None) -> None:
        super().__init__()
        self.pipeline = pipeline or Pipeline()
        self.executor = Executor(self.pipeline)
        self.undo = QUndoStack(self)
        self.undo.setUndoLimit(UNDO_LIMIT)
        self.undo.cleanChanged.connect(self._on_clean_changed)
        self.undo.indexChanged.connect(lambda _: self.schedule_auto_run())
        self._run: RunThread | None = None
        self._run_settled = True
        self._stopping = False              # a stop(wait=True) is spinning a nested loop: defer other work
        self.last_run_outcome = "done"     # how the last run ended: done | stopped | crashed
        self._watcher = QFileSystemWatcher(self)
        self._watcher.fileChanged.connect(self._on_file_changed)
        self._watcher.directoryChanged.connect(self._on_project_dir_changed)
        self._last_saved_text: str | None = None
        self._states_cache: dict[str, NodeState] = {}
        self._held: dict[str, str | None] = {}
        self._states_token = 0             # bumped by every refresh on this thread: older background polls are dropped
        self._poll_task: Task | None = None     # the one states read under way
        self._states_again = False         # a refresh was asked for while it read: read once more after it
        self._closed = False
        self._poll = QTimer(self)
        self._poll.setInterval(POLL_MS)
        self._poll.timeout.connect(self._poll_states)
        self._poll.start()
        self._auto_timer = QTimer(self); self._auto_timer.setSingleShot(True); self._auto_timer.setInterval(AUTO_RUN_DELAY_MS)
        self._source_timer = QTimer(self); self._source_timer.setSingleShot(True); self._source_timer.setInterval(SOURCE_SETTLE_MS)
        self._source_timer.timeout.connect(self._source_changed_settle)
        self._auto_timer.timeout.connect(self._auto_run_now)
        self._auto_pending = False
        self._auto_gen = 0                 # bumped by every request for an automatic run
        self._auto_task: Task | None = None  # the states read that decides whether to run
        self._src_watcher = QFileSystemWatcher(self)
        self._src_watcher.fileChanged.connect(self._on_source_changed)
        self._src_watcher.directoryChanged.connect(self._on_source_dir_changed)
        self.autosave = False              # save the project file every minute; the person turns it on (File → Autosave)
        self.autosave_paused: str | None = None
        self.held: set[str] = set()                # steps changed elsewhere that save files where they should not: run only when asked
        self.last_set_aside: list[str] = []        # hand-edited steps the last answer change left as they were
        self._autosave = QTimer(self); self._autosave.setInterval(AUTOSAVE_SECS * 1000); self._autosave.timeout.connect(self.autosave_now); self._autosave.start()
        self._rewatch()

    def _on_clean_changed(self, clean: bool) -> None:
        try:
            self.dirtyChanged.emit(not clean)
        except RuntimeError:
            pass

    # ------------------------------------------------------------ inputs and column registry (undoable)
    def set_input(self, name: str, value: Any = None, unit: str | None = None, note: str | None = None) -> None:
        before = [i.to_dict() for i in self.pipeline.inputs]
        probe = Pipeline()                              # checked on a copy of the inputs alone
        probe.inputs = [Input(**i) for i in before]
        probe.set_input(name, value, unit, note)
        after = [i.to_dict() for i in probe.inputs]
        if before != after:
            self.undo.push(cmd.SetInputs(self, before, after, f"Set {name}"))

    def remove_input(self, name: str) -> None:
        before = [i.to_dict() for i in self.pipeline.inputs]
        after = [i for i in before if i["name"].lower() != name.lower()]
        if before != after:
            self.undo.push(cmd.SetInputs(self, before, after, f"Remove {name}"))

    def set_column_meta(self, name: str, label: str | None = None, unit: str | None = None) -> None:
        before = json.loads(json.dumps(self.pipeline.columns))
        probe = Pipeline()                              # checked on a copy of the column names alone
        probe.columns = json.loads(json.dumps(before))
        probe.set_column_meta(name, label, unit)
        if probe.columns != before:
            self.undo.push(cmd.SetColumns(self, before, probe.columns, f"Describe {name}"))

    # ------------------------------------------------------------ the project file changing under us
    def _on_file_changed(self, path: str) -> None:
        # Always re-check. Our own writes are recognised by content (see _maybe_reload), so an
        # external write that lands right after a save is never dropped.
        self._rewatch()
        QTimer.singleShot(200, self._maybe_reload)

    def flush_edits(self) -> None:
        """Ask every editor to commit what was typed in the last moment, so saving, reloading or replacing
        the project never loses it, and it never lands on a step of the next project with the same id."""
        self.flushRequested.emit()

    def _maybe_reload(self) -> None:
        self._rewatch()
        if self._stopping:
            # inside stop(wait=True)'s nested event loop: reloading a different project now would re-enter
            # replace_pipeline and stop(); try again once the wait is over
            QTimer.singleShot(1000, self._maybe_reload)
            return
        if not self.pipeline.path or not self.pipeline.path.exists():
            return
        if QApplication.activeModalWidget() is not None or self.running:
            QTimer.singleShot(1000, self._maybe_reload)     # try again once the modal/run is over
            return
        self.flush_edits()                                  # a half-typed setting makes the project dirty: no reload
        try:
            text = self.pipeline.path.read_text(encoding="utf-8")
        except OSError:
            return
        if self._last_saved_text is not None and text == self._last_saved_text:
            return                                          # this is the copy we just wrote
        try:
            new = Pipeline.from_dict(json.loads(text), self.pipeline.path)
        except Exception:  # noqa: BLE001
            log.exception("The project file %s changed on disk but could not be read", self.pipeline.path)
            return
        try:
            if new.to_dict() == self.pipeline.to_dict():
                self._last_saved_text = text                # the same project written differently: nothing to keep apart
                return
        except Exception:  # noqa: BLE001
            log.exception("Could not compare the project on disk with the one in memory")
        if self.dirty:
            self.pause_autosave("the project file changed on disk")
            self.message.emit("The project file changed on disk, but you have unsaved edits."
                              + (" Autosave is paused." if self.autosave else "")
                              + " Save to overwrite the file, or use File → Revert to load the new version.")
            return
        if self.autosave_paused is not None:
            # an earlier version or recovered project is open on purpose: never silently replace it
            self.message.emit("The project file changed on disk. The window still shows your version. Save to overwrite the file, or use File → Revert to load the new version.")
            return
        held = self._held_after(self.pipeline, new)
        self.replace_pipeline(new, disk_text=text)          # exactly the text that was read: a later write is someone else's
        self._hold(held)

    def _held_after(self, old: Pipeline, new: Pipeline) -> set[str]:
        """The steps to hold once ``new`` (the same file, changed elsewhere) replaces ``old``: those it newly points
        where they should not save, and those still held that still would."""
        folder = new.directory.resolve()
        still = {n for n in self.held if n in new.nodes and unsafe_outputs(new, n, folder)}
        return self._risky_outputs(old, new) | still

    def _hold(self, held: set[str]) -> None:
        self.held = held
        if held:
            self.statesChanged.emit()                       # the map marks them
            titles = ", ".join(self.pipeline.nodes[n].title for n in sorted(held))
            self.message.emit(f"Reloaded the project from disk. {titles} would save files outside the project folder or over "
                              "its data, so they weren't run. Run to save them anyway")
        else:
            self.message.emit("Reloaded the project from disk")

    @staticmethod
    def _risky_outputs(old: Pipeline, new: Pipeline) -> set[str]:
        """Steps another program added or pointed at a new file that would save outside the project folder or
        over a file the project reads. The window does not run those by itself (the person has not seen them)."""
        folder = new.directory.resolve()
        out = set()
        for nid in new.nodes:
            files = output_files(new, nid)
            if not files or (nid in old.nodes and old.nodes[nid].type == new.nodes[nid].type
                             and output_files(old, nid) == files):
                continue
            if unsafe_outputs(new, nid, folder):
                out.add(nid)
        return out

    # ------------------------------------------------------------ whole-pipeline replacement, save, shutdown
    def replace_pipeline(self, pipeline: Pipeline, disk_text: str | None = None) -> None:
        """Show another project. ``disk_text`` is the file's text the project was read from; without it the
        file is read now (the project was not read from it: a recovered or earlier version)."""
        self.flush_edits()                                  # into the old project, never the new one
        self.stop(wait=True)
        self.pipeline = pipeline
        self._set_executor(Executor(self.pipeline))
        self.undo.clear()
        self._states_cache = {}
        self.held = set()
        # what is on disk as far as this window knows, so the next watcher event that differs is an external one
        self._last_saved_text = disk_text
        if disk_text is None and self.pipeline.path is not None and self.pipeline.path.exists():
            try:
                self._last_saved_text = self.pipeline.path.read_text(encoding="utf-8")
            except OSError:
                self._last_saved_text = None
        self.resume_autosave()
        self._rewatch()
        self.reloaded.emit()
        self.pathChanged.emit(self.pipeline.path)
        self.refresh_states()

    def load(self, path: Path | str) -> None:
        pipe, text = read_project(Path(path))
        same = self.pipeline.path is not None and self.pipeline.path.resolve() == pipe.path
        held = self._held_after(self.pipeline, pipe) if same else set()     # Revert: what was changed elsewhere stays held
        self.replace_pipeline(pipe, disk_text=text)
        if same:
            self._hold(held)

    def new(self) -> None:
        self.replace_pipeline(Pipeline())

    def shutdown(self) -> None:
        """Stop timers and watchers when the window closes, so a closed document never starts a run.
        Results of a project that was never saved are deleted: nothing refers to them any more.
        Call this after the view pool has been drained: nothing may still be reading the results."""
        self._auto_pending = False
        self._closed = True
        self._auto_timer.stop(); self._poll.stop(); self._autosave.stop(); self._source_timer.stop()
        for w in (self._watcher, self._src_watcher):
            self._watch(w, [])
        self.executor.release()
        if self.pipeline.path is None and self.executor.cache_dir.exists():
            shutil.rmtree(self.executor.cache_dir, ignore_errors=True)

    def _rebase_undo(self, old_dir: Path, new_dir: Path) -> None:
        """Save As moved the project: settings kept by undo commands must keep pointing at the same files."""
        fn = lambda node_type, params: rebase_params(node_type, params, old_dir, new_dir)  # noqa: E731

        def visit(c) -> None:
            if hasattr(c, "rebase"):
                c.rebase(fn)
            for j in range(c.childCount()):
                visit(c.child(j))
        for i in range(self.undo.count()):
            visit(self.undo.command(i))

    def _set_executor(self, ex: Executor) -> None:
        self.executor.release()
        self.executor = ex
        self._held = {}

    def changed_elsewhere(self) -> bool:
        """True when the project file on disk is not what this window last read or wrote: another program (an agent
        over MCP, the command line, another window) changed it, and saving now would silently throw that away."""
        p = self.pipeline.path
        if p is None or not p.exists() or self._last_saved_text is None:
            return False
        try:
            return p.read_text(encoding="utf-8") != self._last_saved_text
        except OSError:
            return False

    def save(self, path: Path | str | None = None, auto: bool = False, overwrite: bool = False) -> Path:
        """Save to the project file (or ``path``). The file's lock is held from the check that nobody else
        changed it until the write, so an agent or a command changing it at the same moment is never lost.
        Raises ChangedOnDisk when it was changed elsewhere, ProjectBusy when another program holds the lock."""
        if path is not None and self.running and Path(path).expanduser().resolve() != self.pipeline.path:
            raise PipelineError("Wait for the run to finish before saving under a new name")
        target = Path(path).expanduser().resolve() if path is not None else self.pipeline.path
        if target is None:
            raise PipelineError("No file path to save to")
        self.flush_edits()
        with project_lock(target, wait=1.0 if auto else 5.0):
            if target == self.pipeline.path and not overwrite and self.changed_elsewhere():
                self.pause_autosave("the project file changed on disk")
                raise ChangedOnDisk("The project file was changed by another program since it was opened here")
            p = self._write(path, auto)
        self._rewatch()
        return p

    def _write(self, path: Path | str | None, auto: bool) -> Path:
        old = self.pipeline.path
        old_dir = self.pipeline.directory.resolve()
        old_cache = self.executor.cache_dir
        p = self.pipeline.save(path, auto=auto)
        if p.parent != old_dir:
            self._rebase_undo(old_dir, p.parent)
        self._last_saved_text = self.pipeline.dumps()       # so the watcher knows this write was ours
        if old != p:
            new_exec = Executor(self.pipeline)
            self.executor.release()                         # before the move: the lease must not travel with the cache
            try:
                if old_cache.exists() and not new_exec.cache_dir.exists():
                    new_exec.cache_dir.parent.mkdir(parents=True, exist_ok=True)
                    if os.stat(old_cache).st_dev == os.stat(new_exec.cache_dir.parent).st_dev:
                        shutil.move(str(old_cache), str(new_exec.cache_dir))     # same drive: a rename, instant
                    else:
                        # another drive: copying gigabytes of results would freeze the window; they are computed
                        # again in the new folder instead; the old copy is swept away on the next start
                        self.message.emit("Saved. The new folder is on another drive, so results will be computed again there")
            except OSError:
                log.exception("Could not move the cached results from %s to %s; the steps will run again", old_cache, new_exec.cache_dir)
            self._set_executor(new_exec)
            self._states_cache = {}
            self.pathChanged.emit(p)
            self.refresh_states()
        self.undo.setClean()
        self.resume_autosave()
        self.clear_recovery()
        return p

    # ------------------------------------------------------------ edits (undoable)
    def add_node(self, type_key: str, x: float, y: float, params: dict[str, Any] | None = None, title: str | None = None,
                 connect_from: str | None = None, port: str | None = None) -> str:
        node = self.pipeline.add_node(type_key, title=title, params=params, x=x, y=y)
        nid = node.id
        self.pipeline.remove_node(nid)  # re-added by the command
        self.undo.push(cmd.AddNode(self, node, connect_from, port))
        return nid

    def remove_nodes(self, ids: list[str]) -> None:
        ids = [i for i in ids if i in self.pipeline.nodes]
        if ids:
            self.undo.push(cmd.RemoveNodes(self, ids))

    def set_params(self, nid: str, changes: dict[str, Any]) -> None:
        if nid not in self.pipeline.nodes:
            return
        node = self.pipeline.nodes[nid]
        nt = registry.get(node.type)
        new = dict(node.params)
        names = {p.name for p in nt.params}
        for k, v in changes.items():
            if k in names:
                new[k] = nt.param(k).coerce(v)
        if new == node.params:
            return
        self.held.discard(nid)                       # the person has seen and changed it
        self.undo.push(cmd.SetParams(self, nid, node.params, new, set(changes)))

    def rename(self, nid: str, title: str) -> None:
        if nid in self.pipeline.nodes and self.pipeline.nodes[nid].title != title:
            self.undo.push(cmd.Rename(self, nid, self.pipeline.nodes[nid].title, title))

    def move_nodes(self, moves: dict[str, tuple[tuple[float, float], tuple[float, float]]]) -> None:
        moves = {k: v for k, v in moves.items() if v[0] != v[1] and k in self.pipeline.nodes}
        if moves:
            self.undo.push(cmd.Move(self, moves))

    def connect(self, source: str, target: str, port: str | None = None) -> None:
        edge = self.pipeline.plan_connect(source, target, port)
        if edge.key() in {e.key() for e in self.pipeline.edges}:
            return
        self.undo.push(cmd.Connect(self, edge))

    def disconnect(self, edge: Edge) -> None:
        if edge.key() in {e.key() for e in self.pipeline.edges}:
            self.undo.push(cmd.Disconnect(self, edge))

    @contextmanager
    def macro(self, text: str) -> Iterator[None]:
        """Group the edits made inside into one undo step. If one of them fails, the ones already made are
        undone and the error goes on: nothing is left half done. A group that changed nothing leaves no
        undo step and does not mark the project as changed."""
        start = self.undo.index()
        self.undo.beginMacro(text)
        try:
            yield
        except BaseException:
            self.undo.endMacro()
            if self.undo.index() > start:
                self.undo.undo()
            raise
        self.undo.endMacro()
        if self.undo.index() > start and self.undo.command(self.undo.index() - 1).childCount() == 0:
            self.undo.undo()                             # empty: step back over it (the next edit drops it)

    def move_edge(self, edge: Edge, target: str, port: str | None) -> None:
        """Move a connection to another input as one undo step. Checked first on a copy, so a move that is
        not allowed (the input is taken, it would make a loop) changes nothing and raises PipelineError."""
        self.pipeline.plan_connect(edge.source, target, port, ignoring=edge)
        with self.macro("Move connection"):
            self.disconnect(edge)
            self.connect(edge.source, target, port)

    def add_note(self, text: str, x: float, y: float) -> str:
        note = self.pipeline.add_note(text, x, y)
        self.pipeline.notes.remove(note)
        self.undo.push(cmd.AddNote(self, note))
        return note.id

    def edit_note(self, nid: str, **changes: Any) -> None:
        note = next((n for n in self.pipeline.notes if n.id == nid), None)
        if note is None:
            return
        old = {k: getattr(note, k) for k in changes}
        if old != changes:
            self.undo.push(cmd.EditNote(self, nid, old, changes))

    def remove_note(self, nid: str) -> None:
        if any(n.id == nid for n in self.pipeline.notes):
            self.undo.push(cmd.RemoveNote(self, nid))

    def duplicate_nodes(self, ids: list[str]) -> list[str]:
        new_ids = []
        with self.macro("Duplicate"):
            for nid in ids:
                if nid not in self.pipeline.nodes:
                    continue
                n = self.pipeline.nodes[nid]
                new_ids.append(self.add_node(n.type, n.x + 40, n.y + 90, params=json.loads(json.dumps(n.params)), title=n.title))
        return new_ids

    # ------------------------------------------------------------ answers
    def build_answer(self, model, spec: dict, answer_id: str | None = None) -> str:
        """Build the steps a question needs (reusing shared ones) and add its Answer, or with ``answer_id``
        change that answer in place. One undo step. Raises PlanError when it cannot be answered."""
        from ..core import answers
        from ..core.planner import apply_plan, protected_nodes
        from ..core.recipes import plan as make_plan
        plan = make_plan(model, spec)
        from ..core.recipes import PlanError
        missing = [s.params["node"] for s in plan.steps if s.type == "@" and s.params["node"] not in self.pipeline.nodes]
        if missing:                                 # checked before anything changes: a failed build leaves no trace
            raise PlanError(f"The table {missing[0]!r} is not in the project any more")
        existing = self.pipeline.answer(answer_id) if answer_id else None
        with self.macro("Change answer" if existing else f"Answer: {plan.title}"):
            applied = apply_plan(self.pipeline, plan, _DocEdits(self),
                                 existing.steps if existing else None, protected_nodes(self.pipeline, answer_id))
            fields = answers.answer_fields(self.pipeline, plan, applied.resolved, applied.record)
            self.last_set_aside = [self.pipeline.nodes[n].title for n in applied.left if n in self.pipeline.nodes]
            if existing is None:
                a = Answer(self.pipeline._new_answer_id(), fields["title"], fields["x"], fields["y"], fields["terminal"],
                           fields["view"], fields["spec"], fields["steps"], fields["assumptions"], fields["rules"])
                self.undo.push(cmd.AddAnswer(self, a))
                aid = a.id
            else:
                before = {k: copy.deepcopy(getattr(existing, k)) for k in fields}
                self.undo.push(cmd.EditAnswer(self, answer_id, before, fields, "Change answer"))
                aid = answer_id
        self.refresh_states()
        return aid

    def set_dataset_meta(self, **changes: Any) -> None:
        """Set the project's dataset-level metadata (creator, license, description…), undoably."""
        before = copy.deepcopy(self.pipeline.meta.get("dataset"))
        probe = Pipeline()                              # validated on a throwaway project
        if before:
            probe.meta = {"dataset": copy.deepcopy(before)}
        probe.set_dataset_meta(**changes)
        after = copy.deepcopy(probe.meta.get("dataset"))
        if before != after:
            self.undo.push(cmd.SetDatasetMeta(self, before, after))

    def ai_steps(self) -> list[str]:
        """The step ids the Assistant built, still present in the project."""
        ids = self.pipeline.meta.get("ai_steps") or []
        return [n for n in ids if n in self.pipeline.nodes] if isinstance(ids, list) else []

    def add_ai_steps(self, ids: list[str]) -> None:
        """Remember that these steps were built by the Assistant (undoable, saved with the project)."""
        before = self.ai_steps()
        after = sorted(set(before) | {n for n in ids if n in self.pipeline.nodes})
        if before != after:
            self.undo.push(cmd.SetAiSteps(self, before, after))

    def clear_ai_steps(self) -> None:
        before = self.ai_steps()
        if before:
            self.undo.push(cmd.SetAiSteps(self, before, []))

    def set_thread(self, thread: dict | None) -> None:
        """Save the Assistant's conversation into the project (undoable, and it marks the project changed)."""
        before = copy.deepcopy(self.pipeline.meta.get("assistant"))
        after = copy.deepcopy(thread) if thread is not None else None
        if before == after:
            return
        self.undo.push(cmd.SetThread(self, before, after))

    def replace_canvas(self, data: dict, text: str = "Replace the canvas") -> None:
        """Swap the whole canvas for another project, as one undo step (the Assistant's overwrite)."""
        before = self.pipeline.to_dict()
        if before == data:
            return
        self.undo.push(cmd.ReplacePipeline(self, before, copy.deepcopy(data), text))

    def delete_answer(self, answer_id: str, remove_steps: bool = False) -> None:
        """Delete an Answer card, and optionally the steps only it uses. Steps another answer needs, and steps
        something else reads from, are always kept."""
        answer = self.pipeline.answer(answer_id)
        if answer is None:
            return
        exclusive = self.answer_exclusive_nodes(answer) if remove_steps else []
        with self.macro("Delete answer"):
            if exclusive:
                self.remove_nodes(exclusive)
            self.undo.push(cmd.RemoveAnswer(self, answer))
        self.refresh_states()

    def answer_exclusive_nodes(self, answer: Answer) -> list[str]:
        from ..core.answers import exclusive_steps
        return exclusive_steps(self.pipeline, answer.id)

    def edit_answer(self, answer_id: str, **changes: Any) -> None:
        answer = self.pipeline.answer(answer_id)
        if answer is None:
            return
        old = {k: getattr(answer, k) for k in changes if hasattr(answer, k)}
        if old and old != changes:
            self.undo.push(cmd.EditAnswer(self, answer_id, old, changes, "Edit answer"))

    def rename_answer(self, answer_id: str, title: str) -> None:
        answer = self.pipeline.answer(answer_id)
        title = title.strip()
        if answer and title and answer.title != title:
            spec = dict(answer.spec, title=title)
            self.undo.push(cmd.EditAnswer(self, answer_id, {"title": answer.title, "spec": answer.spec},
                                          {"title": title, "spec": spec}, "Rename answer"))

    # ------------------------------------------------------------ running
    @property
    def running(self) -> bool:
        """True from run() until the finished handler has refreshed the states on the GUI thread."""
        return self._run is not None and not self._run_settled

    def run(self, targets: list[str] | None = None, force: bool = False, auto: bool = False) -> None:
        """Run the project (or ``targets`` and what they need). A run the person asked for (not ``auto``) also
        runs held steps it includes, and they are held no longer."""
        if self.running:
            return
        runner = self.snapshot_executor()
        snapshot = runner.pipeline
        if targets is not None:
            targets = [t for t in targets if t in snapshot.nodes]
            if not targets:
                return                      # every requested step is gone: nothing to run
        if snapshot.path is None:
            # an unsaved project has no folder: a file named without one would land wherever DANCR was started
            unplaced = {nid for nid in snapshot.nodes if self._needs_folder(snapshot, nid)}
            if unplaced:
                wanted = targets if targets is not None else list(snapshot.nodes)
                targets = [t for t in wanted if t not in unplaced and not unplaced & snapshot.upstream_closure(t)]
                if not auto:
                    titles = ", ".join(snapshot.nodes[n].title for n in sorted(unplaced))
                    self.message.emit(f"Save the project first. {titles} saves its file next to the project")
                if not targets:
                    return
        if not auto and self.held:
            self.held -= set(snapshot.topological_order(targets))
        t = RunThread(runner, targets, force)
        self._run = t
        self._run_settled = False
        # a finished run's last events can still be queued when the project is replaced: only the current
        # run's events reach the states and the views
        t.event.connect(lambda e, t=t: self._on_run_event(e) if t is self._run else None)
        t.failed.connect(lambda text, t=t: self._on_run_failed(text) if t is self._run else None)
        t.finished.connect(lambda: self._on_run_done(t))
        view_pool().hold()
        self.runStarted.emit()
        t.start()

    @staticmethod
    def _needs_folder(p: Pipeline, nid: str) -> bool:
        """A step that saves a file named relative to the project's folder."""
        return any(not Path(v).expanduser().is_absolute() for v in _saved_paths(p, nid))

    def stop(self, wait: bool = False, why: str = "Stopping the current step…") -> None:
        """Ask the run to stop. It ends at the next step boundary: a step already writing its result finishes first.
        With ``wait``, return once it has ended and been settled. That step cannot be cut short and may take
        minutes on a big table, so the window is never frozen meanwhile: events keep flowing and ``busy`` tells
        the window why it must wait (it accepts no other command until ``busy(None)``)."""
        t = self._run
        if t is None or self._stopping:                    # not while another stop(wait=True) is already waiting
            return
        t.cancel.set()
        if not wait or self._run_settled:
            return
        if t.isRunning():
            loop = QEventLoop()
            t.finished.connect(loop.quit)
            check = QTimer(self); check.setInterval(100)    # also covers a finish signalled just before connecting
            check.timeout.connect(lambda: loop.quit() if not t.isRunning() else None)
            check.start()
            self.busy.emit(why)
            self._stopping = True
            try:
                if t.isRunning():
                    loop.exec()
            finally:
                self._stopping = False
                check.stop()
                try:
                    t.finished.disconnect(loop.quit)      # the thread may already be gone if it finished mid-loop
                except (RuntimeError, TypeError):
                    pass
                self.busy.emit(None)
        t.wait()                                            # it has ended: this only joins the thread
        self._on_run_done(t)

    def _on_run_failed(self, text: str) -> None:
        log.error("Run failed:\n%s", text)
        self.message.emit(f"Run failed: {text.splitlines()[0]}")

    def _on_run_event(self, e: dict) -> None:
        t = e.get("type")
        if t == "node_started":
            st = NodeState(e["node"], status="running")
            self._states_cache[e["node"]] = st
            self.nodeState.emit(e["node"], st)
            title = self.pipeline.nodes[e["node"]].title if e["node"] in self.pipeline.nodes else e["node"]
            self.runProgress.emit(e.get("index", 0), e.get("total", 0), title)
        elif t in ("node_finished", "node_failed", "node_cached"):
            st = e["state"]
            self._states_token += 1                          # a background poll read before this is older
            self._states_cache[st.node_id] = st
            self.nodeState.emit(st.node_id, st)

    def _on_run_done(self, t: RunThread) -> None:
        """Settle a finished run. Ignored for any thread but the current one, and only once per run."""
        if t is not self._run or self._run_settled:
            return
        self._run_settled = True
        view_pool().release()
        results = dict(t.results)
        self.last_run_outcome = t.outcome
        # what the run found is known now, without reading a file here; a step it was computing when stopped
        # is no longer "running" (it is read again when asked for); the full re-read follows on a worker
        self._states_cache = {k: v for k, v in self._states_cache.items() if v.status != "running"}
        self._states_cache.update({k: v for k, v in results.items() if k in self.pipeline.nodes})
        self.refresh_states()
        failed = [s for s in results.values() if s.status == "failed"]
        self.runFinished.emit(t.outcome == "done" and not failed, results)
        if self._auto_pending:
            self._auto_timer.start()
        self._run = None                         # release the finished thread (and its snapshot) at once
        try:
            t.deleteLater()
        except RuntimeError:
            pass
