"""Document = pipeline + executor + undo stack + Qt signals. All edits go through here.

Threading: the GUI thread owns ``pipeline``. A run works on a *snapshot* of it in a
RunThread with its own Executor (same cache dir). Previews and chart queries run
in a small pool and read ``executor`` (GUI-side, state reads only). Threads are
always joined before the objects they use are replaced.

Autosave, recovery copies and earlier versions all live here. Autosave pauses whenever
what is in memory should not silently overwrite what is on disk (the file changed under us,
an earlier version was restored, a recovered project was brought back) until the person
saves, saves as, or reverts.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Signal, QTimer, QFileSystemWatcher
from PySide6.QtGui import QUndoStack
from PySide6.QtWidgets import QApplication

from ..core import Pipeline, PipelineError, registry
from ..core.model import Edge, Answer
from ..core.planner import signature, signatures_of, plan_layout
from ..core.executor import Executor, NodeState, _pid_alive
from .workers import RunThread, Task, view_pool
from . import commands as cmd

log = logging.getLogger("dancr.ui")
AUTOSAVE_SECS = 60
UNDO_LIMIT = 200


def recovery_path(pid: int | None = None) -> Path:
    """Where this process keeps a copy of edits that are not in a project file yet (an unsaved project, or
    changes autosave has not written), offered back on the next start if the process dies."""
    from ..logsetup import log_path
    return log_path().parent / f"recovery-{os.getpid() if pid is None else pid}.json"


def dead_recovery_files() -> list[Path]:
    """Recovery copies left by DANCR processes that are no longer running."""
    out = []
    for p in sorted(recovery_path().parent.glob("recovery-*.json")):
        m = re.fullmatch(r"recovery-(\d+)\.json", p.name)
        if m and not _pid_alive(int(m.group(1))):
            out.append(p)
    return out


class Document(QObject):
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
    autosaveChanged = Signal(object)   # the reason autosave is paused, or None
    autosaved = Signal()               # a quiet save just happened
    flushRequested = Signal()          # settings typed but not yet committed (editors wait ~350 ms) must go in now

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
        self._watcher = QFileSystemWatcher(self)
        self._watcher.fileChanged.connect(self._on_file_changed)
        self._last_saved_text: str | None = None
        self._states_cache: dict[str, NodeState] = {}
        self._held: dict[str, str | None] = {}
        self._states_token = 0             # bumped by every refresh on this thread: older background polls are dropped
        self._poll_task: Task | None = None
        self._poll = QTimer(self)
        self._poll.setInterval(2500)
        self._poll.timeout.connect(self._poll_states)
        self._poll.start()
        self._auto_timer = QTimer(self); self._auto_timer.setSingleShot(True); self._auto_timer.setInterval(700)
        self._auto_timer.timeout.connect(self._auto_run_now)
        self._auto_pending = False
        self._src_watcher = QFileSystemWatcher(self)
        self._src_watcher.fileChanged.connect(self._on_source_changed)
        self.autosave_paused: str | None = None
        self._autosave = QTimer(self); self._autosave.setInterval(AUTOSAVE_SECS * 1000); self._autosave.timeout.connect(self.autosave_now); self._autosave.start()
        self._rewatch()

    def _on_clean_changed(self, clean: bool) -> None:
        try:
            self.dirtyChanged.emit(not clean)
        except RuntimeError:
            pass

    # ------------------------------------------------------------ file
    @property
    def path(self) -> Path | None:
        return self.pipeline.path

    @property
    def dirty(self) -> bool:
        return not self.undo.isClean()

    def _rewatch(self) -> None:
        files = self._watcher.files()
        if files:
            self._watcher.removePaths(files)
        if self.pipeline.path and self.pipeline.path.exists():
            self._watcher.addPath(str(self.pipeline.path))
        self._rewatch_sources()

    # ------------------------------------------------------------ sources and auto-run
    def source_paths(self) -> list[Path]:
        out = []
        for n in self.pipeline.nodes.values():
            nt = registry.get(n.type)
            if nt.kind != "source":
                continue
            for p in nt.params:
                if p.kind == "path" and n.params.get(p.name):
                    out.append(self.pipeline.directory / str(n.params[p.name]))
        return out

    def _rewatch_sources(self) -> None:
        files = self._src_watcher.files()
        if files:
            self._src_watcher.removePaths(files)
        for p in self.source_paths():
            if p.exists():
                self._src_watcher.addPath(str(p))

    def _on_source_changed(self, path: str) -> None:
        QTimer.singleShot(1500, self._source_changed_settle)

    def _source_changed_settle(self) -> None:
        self._rewatch_sources()
        self.refresh_states()
        self.message.emit("A data file changed on disk")
        self.schedule_auto_run()

    def source_bytes(self) -> int:
        total = 0
        for p in self.source_paths():
            try:
                total += p.stat().st_size
            except OSError:
                pass
        return total

    @property
    def auto_run(self) -> bool:
        """Small data runs itself after every change; big data waits for Run."""
        forced = self.pipeline.meta.get("auto_run")
        if forced in (True, False):
            return bool(forced)
        return bool(self.pipeline.nodes) and self.source_bytes() <= self.AUTO_RUN_BYTES

    def set_auto_run(self, on: bool | None) -> None:
        if on is None:
            self.pipeline.meta.pop("auto_run", None)
        else:
            self.pipeline.meta["auto_run"] = bool(on)
        self.autoRunChanged.emit(self.auto_run)
        self.schedule_auto_run()

    def schedule_auto_run(self) -> None:
        if self.auto_run:
            self._auto_pending = True
            self._auto_timer.start()

    def _auto_run_now(self) -> None:
        if not self._auto_pending:
            return
        if self.running:
            self._auto_timer.start(); return
        self._auto_pending = False
        if any(s.status != "done" for s in self.executor.states().values()):
            self.run()

    # ------------------------------------------------------------ inputs and column registry (undoable)
    def set_input(self, name: str, value: Any = None, unit: str | None = None, note: str | None = None) -> None:
        before = [i.to_dict() for i in self.pipeline.inputs]
        probe = Pipeline.from_dict(self.pipeline.to_dict())
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
        probe = Pipeline.from_dict(self.pipeline.to_dict())
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
                return
        except Exception:  # noqa: BLE001
            log.exception("Could not compare the project on disk with the one in memory")
        if self.dirty:
            self.pause_autosave("the project file changed on disk")
            self.message.emit("The pipeline file changed on disk but you have unsaved edits. Autosave is paused: Save to overwrite it, or File → Revert to load it.")
            return
        if self.autosave_paused is not None:
            # an earlier version or recovered project is open on purpose: never silently replace it
            self.message.emit("The pipeline file changed on disk. What is in the window is unchanged: Save to overwrite it, or File → Revert to load it.")
            return
        self.replace_pipeline(new)
        self.message.emit("Reloaded the pipeline from disk")

    # ------------------------------------------------------------ autosave, recovery, versions
    def pause_autosave(self, reason: str) -> None:
        if self.autosave_paused != reason:
            self.autosave_paused = reason
            self.autosaveChanged.emit(reason)

    def resume_autosave(self) -> None:
        if self.autosave_paused is not None:
            self.autosave_paused = None
            self.autosaveChanged.emit(None)

    def autosave_now(self) -> None:
        """Once a minute: save quietly when the project has a file (keeping the replaced file as an autosave
        version). Edits that must not go into the file yet — an unsaved project, autosave paused, a run in
        progress, or a dialog open that asks about these very edits (Save changes? Revert?) — are copied
        to the recovery file instead, so a crash or a force-quit loses nothing."""
        self.flush_edits()
        if not self.pipeline.nodes or not self.dirty:
            return
        if self.pipeline.path is None or self.autosave_paused or self.running or QApplication.activeModalWidget() is not None:
            self.write_recovery()
            return
        try:
            self.save(auto=True)
        except (OSError, PipelineError):
            log.exception("Autosave of %s failed", self.pipeline.path)
            self.write_recovery()
            return
        self.autosaved.emit()

    def write_recovery(self) -> None:
        """Keep a copy of edits that are not in the project file (an unsaved project, or unsaved changes to a
        saved one) so a crash, a force-quit or a logout loses nothing. Clears the copy when there are none."""
        if not self.pipeline.nodes or (self.pipeline.path is not None and not self.dirty):
            self.clear_recovery()
            return
        try:
            rp = recovery_path(); rp.parent.mkdir(parents=True, exist_ok=True)
            tmp = rp.with_suffix(".tmp")
            data = {"dancr_recovery": 1, "path": str(self.pipeline.path) if self.pipeline.path else None,
                    "pipeline": self.pipeline.to_dict()}
            tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
            tmp.replace(rp)
        except OSError:
            log.exception("Could not write the recovery copy")

    def clear_recovery(self) -> None:
        try:
            recovery_path().unlink(missing_ok=True)
        except OSError:
            log.exception("Could not remove the recovery copy")

    @staticmethod
    def pending_recovery() -> tuple[Pipeline, Path] | None:
        """Unsaved edits left by a DANCR process that is gone, if any: the project (its ``path`` is the file
        the edits belong to, or None for a project never saved) and the recovery file. Unreadable copies
        are deleted."""
        for rp in dead_recovery_files():
            try:
                data = json.loads(rp.read_text(encoding="utf-8"))
                if not isinstance(data, dict) or data.get("dancr_recovery") != 1:
                    raise ValueError("not a DANCR recovery file")
                path = Path(data["path"]) if data.get("path") else None
                pipe = Pipeline.from_dict(data["pipeline"], path)
            except Exception:  # noqa: BLE001
                log.exception("Recovery file %s is unreadable; deleting it", rp)
                rp.unlink(missing_ok=True)
                continue
            if pipe.nodes:
                return pipe, rp
            rp.unlink(missing_ok=True)
        return None

    def recover(self, pipe: Pipeline, source: Path) -> None:
        """Bring back unsaved edits. Their recovery copy becomes this process's copy, so it stays on disk
        until the project is saved or deliberately closed. Autosave stays paused: the person decides
        whether the recovered edits replace the file."""
        self.replace_pipeline(pipe)
        self.undo.resetClean()
        try:
            source.replace(recovery_path())
        except OSError:
            log.exception("Could not take over the recovery copy %s", source)
        self.pause_autosave("recovered changes — save to keep them" if pipe.path else "recovered project — save it to keep it")

    def versions(self) -> list[Path]:
        return self.pipeline.versions() if self.pipeline.path else []

    def restore_version(self, version: Path) -> None:
        """Load an earlier saved copy into the window without touching the file on disk."""
        p = Pipeline.from_dict(json.loads(version.read_text(encoding="utf-8")), self.pipeline.path)
        self.replace_pipeline(p)
        self.undo.resetClean()
        self.pause_autosave("an earlier version is open — save it to keep it, or Revert")

    # ------------------------------------------------------------ whole-pipeline replacement, save, shutdown
    def replace_pipeline(self, pipeline: Pipeline) -> None:
        self.flush_edits()                                  # into the old project, never the new one
        self.stop(wait=True)
        self.pipeline = pipeline
        self._set_executor(Executor(self.pipeline))
        self.undo.clear()
        self._states_cache = {}
        # the in-memory project now matches what is on disk, so the next watcher event is an external one
        self._last_saved_text = None
        if self.pipeline.path is not None and self.pipeline.path.exists():
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
        self.replace_pipeline(Pipeline.load(path))

    def new(self) -> None:
        self.replace_pipeline(Pipeline())

    def shutdown(self) -> None:
        """Stop timers and watchers when the window closes, so a closed document never starts a run.
        Results of a project that was never saved are deleted: nothing refers to them any more.
        Call this after the view pool has been drained: nothing may still be reading the results."""
        self._auto_pending = False
        self._auto_timer.stop(); self._poll.stop(); self._autosave.stop()
        for w in (self._watcher, self._src_watcher):
            if w.files():
                w.removePaths(w.files())
        self.executor.release()
        if self.pipeline.path is None and self.executor.cache_dir.exists():
            shutil.rmtree(self.executor.cache_dir, ignore_errors=True)

    def _set_executor(self, ex: Executor) -> None:
        self.executor.release()
        self.executor = ex
        self._held = {}

    def save(self, path: Path | str | None = None, auto: bool = False) -> Path:
        if path is not None and self.running and Path(path).expanduser().resolve() != self.pipeline.path:
            raise PipelineError("Wait for the run to finish before saving under a new name")
        self.flush_edits()
        old = self.pipeline.path
        old_cache = self.executor.cache_dir
        p = self.pipeline.save(path, auto=auto)
        self._last_saved_text = self.pipeline.dumps()       # so the watcher knows this write was ours
        if old != p:
            new_exec = Executor(self.pipeline)
            try:
                if old_cache.exists() and not new_exec.cache_dir.exists():
                    new_exec.cache_dir.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(old_cache), str(new_exec.cache_dir))     # keep computed results after Save As
            except OSError:
                log.exception("Could not move the cached results from %s to %s; the steps will run again", old_cache, new_exec.cache_dir)
            self._set_executor(new_exec)
            self._states_cache = {}
            self.pathChanged.emit(p)
            self.refresh_states()
        self.undo.setClean()
        self.resume_autosave()
        self._rewatch()
        self.clear_recovery()
        return p

    # ------------------------------------------------------------ queries
    def state(self, nid: str) -> NodeState:
        st = self._states_cache.get(nid)
        if st is None:
            st = self.executor.state(nid)
            self._states_cache[nid] = st
        return st

    def refresh_states(self) -> None:
        """Re-read every step's state now (after an edit: the person expects to see it at once)."""
        self._states_token += 1
        try:
            new = self.executor.states()
        except Exception:  # noqa: BLE001
            log.exception("Could not read the step states")
            return
        self._apply_states(new)

    def _poll_states(self) -> None:
        """The periodic check for changes made elsewhere (a CLI run, a source file rewritten). Every step's
        state means reading files, which can be slow on a network drive, so it runs on a worker over a
        snapshot of the project; the answer is dropped if the project was edited meanwhile."""
        if self._poll_task is not None:
            return
        snapshot = Pipeline.from_dict(self.pipeline.to_dict(), self.pipeline.path)
        ex = Executor(snapshot, self.executor.cache_dir)
        token, executor = self._states_token, self.executor
        t = Task(ex.states)
        t.waits_for_run = False
        t.signals.done.connect(lambda new: self._apply_states(new) if token == self._states_token and executor is self.executor else None)

        def finished() -> None:
            self._poll_task = None
        t.signals.finished.connect(finished)
        self._poll_task = t
        view_pool().start(t)

    def _apply_states(self, new: dict[str, NodeState]) -> None:
        # keep "running" markers from the live run
        for nid, st in self._states_cache.items():
            if st.status == "running" and nid in new and new[nid].status != "done":
                new[nid] = st
        changed = set(new) != set(self._states_cache) or any(
            new[k].status != self._states_cache[k].status or new[k].hash != self._states_cache[k].hash for k in new)
        self._states_cache = new
        held = {k: st.hash for k, st in new.items()}
        if held != self._held:                       # other processes' cache sweeps keep what this window shows
            self._held = held
            self.executor.hold(held)
        if changed:
            self.statesChanged.emit()

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
        self.undo.push(cmd.SetParams(self, nid, node.params, new, set(changes)))

    def rename(self, nid: str, title: str) -> None:
        if nid in self.pipeline.nodes and self.pipeline.nodes[nid].title != title:
            self.undo.push(cmd.Rename(self, nid, self.pipeline.nodes[nid].title, title))

    def move_nodes(self, moves: dict[str, tuple[tuple[float, float], tuple[float, float]]]) -> None:
        moves = {k: v for k, v in moves.items() if v[0] != v[1] and k in self.pipeline.nodes}
        if moves:
            self.undo.push(cmd.Move(self, moves))

    def connect(self, source: str, target: str, port: str | None = None) -> None:
        probe = Pipeline.from_dict(self.pipeline.to_dict())
        edge = probe.connect(source, target, port)
        if edge.key() in {e.key() for e in self.pipeline.edges}:
            return
        self.undo.push(cmd.Connect(self, edge))

    def disconnect(self, edge: Edge) -> None:
        if edge.key() in {e.key() for e in self.pipeline.edges}:
            self.undo.push(cmd.Disconnect(self, edge))

    def move_edge(self, edge: Edge, target: str, port: str | None) -> None:
        """Move a connection to another input as one undo step. Checked first on a copy, so a move that is
        not allowed (the input is taken, it would make a loop) changes nothing and raises PipelineError."""
        probe = Pipeline.from_dict(self.pipeline.to_dict())
        probe.disconnect(edge.source, edge.target, edge.port)
        probe.connect(edge.source, target, port)
        self.undo.beginMacro("Move connection")
        try:
            self.disconnect(edge)
            self.connect(edge.source, target, port)
        finally:
            self.undo.endMacro()

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
        self.undo.beginMacro("Duplicate")
        try:
            for nid in ids:
                if nid not in self.pipeline.nodes:
                    continue
                n = self.pipeline.nodes[nid]
                new_ids.append(self.add_node(n.type, n.x + 40, n.y + 90, params=json.loads(json.dumps(n.params)), title=n.title))
        finally:
            self.undo.endMacro()
        return new_ids

    # ------------------------------------------------------------ answers (guided build)
    def apply_plan(self, plan) -> str:
        """Build the steps a guided question needs — reusing shared ones — and add an Answer card.

        The whole build is one undo entry. The Answer is *not* connected to the dataflow; it points
        at the step whose output answers the question."""
        self.undo.beginMacro("Build answer")
        try:
            resolved = self._apply_steps(plan)
            terminal = resolved[plan.terminal]
            node = self.pipeline.nodes[terminal]
            answer = Answer(self.pipeline._new_answer_id(), plan.title, node.x + 300.0, node.y + 40.0,
                            terminal, plan.view, dict(plan.config))
            self.undo.push(cmd.AddAnswer(self, answer))
        finally:
            self.undo.endMacro()
        self.refresh_states()
        return answer.id

    def rebuild_answer(self, answer_id: str, plan) -> str:
        """Change a question: replace the steps it alone built with a freshly planned branch,
        keeping any steps shared with other answers. Callers warn first (hand edits are replaced)."""
        answer = self.pipeline.answer(answer_id)
        if answer is None:
            return answer_id
        exclusive = self.answer_exclusive_nodes(answer)
        self.undo.beginMacro("Change answer")
        try:
            if exclusive:
                self.remove_nodes(exclusive)
            resolved = self._apply_steps(plan)
            terminal = resolved[plan.terminal]
            node = self.pipeline.nodes[terminal]
            before = answer.to_dict()
            after = {"title": plan.title, "terminal": terminal, "view": plan.view,
                     "config": dict(plan.config), "x": node.x + 300.0, "y": node.y + 40.0}
            self.undo.push(cmd.EditAnswer(self, answer_id, {k: before.get(k) for k in after}, after, "Change answer"))
        finally:
            self.undo.endMacro()
        self.refresh_states()
        return answer_id

    def delete_answer(self, answer_id: str, remove_steps: bool = False) -> None:
        """Delete an Answer card, and optionally the steps that exist only because of it. Steps shared
        with another answer (or the data layer another answer needs) are always kept."""
        answer = self.pipeline.answer(answer_id)
        if answer is None:
            return
        exclusive = self.answer_exclusive_nodes(answer) if remove_steps else []
        self.undo.beginMacro("Delete answer")
        try:
            if exclusive:
                self.remove_nodes(exclusive)
            self.undo.push(cmd.RemoveAnswer(self, answer))
        finally:
            self.undo.endMacro()
        self.refresh_states()

    def answer_exclusive_nodes(self, answer: Answer) -> list[str]:
        """Node ids that exist only because of this answer: its upstream branch minus anything another
        answer also depends on. Never includes a node another answer needs, so deletion is safe."""
        p = self.pipeline
        if answer.terminal not in p.nodes:
            return []
        branch = p.upstream_closure(answer.terminal) | {answer.terminal}
        shared: set[str] = set()
        for other in p.answers:
            if other.id != answer.id and other.terminal in p.nodes:
                shared |= p.upstream_closure(other.terminal) | {other.terminal}
        return sorted(branch - shared)

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
            self.undo.push(cmd.EditAnswer(self, answer_id, {"title": answer.title}, {"title": title}, "Rename answer"))

    def _apply_steps(self, plan) -> dict[str, str]:
        """Resolve a plan to node ids, creating only the steps that do not already exist. Returns
        {plan key: node id}. Shared steps (same type, settings and upstream) are reused."""
        sigs = signatures_of(self.pipeline)
        resolved: dict[str, str] = {}
        positions = plan_layout(plan)
        for step in plan.steps:
            ins = {port: [resolved[k] for k in keys] for port, keys in step.inputs.items()}
            if step.type == "load_file":
                existing = self._find_load(step.params.get("path"))
                if existing:
                    resolved[step.key] = existing
                    continue
            sig = signature(step.type, step.params, ins)
            existing = sigs.get(sig)
            if existing and existing in self.pipeline.nodes:
                resolved[step.key] = existing
                continue
            x, y = positions.get(step.key, (60.0, 200.0))
            nid = self.add_node(step.type, x, y, params=step.params, title=step.title)
            for port, srcs in ins.items():
                for s in srcs:
                    try:
                        self.connect(s, nid, port)
                    except PipelineError:
                        pass
            resolved[step.key] = nid
            sigs[sig] = nid
        return resolved

    def _find_load(self, path) -> str | None:
        """An existing load step for the same file, so a guided build over files already on the map
        reads them once instead of adding a second loader."""
        if not path:
            return None
        directory = self.pipeline.directory

        def resolve(p):
            q = Path(str(p)).expanduser()
            return q if q.is_absolute() else (directory / q)

        target = resolve(path)
        for nid, node in self.pipeline.nodes.items():
            if node.type == "load_file" and node.params.get("path") and resolve(node.params["path"]) == target:
                return nid
        return None

    # ------------------------------------------------------------ running
    @property
    def running(self) -> bool:
        """True from run() until the finished handler has refreshed the states on the GUI thread."""
        return self._run is not None and not self._run_settled

    def run(self, targets: list[str] | None = None, force: bool = False) -> None:
        if self.running:
            return
        snapshot = Pipeline.from_dict(self.pipeline.to_dict(), self.pipeline.path)
        runner = Executor(snapshot, self.executor.cache_dir)
        if targets is not None:
            targets = [t for t in targets if t in snapshot.nodes]
            if not targets:
                return                      # every requested step is gone: nothing to run
        t = RunThread(runner, targets, force)
        self._run = t
        self._run_settled = False
        t.event.connect(self._on_run_event)
        t.failed.connect(self._on_run_failed)
        t.finished.connect(lambda: self._on_run_done(t))
        view_pool().hold()
        self.runStarted.emit()
        t.start()

    def stop(self, wait: bool = False) -> None:
        t = self._run
        if t is None:
            return
        t.cancel.set()
        if wait:
            if t.isRunning():
                t.wait()
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
            self._states_cache[st.node_id] = st
            self.nodeState.emit(st.node_id, st)

    def _on_run_done(self, t: RunThread) -> None:
        """Settle a finished run. Ignored for any thread but the current one, and only once per run."""
        if t is not self._run or self._run_settled:
            return
        self._run_settled = True
        view_pool().release()
        results = dict(t.results)
        self.refresh_states()
        failed = [s for s in results.values() if s.status == "failed"]
        self.runFinished.emit(not failed, results)
        if self._auto_pending:
            self._auto_timer.start()
