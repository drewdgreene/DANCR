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
import time
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Signal, QTimer, QFileSystemWatcher
from PySide6.QtGui import QUndoStack
from PySide6.QtWidgets import QApplication

from ..core import Pipeline, PipelineError, registry
from ..core.model import Edge
from ..core.executor import Executor, NodeState, _pid_alive
from .workers import RunThread, view_pool
from . import commands as cmd

log = logging.getLogger("dancr.ui")
AUTOSAVE_SECS = 60
UNDO_LIMIT = 200


def recovery_path(pid: int | None = None) -> Path:
    """Where this process keeps its unsaved (never saved) project between sessions."""
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
    autoRunChanged = Signal(bool)
    autosaveChanged = Signal(object)   # the reason autosave is paused, or None
    autosaved = Signal()               # a quiet save just happened

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
        self._run_started_at = 0.0
        self._watcher = QFileSystemWatcher(self)
        self._watcher.fileChanged.connect(self._on_file_changed)
        self._last_saved_text: str | None = None
        self._states_cache: dict[str, NodeState] = {}
        self._poll = QTimer(self)
        self._poll.setInterval(2500)
        self._poll.timeout.connect(self.refresh_states)
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

    def _maybe_reload(self) -> None:
        self._rewatch()
        if not self.pipeline.path or not self.pipeline.path.exists():
            return
        if QApplication.activeModalWidget() is not None or self.running:
            QTimer.singleShot(1000, self._maybe_reload)     # try again once the modal/run is over
            return
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
        """Save quietly once a minute when the project has a file (every save keeps an earlier version).
        An unsaved project is copied to a recovery file instead, offered back on the next start.
        Nothing is written while autosave is paused."""
        if self.autosave_paused or self.running or not self.pipeline.nodes or not self.dirty:
            return
        if self.pipeline.path is None:
            self.write_recovery()
            return
        try:
            self.save()
        except (OSError, PipelineError):
            log.exception("Autosave of %s failed", self.pipeline.path)
            return
        self.autosaved.emit()

    def write_recovery(self) -> None:
        """Keep a copy of an unsaved project so a crash or a force-quit loses nothing."""
        if self.pipeline.path is not None or not self.pipeline.nodes:
            return
        try:
            rp = recovery_path(); rp.parent.mkdir(parents=True, exist_ok=True)
            tmp = rp.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.pipeline.to_dict(), indent=1), encoding="utf-8")
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
        """The unsaved project of a DANCR process that is gone, if any. Unreadable copies are deleted."""
        for rp in dead_recovery_files():
            try:
                pipe = Pipeline.from_dict(json.loads(rp.read_text(encoding="utf-8")), None)
            except Exception:  # noqa: BLE001
                log.exception("Recovery file %s is unreadable; deleting it", rp)
                rp.unlink(missing_ok=True)
                continue
            if pipe.nodes:
                return pipe, rp
            rp.unlink(missing_ok=True)
        return None

    def recover(self, pipe: Pipeline, source: Path) -> None:
        """Bring back an unsaved project. Its recovery copy becomes this process's copy, so it stays
        on disk until the project is saved or deliberately closed."""
        self.replace_pipeline(pipe)
        self.undo.resetClean()
        try:
            source.replace(recovery_path())
        except OSError:
            log.exception("Could not take over the recovery copy %s", source)
        self.pause_autosave("recovered project — save it to keep it")

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
        self.stop(wait=True)
        self.pipeline = pipeline
        self.executor = Executor(self.pipeline)
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
        if self.pipeline.path is None and self.executor.cache_dir.exists():
            shutil.rmtree(self.executor.cache_dir, ignore_errors=True)

    def save(self, path: Path | str | None = None) -> Path:
        if path is not None and self.running and Path(path).expanduser().resolve() != self.pipeline.path:
            raise PipelineError("Wait for the run to finish before saving under a new name")
        old = self.pipeline.path
        old_cache = self.executor.cache_dir
        p = self.pipeline.save(path)
        self._last_saved_text = self.pipeline.dumps()       # so the watcher knows this write was ours
        if old != p:
            new_exec = Executor(self.pipeline)
            try:
                if old_cache.exists() and not new_exec.cache_dir.exists():
                    new_exec.cache_dir.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(old_cache), str(new_exec.cache_dir))     # keep computed results after Save As
            except OSError:
                log.exception("Could not move the cached results from %s to %s; the steps will run again", old_cache, new_exec.cache_dir)
            self.executor = new_exec
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
        try:
            new = self.executor.states()
        except Exception:  # noqa: BLE001
            log.exception("Could not read the step states")
            return
        # keep "running" markers from the live run
        for nid, st in self._states_cache.items():
            if st.status == "running" and nid in new and new[nid].status != "done":
                new[nid] = st
        changed = set(new) != set(self._states_cache) or any(
            new[k].status != self._states_cache[k].status or new[k].hash != self._states_cache[k].hash for k in new)
        self._states_cache = new
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
        self._run_started_at = time.time()
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
