"""Document watch methods."""
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
from .doc_common import AUTOSAVE_SECS, AUTO_RUN_DELAY_MS, POLL_MS, SOURCE_SETTLE_MS, log  # noqa: F401


class DocWatch:
    # ------------------------------------------------------------ file
    @property
    def path(self) -> Path | None:
        return self.pipeline.path

    @property
    def dirty(self) -> bool:
        return not self.undo.isClean()

    @staticmethod
    def _watch(watcher: QFileSystemWatcher, paths: list[Path]) -> None:
        """Watch exactly these files; for one that is missing (deleted, or being replaced), its folder instead,
        so the file coming back is noticed however long it was gone."""
        old = watcher.files() + watcher.directories()
        if old:
            watcher.removePaths(old)
        want: set[str] = set()
        for p in paths:
            if p.exists():
                want.add(str(p))
            elif p.parent.is_dir():
                want.add(str(p.parent))
        if want:
            watcher.addPaths(sorted(want))

    def _rewatch(self) -> None:
        self._watch(self._watcher, [self.pipeline.path] if self.pipeline.path else [])
        self._rewatch_sources()

    def _on_project_dir_changed(self, _dir: str) -> None:
        p = self.pipeline.path
        if p is not None and p.exists() and str(p) not in self._watcher.files():
            self._on_file_changed(str(p))           # the project file is back: read it as any other change

    # ------------------------------------------------------------ sources and auto-run
    def source_paths(self) -> list[Path]:
        """The files a project reads, for watching: a path or folder setting, and every member a folder source
        matches (so adding, changing or removing a file in the folder is noticed)."""
        out: list[Path] = []
        seen: set[str] = set()

        def add(p: Path) -> None:
            key = str(p)
            if key not in seen:
                seen.add(key)
                out.append(p)

        for n in self.pipeline.nodes.values():
            nt = registry.get(n.type)
            if nt.kind != "source":
                continue
            for p in nt.params:
                if p.kind in ("path", "dir") and n.params.get(p.name):
                    v = Path(str(n.params[p.name])).expanduser()
                    add(v if v.is_absolute() else self.pipeline.directory / v)
            if nt.source_files is not None:
                try:
                    for f in nt.source_files(self.pipeline.directory, n.params):
                        add(Path(f))
                except Exception:  # noqa: BLE001 - an unreadable folder simply contributes no paths to watch
                    pass
        return out

    def _rewatch_sources(self) -> None:
        self._watch(self._src_watcher, self.source_paths())

    def _on_source_changed(self, path: str) -> None:
        self._source_timer.start()                  # restarted by each event: one refresh once writing stops

    def _on_source_dir_changed(self, _dir: str) -> None:
        watched = set(self._src_watcher.files())
        if any(p.exists() and str(p) not in watched for p in self.source_paths()):
            self._source_timer.start()              # a data file that was missing is there now

    def _source_changed_settle(self) -> None:
        self._rewatch_sources()
        self.refresh_states()
        self.message.emit("A data file changed on disk")
        self.schedule_auto_run()

    def source_bytes(self) -> int:
        # stat-walking a folder source on every keystroke-undo is wasteful (and slow on a network drive): a short
        # cache means a burst of calls within one edit walks once. Disk changes refresh it within a second.
        now = time.monotonic()
        cached = getattr(self, "_src_bytes_cache", None)
        if cached is not None and now - cached[0] < 1.0:
            return cached[1]
        total = 0
        for p in self.source_paths():
            try:
                total += p.stat().st_size
            except OSError:
                pass
        self._src_bytes_cache = (now, total)
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
            self._auto_gen += 1
            self._auto_timer.start()

    def _auto_run_now(self) -> None:
        """Run what is not computed yet. Which steps those are means reading every step's state (files, maybe
        on a network drive), so it is read on a worker; a change made meanwhile asks again once it is read."""
        if not self._auto_pending or self._auto_task is not None or self._closed:
            return
        if self.running:
            self._auto_timer.start(); return
        gen, executor = self._auto_gen, self.executor
        t = Task(self.snapshot_executor().states)
        t.waits_for_run = False

        def decide(states: dict[str, NodeState]) -> None:
            if gen != self._auto_gen or executor is not self.executor or self.running:
                return                              # changed meanwhile: asked again below
            self._auto_pending = False
            todo = [nid for nid in self.pipeline.nodes
                    if nid not in self.held and not (self.held and self.held & self.pipeline.upstream_closure(nid))]
            if any(states[nid].status != "done" for nid in todo if nid in states):
                self.run(todo if self.held else None, auto=True)

        def failed(_msg: str) -> None:
            if gen == self._auto_gen:
                self._auto_pending = False          # logged by the task; the next edit tries again

        def finished() -> None:
            self._auto_task = None
            if self._auto_pending and not self._closed:
                self._auto_timer.start()
        t.signals.done.connect(decide)
        t.signals.failed.connect(failed)
        t.signals.finished.connect(finished)
        self._auto_task = t
        view_pool().start(t)

