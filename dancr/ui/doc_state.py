"""Document state methods."""
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


class DocState:
    # ------------------------------------------------------------ queries
    def state(self, nid: str) -> NodeState:
        st = self._states_cache.get(nid)
        if st is None:
            st = self.executor.state(nid)
            self._states_cache[nid] = st
        return st

    def refresh_states(self) -> None:
        """Re-read every step's state (after an edit, a reload, a run). Any answer read before this call is
        dropped; see ``_read_states``."""
        self._states_token += 1
        self._states_again = self._poll_task is not None
        self._read_states()

    def snapshot_executor(self) -> Executor:
        """An Executor over a copy of the project as it is now, for work off the GUI thread: the live project
        keeps changing under the person's hands, the copy does not. It shares the cache folder."""
        return Executor(Pipeline.from_dict(self.pipeline.to_dict(), self.pipeline.path), self.executor.cache_dir)

    def _poll_states(self) -> None:
        """The periodic check for changes made elsewhere (a CLI run, a source file rewritten)."""
        self._read_states()

    def _read_states(self) -> None:
        """Every step's state means reading files (sources are stat'ed and sampled), which can be slow on a
        network drive, so it runs on a worker over a snapshot of the project, one read at a time; the answer is
        dropped if the project was edited meanwhile, and a refresh asked for during a read gets one more read
        (however many were asked for) once it ends."""
        if self._poll_task is not None or self._closed:
            return
        ex = self.snapshot_executor()
        token, executor = self._states_token, self.executor
        t = Task(ex.states)
        t.waits_for_run = False
        t.signals.done.connect(lambda new: self._apply_states(new) if token == self._states_token and executor is self.executor else None)

        def finished() -> None:
            self._poll_task = None
            if self._states_again:
                self._states_again = False
                self._read_states()
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
        # While a run is in flight this read is over a snapshot taken before it: its hashes are not what the live
        # executor now holds, so writing them would drop the run's own held results from this window's lease and let
        # another process's cache sweep delete them. The run holds its own lease; refresh this window's right after
        # it finishes (`_on_run_done` -> `refresh_states`).
        if not self.running:
            held = {k: st.hash for k, st in new.items()}
            if held != self._held:                   # other processes' cache sweeps keep what this window shows
                self._held = held
                self.executor.hold(held)
        if changed:
            self.statesChanged.emit()

