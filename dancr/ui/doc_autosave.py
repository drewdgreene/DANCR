"""Document autosave methods."""
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
from .doc_common import (  # noqa: F401
    AUTOSAVE_SECS, AUTO_RUN_DELAY_MS, POLL_MS, SOURCE_SETTLE_MS, ChangedOnDisk, log,
)


class DocAutosave:
    # ------------------------------------------------------------ autosave, recovery, versions
    def set_autosave(self, on: bool) -> None:
        if self.autosave != on:
            self.autosave = on
            self.autosaveChanged.emit(self.autosave_paused)

    def pause_autosave(self, reason: str) -> None:
        if self.autosave_paused != reason:
            self.autosave_paused = reason
            self.autosaveChanged.emit(reason)

    def resume_autosave(self) -> None:
        if self.autosave_paused is not None:
            self.autosave_paused = None
            self.autosaveChanged.emit(None)

    def autosave_now(self) -> None:
        """Once a minute: with autosave on, save quietly when the project has a file (keeping the replaced file
        as an autosave version). Edits that must not go into the file — autosave off, an unsaved project,
        autosave paused, a run in progress, or a dialog open that asks about these very edits (Save changes?
        Revert?) — are copied to the recovery file instead, so a crash or a force-quit loses nothing."""
        self.flush_edits()
        if not self.pipeline.nodes or not self.dirty:
            return
        if (not self.autosave or self.pipeline.path is None or self.autosave_paused or self.running
                or QApplication.activeModalWidget() is not None):
            self.write_recovery()
            return
        try:
            self.save(auto=True)
        except (ChangedOnDisk, ProjectBusy):
            self.write_recovery()            # never over someone else's change; the person decides when saving
            return
        except (OSError, PipelineError):
            log.exception("Autosave of %s failed", self.pipeline.path)
            self.write_recovery()
            return
        self.autosaved.emit()

    def write_recovery(self) -> None:
        """Keep a copy of edits that are not in the project file (an unsaved project, or unsaved changes to a
        saved one) so a crash, a force-quit or a logout loses nothing. Clears the copy when there are none."""
        self.flush_edits()
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
        self.pause_autosave("recovered changes not saved yet" if pipe.path else "recovered project not saved yet")

    def versions(self) -> list[Path]:
        return self.pipeline.versions() if self.pipeline.path else []

    def restore_version(self, version: Path) -> None:
        """Load an earlier saved copy into the window without touching the file on disk."""
        p = Pipeline.from_dict(json.loads(version.read_text(encoding="utf-8")), self.pipeline.path)
        self.replace_pipeline(p)
        self.undo.resetClean()
        self.pause_autosave("an earlier version is open")

