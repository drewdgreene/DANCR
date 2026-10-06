"""MainWindow run methods."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from PySide6.QtCore import Qt, QPointF, QPoint, QSettings, QTimer, QSize, QUrl
from PySide6.QtGui import QAction, QActionGroup, QKeySequence, QCloseEvent, QDesktopServices
from PySide6.QtWidgets import (QMainWindow, QMenu, QFileDialog, QMessageBox, QSplitter, QToolBar, QStatusBar, QInputDialog, QLabel,
                               QProgressDialog, QApplication, QToolButton, QStackedWidget, QWidget, QVBoxLayout, QHBoxLayout,
                               QPushButton, QFrame, QProgressBar, QDialog, QSizePolicy, QCheckBox)

from ..core import registry, PipelineError
from ..core.model import Pipeline
from ..core.samples import write_sample, build_template
from .document import Document
from .common import listen
from .dialogs import Toast, VersionsDialog
from .stepfactory import StepFactory
from .workers import view_pool
from .canvas import CanvasScene, CanvasView, NODE_W, NODE_H
from .inspector import InspectorPanel
from .steppicker import StepPicker
from .rail import Rail, VIEW_TYPES, REPORT_TYPES
from .tableview import TableView
from .chartview import ChartView
from .mapview import MapView
from .reportview import ReportView
from .inputsview import InputsView
from .enterdata import EnterDataView
from .answering import Understanding, AskBar, AnswerPanel
from .assistant import AssistantPanel, SideDock
from .startpage import StartPage
from .insight import InsightBar
from .sources import SourcesTray
from .theme import T
from .icons import icon

from .window_common import DATA_FILTER, FILE_FILTER, log  # noqa: F401


class WindowRun:
    # ------------------------------------------------------------ running
    def run(self, targets: list[str] | None = None, force: bool = False) -> None:
        if self.doc.running:
            self.status.showMessage("A run is already in progress", 4000); return
        probs = self.doc.pipeline.problems()
        if targets:
            need = set(targets)
            for t in targets:
                need |= self.doc.pipeline.upstream_closure(t)
            titles = {self.doc.pipeline.nodes[n].title for n in need if n in self.doc.pipeline.nodes}
            probs = [p for p in probs if p.split(":")[0] in titles]
        if probs:
            self.status.showMessage("Fix first: " + "; ".join(probs[:3]), 10000)
            self.inspector.show_problems(probs)
        self.doc.run(targets, force=force)

    def run_selected(self) -> None:
        cur = self._current if self._current in self.doc.pipeline.nodes else None
        self.run([cur] if cur else None)

    def stop(self) -> None:
        if self.doc.running:
            self.doc.stop()
            self.status.showMessage("Stopping after the current step…", 5000)

    def _on_run_started(self) -> None:
        for a in (self.a_run, self.a_run_sel, self.a_run_force, self.a_clear_cache, self.a_save_as):
            a.setEnabled(False)
        self.a_stop.setEnabled(True); self.a_stop.setVisible(True)
        self._run_total = 0; self._run_index = 0; self._run_durations = []
        self._run_step_start = time.monotonic(); self._run_est = 2.0
        self.progress_bar.setRange(0, 1000); self.progress_bar.setValue(0)
        self.progress_label.setText("Starting…"); self.progress.show(); self._t0 = time.monotonic(); self._tick.start()

    def _run_progress(self, index: int, total: int, title: str) -> None:
        """A step started: the previous one just finished, so use its duration to estimate the next."""
        now = time.monotonic()
        if self._run_total and index > 0:
            self._run_durations.append(max(0.0, now - self._run_step_start))
            recent = self._run_durations[-5:]
            self._run_est = max(0.1, sum(recent) / len(recent))
        self._run_total = max(1, total)
        self._run_index = index
        self._run_step_start = now
        self._progress_text = f"Step {index + 1} of {total}: {title}"
        self._tick_progress()

    def _tick_progress(self) -> None:
        """Advance the bar smoothly: whole steps are exact, and the current step eases toward its end
        on a clock (a step's duration is unknown up front), so the bar never sits still."""
        now = time.monotonic()
        secs = int(now - getattr(self, "_t0", now))
        elapsed = now - self._run_step_start
        frac = min(0.97, elapsed / (elapsed + max(0.1, self._run_est)))
        value = int(1000 * (self._run_index + frac) / max(1, self._run_total))
        self.progress_bar.setValue(max(0, min(1000, value)))
        self.progress_label.setText(f"{self._progress_text}   ·   {secs // 60}:{secs % 60:02d}")

    def _hide_progress(self) -> None:
        if not self.doc.running:
            self.progress.hide()

    def _on_run_finished(self, ok: bool, results: dict) -> None:
        for a in (self.a_run, self.a_run_sel, self.a_run_force, self.a_clear_cache, self.a_save_as):
            a.setEnabled(True)
        self.a_stop.setEnabled(False); self.a_stop.setVisible(False)
        self._tick.stop(); self.progress_bar.setValue(1000)
        QTimer.singleShot(350, self._hide_progress)      # let the full bar be seen, then hide
        self._refresh_mode()
        self._refresh_insight()
        self._on_assistant_run_finished(ok, results)
        failed = [k for k, s in results.items() if s.status == "failed" and k in self.doc.pipeline.nodes]
        if self.doc.last_run_outcome == "stopped":
            self.status.showMessage("Stopped", 5000)
        elif self.doc.last_run_outcome == "crashed":
            pass                                         # "Run failed: …" is already on the status line
        elif ok:
            secs = time.monotonic() - getattr(self, "_t0", time.monotonic())
            self.status.showMessage(f"Done in {secs:.1f} s" if secs >= 1 else "Done", 5000)
        else:
            names = [self.doc.pipeline.nodes[k].title for k in failed]
            self.status.showMessage(f"Could not finish: {', '.join(names[:3])}. Open the step to see why.", 10000)
            if failed and self._current not in failed:
                self.show_node(failed[0])

    def clear_cache(self) -> None:
        if self.doc.running:
            return
        mb = self.doc.executor.cache_size() / 1e6
        if QMessageBox.question(self, "Clear cached results", f"Delete {mb:.0f} MB of cached results? Every step will need to run again.") == QMessageBox.Yes:
            self.doc.executor.clear_cache()
            self.doc.refresh_states()

