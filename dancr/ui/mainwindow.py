"""Main window: left rail (Tables / Charts / Reports / Inputs), centre page for the selected item,
a Map drawer showing the steps, and a Settings dock on the right."""
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

from .window_common import DATA_FILTER, FILE_FILTER, log  # noqa: F401 - re-exported
from .window_actions import WindowActions
from .window_pages import WindowPages
from .window_answers import WindowAnswers
from .window_session import WindowSession
from .window_build import WindowBuild
from .window_run import WindowRun
from .window_help import WindowHelp
class MainWindow(WindowActions, WindowPages, WindowAnswers, WindowSession, WindowBuild, WindowRun, WindowHelp, QMainWindow):
    def __init__(self, doc: Document | None = None) -> None:
        super().__init__()
        self.setWindowTitle("DANCR")
        self.resize(1440, 900)
        self.settings = QSettings()
        self.doc = doc if doc is not None else Document()
        self.steps = StepFactory(self)
        self._terminating = False
        self._disposed = False
        self._toast_index = -1
        self.scene = CanvasScene(self.doc, self)          # parented: it goes when the window goes (theme switch)
        self.view = CanvasView(self.scene)
        self.rail = Rail(self.doc)
        # result views live in the result drawer, not the centre; the start page is its own full window
        self.pages = QStackedWidget()
        self.pages.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)     # hidden pages must not set the window's minimum width
        self.start = StartPage()
        self.table = TableView(self.doc); self.chart = ChartView(self.doc); self.map = MapView(self.doc)
        self.report = ReportView(self.doc); self.inputs = InputsView(self.doc); self.entry = EnterDataView(self.doc)
        for p in (self.table, self.chart, self.map, self.report, self.inputs, self.entry):
            self.pages.addWidget(p)
        self.understanding = Understanding(self.doc, self)
        self.askbar = AskBar(self.doc, self.understanding)
        self.answer_bar = AnswerPanel(self.doc, self.understanding)
        self.answer_bar.set_answer(None)
        # the canvas is home: the insight bar on top, the graph in the middle, the sources tray below
        self.insight = InsightBar()
        self.sources = SourcesTray(self.doc, self)
        self.canvas_box = QWidget(); cbl = QVBoxLayout(self.canvas_box)
        cbl.setContentsMargins(0, 0, 0, 0); cbl.setSpacing(0)
        cbl.addWidget(self.insight); cbl.addWidget(self.view, 1); cbl.addWidget(self.sources)
        self.toast = Toast(self.canvas_box)
        self.inspector = InspectorPanel(self.doc, self)
        self.assistant = AssistantPanel(self.doc, self.understanding)
        self.side = SideDock(self.inspector, self.assistant)
        self.top_split = QSplitter(Qt.Horizontal)         # rail | canvas | side
        self.top_split.addWidget(self.rail); self.top_split.addWidget(self.canvas_box); self.top_split.addWidget(self.side)
        self.top_split.setStretchFactor(0, 0); self.top_split.setStretchFactor(1, 1); self.top_split.setStretchFactor(2, 0)
        self.top_split.setSizes([210, 760, 330])   # settings open by default; the canvas still leads
        self.top_split.setCollapsible(0, True); self.top_split.setCollapsible(1, False); self.top_split.setCollapsible(2, True)
        # the result drawer (bottom): the answers bar, the selected answer's chips, and the result itself
        self.result_box = QWidget(); self.result_box.setMinimumHeight(180)
        rl = QVBoxLayout(self.result_box); rl.setContentsMargins(0, 0, 0, 0); rl.setSpacing(0)
        result_head = QFrame(); result_head.setStyleSheet(f"QFrame {{ background: {T.bg}; border-top: 1px solid {T.border}; border-bottom: 1px solid {T.border}; }}")
        rh = QHBoxLayout(result_head); rh.setContentsMargins(10, 3, 6, 3)
        self.result_label = QLabel("Result"); self.result_label.setObjectName("muted"); self.result_label.setWordWrap(True)
        self.result_expand = QToolButton(); self.result_expand.setObjectName("quiet"); self.result_expand.setCheckable(True)
        self.result_expand.setIcon(icon("arrows-out", T.muted, 14)); self.result_expand.setToolTip("Expand or restore the result panel")
        self.result_expand.toggled.connect(self._toggle_result_expand)
        self.result_close = QToolButton(); self.result_close.setObjectName("quiet"); self.result_close.setIcon(icon("x", T.muted, 14))
        self.result_close.setToolTip("Hide the result panel"); self.result_close.clicked.connect(lambda: self.a_result.setChecked(False))
        rh.addWidget(self.result_label, 1); rh.addWidget(self.result_expand); rh.addWidget(self.result_close)
        rl.addWidget(result_head); rl.addWidget(self.askbar); rl.addWidget(self.answer_bar); rl.addWidget(self.pages, 1)
        # when the result panel is closed it collapses to this handle, always one click away
        self.result_handle = QToolButton(); self.result_handle.setObjectName("quiet"); self.result_handle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.result_handle.setIcon(icon("table", T.muted, 14)); self.result_handle.setText("Show the result panel (Ctrl+M)")
        self.result_handle.setStyleSheet(f"QToolButton {{ border: none; border-top: 1px solid {T.border}; background: {T.bg}; padding: 3px 10px; text-align: left; }} QToolButton:hover {{ color: {T.accent}; }}")
        self.result_handle.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.result_handle.clicked.connect(lambda: self.a_result.setChecked(True))
        self.result_handle.hide()
        # run strip (full width, above the log bar)
        self.progress = QFrame(); self.progress.setStyleSheet(f"QFrame {{ background: {T.panel}; border-top: 1px solid {T.border}; }}")
        pl_ = QHBoxLayout(self.progress); pl_.setContentsMargins(10, 4, 10, 4); pl_.setSpacing(10)
        self.progress_label = QLabel("")
        self.progress_bar = QProgressBar(); self.progress_bar.setTextVisible(False); self.progress_bar.setFixedHeight(4); self.progress_bar.setRange(0, 0)
        self.stop_btn = QPushButton("Stop"); self.stop_btn.setIcon(icon("stop", T.text, 14)); self.stop_btn.clicked.connect(self.stop)
        pl_.addWidget(self.progress_label, 1); pl_.addWidget(self.progress_bar, 2); pl_.addWidget(self.stop_btn)
        self.progress.hide()
        # a run being torn down: a non-blocking banner, not a modal dialog (the window keeps repainting)
        self.busy_bar = QFrame(); self.busy_bar.setStyleSheet(f"QFrame {{ background: {T.panel}; border-bottom: 1px solid {T.border}; }}")
        bl = QHBoxLayout(self.busy_bar); bl.setContentsMargins(12, 4, 12, 4); bl.setSpacing(10)
        self.busy_label = QLabel("")
        self.busy_progress = QProgressBar(); self.busy_progress.setTextVisible(False); self.busy_progress.setFixedHeight(4); self.busy_progress.setRange(0, 0)
        bl.addWidget(self.busy_label, 1); bl.addWidget(self.busy_progress, 2)
        self.busy_bar.hide()
        self.outer_split = QSplitter(Qt.Vertical)          # canvas on top, result drawer below
        self.outer_split.addWidget(self.top_split)
        self.outer_split.addWidget(self.result_box)
        self.outer_split.setStretchFactor(0, 4); self.outer_split.setStretchFactor(1, 1); self.outer_split.setSizes([660, 240])
        self.outer_split.setCollapsible(0, False); self.outer_split.setCollapsible(1, True)
        self.workspace = QWidget(); cw = QVBoxLayout(self.workspace); cw.setContentsMargins(0, 0, 0, 0); cw.setSpacing(0)
        cw.addWidget(self.busy_bar); cw.addWidget(self.outer_split, 1); cw.addWidget(self.result_handle); cw.addWidget(self.progress)
        self.root = QStackedWidget()
        self.root.addWidget(self.start); self.root.addWidget(self.workspace)
        self._entered = False                              # False until the person leaves the start page for good
        self.setCentralWidget(self.root)
        self.picker = StepPicker(self)
        self._picker_ctx: dict = {}
        self.status = QStatusBar(); self.setStatusBar(self.status)
        self.autosave_label = QLabel(""); self.autosave_label.setObjectName("faint"); self.status.addPermanentWidget(self.autosave_label)
        self.mode_label = QLabel(""); self.mode_label.setObjectName("faint"); self.status.addPermanentWidget(self.mode_label)
        self._current: str | None = None
        self._current_answer: str | None = None
        self._assistant_terminal: str | None = None
        self._assistant_answer: str | None = None
        self._ask_open = False                    # the ask bar shows only when asked for
        self._building: set[str] = set()          # answers queued until every row has been read
        self._tick = QTimer(self); self._tick.setInterval(100); self._tick.timeout.connect(self._tick_progress)
        self._progress_text = ""
        self._run_total = 0
        self._run_index = 0
        self._run_step_start = time.monotonic()
        self._run_est = 2.0
        self._run_durations: list[float] = []
        self._build_actions()
        self._wire()
        self._update_title()
        self._restore_layout()
        self._show_page()
        if self.doc.running:                     # rebuilt (theme switch) during a run: show it, keep Stop working
            self._on_run_started()
        QTimer.singleShot(200, self._maybe_recover)
        QTimer.singleShot(300, self._maybe_first_run)

