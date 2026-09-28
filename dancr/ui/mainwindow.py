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
from .reportview import ReportView
from .inputsview import InputsView
from .enterdata import EnterDataView
from .answering import Understanding, AskBar, AnswerPanel
from .startpage import StartPage
from .theme import T
from .icons import icon

FILE_FILTER = "DANCR project (*.json)"
DATA_FILTER = "Data files (*.csv *.tsv *.txt *.dat *.xlsx *.xlsm *.xls *.parquet);;All files (*)"


log = logging.getLogger("dancr.ui")

class MainWindow(QMainWindow):
    def __init__(self, doc: Document | None = None) -> None:
        super().__init__()
        self.setWindowTitle("DANCR")
        self.resize(1440, 900)
        self.settings = QSettings()
        self.doc = doc if doc is not None else Document()
        self.steps = StepFactory(self)
        self._terminating = False
        self._disposed = False
        self._busy_dlg: QProgressDialog | None = None
        self._toast_index = -1
        self.scene = CanvasScene(self.doc, self)          # parented: it goes when the window goes (theme switch)
        self.view = CanvasView(self.scene)
        self.rail = Rail(self.doc)
        self.pages = QStackedWidget()
        self.pages.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)     # hidden pages must not set the window's minimum width
        self.start = StartPage(); self.table = TableView(self.doc); self.chart = ChartView(self.doc)
        self.report = ReportView(self.doc); self.inputs = InputsView(self.doc); self.entry = EnterDataView(self.doc)
        for p in (self.start, self.table, self.chart, self.report, self.inputs, self.entry):
            self.pages.addWidget(p)
        # Top row: project rail | content pages | settings. Below it the map runs the full width,
        # so the graph gets the whole window and the side panels stop at the top of the map.
        centre = QWidget(); cl = QVBoxLayout(centre); cl.setContentsMargins(0, 0, 0, 0); cl.setSpacing(0)
        self.understanding = Understanding(self.doc, self)
        self.askbar = AskBar(self.doc, self.understanding)
        self.answer_bar = AnswerPanel(self.doc, self.understanding)
        self.answer_bar.set_answer(None)
        cl.addWidget(self.askbar)
        cl.addWidget(self.answer_bar)
        cl.addWidget(self.pages, 1)
        self.toast = Toast(centre)
        self.inspector = InspectorPanel(self.doc, self)
        self.top_split = QSplitter(Qt.Horizontal)
        self.top_split.addWidget(self.rail); self.top_split.addWidget(centre); self.top_split.addWidget(self.inspector)
        self.top_split.setStretchFactor(0, 0); self.top_split.setStretchFactor(1, 1); self.top_split.setStretchFactor(2, 0)
        self.top_split.setSizes([250, 820, 370]); self.top_split.setCollapsible(1, False)
        # the map drawer, full width
        self.map_box = QWidget(); self.map_box.setMinimumHeight(140); ml = QVBoxLayout(self.map_box); ml.setContentsMargins(0, 0, 0, 0); ml.setSpacing(0)
        map_head = QFrame(); map_head.setStyleSheet(f"QFrame {{ background: {T.bg}; border-top: 1px solid {T.border}; border-bottom: 1px solid {T.border}; }}")
        mh = QHBoxLayout(map_head); mh.setContentsMargins(10, 3, 6, 3)
        ml_lab = QLabel("The map shows every step in this project, in order. Drag a step to move it, or click one to see it."); ml_lab.setObjectName("muted"); ml_lab.setWordWrap(True)
        self.map_close = QToolButton(); self.map_close.setObjectName("quiet"); self.map_close.setIcon(icon("x", T.muted, 14)); self.map_close.clicked.connect(lambda: self.a_map.setChecked(False))
        mh.addWidget(ml_lab, 1); mh.addWidget(self.map_close)
        ml.addWidget(map_head); ml.addWidget(self.view, 1)
        # when the map is closed it collapses to this handle, so it is always one click away
        self.map_handle = QToolButton(); self.map_handle.setObjectName("quiet"); self.map_handle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.map_handle.setIcon(icon("map-trifold", T.muted, 14)); self.map_handle.setText("Show the map of steps (Ctrl+M)")
        self.map_handle.setStyleSheet(f"QToolButton {{ border: none; border-top: 1px solid {T.border}; background: {T.bg}; padding: 3px 10px; text-align: left; }} QToolButton:hover {{ color: {T.accent}; }}")
        self.map_handle.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.map_handle.clicked.connect(lambda: self.a_map.setChecked(True))
        self.map_handle.hide()
        # run strip (full width, above the log bar)
        self.progress = QFrame(); self.progress.setStyleSheet(f"QFrame {{ background: {T.panel}; border-top: 1px solid {T.border}; }}")
        pl_ = QHBoxLayout(self.progress); pl_.setContentsMargins(10, 4, 10, 4); pl_.setSpacing(10)
        self.progress_label = QLabel("")
        self.progress_bar = QProgressBar(); self.progress_bar.setTextVisible(False); self.progress_bar.setFixedHeight(4); self.progress_bar.setRange(0, 0)
        self.stop_btn = QPushButton("Stop"); self.stop_btn.setIcon(icon("stop", T.text, 14)); self.stop_btn.clicked.connect(self.stop)
        pl_.addWidget(self.progress_label, 1); pl_.addWidget(self.progress_bar, 2); pl_.addWidget(self.stop_btn)
        self.progress.hide()
        self.outer_split = QSplitter(Qt.Vertical)
        self.outer_split.addWidget(self.top_split)
        self.outer_split.addWidget(self.map_box)
        self.outer_split.setStretchFactor(0, 3); self.outer_split.setStretchFactor(1, 1); self.outer_split.setSizes([620, 260])
        self.outer_split.setCollapsible(0, False)
        central = QWidget(); cw = QVBoxLayout(central); cw.setContentsMargins(0, 0, 0, 0); cw.setSpacing(0)
        cw.addWidget(self.outer_split, 1); cw.addWidget(self.map_handle); cw.addWidget(self.progress)
        self.setCentralWidget(central)
        self.picker = StepPicker(self)
        self._picker_ctx: dict = {}
        self.status = QStatusBar(); self.setStatusBar(self.status)
        self.autosave_label = QLabel(""); self.autosave_label.setObjectName("faint"); self.status.addPermanentWidget(self.autosave_label)
        self.mode_label = QLabel(""); self.mode_label.setObjectName("faint"); self.status.addPermanentWidget(self.mode_label)
        self._current: str | None = None
        self._current_answer: str | None = None
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

    # ------------------------------------------------------------ actions
    def _act(self, text: str, icon_name: str | None, shortcut=None, slot=None, tip: str | None = None) -> QAction:
        a = QAction(text, self)
        if icon_name:
            a.setIcon(icon(icon_name, T.text, 16))
        if shortcut:
            a.setShortcuts(shortcut if isinstance(shortcut, list) else [shortcut])
        if slot:
            a.triggered.connect(slot)
        if tip:
            a.setToolTip(tip); a.setStatusTip(tip)
        return a

    def _build_actions(self) -> None:
        mb = self.menuBar()
        file_m = mb.addMenu("&File")
        self.a_new = self._act("&New project", None, QKeySequence.New, self.new_pipeline)
        self.a_open = self._act("&Open project…", None, QKeySequence.Open, self.open_dialog)
        self.a_save = self._act("&Save", "floppy-disk", QKeySequence.Save, self.save, "Save the project (Ctrl+S)")
        self.a_save_as = self._act("Save &as…", None, QKeySequence.SaveAs, self.save_as)
        self.a_versions = self._act("Earlier &versions…", "clock-counter-clockwise", None, self.show_versions, "Go back to an earlier saved version")
        self.a_revert = self._act("Re&vert to saved", None, None, self.revert)
        self.a_autosave = QAction("A&utosave", self, checkable=True)
        self.a_autosave.setToolTip("Save the project to its file every minute. When off, the file changes only when you save.")
        self.a_autosave.setChecked(self.settings.value("autosave", False, type=bool))
        self.a_autosave.triggered.connect(self._set_autosave)
        self.doc.set_autosave(self.a_autosave.isChecked())
        self.a_open_data = self._act("Open data file…", "folder-open", "Ctrl+I", self.add_data_file, "Open a CSV, Excel or Parquet file as a new table (Ctrl+I)")
        self.a_quit = self._act("&Quit", None, QKeySequence.Quit, self.close)
        self.recent_menu = file_m.addMenu("Open &recent")
        self.template_menu = QMenu("New from &template", self)
        from ..core.samples import TEMPLATES
        for t in TEMPLATES:
            self.template_menu.addAction(t["title"], lambda k=t["key"]: self._start_template(k))
        self.examples_menu = QMenu("&Examples", self)
        from ..core.samples import EXAMPLES
        for e in EXAMPLES:
            self.examples_menu.addAction(e["title"], lambda k=e["key"]: self.open_example(k))
        file_m.addAction(self.a_new); file_m.addMenu(self.template_menu); file_m.addMenu(self.examples_menu); file_m.addAction(self.a_open); file_m.addMenu(self.recent_menu)
        file_m.addSeparator(); file_m.addAction(self.a_open_data); file_m.addSeparator()
        file_m.addAction(self.a_save); file_m.addAction(self.a_save_as); file_m.addAction(self.a_autosave)
        file_m.addAction(self.a_versions); file_m.addAction(self.a_revert)
        file_m.addSeparator(); file_m.addAction(self.a_quit)
        self._refresh_recent()

        edit_m = mb.addMenu("&Edit")
        self.a_undo = self._act("&Undo", "arrow-u-up-left", QKeySequence.Undo, self.doc.undo.undo, "Undo (Ctrl+Z)")
        self.a_redo = self._act("&Redo", "arrow-u-up-right", [QKeySequence.Redo, "Ctrl+Y"], self.doc.undo.redo, "Redo (Ctrl+Y)")
        self.doc.undo.canUndoChanged.connect(self.a_undo.setEnabled); self.doc.undo.canRedoChanged.connect(self.a_redo.setEnabled)
        self.doc.undo.undoTextChanged.connect(self._undo_text); self.doc.undo.redoTextChanged.connect(self._redo_text)
        self.a_undo.setEnabled(False); self.a_redo.setEnabled(False)
        self.a_add = self._act("Add &step…", "plus", ["Ctrl+K", "Insert"], lambda: self.open_picker(), "Add a step after the current table (Ctrl+K)")
        self.a_ask = self._act("Ask a question…", "sparkle", "Ctrl+J", self.focus_ask,
                                  "Ask about your data in plain words, or pick an answer DANCR offers (Ctrl+J)")
        self.a_delete = self._act("&Delete step", "trash", None, self.delete_current, "Delete the selected step (Delete in the project list or the map)")
        self.a_dup = self._act("D&uplicate step", "copy", "Ctrl+D", lambda: self.doc.duplicate_nodes(self.scene.selected_node_ids()))
        self.a_note = self._act("Add &note to the map", "note-pencil", "Ctrl+Shift+N", lambda: self._add_note(self.view.mapToScene(self.view.viewport().rect().center())))
        self.a_inputs = self._act("&Inputs (named values)…", "gear", "Ctrl+Shift+I", lambda: self.rail.select("inputs", "inputs"))
        for a in (self.a_undo, self.a_redo):
            edit_m.addAction(a)
        edit_m.addSeparator()
        for a in (self.a_add, self.a_delete, self.a_dup, self.a_note, self.a_inputs):
            edit_m.addAction(a)
        edit_m.insertAction(self.a_delete, self.a_ask)
        run_m = mb.addMenu("&Run")
        self.a_run = self._act("&Run everything", "play", ["Ctrl+R", "F5"], lambda: self.run(), "Compute every step on the full data (Ctrl+R)")
        self.a_run_sel = self._act("Run up to &this step", None, "Ctrl+Shift+R", self.run_selected, "Run the current step and what it needs (Ctrl+Shift+R)")
        self.a_run_force = self._act("Run everything again (ignore cached results)", None, None, lambda: self.run(force=True))
        self.a_stop = self._act("&Stop", "stop", "Ctrl+.", self.stop, "Stop after the current step (Ctrl+.)")   # not Escape: Escape closes find bars and editors
        self.a_stop.setEnabled(False); self.a_stop.setVisible(False)
        self.a_auto = QAction("Run automatically after every change", self, checkable=True)
        self.a_auto.setToolTip("On for small data. Turn it off for very large files, and press Run when you are ready.")
        self.a_auto.triggered.connect(lambda on: self.doc.set_auto_run(on))
        self.a_clear_cache = self._act("Clear cached results…", None, None, self.clear_cache)
        for a in (self.a_run, self.a_run_sel, self.a_run_force, self.a_stop):
            run_m.addAction(a)
        run_m.addSeparator(); run_m.addAction(self.a_auto); run_m.addAction(self.a_clear_cache)

        view_m = mb.addMenu("&View")
        self.a_map = QAction("Show the &map of steps", self, checkable=True, checked=True); self.a_map.setShortcut("Ctrl+M")
        self.a_map.setIcon(icon("map-trifold", T.text, 16)); self.a_map.setToolTip("Show or hide the map of steps (Ctrl+M)")
        self.a_map.toggled.connect(self._toggle_map)
        self.a_settings = QAction("Show &settings", self, checkable=True, checked=True)
        self.a_settings.setIcon(icon("sliders", T.text, 16)); self.a_settings.setShortcut("Ctrl+,")
        self.a_settings.setToolTip("Show or hide the settings panel (Ctrl+,)")
        self.a_settings.toggled.connect(lambda _: self._apply_side_panels())
        self.a_fit = self._act("&Fit the map in view", "arrows-out", "Ctrl+0", self.view.fit_all)
        self.a_zoom_in = self._act("Zoom map in", None, [QKeySequence.ZoomIn, "Ctrl+="], lambda: self.view.zoom_by(1.2))
        self.a_zoom_out = self._act("Zoom map out", None, QKeySequence.ZoomOut, lambda: self.view.zoom_by(1 / 1.2))
        for a in (self.a_map, self.a_settings, self.a_fit, self.a_zoom_in, self.a_zoom_out):
            view_m.addAction(a)
        view_m.addSeparator()
        self.theme_menu = view_m.addMenu("Appearance")
        grp = QActionGroup(self); grp.setExclusive(True)
        from .theme import preference
        current_theme = preference()
        self._theme_actions: dict[str, QAction] = {}
        for value, label in (("system", "Follow the system"), ("light", "Light"), ("dark", "Dark")):
            a = QAction(label, self); a.setCheckable(True)
            a.setChecked(value == current_theme)
            a.triggered.connect(lambda _=False, v=value: self._set_theme(v))
            grp.addAction(a); self.theme_menu.addAction(a)
            self._theme_actions[value] = a

        help_m = mb.addMenu("&Help")
        help_m.addAction(self._act("&User guide", "question", "F1", self.show_help))
        help_m.addAction(self._act("Formula &functions", None, None, lambda: self.show_help("formulas")))
        help_m.addAction(self._act("For AI agents and the command line", None, None, lambda: self.show_help("agents")))
        help_m.addSeparator()
        help_m.addAction(self._act("Show &log file", None, None, self.show_log))
        help_m.addAction(self._act("&About DANCR", None, None, self.about))

        tb = QToolBar("Main"); tb.setObjectName("maintoolbar"); tb.setMovable(False); tb.setIconSize(QSize(16, 16))
        tb.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.addToolBar(tb)
        tb.addAction(self.a_open_data); tb.addAction(self.a_add); tb.addAction(self.a_ask); tb.addSeparator()
        tb.addAction(self.a_run); tb.addAction(self.a_stop); tb.addSeparator()
        tb.addAction(self.a_undo); tb.addAction(self.a_redo); tb.addSeparator()
        tb.addAction(self.a_save); tb.addAction(self.a_versions)
        spacer = QWidget(); spacer.setSizePolicy(spacer.sizePolicy().horizontalPolicy().Expanding, spacer.sizePolicy().verticalPolicy()); tb.addWidget(spacer)
        tb.addAction(self.a_map); tb.addAction(self.a_settings)
        for a in (self.a_undo, self.a_redo, self.a_save, self.a_versions, self.a_stop, self.a_map, self.a_settings):
            btn = tb.widgetForAction(a)
            if isinstance(btn, QToolButton):
                btn.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self.toolbar = tb

    # ------------------------------------------------------------ appearance / rebuild
    def _set_theme(self, value: str) -> None:
        """Switch Light/Dark/System. The manager re-applies the palette and asks the app to rebuild
        this window, so every widget picks up the new colours."""
        from .theme import set_preference, theme_manager
        set_preference(value)
        for v, a in self._theme_actions.items():
            a.setChecked(v == value)
        manager = theme_manager(QApplication.instance())
        if manager is not None:
            manager.apply()          # emits changed -> app rebuilds the window with the new palette
        else:                        # no manager (e.g. a bare window in a test): apply in place
            from .theme import apply_app_style
            apply_app_style(QApplication.instance())

    def capture_ui_state(self) -> dict:
        """Everything worth carrying across a theme rebuild."""
        return {"geometry": self.saveGeometry(), "current": self._current, "answer": self._current_answer,
                "map": self.a_map.isChecked(), "tab": self.table.tabs.currentIndex(),
                "top_split": self.top_split.saveState(), "outer_split": self.outer_split.saveState(),
                "settings": self.a_settings.isChecked()}

    def restore_ui_state(self, state: dict) -> None:
        g = state.get("geometry")
        if g is not None:
            self.restoreGeometry(g)
        if state.get("top_split"):
            self.top_split.restoreState(state["top_split"])
        if state.get("outer_split"):
            self.outer_split.restoreState(state["outer_split"])
        self.a_settings.setChecked(bool(state.get("settings", True)))
        self.a_map.setChecked(bool(state.get("map", True)))
        if state.get("tab"):
            self.table.tabs.setCurrentIndex(int(state["tab"]))
        answer = state.get("answer")
        if answer and self.doc.pipeline.answer(answer) is not None:
            self.show_answer(answer)
            return
        cur = state.get("current")
        if cur == "inputs" or (cur and cur in self.doc.pipeline.nodes):
            self.show_node(cur)
        else:
            self._current = None
            self._show_page()

    def dispose_for_theme(self) -> None:
        """Delete this window without running closeEvent: the Document and its cache now belong to the
        replacement window, so they must not be shut down here."""
        self._disposed = True
        try:
            self._tick.stop()
        except RuntimeError:
            pass
        self.hide()
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.deleteLater()

    def _on_undo_index(self, i: int) -> None:
        """Hide the undo toast when the stack moves past it. Guarded: it can fire during teardown
        after the Toast (a child widget) has already been deleted."""
        try:
            if self.toast.isVisible() and i != self._toast_index:
                self.toast.hide()
        except RuntimeError:
            pass

    def _undo_text(self, t: str) -> None:
        try:
            self.a_undo.setToolTip(f"Undo {t}" if t else "Undo (Ctrl+Z)")
        except RuntimeError:
            pass

    def _redo_text(self, t: str) -> None:
        try:
            self.a_redo.setToolTip(f"Redo {t}" if t else "Redo (Ctrl+Y)")
        except RuntimeError:
            pass

    def _wire(self) -> None:
        d = self.doc
        listen(self, d.dirtyChanged, lambda _: self._update_title())
        listen(self, d.pathChanged, lambda _: self._update_title())
        listen(self, d.autosaveChanged, lambda _: self._update_title())
        listen(self, d.autosaved, lambda: (self._update_title(), self.status.showMessage("Saved automatically", 2000)))
        listen(self, d.undo.indexChanged, self._on_undo_index)
        listen(self, d.reloaded, self._on_reloaded)
        listen(self, d.message, lambda m: self.status.showMessage(m, 8000))
        listen(self, d.busy, self._on_busy)
        listen(self, d.runStarted, self._on_run_started)
        listen(self, d.runProgress, self._run_progress)
        listen(self, d.runFinished, self._on_run_finished)
        listen(self, d.nodeAdded, lambda _: self._show_page())
        listen(self, d.nodeRemoved, self._on_node_removed)
        listen(self, d.autoRunChanged, lambda _: self._refresh_mode())
        for sig in (d.nodeAdded, d.nodeRemoved, d.nodeChanged):
            listen(self, sig, lambda _: self._refresh_mode())
        self.scene.status.connect(lambda m: self.status.showMessage(m, 6000))
        self.scene.selectionChangedTo.connect(self._on_scene_select)
        self.scene.nodeActivated.connect(self.show_node)
        self.scene.addAfterRequested.connect(self._add_after)
        self.scene.runRequested.connect(lambda targets: self.run(targets))
        self.scene.answerActivated.connect(self.show_answer)
        self.scene.answerChangeRequested.connect(self.show_answer)
        self.scene.answerDeleteRequested.connect(self.delete_answer_dialog)
        self.askbar.build.connect(lambda spec: self.build_answer(spec))
        self.askbar.closed.connect(self.close_ask)
        self.answer_bar.change.connect(self.change_answer)
        self.answer_bar.showSteps.connect(self._show_answer_steps)
        self.answer_bar.delete.connect(self.delete_answer_dialog)
        self.answer_bar.rename.connect(self.doc.rename_answer)
        self.view.fileDropped.connect(self._file_dropped)
        self.view.nodeTypeDropped.connect(lambda k, p: self.add_node(k, p))
        self.view.addStepRequested.connect(lambda gp, sp: self.open_picker(gp, sp))
        self.view.addNoteRequested.connect(self._add_note)
        self.view.deleteRequested.connect(self.delete_current)
        self.picker.chosen.connect(self._picker_chose)
        self.inspector.runRequested.connect(lambda nid: self.run([nid]))
        self.rail.selected.connect(self._on_rail_select)
        self.rail.deleteRequested.connect(self.delete_current)
        self.table.runRequested.connect(lambda nid: self.run([nid]))
        self.table.columnAction.connect(self.steps.column_action)
        self.table.cellAction.connect(self.steps.cell_action)
        self.table.chartColumns.connect(self.steps.chart_columns)
        self.chart.addToReport.connect(self.steps.add_to_report)
        self.report.runRequested.connect(lambda nid: self.run([nid]))
        self.start.openProject.connect(self.open_dialog)
        self.start.openFiles.connect(self.add_data_files)
        self.start.openRecent.connect(lambda p: self._confirm_stop_run("open another project") and self.maybe_save() and self.open_path(p))
        self.start.blank.connect(lambda: self.add_node("enter_data", None))
        self.start.example.connect(self.open_example)
        self.start.removeRecent.connect(self._remove_recent)
        self.setAcceptDrops(True)

    # ------------------------------------------------------------ pages and selection
    def _show_page(self) -> None:
        """Pick the page for the current selection (start page when the project is empty)."""
        if self._on_start_page():
            self.start.set_recent(self._recent())
            self.pages.setCurrentWidget(self.start)
            self.map_box.setVisible(False); self.map_handle.setVisible(False)
            self.askbar.setVisible(False)
            self._apply_side_panels()
            return
        self._apply_side_panels()
        self.askbar.setVisible(self._ask_open)       # only when asked for: Ask a question, or an answer selected
        self._apply_map_visibility()
        nid = self._current
        if nid == "inputs":
            self.pages.setCurrentWidget(self.inputs); return
        if nid is None or nid not in self.doc.pipeline.nodes:
            self.pages.setCurrentWidget(self.table)
            if self.table.nid is not None:
                self.table.set_node(None)
            return
        t = self.doc.pipeline.nodes[nid].type
        page = {"chart": self.chart, "report": self.report, "enter_data": self.entry}.get(t, self.table)
        if page.nid != nid:                     # the same step again: keep what is shown, no re-query
            page.set_node(nid)
        self.pages.setCurrentWidget(page)

    def show_node(self, nid: str | None) -> None:
        if nid != self._current:
            self._current_answer = None
            self.scene.clear_highlight()
            self.answer_bar.set_answer(None)
        if nid == self._current:
            self._show_page(); return
        self._current = nid
        self._focus_answers(nid)
        self.inspector.set_node(nid if nid != "inputs" else None)
        if nid and nid != "inputs":
            self.rail.select("node", nid, emit=False)
            if self.scene.selected_node_ids() != [nid]:
                self.scene.select_node(nid)
        elif nid == "inputs":
            self.rail.select("inputs", "inputs", emit=False)
        self._show_page()
        self._refresh_mode()

    def _on_rail_select(self, kind: str, ident) -> None:
        if kind == "node":
            self.show_node(ident)
            self.view.focus_node(ident)      # picking a step in the rail centres the map on it
        elif kind == "inputs":
            self.show_node("inputs")
        elif kind == "answer":
            self.show_answer(ident)

    def _on_scene_select(self, nid: str | None) -> None:
        if nid is not None:
            self.show_node(nid)

    def _on_node_removed(self, nid: str) -> None:
        if nid == self._current:
            self._current = None
            self.inspector.set_node(None)
        self._show_page()

    # ------------------------------------------------------------ answers
    def focus_ask(self) -> None:
        """Ask a question: opens the ask bar (or closes it when it is already open)."""
        if not self.doc.pipeline.nodes:
            self.add_data_files(); return
        if self._ask_open and self.askbar.isVisible():
            self.close_ask(); return
        self.open_ask(focus=True)

    def open_ask(self, focus: bool = False) -> None:
        self._ask_open = True
        if self.doc.pipeline.nodes:
            self.askbar.setVisible(True)
        if focus:
            self.askbar.focus_edit()

    def close_ask(self) -> None:
        self._ask_open = False
        self.askbar.setVisible(False)

    def build_answer(self, spec: dict, answer_id: str | None = None) -> None:
        """Build (or change) an answer once every row of the tables has been read: answers are never planned
        from a sample. Shows the answer when it is built."""
        from ..core.recipes import PlanError
        same = self._same_answer(spec) if answer_id is None else None
        if same is not None:                       # asked again (or a double click): the answer is already here
            self.show_answer(same)
            return
        key = json.dumps([spec, answer_id], sort_keys=True, default=str)
        if key in self._building:
            return
        self._building.add(key)
        if not self.understanding.full:
            self.askbar.show_status("Reading every row of your tables first…")

        def go(model) -> None:
            self._building.discard(key)
            if self._disposed:
                return
            if answer_id is None and self._same_answer(spec) is not None:
                self.show_answer(self._same_answer(spec)); return
            try:
                aid = self.doc.build_answer(model, spec, answer_id)
            except (PlanError, KeyError) as e:
                self.askbar.show_status(str(e).strip("'\""), error=True)     # inline, never a dialog out of the blue
                return
            self.askbar.show_status("")
            self.doc.schedule_auto_run()
            self.show_answer(aid)
            a = self.doc.pipeline.answer(aid)
            if a is not None and a.terminal in self.scene.nodes:
                QTimer.singleShot(0, lambda: self.view.reveal(a.terminal) if not self._disposed else None)
            a = self.doc.pipeline.answer(aid)
            if a is not None and not self.doc.auto_run and not self.doc.running:
                self.run([a.terminal])
            from ..core.answers import set_aside_note
            note = set_aside_note(self.doc.last_set_aside)
            if note:
                self.toast.show_message(note, "OK", lambda: None)
            self.status.showMessage(note or "Built the answer. Change its choices above, or ask another question", 8000)
        self.understanding.when_full(go)

    def _same_answer(self, spec: dict) -> str | None:
        want = json.dumps({k: v for k, v in spec.items() if k != "title"}, sort_keys=True, default=str)
        for a in self.doc.pipeline.answers:
            if a.terminal in self.doc.pipeline.nodes and json.dumps({k: v for k, v in a.spec.items() if k != "title"},
                                                                     sort_keys=True, default=str) == want:
                return a.id
        return None

    def change_answer(self, aid: str, key: str, value) -> None:
        from ..core.recipes import apply_choice
        a = self.doc.pipeline.answer(aid)
        if a is not None:
            self.build_answer(apply_choice(a.spec, key, value), aid)

    def _show_answer_steps(self) -> None:
        a = self.doc.pipeline.answer(self._current_answer) if self._current_answer else None
        if not self.a_map.isChecked():
            self.a_map.setChecked(True)
        if a is not None and a.terminal in self.doc.pipeline.nodes:
            self.view.focus_node(a.terminal)
        else:
            self.view.fit_all()

    def _focus_answers(self, nid: str | None) -> None:
        """The tray offers answers about the table being looked at; a chart or report is not a table to ask about."""
        n = self.doc.pipeline.nodes.get(nid) if nid else None
        self.understanding.set_focus(nid if n is not None and registry.get(n.type).kind != "sink" and n.type != "chart" else None)

    def show_answer(self, aid: str) -> None:
        """Select an Answer: show its result in the centre and highlight its branch on the map."""
        answer = self.doc.pipeline.answer(aid)
        if answer is None:
            return
        self.understanding.set_focus(None)
        self.open_ask()
        self._current_answer = aid
        self._current = answer.terminal if answer.terminal in self.doc.pipeline.nodes else None
        self.rail.select("answer", aid, emit=False)
        self.scene.select_answer(aid)
        if self._current:
            self.scene.highlight_branch(self.doc.pipeline.upstream_closure(self._current) | {self._current})
            self.inspector.set_node(self._current)
        else:
            self.scene.clear_highlight()
        self._sync_answer_bar()
        self._show_page()

    def delete_answer_dialog(self, aid: str) -> None:
        answer = self.doc.pipeline.answer(aid)
        if answer is None:
            return
        exclusive = self.doc.answer_exclusive_nodes(answer)
        box = QMessageBox(self)
        box.setWindowTitle("Delete answer")
        box.setIcon(QMessageBox.Question)
        box.setText(f"Delete “{answer.title}”?")
        cb = QCheckBox("Also remove the steps it built")
        if exclusive:
            cb.setText(f"Also remove the {len(exclusive)} step(s) it built")
        else:
            cb.setText("Its steps are shared with other answers, so they are kept")
            cb.setEnabled(False)
        box.setCheckBox(cb)
        box.setStandardButtons(QMessageBox.Cancel | QMessageBox.Yes)
        box.setDefaultButton(QMessageBox.Cancel)
        if box.exec() != QMessageBox.Yes:
            return
        self.doc.delete_answer(aid, remove_steps=cb.isChecked())
        if self._current_answer == aid:
            self._current_answer = None
            self.scene.clear_highlight()
            self.answer_bar.set_answer(None)
            self._current = None
            self._show_page()

    def _sync_answer_bar(self) -> None:
        answer = self.doc.pipeline.answer(self._current_answer) if self._current_answer else None
        self.answer_bar.set_answer(answer.id if answer is not None else None)

    def current_table(self) -> str | None:
        """The table the person is looking at (charts and reports resolve to their input)."""
        nid = self._current
        if nid in (None, "inputs") or nid not in self.doc.pipeline.nodes:
            return None
        n = self.doc.pipeline.nodes[nid]
        if n.type in VIEW_TYPES or n.type in REPORT_TYPES:
            ins = self.doc.pipeline.inputs_of(nid)
            for lst in ins.values():
                if lst:
                    return lst[0]
            return None
        return nid

    def _on_start_page(self) -> bool:
        return not self.doc.pipeline.nodes and self._current is None

    def _apply_side_panels(self) -> None:
        """The project list and the settings panel have nothing to show until the project has a step."""
        start = self._on_start_page()
        self.rail.setVisible(not start)
        self.inspector.setVisible(not start and self.a_settings.isChecked())
        for a in (self.a_settings, self.a_map, self.a_add, self.a_ask, self.a_run):
            a.setEnabled(not start)

    def _toggle_map(self, on: bool) -> None:
        self._apply_map_visibility()
        if on:
            QTimer.singleShot(0, self.view.fit_all)

    def _apply_map_visibility(self) -> None:
        have = bool(self.doc.pipeline.nodes) or self._current is not None
        on = self.a_map.isChecked()
        self.map_box.setVisible(on and have)
        self.map_handle.setVisible(have and not on)
        if on and have:
            sizes = self.outer_split.sizes()
            if len(sizes) == 2 and sizes[1] < 120:          # reopened into a collapsed slot
                total = sum(sizes) or self.outer_split.height()
                self.outer_split.setSizes([max(200, total - 260), 260])

    def _refresh_mode(self) -> None:
        auto = self.doc.auto_run and bool(self.doc.pipeline.nodes)
        self.a_auto.blockSignals(True); self.a_auto.setChecked(self.doc.auto_run); self.a_auto.blockSignals(False)
        self.a_run.setVisible(not auto or self.doc.running)
        self.a_run_sel.setEnabled(not auto)
        if not self.doc.pipeline.nodes:
            self.mode_label.setText("")
        elif auto:
            self.mode_label.setText("runs automatically")
        else:
            mb = self.doc.source_bytes() / 1e6
            self.mode_label.setText(f"large data ({mb:,.0f} MB), press Run to compute")
        if self.table.nid:
            self.table._refresh_header()

    def delete_current(self) -> None:
        aids = self.scene.selected_answer_ids()
        if aids:
            self.delete_answer_dialog(aids[0])
            return
        if not self.scene.selected_node_ids() and self.scene.selectedItems():
            self.scene.delete_selection()          # an arrow or a note is selected on the map: that is what goes
            return
        ids = self.scene.selected_node_ids() or ([self._current] if self._current and self._current in self.doc.pipeline.nodes else [])
        if not ids:
            return
        titles = [self.doc.pipeline.nodes[i].title for i in ids]
        if self.scene.selected_node_ids():
            self.scene.delete_selection()          # nodes plus any selected arrows and notes, one undo entry
        else:
            self.doc.remove_nodes(ids)
        idx = self._toast_index = self.doc.undo.index()
        self.toast.show_message(f"Deleted {titles[0] if len(titles) == 1 else f'{len(titles)} steps'}", "Undo",
                                lambda: self.doc.undo.undo() if self.doc.undo.index() == idx else None)

    # ------------------------------------------------------------ file ops
    def _set_autosave(self, on: bool) -> None:
        """Autosave is the person's choice for every project (remembered between sessions); off to begin with."""
        self.settings.setValue("autosave", on)
        self.doc.set_autosave(on)
        self.status.showMessage("Autosave on: the project is saved every minute" if on
                                else "Autosave off: the project file changes only when you save", 4000)

    def _update_title(self) -> None:
        name = self.doc.path.stem if self.doc.path else "Untitled"
        paused = self.doc.autosave_paused if self.doc.autosave else None
        self.setWindowTitle(f"{'• ' if self.doc.dirty else ''}{name}{' (autosave paused)' if paused else ''} — DANCR")
        self.autosave_label.setText(f"Autosave paused: {paused}" if paused else "")
        self.a_revert.setEnabled(self.doc.path is not None and self.doc.dirty)

    def _on_reloaded(self) -> None:
        self._building.clear()
        self._update_title()
        self._current = None
        self._current_answer = None
        self.scene.clear_highlight()
        self.answer_bar.set_answer(None)
        self.inspector.set_node(None)
        QTimer.singleShot(0, self.view.fit_all)
        self.doc.schedule_auto_run()
        if self.doc.path:
            self._push_recent(self.doc.path)
        first = next(iter(self.doc.pipeline.topological_order()), None) if self.doc.pipeline.nodes else None
        if first:
            self.show_node(first)
        else:
            self._show_page()
        self._refresh_mode()

    def maybe_save(self) -> bool:
        self.doc.flush_edits()
        if not self.doc.dirty:
            return True
        if self.doc.autosave_paused:
            why = f"Autosave is paused ({self.doc.autosave_paused})" if self.doc.autosave else self.doc.autosave_paused.capitalize()
            what = f"{why}. Save replaces {self.doc.path.name if self.doc.path else 'nothing'} with what you see now; Discard keeps the file on disk as it is."
        else:
            what = "Save changes to this project?"
        r = QMessageBox.question(self, "Unsaved changes", what, QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        if r == QMessageBox.Save:
            return self.save()
        return r == QMessageBox.Discard

    def _confirm_stop_run(self, what: str) -> bool:
        if not self.doc.running:
            return True
        r = QMessageBox.question(self, "A run is in progress", f"Stop the run and {what}?", QMessageBox.Yes | QMessageBox.No)
        if r != QMessageBox.Yes:
            return False
        self.doc.stop(wait=True)
        return True

    def _on_busy(self, why: str | None) -> None:
        """While the document waits for a run to stop, a window-modal note says why; it takes every click, so
        nothing can open, save or close the project while the run is torn down."""
        if self._busy_dlg is not None:
            self._busy_dlg.close(); self._busy_dlg.deleteLater(); self._busy_dlg = None
        if why:
            dlg = self._busy_dlg = QProgressDialog(why, None, 0, 0, self)
            dlg.setWindowTitle("DANCR"); dlg.setWindowModality(Qt.WindowModal); dlg.setMinimumDuration(0)
            dlg.show(); dlg.setValue(0)

    def new_pipeline(self) -> None:
        if self._confirm_stop_run("start a new project") and self.maybe_save():
            self.doc.new()

    def open_dialog(self) -> None:
        if not self._confirm_stop_run("open another project") or not self.maybe_save():
            return
        start = str(self.doc.path.parent) if self.doc.path else str(Path.home())
        f, _ = QFileDialog.getOpenFileName(self, "Open project", start, FILE_FILTER)
        if f:
            self.open_path(f)

    def open_path(self, path: str | Path) -> None:
        try:
            self.doc.load(path)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Cannot open", str(e))
            return
        self.status.showMessage(f"Opened {path}", 5000)

    def save(self) -> bool:
        if self.doc.path is None:
            return self.save_as()
        from .document import ChangedOnDisk
        try:
            self.doc.save()
        except ChangedOnDisk:
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Warning)
            box.setWindowTitle("The project changed on disk")
            box.setText(f"{self.doc.path.name} was changed by another program (an agent, the command line or another "
                        "window) since it was opened here.")
            box.setInformativeText("Keep mine overwrites their change (it stays under File → Earlier versions). "
                                   "Load theirs throws away the unsaved changes in this window.")
            mine = box.addButton("Keep mine", QMessageBox.AcceptRole)
            theirs = box.addButton("Load theirs", QMessageBox.DestructiveRole)
            box.addButton(QMessageBox.Cancel)
            box.exec()
            if box.clickedButton() is mine:
                try:
                    self.doc.save(overwrite=True)
                except (OSError, PipelineError) as e:
                    QMessageBox.critical(self, "Cannot save", str(e)); return False
            elif box.clickedButton() is theirs:
                self.doc.load(self.doc.path)
                return True
            else:
                return False
        except (OSError, PipelineError) as e:
            QMessageBox.critical(self, "Cannot save", str(e)); return False
        self.status.showMessage(f"Saved {self.doc.path.name}", 3000)
        self._update_title()
        return True

    def save_as(self) -> bool:
        if self.doc.running:
            QMessageBox.information(self, "Save as", "Wait for the run to finish before saving under a new name.")
            return False
        start = str(self.doc.path) if self.doc.path else str(Path.home() / "project.json")
        f, _ = QFileDialog.getSaveFileName(self, "Save project as", start, FILE_FILTER)
        if not f:
            return False
        if not f.lower().endswith(".json"):
            f += ".json"
        try:
            self.doc.save(f)
        except (OSError, PipelineError) as e:
            QMessageBox.critical(self, "Cannot save", str(e)); return False
        self._push_recent(Path(f))
        self._update_title()
        self.status.showMessage(f"Saved {f}", 3000)
        return True

    def _maybe_recover(self) -> None:
        if self._disposed or self.doc.pipeline.nodes:
            return
        found = self.doc.pending_recovery()
        if found is None:
            return
        pipe, rp = found
        if pipe.path is not None:
            text = (f"Last time DANCR closed with unsaved changes to {pipe.path.name}. Bring them back? "
                    "The file itself is unchanged until you save.")
        else:
            text = f"Last time DANCR closed with an unsaved project of {len(pipe.nodes)} steps. Bring it back?"
        r = QMessageBox.question(self, "Unsaved changes", text, QMessageBox.Yes | QMessageBox.No)
        if r == QMessageBox.Yes:
            self.doc.recover(pipe, rp)
            self.status.showMessage("Recovered. Save it to keep it", 8000)
        else:
            rp.unlink(missing_ok=True)

    def revert(self) -> None:
        if self.doc.path and self._confirm_stop_run("reload from disk") and \
                QMessageBox.question(self, "Revert", "Discard your changes and reload from disk?") == QMessageBox.Yes:
            self.open_path(self.doc.path)

    def show_versions(self) -> None:
        if self.doc.path is None:
            QMessageBox.information(self, "Earlier versions", "Save the project first. From then on every save keeps an earlier version you can go back to.")
            return
        dlg = VersionsDialog(self, self.doc)
        if dlg.exec() == QDialog.Accepted and dlg.chosen() and self._confirm_stop_run("restore an earlier version"):
            try:
                self.doc.restore_version(dlg.chosen())
            except Exception as e:  # noqa: BLE001
                QMessageBox.critical(self, "Cannot restore", str(e)); return
            self.status.showMessage("Restored. Save to keep it, or Revert to go back."
                                    + (" Autosave is paused until then." if self.doc.autosave else ""), 8000)

    def terminate(self) -> None:
        """The system asked us to quit (SIGTERM): no questions, keep an unsaved project recoverable, close cleanly."""
        self._terminating = True
        self.close()
        if self.isVisible():
            QApplication.quit()

    def closeEvent(self, e: QCloseEvent) -> None:
        if self._terminating:
            self.doc.stop(wait=True, why="Stopping the current step before quitting…")
            self.doc.write_recovery()             # unsaved edits of any project; offered back on the next start
        else:
            if self.doc.running:
                r = QMessageBox.question(self, "A run is in progress", "Stop the run and quit?", QMessageBox.Yes | QMessageBox.No)
                if r != QMessageBox.Yes:
                    e.ignore(); return
                self.doc.stop(wait=True, why="Stopping the current step before quitting…")
            if not self.maybe_save():
                e.ignore(); return
            self.doc.clear_recovery()             # saved or deliberately discarded
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("top_split", self.top_split.saveState())
        self.settings.setValue("outer_split", self.outer_split.saveState())
        self.settings.setValue("map", self.a_map.isChecked())
        view_pool().shutdown(5000)                # nothing may still read the results when the cache is removed
        self.doc.shutdown()
        e.accept()

    def _restore_layout(self) -> None:
        g = self.settings.value("geometry")
        if g:
            self.restoreGeometry(g)
        t = self.settings.value("top_split")
        if t:
            self.top_split.restoreState(t)
        o = self.settings.value("outer_split")
        if o:
            self.outer_split.restoreState(o)
        if self.inspector.width() < 260:                    # a saved layout that squeezed it too far
            sizes = self.top_split.sizes()
            if len(sizes) == 3:
                self.top_split.setSizes([sizes[0], max(300, sizes[1] - 120), 360])
        m = self.settings.value("map")
        if m is not None:
            self.a_map.setChecked(m in (True, "true", "True", 1))

    def _recent(self) -> list[str]:
        v = self.settings.value("recent")
        if isinstance(v, str):
            return [v] if v else []
        return [str(x) for x in (v or [])]

    def _remove_recent(self, p: str) -> None:
        self.settings.setValue("recent", [r for r in self._recent() if r != p])
        self._refresh_recent()
        if self._on_start_page():
            self.start.set_recent(self._recent())

    def _push_recent(self, p: Path) -> None:
        rec = [str(p)] + [r for r in self._recent() if r != str(p)]
        self.settings.setValue("recent", rec[:10])
        self._refresh_recent()

    def _refresh_recent(self) -> None:
        self.recent_menu.clear()
        for r in self._recent():
            self.recent_menu.addAction(QAction(r, self, triggered=lambda checked=False, p=r: self._confirm_stop_run("open another project") and self.maybe_save() and self.open_path(p)))

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self.toast.reposition()

    # ------------------------------------------------------------ drag and drop onto the whole window
    def dragEnterEvent(self, e) -> None:
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e) -> None:
        paths = [u.toLocalFile() for u in e.mimeData().urls() if u.toLocalFile()]
        projects = [p for p in paths if p.lower().endswith(".json")]
        data = [p for p in paths if not p.lower().endswith(".json")]
        if projects:
            if self._confirm_stop_run("open another project") and self.maybe_save():
                self.open_path(projects[0])
            e.acceptProposedAction(); return
        self._add_files(data)
        e.acceptProposedAction()

    # ------------------------------------------------------------ building
    def open_picker(self, global_pos: QPoint | None = None, scene_pos: QPointF | None = None, after: str | None = None) -> None:
        self._picker_ctx = {"scene_pos": scene_pos, "after": after or self.current_table()}
        if global_pos is None:
            btn = self.toolbar.widgetForAction(self.a_add)
            global_pos = btn.mapToGlobal(QPoint(0, btn.height())) if btn else self.mapToGlobal(QPoint(200, 80))
        self.picker.open_at(global_pos)

    def _add_after(self, nid: str, screen_pos: QPoint) -> None:
        self.scene.select_node(nid)
        self.open_picker(screen_pos, None, after=nid)

    def _picker_chose(self, type_key: str) -> None:
        ctx, self._picker_ctx = self._picker_ctx, {}
        self.add_node(type_key, ctx.get("scene_pos"), connect_from=ctx.get("after"))

    def _free_position(self, near: str | None = None) -> QPointF:
        if near and near in self.scene.nodes:
            it = self.scene.nodes[near]
            x, y = it.x() + NODE_W + 90, it.y()
        elif self.scene.nodes:
            x = max(i.x() for i in self.scene.nodes.values()) + NODE_W + 90
            y = min(i.y() for i in self.scene.nodes.values())
        else:
            x, y = 60.0, 200.0
        taken = [(i.x(), i.y()) for i in self.scene.nodes.values()]
        while any(abs(tx - x) < NODE_W and abs(ty - y) < NODE_H for tx, ty in taken):
            y += NODE_H + 30
        return QPointF(x, y)

    def _source_position(self) -> QPointF:
        """Tables go in the first column of the map, one under another."""
        p = self.doc.pipeline
        sources = [n for n in p.nodes.values() if registry.get(n.type).kind == "source"]
        if not sources:
            return self._free_position(None) if p.nodes else QPointF(60.0, 140.0)
        x = min(n.x for n in sources)
        y = max(n.y for n in sources) + NODE_H + 40
        while any(abs(n.x - x) < NODE_W and abs(n.y - y) < NODE_H for n in p.nodes.values()):
            y += NODE_H + 30
        return QPointF(x, y)

    def add_node(self, type_key: str, pos: QPointF | None = None, params: dict | None = None, title: str | None = None,
                 connect_from: str | None = None, port: str | None = None, show: bool = True) -> str:
        nt = registry.get(type_key)
        if connect_from is None and nt.inputs:
            connect_from = self.current_table()
        if not nt.inputs:
            connect_from = None
        if pos is None:
            pos = self._free_position(connect_from)
        nid = self.doc.add_node(type_key, pos.x(), pos.y(), params=params, title=title, connect_from=connect_from, port=port)
        if show:
            self.show_node(nid)
            self.view.reveal(nid)
        self.doc.schedule_auto_run()
        return nid

    def _add_note(self, pos: QPointF) -> None:
        text, ok = QInputDialog.getMultiLineText(self, "Note", "Text:", "")
        if ok and text.strip():
            self.doc.add_note(text.strip(), pos.x(), pos.y())

    def _file_dropped(self, path: str, pos: QPointF) -> None:
        self._add_load_node(path, pos)

    def add_data_file(self) -> None:
        self.add_data_files()

    def add_data_files(self) -> None:
        start = str(self.doc.path.parent) if self.doc.path else str(Path.home())
        files, _ = QFileDialog.getOpenFileNames(self, "Open data files", start, DATA_FILTER)
        self._add_files(files)

    def _add_files(self, files: list[str]) -> None:
        """Load each file (as one undo step), show the first, and let the tray offer answers about them all. A big
        workbook is looked through for its tables on a worker, so the window stays responsive meanwhile."""
        if not files:
            return
        from ..core.nodes.load import tables_in, EXCEL_EXT

        def listed() -> list[tuple[str, str, dict]]:
            return [(f, title, extra) for f in files for title, extra in tables_in(f)]

        def big(f: str) -> bool:
            try:
                return Path(f).suffix.lower() in EXCEL_EXT and Path(f).stat().st_size > 2_000_000
            except OSError:
                return False
        if not any(big(f) for f in files):
            self._add_listed(listed())
            return
        from .workers import Task
        self.status.showMessage(f"Looking through {Path(next(f for f in files if big(f))).name}…")
        t = Task(listed)
        t.waits_for_run = False                        # reads only the files dropped, never a result being written
        pending = getattr(self, "_listing", None)
        if pending is None:
            pending = self._listing = set()
        pending.add(t)                                 # kept until it reports, so every drop is added, in order
        t.signals.done.connect(lambda r, t=t: (pending.discard(t), self._add_listed(r)))
        t.signals.failed.connect(lambda m, t=t: (pending.discard(t), self.status.showMessage(f"Could not open the files: {m}", 8000)))
        view_pool().start(t)

    def _add_listed(self, tables: list[tuple[str, str, dict]]) -> None:
        if not tables:
            return
        with self.doc.macro("Open data" if len(tables) == 1 else f"Open {len(tables)} tables"):
            ids = [self._add_load_node(f, None, run=False, show=False, extra=extra, title=title) for f, title, extra in tables]
        if not self.doc.auto_run and not self.doc.running:
            self.run(ids)
        self.show_node(ids[0])
        if len(ids) > 1:
            self.status.showMessage(f"Opened {len(ids)} tables. The answers above cover all of them", 8000)

    def _add_load_node(self, path: str, pos: QPointF | None, run: bool = True, show: bool = True,
                       extra: dict | None = None, title: str | None = None) -> str:
        p = Path(path)
        rel = p
        if self.doc.path:
            try:
                rel = p.relative_to(self.doc.path.parent)
            except ValueError:
                rel = p
        self.scene.clearSelection()
        if pos is None:
            pos = self._source_position()
        params = {"path": str(rel), **{k: v for k, v in (extra or {}).items() if v not in (None, "")}}
        nid = self.add_node("load_file", pos, params=params, title=title or p.stem, connect_from=None, show=show)
        if run and not self.doc.auto_run and not self.doc.running:
            self.run([nid])
        elif self.doc.running:
            self.status.showMessage(f"Added {p.name}. It will load when you next run.", 6000)
        return nid

    def open_example(self, key: str) -> None:
        """Open an example project, making it first if it is not in ~/DANCR samples yet."""
        from ..core.samples import write_example, example
        if not (self._confirm_stop_run("open the example") and self.maybe_save()):
            return
        try:
            path = write_example(key, Path.home() / "DANCR samples" / example(key)["title"])
        except Exception as e:  # noqa: BLE001
            log.exception("Could not make the example project")
            QMessageBox.critical(self, "Example project", f"Couldn't make the example project: {e}"); return
        self.open_path(str(path))

    def _start_template(self, key: str) -> None:
        # a template is a new project: it never replaces the open one's file (even one whose steps were all deleted)
        if not self.maybe_save():
            return
        base = self.doc.path.parent if self.doc.path else Path.home() / "DANCR samples"
        try:
            data = write_sample(base)
        except OSError as e:
            QMessageBox.critical(self, "Sample data", f"Could not write the sample file: {e}"); return
        pipe = Pipeline("Untitled")
        try:
            build_template(key, pipe, data)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Template", str(e)); return
        self.doc.replace_pipeline(pipe)             # the files it saves wait for the project to be saved: they go next to it
        self.doc.undo.resetClean()
        self.doc.schedule_auto_run()
        self.status.showMessage(f"Sample data saved to {data}", 8000)

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

    # ------------------------------------------------------------ help
    def show_help(self, section: str = "") -> None:
        from .helpdialog import HelpDialog
        HelpDialog(self, section).show()

    def show_log(self) -> None:
        from ..logsetup import log_path
        lp = log_path()
        if lp.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(lp)))
        self.status.showMessage(f"Log file: {lp}", 10000)

    def about(self) -> None:
        from .. import __version__
        QMessageBox.about(self, "About DANCR",
                          f"<b>DANCR {__version__}</b><br>Data Analysis Node-based Canvas for Research<br><br>"
                          "Designed and developed by Drew Greene. Commissioned by Jens Dancer.<br>"
                          "Engine: Polars. UI: Qt / PySide6 / pyqtgraph. Icons: Phosphor (MIT).")
