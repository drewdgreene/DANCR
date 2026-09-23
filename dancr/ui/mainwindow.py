"""Main window: left rail (Tables / Charts / Reports / Inputs), centre page for the selected item,
a Map drawer showing the steps, and a Settings dock on the right."""
from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtCore import Qt, QPointF, QPoint, QSettings, QTimer, QSize, QUrl
from PySide6.QtGui import QAction, QActionGroup, QKeySequence, QCloseEvent, QDesktopServices
from PySide6.QtWidgets import (QMainWindow, QFileDialog, QMessageBox, QSplitter, QToolBar, QStatusBar, QInputDialog, QLabel,
                               QProgressDialog, QApplication, QToolButton, QStackedWidget, QWidget, QVBoxLayout, QHBoxLayout,
                               QPushButton, QFrame, QProgressBar, QDialog, QSizePolicy)

from ..core import registry, PipelineError
from ..core.model import Pipeline
from ..core.samples import write_sample, build_template
from .document import Document
from .dialogs import Toast, VersionsDialog
from .stepfactory import StepFactory
from .workers import view_pool
from .canvas import CanvasScene, CanvasView, NODE_W, NODE_H
from .inspector import InspectorDock
from .steppicker import StepPicker
from .rail import Rail, VIEW_TYPES, REPORT_TYPES
from .tableview import TableView
from .chartview import ChartView
from .reportview import ReportView
from .inputsview import InputsView
from .enterdata import EnterDataView
from .startpage import StartPage
from .theme import T
from .icons import icon

FILE_FILTER = "DANCR project (*.json)"
DATA_FILTER = "Data files (*.csv *.tsv *.txt *.dat *.xlsx *.xlsm *.xls *.parquet);;All files (*)"


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
        self._toast_index = -1
        self.scene = CanvasScene(self.doc)
        self.view = CanvasView(self.scene)
        self.rail = Rail(self.doc)
        self.pages = QStackedWidget()
        self.pages.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)     # hidden pages must not set the window's minimum width
        self.start = StartPage(); self.table = TableView(self.doc); self.chart = ChartView(self.doc)
        self.report = ReportView(self.doc); self.inputs = InputsView(self.doc); self.entry = EnterDataView(self.doc)
        for p in (self.start, self.table, self.chart, self.report, self.inputs, self.entry):
            self.pages.addWidget(p)
        # centre: pages over the map drawer
        centre = QWidget(); cl = QVBoxLayout(centre); cl.setContentsMargins(0, 0, 0, 0); cl.setSpacing(0)
        self.centre_split = QSplitter(Qt.Vertical)
        self.centre_split.addWidget(self.pages)
        self.map_box = QWidget(); ml = QVBoxLayout(self.map_box); ml.setContentsMargins(0, 0, 0, 0); ml.setSpacing(0)
        map_head = QFrame(); map_head.setStyleSheet(f"QFrame {{ background: {T.bg}; border-top: 1px solid {T.border}; border-bottom: 1px solid {T.border}; }}")
        mh = QHBoxLayout(map_head); mh.setContentsMargins(10, 3, 6, 3)
        ml_lab = QLabel("Map — every step in this project, in order. Drag a step to move it; click one to see it."); ml_lab.setObjectName("muted"); ml_lab.setWordWrap(True)
        self.map_close = QToolButton(); self.map_close.setObjectName("quiet"); self.map_close.setIcon(icon("x", T.muted, 14)); self.map_close.clicked.connect(lambda: self.a_map.setChecked(False))
        mh.addWidget(ml_lab, 1); mh.addWidget(self.map_close)
        ml.addWidget(map_head); ml.addWidget(self.view, 1)
        self.centre_split.addWidget(self.map_box)
        self.centre_split.setStretchFactor(0, 3); self.centre_split.setStretchFactor(1, 1); self.centre_split.setSizes([620, 260])
        cl.addWidget(self.centre_split, 1)
        # when the map is closed it collapses to this handle, so it is always one click away
        self.map_handle = QToolButton(); self.map_handle.setObjectName("quiet"); self.map_handle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.map_handle.setIcon(icon("map-trifold", T.muted, 14)); self.map_handle.setText("Show the map of steps (Ctrl+M)")
        self.map_handle.setStyleSheet(f"QToolButton {{ border: none; border-top: 1px solid {T.border}; background: {T.bg}; padding: 3px 10px; text-align: left; }} QToolButton:hover {{ color: {T.accent}; }}")
        self.map_handle.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.map_handle.clicked.connect(lambda: self.a_map.setChecked(True))
        self.map_handle.hide()
        cl.addWidget(self.map_handle)
        self.toast = Toast(centre)
        # run strip (bottom of centre)
        self.progress = QFrame(); self.progress.setStyleSheet(f"QFrame {{ background: {T.panel}; border-top: 1px solid {T.border}; }}")
        pl_ = QHBoxLayout(self.progress); pl_.setContentsMargins(10, 4, 10, 4); pl_.setSpacing(10)
        self.progress_label = QLabel("")
        self.progress_bar = QProgressBar(); self.progress_bar.setTextVisible(False); self.progress_bar.setFixedHeight(4); self.progress_bar.setRange(0, 0)
        self.stop_btn = QPushButton("Stop"); self.stop_btn.setIcon(icon("stop", T.text, 14)); self.stop_btn.clicked.connect(self.stop)
        pl_.addWidget(self.progress_label, 1); pl_.addWidget(self.progress_bar, 2); pl_.addWidget(self.stop_btn)
        self.progress.hide(); cl.addWidget(self.progress)
        split = QSplitter(Qt.Horizontal); split.addWidget(self.rail); split.addWidget(centre)
        split.setStretchFactor(0, 0); split.setStretchFactor(1, 1); split.setSizes([250, 1170]); split.setCollapsible(1, False)
        self.setCentralWidget(split)
        self.inspector = InspectorDock(self.doc, self)
        self.addDockWidget(Qt.RightDockWidgetArea, self.inspector)
        self.resizeDocks([self.inspector], [370], Qt.Horizontal)
        self.picker = StepPicker(self)
        self._picker_ctx: dict = {}
        self.status = QStatusBar(); self.setStatusBar(self.status)
        self.autosave_label = QLabel(""); self.autosave_label.setObjectName("faint"); self.status.addPermanentWidget(self.autosave_label)
        self.mode_label = QLabel(""); self.mode_label.setObjectName("faint"); self.status.addPermanentWidget(self.mode_label)
        self._current: str | None = None
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
        QTimer.singleShot(200, self._maybe_recover)
        QTimer.singleShot(400, self._maybe_tour)

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
        self.a_open_data = self._act("Open data file…", "folder-open", "Ctrl+I", self.add_data_file, "Open a CSV, Excel or Parquet file as a new table (Ctrl+I)")
        self.a_quit = self._act("&Quit", None, QKeySequence.Quit, self.close)
        self.recent_menu = file_m.addMenu("Open &recent")
        file_m.addAction(self.a_new); file_m.addAction(self.a_open); file_m.addMenu(self.recent_menu)
        file_m.addSeparator(); file_m.addAction(self.a_open_data); file_m.addSeparator()
        file_m.addAction(self.a_save); file_m.addAction(self.a_save_as); file_m.addAction(self.a_versions); file_m.addAction(self.a_revert)
        file_m.addSeparator(); file_m.addAction(self.a_quit)
        self._refresh_recent()

        edit_m = mb.addMenu("&Edit")
        self.a_undo = self._act("&Undo", "arrow-u-up-left", QKeySequence.Undo, self.doc.undo.undo, "Undo (Ctrl+Z)")
        self.a_redo = self._act("&Redo", "arrow-u-up-right", [QKeySequence.Redo, "Ctrl+Y"], self.doc.undo.redo, "Redo (Ctrl+Y)")
        self.doc.undo.canUndoChanged.connect(self.a_undo.setEnabled); self.doc.undo.canRedoChanged.connect(self.a_redo.setEnabled)
        self.doc.undo.undoTextChanged.connect(self._undo_text); self.doc.undo.redoTextChanged.connect(self._redo_text)
        self.a_undo.setEnabled(False); self.a_redo.setEnabled(False)
        self.a_add = self._act("Add &step…", "plus", ["Ctrl+K", "Insert"], lambda: self.open_picker(), "Add a step after the current table (Ctrl+K)")
        self.a_delete = self._act("&Delete step", "trash", None, self.delete_current, "Delete the selected step (Delete in the project list or the map)")
        self.a_dup = self._act("D&uplicate step", "copy", "Ctrl+D", lambda: self.doc.duplicate_nodes(self.scene.selected_node_ids()))
        self.a_note = self._act("Add &note to the map", "note-pencil", "Ctrl+Shift+N", lambda: self._add_note(self.view.mapToScene(self.view.viewport().rect().center())))
        self.a_inputs = self._act("&Inputs (named values)…", "gear", "Ctrl+Shift+I", lambda: self.rail.select("inputs", "inputs"))
        for a in (self.a_undo, self.a_redo):
            edit_m.addAction(a)
        edit_m.addSeparator()
        for a in (self.a_add, self.a_delete, self.a_dup, self.a_note, self.a_inputs):
            edit_m.addAction(a)

        run_m = mb.addMenu("&Run")
        self.a_run = self._act("&Run everything", "play", ["Ctrl+R", "F5"], lambda: self.run(), "Compute every step on the full data (Ctrl+R)")
        self.a_run_sel = self._act("Run up to &this step", None, "Ctrl+Shift+R", self.run_selected, "Run the current step and what it needs (Ctrl+Shift+R)")
        self.a_run_force = self._act("Run everything again (ignore cached results)", None, None, lambda: self.run(force=True))
        self.a_stop = self._act("&Stop", "stop", "Escape", self.stop, "Stop after the current step")
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
        self.a_settings = self.inspector.toggleViewAction(); self.a_settings.setText("Show &settings"); self.a_settings.setIcon(icon("sliders", T.text, 16)); self.a_settings.setShortcut("Ctrl+,")
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
        help_m.addAction(self._act("Show the &tour again", None, None, lambda: self._maybe_tour(force=True)))
        help_m.addSeparator()
        help_m.addAction(self._act("Show &log file", None, None, self.show_log))
        help_m.addAction(self._act("&About DANCR", None, None, self.about))

        tb = QToolBar("Main"); tb.setObjectName("maintoolbar"); tb.setMovable(False); tb.setIconSize(QSize(16, 16))
        tb.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.addToolBar(tb)
        tb.addAction(self.a_open_data); tb.addAction(self.a_add); tb.addSeparator()
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
        return {"geometry": self.saveGeometry(), "current": self._current,
                "map": self.a_map.isChecked(), "tab": self.table.tabs.currentIndex()}

    def restore_ui_state(self, state: dict) -> None:
        g = state.get("geometry")
        if g is not None:
            self.restoreGeometry(g)
        self.a_map.setChecked(bool(state.get("map", True)))
        if state.get("tab"):
            self.table.tabs.setCurrentIndex(int(state["tab"]))
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
        d.dirtyChanged.connect(lambda _: self._update_title())
        d.pathChanged.connect(lambda _: self._update_title())
        d.autosaveChanged.connect(lambda _: self._update_title())
        d.autosaved.connect(lambda: (self._update_title(), self.status.showMessage("Saved automatically", 2000)))
        d.undo.indexChanged.connect(self._on_undo_index)
        d.reloaded.connect(self._on_reloaded)
        d.message.connect(lambda m: self.status.showMessage(m, 8000))
        d.runStarted.connect(self._on_run_started)
        d.runProgress.connect(self._run_progress)
        d.runFinished.connect(self._on_run_finished)
        d.nodeAdded.connect(lambda _: self._show_page())
        d.nodeRemoved.connect(self._on_node_removed)
        d.autoRunChanged.connect(lambda _: self._refresh_mode())
        d.nodeAdded.connect(lambda _: self._refresh_mode()); d.nodeRemoved.connect(lambda _: self._refresh_mode()); d.nodeChanged.connect(lambda _: self._refresh_mode())
        self.scene.status.connect(lambda m: self.status.showMessage(m, 6000))
        self.scene.selectionChangedTo.connect(self._on_scene_select)
        self.scene.nodeActivated.connect(self.show_node)
        self.scene.addAfterRequested.connect(self._add_after)
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
        self.start.openData.connect(self.add_data_file); self.start.openProject.connect(self.open_dialog)
        self.start.openRecent.connect(lambda p: self._confirm_stop_run("open another project") and self.maybe_save() and self.open_path(p))
        self.start.template.connect(self._start_template); self.start.blank.connect(lambda: self.add_node("enter_data", None))
        self.setAcceptDrops(True)

    # ------------------------------------------------------------ pages and selection
    def _show_page(self) -> None:
        """Pick the page for the current selection (start page when the project is empty)."""
        if not self.doc.pipeline.nodes and self._current is None:
            self.start.set_recent(self._recent())
            self.pages.setCurrentWidget(self.start)
            self.map_box.setVisible(False); self.map_handle.setVisible(False)
            return
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
        if nid == self._current:
            self._show_page(); return
        self._current = nid
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

    def _on_scene_select(self, nid: str | None) -> None:
        if nid is not None:
            self.show_node(nid)

    def _on_node_removed(self, nid: str) -> None:
        if nid == self._current:
            self._current = None
            self.inspector.set_node(None)
        self._show_page()

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
            sizes = self.centre_split.sizes()
            if len(sizes) == 2 and sizes[1] < 120:          # reopened into a collapsed slot
                total = sum(sizes) or self.centre_split.height()
                self.centre_split.setSizes([max(200, total - 260), 260])

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
            self.mode_label.setText(f"large data ({mb:,.0f} MB): press Run to compute")
        if self.table.nid:
            self.table._refresh_header()

    def delete_current(self) -> None:
        ids = self.scene.selected_node_ids() or ([self._current] if self._current and self._current in self.doc.pipeline.nodes else [])
        if not ids:
            self.scene.delete_selection()          # a selected arrow or note on the map
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
    def _update_title(self) -> None:
        name = self.doc.path.stem if self.doc.path else "Untitled"
        paused = self.doc.autosave_paused
        self.setWindowTitle(f"{'• ' if self.doc.dirty else ''}{name}{' (autosave paused)' if paused else ''} — DANCR")
        self.autosave_label.setText(f"Autosave paused: {paused}" if paused else "")
        self.a_revert.setEnabled(self.doc.path is not None and self.doc.dirty)

    def _on_reloaded(self) -> None:
        self._update_title()
        self._current = None
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
        if not self.doc.dirty:
            return True
        if self.doc.autosave_paused:
            what = f"Autosave is paused ({self.doc.autosave_paused}). Save replaces {self.doc.path.name if self.doc.path else 'nothing'} with what you see now; Discard keeps the file on disk as it is."
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
        self._wait_for_run("Stopping the current step…")
        return True

    def _wait_for_run(self, text: str) -> None:
        if not self.doc.running:
            return
        self.doc.stop()
        dlg = QProgressDialog(text, None, 0, 0, self)
        dlg.setWindowTitle("DANCR"); dlg.setWindowModality(Qt.WindowModal); dlg.setMinimumDuration(0)
        dlg.show(); dlg.setValue(0)
        while self.doc.running:
            QApplication.processEvents()
            self.doc._run.wait(50)
        dlg.close()

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
        try:
            self.doc.save()
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
        r = QMessageBox.question(self, "Unsaved project", f"Last time DANCR closed with an unsaved project of {len(pipe.nodes)} steps. Bring it back?",
                                 QMessageBox.Yes | QMessageBox.No)
        if r == QMessageBox.Yes:
            self.doc.recover(pipe, rp)
            self.status.showMessage("Recovered — save it to keep it", 8000)
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
            self.status.showMessage("Restored — save to keep it, or Revert to go back. Autosave is paused until then.", 8000)

    def terminate(self) -> None:
        """The system asked us to quit (SIGTERM): no questions, keep an unsaved project recoverable, close cleanly."""
        self._terminating = True
        self.close()
        if self.isVisible():
            QApplication.quit()

    def closeEvent(self, e: QCloseEvent) -> None:
        if self._terminating:
            self.doc.stop(wait=True)
            self.doc.write_recovery()             # untitled projects only; offered back on the next start
        else:
            if self.doc.running:
                r = QMessageBox.question(self, "A run is in progress", "Stop the run and quit?", QMessageBox.Yes | QMessageBox.No)
                if r != QMessageBox.Yes:
                    e.ignore(); return
                self._wait_for_run("Stopping the current step before quitting…")
            if not self.maybe_save():
                e.ignore(); return
            self.doc.clear_recovery()             # saved or deliberately discarded
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("state", self.saveState())
        self.settings.setValue("map", self.a_map.isChecked())
        view_pool().shutdown(5000)                # nothing may still read the results when the cache is removed
        self.doc.shutdown()
        e.accept()

    def _restore_layout(self) -> None:
        g = self.settings.value("geometry")
        if g:
            self.restoreGeometry(g)
        s = self.settings.value("state")
        if s:
            self.restoreState(s)
        if self.inspector.width() < 320:
            self.resizeDocks([self.inspector], [370], Qt.Horizontal)
        m = self.settings.value("map")
        if m is not None:
            self.a_map.setChecked(m in (True, "true", "True", 1))

    def _recent(self) -> list[str]:
        v = self.settings.value("recent")
        if isinstance(v, str):
            return [v] if v else []
        return [str(x) for x in (v or [])]

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
        for u in e.mimeData().urls():
            p = u.toLocalFile()
            if p.lower().endswith(".json"):
                if self._confirm_stop_run("open another project") and self.maybe_save():
                    self.open_path(p)
            elif p:
                self._add_load_node(p, None)
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
        start = str(self.doc.path.parent) if self.doc.path else str(Path.home())
        f, _ = QFileDialog.getOpenFileName(self, "Open data file", start, DATA_FILTER)
        if f:
            self._add_load_node(f, None)

    def _add_load_node(self, path: str, pos: QPointF | None) -> None:
        p = Path(path)
        rel = p
        if self.doc.path:
            try:
                rel = p.relative_to(self.doc.path.parent)
            except ValueError:
                rel = p
        self.scene.clearSelection()
        nid = self.add_node("load_file", pos, params={"path": str(rel)}, title=p.stem)
        if not self.doc.auto_run and not self.doc.running:
            self.run([nid])
        elif self.doc.running:
            self.status.showMessage(f"Added {p.name}. It will load when you next run.", 6000)

    def _start_template(self, key: str) -> None:
        base = self.doc.path.parent if self.doc.path else Path.home() / "DANCR samples"
        try:
            data = write_sample(base)
        except OSError as e:
            QMessageBox.critical(self, "Sample data", f"Could not write the sample file: {e}"); return
        pipe = Pipeline(self.doc.pipeline.name); pipe.path = self.doc.path
        try:
            build_template(key, pipe, data)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Template", str(e)); return
        self.doc.replace_pipeline(pipe)
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
        if ok:
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

    def _maybe_tour(self, force: bool = False) -> None:
        if self._disposed or (not force and self.settings.value("tour_shown")):
            return
        self.settings.setValue("tour_shown", True)
        from .helpdialog import TourDialog
        TourDialog(self).show()

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
