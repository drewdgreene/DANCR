"""MainWindow actions methods."""
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


class WindowActions:
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
        self.a_catalog = self._act("Browse &project folder…", None, None, self.browse_catalog,
                                   "See every DANCR project in a folder and the datasets in each")
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
        file_m.addSeparator(); file_m.addAction(self.a_open_data); file_m.addAction(self.a_catalog); file_m.addSeparator()
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
        self.a_ask = self._act("&Auto", "sparkle", "Ctrl+J", self.focus_ask,
                                  "Ask a question in plain words, or pick an answer DANCR offers (Ctrl+J)")
        self.a_assistant = self._act("Assistant", "magic-wand", "Ctrl+Shift+J", self.focus_assistant,
                                     "Talk to the Assistant: it builds real steps you approve (Ctrl+Shift+J)")
        self.a_assistant.setCheckable(True)
        self.a_delete = self._act("&Delete step", "trash", None, self.delete_current, "Delete the selected step (Delete in the project list or the map)")
        self.a_dup = self._act("D&uplicate step", "copy", "Ctrl+D", lambda: self.doc.duplicate_nodes(self.scene.selected_node_ids()))
        self.a_note = self._act("Add &note to the map", "note-pencil", "Ctrl+Shift+N", lambda: self._add_note(self.view.mapToScene(self.view.viewport().rect().center())))
        self.a_inputs = self._act("&Inputs (named values)…", "gear", "Ctrl+Shift+I", lambda: self.rail.select("inputs", "inputs"))
        self.a_remove_ai = self._act("Remove AI-built steps", None, None, self.remove_ai_steps,
                                     "Delete every step the Assistant built, in one undo")
        self.a_remove_ai.setEnabled(False)
        for a in (self.a_undo, self.a_redo):
            edit_m.addAction(a)
        edit_m.addSeparator()
        for a in (self.a_add, self.a_delete, self.a_dup, self.a_note, self.a_inputs, self.a_remove_ai):
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
        self.a_result = QAction("Show the &result panel", self, checkable=True, checked=True); self.a_result.setShortcut("Ctrl+M")
        self.a_result.setIcon(icon("table", T.text, 16)); self.a_result.setToolTip("Show or hide the result panel (Ctrl+M)")
        self.a_result.toggled.connect(self._toggle_result)
        self.a_settings = QAction("Show &settings", self, checkable=True, checked=True)
        self.a_settings.setIcon(icon("sliders", T.text, 16)); self.a_settings.setShortcut("Ctrl+,")
        self.a_settings.setToolTip("Show or hide the settings panel (Ctrl+,)")
        self.a_settings.toggled.connect(lambda _: self._apply_side_panels())
        self.a_fit = self._act("&Fit the map in view", "arrows-out", "Ctrl+0", self.view.fit_all)
        self.a_zoom_in = self._act("Zoom map in", None, [QKeySequence.ZoomIn, "Ctrl+="], lambda: self.view.zoom_by(1.2))
        self.a_zoom_out = self._act("Zoom map out", None, QKeySequence.ZoomOut, lambda: self.view.zoom_by(1 / 1.2))
        for a in (self.a_result, self.a_settings, self.a_fit, self.a_zoom_in, self.a_zoom_out):
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

        share_m = mb.addMenu("&Share")
        self.a_dataset = self._act("&Dataset details…", "article", None, self.dataset_details,
                                   "Who made the data, the license, how to cite it — the header of a FAIR record")
        export_m = share_m.addMenu("Export metadata as")
        for fmt, label in (("schema.org", "schema.org / JSON-LD (Dataset Search)"),
                           ("frictionless", "Frictionless Data Package"),
                           ("manifest", "Run manifest (provenance)"),
                           ("rocrate", "RO-Crate (metadata graph)")):
            export_m.addAction(self._act(label, None, None, lambda _=False, f=fmt: self.export_metadata(f)))
        self.a_package = self._act("Package as RO-Crate…", None, None, self.package_crate,
                                   "Write a self-contained RO-Crate: the FAIR descriptors, the project and its run manifest")
        share_m.addAction(self.a_dataset)
        share_m.addSeparator(); share_m.addAction(export_m.menuAction()); share_m.addAction(self.a_package)

        help_m = mb.addMenu("&Help")
        help_m.addAction(self._act("&Getting started", "hand", None, self.show_onboarding))
        help_m.addSeparator()
        help_m.addAction(self._act("&User guide", "question", "F1", self.show_help))
        help_m.addAction(self._act("&What DANCR can read…", None, None, lambda: self.show_help("formats")))
        help_m.addAction(self._act("Formula &functions", None, None, lambda: self.show_help("formulas")))
        help_m.addAction(self._act("For AI agents and the command line", None, None, lambda: self.show_help("agents")))
        help_m.addSeparator()
        help_m.addAction(self._act("Connect the &Assistant…", "magic-wand", None, self.setup_assistant))
        help_m.addAction(self._act("Use from a &coding agent…", "list-bullets", None, self.setup_agents))
        help_m.addAction(self._act("&Check this build…", "check-circle", None, self.check_build))
        help_m.addSeparator()
        help_m.addAction(self._act("Show &log file", None, None, self.show_log))
        help_m.addAction(self._act("&About DANCR", None, None, self.about))

        tb = QToolBar("Main"); tb.setObjectName("maintoolbar"); tb.setMovable(False); tb.setIconSize(QSize(16, 16))
        tb.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.addToolBar(tb)
        tb.addAction(self.a_open_data); tb.addAction(self.a_add); tb.addAction(self.a_ask); tb.addAction(self.a_assistant); tb.addSeparator()
        tb.addAction(self.a_run); tb.addAction(self.a_stop); tb.addSeparator()
        tb.addAction(self.a_undo); tb.addAction(self.a_redo); tb.addSeparator()
        tb.addAction(self.a_save); tb.addAction(self.a_versions)
        spacer = QWidget(); spacer.setSizePolicy(spacer.sizePolicy().horizontalPolicy().Expanding, spacer.sizePolicy().verticalPolicy()); tb.addWidget(spacer)
        tb.addAction(self.a_result); tb.addAction(self.a_settings)
        for a in (self.a_undo, self.a_redo, self.a_save, self.a_versions, self.a_stop, self.a_result, self.a_settings):
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
                "result": self.a_result.isChecked(), "tab": self.table.tabs.currentIndex(),
                "top_split": self.top_split.saveState(), "outer_split": self.outer_split.saveState(),
                "settings": self.a_settings.isChecked(), "assistant": self.a_assistant.isChecked()}

    def restore_ui_state(self, state: dict) -> None:
        g = state.get("geometry")
        if g is not None:
            self.restoreGeometry(g)
        if state.get("top_split"):
            self.top_split.restoreState(state["top_split"])
        if state.get("outer_split"):
            self.outer_split.restoreState(state["outer_split"])
        self.a_settings.setChecked(bool(state.get("settings", True)))
        self.a_assistant.setChecked(bool(state.get("assistant", False)))
        self.a_result.setChecked(bool(state.get("result", True)))
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
        try:                                   # cancel an in-flight Assistant turn: a queued reply must not reach us
            self.assistant.stop()
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
        listen(self, d.statesChanged, self._refresh_insight)      # findings update when the poll refreshes states
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
        self.side.tabChanged.connect(self._on_side_tab)
        self.assistant.buildRequested.connect(self.apply_assistant_proposal)
        self.assistant.proposalPreview.connect(self._preview_proposal)
        self.assistant.proposalCleared.connect(lambda: self.scene.clear_ghost())
        self.assistant.applyEditsRequested.connect(self.assistant_apply_edits)
        self.assistant.revealRequested.connect(self._assistant_reveal)
        self.assistant.saveProjectRequested.connect(self.assistant_save_as_project)
        self.assistant.replaceCanvasRequested.connect(self.assistant_replace_canvas)
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
        self.map.addToReport.connect(self.steps.add_to_report)
        self.report.runRequested.connect(lambda nid: self.run([nid]))
        self.start.openProject.connect(self.open_dialog)
        self.start.openFiles.connect(self.add_data_files)
        self.start.openRecent.connect(lambda p: self._confirm_stop_run("open another project") and self.maybe_save() and self.open_path(p))
        self.start.blank.connect(lambda: self.add_node("enter_data", None))
        self.start.example.connect(self.open_example)
        self.start.template.connect(self._start_template)
        self.start.removeRecent.connect(self._remove_recent)
        self.start.formatsHelp.connect(lambda: self.show_help("formats"))
        self.start.guide.connect(lambda: self.show_help())
        self.start.tour.connect(self.show_onboarding)
        self.start.assistantSetup.connect(self.setup_assistant)
        self.start.agentSetup.connect(self.setup_agents)
        self.start.diagnostics.connect(self.check_build)
        listen(self, d.statesChanged, lambda: self.a_remove_ai.setEnabled(bool(self.doc.ai_steps())))
        self.insight.jumped.connect(lambda nid: (self.show_node(nid), self.view.focus_node(nid)))
        self.sources.addRequested.connect(self.add_data_files)
        self.sources.selected.connect(lambda nid: (self.show_node(nid), self.view.focus_node(nid)))
        self.sources.relateRequested.connect(self.show_relations)
        self.setAcceptDrops(True)

