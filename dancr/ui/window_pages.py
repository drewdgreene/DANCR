"""MainWindow pages methods."""
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


class WindowPages:
    # ------------------------------------------------------------ pages and selection
    def _show_page(self) -> None:
        """Pick the page for the current selection (start page when the project is empty)."""
        if self._on_start_page():
            self.start.set_recent(self._recent())
            self.root.setCurrentWidget(self.start)
            self.close_ask()
            self._apply_side_panels()
            return
        self.root.setCurrentWidget(self.workspace)
        self._apply_side_panels()
        self._sync_ask()                            # the ask overlay, only when asked for
        self._apply_result_visibility()
        self._update_result_label()
        nid = self._current
        if nid == "inputs":
            self.pages.setCurrentWidget(self.inputs); self._refresh_insight(); return
        if nid is None or nid not in self.doc.pipeline.nodes:
            self.pages.setCurrentWidget(self.table)
            if self.table.nid is not None:
                self.table.set_node(None)
            self._refresh_insight(); return
        t = self.doc.pipeline.nodes[nid].type
        page = {"chart": self.chart, "map": self.map, "report": self.report, "enter_data": self.entry}.get(t, self.table)
        if page.nid != nid:                     # the same step again: keep what is shown, no re-query
            page.set_node(nid)
        self.pages.setCurrentWidget(page)
        self._refresh_insight()

    def show_node(self, nid: str | None) -> None:
        if nid != self._current:
            self._current_answer = None
            self.scene.clear_highlight()
            self.answer_bar.set_answer(None)
        if nid == self._current:
            self._show_page(); return
        self._current = nid
        if nid and nid != "inputs":
            self.a_result.setChecked(True)      # selecting a step opens the result drawer on it
        self._focus_answers(nid)
        self.inspector.set_node(nid if nid != "inputs" else None)
        self.assistant.set_focus(nid if nid != "inputs" else None)
        if nid and nid != "inputs":
            self.rail.select("node", nid, emit=False)
            if self.scene.selected_node_ids() != [nid]:
                self.scene.select_node(nid)
        elif nid == "inputs":
            self.rail.select("inputs", "inputs", emit=False)
        self._show_page()
        self._refresh_mode()

    def _update_result_label(self) -> None:
        """The result header names what is shown: the selected answer, else the selected step, else 'Result'."""
        label = "Result"
        a = self.doc.pipeline.answer(self._current_answer) if self._current_answer else None
        if a is not None:
            label = f"Answer · {a.title}"
        elif self._current and self._current in self.doc.pipeline.nodes:
            label = self.doc.pipeline.nodes[self._current].title
        self.result_label.setText(label)

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

