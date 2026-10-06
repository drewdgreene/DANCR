"""MainWindow build methods."""
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


class WindowBuild:
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
        self._entered = True                     # the project now has content: past the start page
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
        from .workers import Task, alive
        self.status.showMessage(f"Looking through {Path(next(f for f in files if big(f))).name}…")
        t = Task(listed)
        t.waits_for_run = False                        # reads only the files dropped, never a result being written
        pending = getattr(self, "_listing", None)
        if pending is None:
            pending = self._listing = set()
        pending.add(t)                                 # kept until it reports, so every drop is added, in order
        # guard against a result arriving after the window was disposed (a theme switch): do not touch a dead widget
        t.signals.done.connect(lambda r, t=t: (pending.discard(t), self._add_listed(r) if alive(self) else None))
        t.signals.failed.connect(lambda m, t=t: (pending.discard(t),
                                                 self.status.showMessage(f"Could not open the files: {m}", 8000) if alive(self) else None))
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

