"""Centre page for a table: status header, grid, and a summary tab."""
from __future__ import annotations

from typing import Callable

import polars as pl
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QTabWidget, QTableView, QAbstractItemView, QStackedLayout, QToolButton

from ..core.executor import Executor
from ..views.table import TablePager
from ..views.stats import column_summary
from .document import Document
from .grid import Grid, TableModel
from .workers import Serial
from .theme import T
from .common import page_header, status_dot
from .icons import icon


class TableView(QWidget):
    runRequested = Signal(str)
    columnAction = Signal(str, str)
    cellAction = Signal(str, int, str, object)
    chartColumns = Signal(list)

    def __init__(self, doc: Document, parent=None) -> None:
        super().__init__(parent)
        self.doc = doc
        self.nid: str | None = None
        self._preview_serial = Serial()
        self._summary_serial = Serial()
        self._shown_hash: str | None = None
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(0)
        head, h = page_header()
        self.title = QLabel(""); self.title.setObjectName("heading")
        self.status = QLabel(""); self.status.setWordWrap(True); self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.open_btn = QPushButton("Open file"); self.open_btn.setIcon(icon("arrow-square-out", T.text, 14)); self.open_btn.hide()
        self.open_btn.clicked.connect(self._open_saved)
        self.run_btn = QPushButton("Run"); self.run_btn.setObjectName("primary"); self.run_btn.setIcon(icon("play", "#ffffff", 14))
        self.run_btn.clicked.connect(lambda: self.runRequested.emit(self.nid))
        self.copy_btn = QToolButton(); self.copy_btn.setObjectName("quiet"); self.copy_btn.setIcon(icon("copy", T.muted, 16)); self.copy_btn.setToolTip("Copy the selected cells (Ctrl+C), or everything shown if nothing is selected")
        self.find_btn = QToolButton(); self.find_btn.setObjectName("quiet"); self.find_btn.setIcon(icon("magnifying-glass", T.muted, 16)); self.find_btn.setToolTip("Find or jump to a time (Ctrl+F)")
        h.addWidget(self.title); h.addWidget(self.status, 1); h.addWidget(self.find_btn); h.addWidget(self.copy_btn); h.addWidget(self.open_btn); h.addWidget(self.run_btn)
        lay.addWidget(head)
        self.tabs = QTabWidget(); self.tabs.setDocumentMode(True)
        self.grid = Grid()
        self.tabs.addTab(self.grid, "Rows")
        summary_holder = QWidget(); self.summary_stack = QStackedLayout(summary_holder); self.summary_stack.setStackingMode(QStackedLayout.StackAll)
        self.summary = QTableView(); self.summary_model = TableModel(); self.summary.setModel(self.summary_model)
        self.summary.setAlternatingRowColors(True); self.summary.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.summary_overlay = QLabel(""); self.summary_overlay.setAlignment(Qt.AlignCenter); self.summary_overlay.setObjectName("muted"); self.summary_overlay.hide()
        self.summary_stack.addWidget(self.summary); self.summary_stack.addWidget(self.summary_overlay); self.summary_stack.setCurrentWidget(self.summary_overlay)
        self.tabs.addTab(summary_holder, "Describe")
        lay.addWidget(self.tabs, 1)
        self.tabs.currentChanged.connect(lambda _: self.refresh())
        self.grid.columnAction.connect(self.columnAction.emit)
        self.grid.cellAction.connect(self.cellAction.emit)
        self.grid.chartColumns.connect(self.chartColumns.emit)
        self.copy_btn.clicked.connect(self._copy)
        self.find_btn.clicked.connect(self.grid.show_find)
        doc.nodeState.connect(self._on_state)
        doc.statesChanged.connect(self._refresh_if_changed)
        doc.nodeChanged.connect(lambda nid: self._schedule() if nid == self.nid else None)
        doc.columnsChanged.connect(lambda: self.grid.model.set_column_meta(self.doc.pipeline.columns))
        doc.edgeAdded.connect(lambda e: self._schedule() if e.target == self.nid else None)
        doc.edgeRemoved.connect(lambda e: self._schedule() if e.target == self.nid else None)
        doc.runStarted.connect(self._refresh_header)
        doc.runFinished.connect(lambda ok, res: self.refresh())
        doc.reloaded.connect(lambda: self.set_node(None))
        self._timer = QTimer(self); self._timer.setSingleShot(True); self._timer.setInterval(450); self._timer.timeout.connect(self.refresh)

    def set_node(self, nid: str | None) -> None:
        self.nid = nid if nid in self.doc.pipeline.nodes else None
        self._shown_hash = None
        self.refresh()

    def _open_saved(self) -> None:
        from .external import open_external
        p = (self.doc.state(self.nid).report or {}).get("path") if self.nid else None
        if p:
            open_external(p)

    def _copy(self) -> None:
        if self.grid.table.selectionModel() and self.grid.table.selectionModel().selectedIndexes():
            self.grid.copy_selection()
        else:
            self.grid.copy_all_visible()
        self.doc.message.emit("Copied to the clipboard")

    def _schedule(self) -> None:
        self._timer.start()

    def _on_state(self, nid: str, st) -> None:
        if nid == self.nid and st.status != "running":
            self._shown_hash = None; self.refresh()
        elif nid == self.nid:
            self._refresh_header()

    def _refresh_if_changed(self) -> None:
        if self.nid and self.doc.state(self.nid).hash != self._shown_hash:
            self.refresh()

    def _refresh_header(self) -> None:
        if self.nid is None or self.nid not in self.doc.pipeline.nodes:
            return
        node = self.doc.pipeline.nodes[self.nid]
        st = self.doc.state(self.nid)
        self.title.setText(node.title)
        saved = (st.report or {}).get("path") if st.status == "done" else None
        self.open_btn.setVisible(bool(saved))
        if saved:
            txt = status_dot(st.status, f"saved to {saved}")
        elif st.status == "done":
            txt = status_dot(st.status, f"{st.rows:,} rows × {len(st.columns)} columns")
        elif st.status == "failed":
            txt = status_dot(st.status, f"<span style='color:{T.danger}'>{st.error}</span>")
        elif st.status == "running":
            txt = status_dot(st.status, "running…")
        else:
            txt = status_dot(st.status, "preview" + ("" if self.doc.auto_run else " — press Run to compute everything"))
        self.status.setText(txt)
        self.run_btn.setVisible(st.status not in ("done", "running") and not self.doc.auto_run)
        self.run_btn.setEnabled(not self.doc.running)

    def refresh(self) -> None:
        if self.nid is not None and self.nid not in self.doc.pipeline.nodes:
            self.nid = None
        if self.nid is None:
            self.title.setText(""); self.status.setText(""); self.run_btn.hide(); self.open_btn.hide()
            self.grid.set_content(None, None, True, None, None); self.summary_model.set_pager(None)
            return
        st = self.doc.state(self.nid)
        self._shown_hash = st.hash
        self._refresh_header()
        tab = self.tabs.currentIndex()
        if st.status == "done" and st.output:
            nid, output, rows = self.nid, st.output, st.rows
            if tab == 0:
                self.grid.set_overlay("")

                def open_result():                      # reads the Parquet footer: off the GUI thread
                    lf = pl.scan_parquet(output)
                    return nid, lf, TablePager(lf, rows=rows)

                def ready(r):
                    n, lf, pager = r
                    if n != self.nid or self.doc.state(n).output != output:
                        return
                    self.grid.set_content(lf, rows, False, st.column_stats, self.doc.pipeline.columns, pager=pager)
                    if rows and rows > 50_000_000:
                        self.status.setText(self.status.text() + " · showing the first 50,000,000")
                self._preview_serial.submit(open_result, ready, lambda m: self.doc.refresh_states())
            else:
                self._load_summary(lambda: pl.scan_parquet(output), rows or 0)
            return
        if st.status == "running":
            self.grid.set_overlay("Running…"); return
        if self.doc.running:
            msg = "Waiting for the run to finish…"      # both tabs say so while the run holds the gate
            if tab == 0:
                self.grid.set_overlay(msg)
            else:
                self.summary_overlay.setText(msg); self.summary_overlay.show()
        elif tab == 0:
            self.grid.set_overlay("Building a preview…")
        nid = self.nid

        def work():
            try:
                df, res, kind = self.doc.executor.preview(nid, Executor.PREVIEW_ROWS)
                return nid, df, kind, None
            except Exception as e:  # noqa: BLE001
                from ..core.executor import friendly_error
                return nid, None, None, friendly_error(e)
        self._preview_serial.submit(work, self._preview_ready, lambda m: self.grid.set_overlay(m))

    def _preview_ready(self, r) -> None:
        nid, df, kind, err = r
        if nid != self.nid:
            return
        self.grid.set_overlay("")
        if err is not None:
            self.grid.set_content(None, None, True, None, None)
            self.status.setText(f"<span style='color:{T.danger}'>{err}</span>")
            return
        lf = df.lazy()
        if self.tabs.currentIndex() == 0:
            self.grid.set_content(lf, len(df), True, None, self.doc.pipeline.columns)
        else:
            self._load_summary(lambda: lf, len(df))
        n = Executor.PREVIEW_ROWS
        how = {"spread": f"a sample of {n:,} rows spread across the data (counts and totals are approximate)",
               "head": f"the first {n:,} rows of the input", "all": "all rows"}.get(kind, "a sample")
        st = self.doc.state(nid)
        if st.status != "failed":
            self.status.setText(status_dot("preview", f"preview: {len(df):,} rows from {how}" + ("" if self.doc.auto_run else " — press Run to compute everything")))

    def _load_summary(self, open_frame: Callable[[], pl.LazyFrame], rows: int) -> None:
        """Describe every column on a worker; ``open_frame`` runs there too, so no file is touched here."""
        nid = self.nid
        self.summary_overlay.setText("Waiting for the run to finish…" if self.doc.running else f"Describing {rows:,} rows…"); self.summary_overlay.show()
        self._summary_serial.submit(lambda: column_summary(open_frame()), lambda df: self._summary_ready(nid, df), lambda m: self.summary_overlay.setText(m))

    def _summary_ready(self, nid: str, df: pl.DataFrame) -> None:
        if nid != self.nid:
            return
        self.summary_overlay.hide()
        self.summary_model.set_pager(TablePager(df.lazy(), rows=len(df)), in_memory=True)
        self.summary.resizeColumnsToContents()
