"""The grid: a table view with Excel-like header menus that create steps, find, jump-to-time and copy."""
from __future__ import annotations


import logging
from typing import Any

import polars as pl
from PySide6.QtCore import Qt, Signal, QAbstractTableModel, QModelIndex, QPoint
from PySide6.QtGui import QColor, QFont, QBrush, QKeySequence, QAction, QGuiApplication
from PySide6.QtWidgets import (QTableView, QAbstractItemView, QMenu, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QLabel,
                               QToolButton, QStackedLayout)

from ..views.table import TablePager, format_value
from ..views.stats import quick_column_info
from ..core.expr import NUM, TIME, STR, BOOL
from .theme import T
from .icons import icon
from .workers import Task, view_pool

log = logging.getLogger("dancr.ui")
MAX_TABLE_ROWS = 50_000_000     # QHeaderView length is a 32-bit int
KIND_GLYPH = {NUM: "#", TIME: "◷", STR: "Aa", BOOL: "✓"}


def fmt_number(v: float, decimals: int | None) -> str:
    if v != v:
        return "–"
    if decimals is not None:
        return f"{v:,.{decimals}f}"
    return format_value(v)


class TableModel(QAbstractTableModel):
    """Grid over a TablePager. Pages are read on a worker thread; cells show a faint dot until they arrive."""

    def __init__(self) -> None:
        super().__init__()
        self.pager: TablePager | None = None
        self._rows = 0
        self._info_cache: dict[str, str] = {}
        self._pending: set[int] = set()
        self._generation = 0
        self.column_stats: dict[str, dict] = {}
        self.column_meta: dict[str, dict] = {}
        self.in_memory = False
        self._decimals: dict[str, int | None] = {}
        self._keep: set[Task] = set()
        self._logged_error = False       # one traceback per table, not one per cell

    def set_pager(self, pager: TablePager | None, in_memory: bool = False, column_stats: dict | None = None,
                  column_meta: dict | None = None) -> None:
        self.beginResetModel()
        self.pager = pager
        self.in_memory = in_memory
        self._rows = min(pager.rows, MAX_TABLE_ROWS) if pager else 0
        self._info_cache.clear(); self._pending.clear(); self._decimals.clear()
        self._generation += 1
        self._logged_error = False
        self.column_stats = column_stats or {}
        self.column_meta = column_meta or {}
        self.endResetModel()

    def set_column_meta(self, meta: dict) -> None:
        self.column_meta = meta or {}
        if self.pager:
            self.headerDataChanged.emit(Qt.Horizontal, 0, max(0, len(self.pager.columns) - 1))

    def _request_page(self, p: int) -> None:
        if p in self._pending or self.pager is None:
            return
        self._pending.add(p)
        pager, gen = self.pager, self._generation

        def done(df):
            if gen != self._generation or self.pager is not pager:
                return
            pager.store_page(p, df)
            self._pending.discard(p)
            top = p * pager.page_size
            bottom = min(self._rows - 1, top + pager.page_size - 1)
            if bottom >= top:
                self.dataChanged.emit(self.index(top, 0), self.index(bottom, max(0, self.columnCount() - 1)))
        t = Task(pager.fetch_page, p)
        t.waits_for_run = False
        t.signals.done.connect(done)
        t.signals.failed.connect(lambda m: self._pending.discard(p))
        self._keep.add(t)
        t.signals.finished.connect(lambda t=t: self._keep.discard(t))
        view_pool().start(t)

    def _cell(self, row: int, col: int) -> tuple[bool, Any]:
        if self.pager is None:
            return True, None
        if self.in_memory:
            return True, self.pager.value(row, col)
        ok, v = self.pager.cached_value(row, col)
        if not ok:
            self._request_page(row // self.pager.page_size)
        return ok, v

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else self._rows

    def columnCount(self, parent=QModelIndex()) -> int:
        return 0 if (parent.isValid() or not self.pager) else len(self.pager.columns)

    def _decimals_for(self, col: str) -> int | None:
        """Sensible fixed decimals per numeric column from its magnitude and spread."""
        if col in self._decimals:
            return self._decimals[col]
        st = self.column_stats.get(col) or {}
        d: int | None = None
        if not st and self.pager is not None and 0 in self.pager._pages and col in self.pager._pages[0].columns:
            try:
                ser = self.pager._pages[0][col]
                if ser.dtype.is_numeric():
                    st = {"min": ser.min(), "max": ser.max()}
            except Exception:
                st = {}
        try:
            lo, hi = st.get("min"), st.get("max")
            if lo is not None and hi is not None:
                span = abs(float(hi) - float(lo))
                mag = max(abs(float(hi)), abs(float(lo)), 1e-12)
                if span == 0:
                    d = 2
                elif span < 0.01:
                    d = 6
                elif span < 1:
                    d = 4
                elif span < 100:
                    d = 3 if mag < 1000 else 2
                else:
                    d = 2 if mag < 1e6 else 0
        except (TypeError, ValueError):
            d = None
        self._decimals[col] = d
        return d

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        if not index.isValid() or not self.pager:
            return None
        if role == Qt.DisplayRole:
            try:
                ok, v = self._cell(index.row(), index.column())
            except Exception:  # noqa: BLE001
                self._log_cell_error(index)
                return "…"
            if not ok:
                return "·"
            if v is None or (isinstance(v, float) and v != v):
                return "–"
            if isinstance(v, float):
                return fmt_number(v, self._decimals_for(self.pager.columns[index.column()]))
            return format_value(v)
        if role == Qt.TextAlignmentRole:
            k = self.pager.kinds.get(self.pager.columns[index.column()])
            return int(Qt.AlignRight | Qt.AlignVCenter) if k == NUM else int(Qt.AlignLeft | Qt.AlignVCenter)
        if role == Qt.ForegroundRole:
            try:
                ok, v = self._cell(index.row(), index.column())
            except Exception:  # noqa: BLE001
                self._log_cell_error(index)
                return None
            if not ok or v is None or (isinstance(v, float) and v != v):
                return QBrush(QColor(T.faint))
        return None

    def _log_cell_error(self, index: QModelIndex) -> None:
        if not self._logged_error:
            self._logged_error = True
            log.exception("Could not read cell row %d column %d of the table", index.row(), index.column())

    def raw(self, row: int, col: int) -> Any:
        ok, v = self._cell(row, col)
        return v if ok else None

    def headerData(self, section: int, orientation, role: int = Qt.DisplayRole) -> Any:
        if not self.pager:
            return None
        if orientation == Qt.Horizontal:
            c = self.pager.columns[section]
            if role == Qt.DisplayRole:
                meta = self.column_meta.get(c) or {}
                label = meta.get("label") or c
                unit = meta.get("unit")
                glyph = KIND_GLYPH.get(self.pager.kinds.get(c, ""), "")
                text = f"{label} ({unit})" if unit else label
                return f"{glyph} {text}" if glyph else text
            if role == Qt.ToolTipRole:
                if c not in self._info_cache:
                    kind = self.pager.kinds.get(c, "?")
                    st = self.column_stats.get(c) or {}
                    meta = self.column_meta.get(c) or {}
                    bits = [c + (f" — {meta['label']}" if meta.get("label") else ""), f"type: {kind}" + (f", unit: {meta['unit']}" if meta.get("unit") else "")]
                    if "nulls" in st:
                        bits.append(f"blank: {int(st['nulls']):,}")
                    for k in ("min", "max", "mean"):
                        if st.get(k) is not None:
                            v = st[k]
                            bits.append(f"{k}: {format_value(v) if not isinstance(v, str) else v}")
                    if not st and self.in_memory:
                        try:
                            info = quick_column_info(self.pager.lf, c, 100_000)
                            bits.append(f"blank: {info['missing']:,}")
                            for k in ("min", "max", "mean", "earliest", "latest", "distinct"):
                                if k in info:
                                    bits.append(f"{k}: {format_value(info[k]) if not isinstance(info[k], str) else info[k]}")
                        except Exception:
                            pass
                    bits.append("Right-click for filter, sort, chart and more")
                    self._info_cache[c] = "\n".join(bits)
                return self._info_cache[c]
        elif role == Qt.DisplayRole:
            return f"{section + 1:,}"
        return None


class Grid(QWidget):
    """Table view + find bar. Emits column/cell actions that the window turns into steps."""
    columnAction = Signal(str, str)          # action, column
    cellAction = Signal(str, int, str, object)   # action, row (1-based), column, value
    chartColumns = Signal(list)              # selected numeric columns -> chart these

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(0)
        # find bar
        self.find_bar = QWidget(); fb = QHBoxLayout(self.find_bar); fb.setContentsMargins(8, 4, 8, 4); fb.setSpacing(6)
        self.find_edit = QLineEdit(); self.find_edit.setPlaceholderText("Find text or a value, or jump to a time like 2024-06-05 14:00")
        self.find_edit.setClearButtonEnabled(True)
        self.find_status = QLabel(""); self.find_status.setObjectName("muted")
        close = QToolButton(); close.setObjectName("quiet"); close.setIcon(icon("x", T.muted, 14)); close.clicked.connect(self.hide_find)
        fb.addWidget(self.find_edit, 1); fb.addWidget(self.find_status); fb.addWidget(close)
        self.find_bar.setStyleSheet(f"QWidget {{ background: {T.bg}; border-bottom: 1px solid {T.border}; }}")
        self.find_bar.hide()
        lay.addWidget(self.find_bar)
        holder = QWidget(); self.stack = QStackedLayout(holder); self.stack.setStackingMode(QStackedLayout.StackAll)
        self.table = QTableView(); self.model = TableModel(); self.table.setModel(self.model)
        self.table.setAlternatingRowColors(True); self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.horizontalHeader().setDefaultSectionSize(130); self.table.horizontalHeader().setStretchLastSection(False)
        self.table.horizontalHeader().setSectionsClickable(True); self.table.horizontalHeader().setHighlightSections(False)
        self.table.horizontalHeader().setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.horizontalHeader().customContextMenuRequested.connect(self._header_menu)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._cell_menu)
        self.table.verticalHeader().setDefaultSectionSize(22); self.table.verticalHeader().setMinimumWidth(64)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        f = QFont(T.ui_font); f.setPointSizeF(9.5); self.table.setFont(f)
        self.overlay = QLabel(""); self.overlay.setAlignment(Qt.AlignCenter); self.overlay.setObjectName("muted"); self.overlay.hide()
        self.overlay.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.stack.addWidget(self.table); self.stack.addWidget(self.overlay); self.stack.setCurrentWidget(self.overlay)
        lay.addWidget(holder, 1)
        self.find_edit.returnPressed.connect(self.find_next)
        a = QAction(self); a.setShortcut(QKeySequence.Find); a.setShortcutContext(Qt.WidgetWithChildrenShortcut); a.triggered.connect(self.show_find); self.addAction(a)
        c = QAction(self); c.setShortcut(QKeySequence.Copy); c.setShortcutContext(Qt.WidgetWithChildrenShortcut); c.triggered.connect(self.copy_selection); self.table.addAction(c)
        self._lf: pl.LazyFrame | None = None
        self._keep: set[Task] = set()

    # ---- content
    def set_content(self, lf: pl.LazyFrame | None, rows: int | None, in_memory: bool, column_stats: dict | None, column_meta: dict | None,
                    pager: TablePager | None = None) -> None:
        """Show a frame. Pass a prebuilt `pager` (made on a worker thread) for on-disk results so no file is read here."""
        self._lf = lf
        if lf is None:
            self.model.set_pager(None); return
        self.model.set_pager(pager or TablePager(lf, rows=rows), in_memory=in_memory, column_stats=column_stats, column_meta=column_meta)
        self._size_columns()

    def set_overlay(self, text: str) -> None:
        self.overlay.setText(text); self.overlay.setVisible(bool(text))

    def _size_columns(self) -> None:
        h = self.table.horizontalHeader()
        if not self.model.pager:
            return
        for i in range(min(self.model.columnCount(), 80)):
            name = self.model.pager.columns[i]
            kind = self.model.pager.kinds.get(name, "any")
            base = 178 if kind == TIME else 96
            h.resizeSection(i, max(base, min(300, 24 + 7 * len(str(self.model.headerData(i, Qt.Horizontal))))))

    # ---- menus
    def _column_at(self, pos: QPoint) -> str | None:
        idx = self.table.horizontalHeader().logicalIndexAt(pos)
        if idx < 0 or not self.model.pager:
            return None
        return self.model.pager.columns[idx]

    def _header_menu(self, pos: QPoint) -> None:
        col = self._column_at(pos)
        if col is None or not self.model.pager:
            return
        kind = self.model.pager.kinds.get(col, "any")
        m = QMenu(self)
        def add(text, action, ic=None):
            a = m.addAction(icon(ic, T.text, 14), text) if ic else m.addAction(text)
            a.triggered.connect(lambda: self.columnAction.emit(action, col))
        add("Filter rows by this column…", "filter", "funnel")
        add("Sort by this column (smallest first)", "sort_asc", "sort-ascending")
        add("Sort by this column (largest first)", "sort_desc", "sort-ascending")
        m.addSeparator()
        if kind == NUM:
            add("Chart this column", "chart", "chart-line")
            add("Check against a limit…", "limit", "check-circle")
        add("New column from a formula…", "formula", "function")
        m.addSeparator()
        add("Rename, set units…", "describe", "note-pencil")
        if kind != NUM:
            add("Fix numbers and dates…", "fixtype", "text-aa")
        add("Hide this column", "hide", "x")
        m.addSeparator()
        add("Copy column name", "copyname", "copy")
        m.exec(self.table.horizontalHeader().mapToGlobal(pos))

    def _cell_menu(self, pos: QPoint) -> None:
        idx = self.table.indexAt(pos)
        if not idx.isValid() or not self.model.pager:
            return
        col = self.model.pager.columns[idx.column()]
        value = self.model.raw(idx.row(), idx.column())
        m = QMenu(self)
        a = m.addAction(icon("copy", T.text, 14), "Copy"); a.triggered.connect(self.copy_selection)
        f = m.addAction(icon("note-pencil", T.text, 14), "Fix this value…"); f.triggered.connect(lambda: self.cellAction.emit("fix", idx.row() + 1, col, value))
        sel_cols = sorted({i.column() for i in self.table.selectionModel().selectedIndexes()})
        nums = [self.model.pager.columns[i] for i in sel_cols if self.model.pager.kinds.get(self.model.pager.columns[i]) == NUM]
        if len(nums) >= 1:
            m.addSeparator()
            c = m.addAction(icon("chart-line", T.text, 14), "Chart selected columns" if len(nums) > 1 else f"Chart {nums[0]}")
            c.triggered.connect(lambda: self.chartColumns.emit(nums))
            if len(nums) == 2:
                fit = m.addAction(icon("chart-scatter", T.text, 14), f"Fit a curve: {nums[1]} against {nums[0]}")
                fit.triggered.connect(lambda: self.columnAction.emit(f"fit:{nums[0]}:{nums[1]}", nums[1]))
        m.exec(self.table.viewport().mapToGlobal(pos))

    # ---- clipboard
    def copy_selection(self) -> None:
        if not self.model.pager:
            return
        sel = self.table.selectionModel().selectedIndexes()
        if not sel:
            return
        rows = sorted({i.row() for i in sel}); cols = sorted({i.column() for i in sel})
        if len(rows) > 200_000:
            rows = rows[:200_000]
        header = [str(self.model.headerData(c, Qt.Horizontal)) for c in cols]
        lines = ["\t".join(header)]
        for r in rows:
            lines.append("\t".join(str(self.model.data(self.model.index(r, c), Qt.DisplayRole)) for c in cols))
        QGuiApplication.clipboard().setText("\n".join(lines))

    def copy_all_visible(self) -> None:
        """Copy the first page(s) of the table as tab-separated text."""
        if not self.model.pager:
            return
        self.table.selectAll(); self.copy_selection(); self.table.clearSelection()

    # ---- find / jump
    def show_find(self) -> None:
        self.find_bar.show(); self.find_edit.setFocus(); self.find_edit.selectAll()

    def hide_find(self) -> None:
        self.find_bar.hide(); self.find_status.setText("")

    def find_next(self) -> None:
        q = self.find_edit.text().strip()
        if not q or self._lf is None or not self.model.pager:
            return
        lf, pager = self._lf, self.model.pager
        start = self.table.currentIndex().row() + 1 if self.table.currentIndex().isValid() else 0
        self.find_status.setText("searching…")
        from ..core.timeutil import detect_datetime_format
        from ..core.dtypes import datetime_literal

        def work():
            schema = pager.schema
            times = [c for c, k in pager.kinds.items() if k == TIME]
            fmt = detect_datetime_format(pl.Series([q]), 1.0) if times else None
            base = lf.with_row_index("__r")
            if fmt and times:
                t = times[0]
                hit = base.filter(pl.col(t) >= datetime_literal(q, schema[t])).select("__r").head(1).collect(engine="streaming")
                return int(hit[0, 0]) if hit.height else None
            conds = [pl.col(c).cast(pl.Utf8).str.contains(q, literal=True) for c in pager.columns]
            hit = base.filter(pl.col("__r") >= start).filter(pl.any_horizontal(conds)).select("__r").head(1).collect(engine="streaming")
            if hit.height:
                return int(hit[0, 0])
            hit = base.filter(pl.any_horizontal(conds)).select("__r").head(1).collect(engine="streaming")
            return int(hit[0, 0]) if hit.height else None

        def done(r):
            if self.model.pager is not pager:       # the table changed while searching
                return
            if r is None:
                self.find_status.setText("not found"); return
            self.find_status.setText(f"row {r + 1:,}")
            idx = self.model.index(min(r, self.model.rowCount() - 1), 0)
            self.table.scrollTo(idx, QAbstractItemView.PositionAtCenter)
            self.table.setCurrentIndex(idx); self.table.selectRow(idx.row())
        t = Task(work); t.waits_for_run = False
        t.signals.done.connect(done); t.signals.failed.connect(lambda m: self.find_status.setText(m[:60]) if self.model.pager is pager else None)
        self._keep.add(t); t.signals.finished.connect(lambda t=t: self._keep.discard(t))
        view_pool().start(t)
