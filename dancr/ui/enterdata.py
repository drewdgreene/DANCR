"""Centre page for a 'Type in a table' step: an editable grid with paste from Excel."""
from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeySequence, QAction, QGuiApplication
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QTableWidget, QTableWidgetItem, QPushButton, QInputDialog, QMenu

from .document import Document
from .theme import T
from .common import page_header
from .icons import icon

TYPES = [("text", "text"), ("number", "number"), ("datetime", "date / time"), ("bool", "true / false")]


class EnterDataView(QWidget):
    def __init__(self, doc: Document, parent=None) -> None:
        super().__init__(parent)
        self.doc = doc
        self.nid: str | None = None
        self._suppress = False
        self._timer = QTimer(self); self._timer.setSingleShot(True); self._timer.setInterval(400); self._timer.timeout.connect(self._commit)
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(0)
        head, h = page_header(spacing=6)
        self.title = QLabel(""); self.title.setObjectName("heading")
        hint = QLabel("Type or paste from Excel. Right-click a column header to rename it or set its type."); hint.setObjectName("muted"); hint.setWordWrap(True)
        self.add_col = QPushButton("Add column"); self.add_col.setIcon(icon("plus", T.text, 14)); self.add_col.clicked.connect(self._add_column)
        self.add_row = QPushButton("Add rows"); self.add_row.clicked.connect(lambda: self._ensure_rows(self.table.rowCount() + 10))
        self.paste_btn = QPushButton("Paste"); self.paste_btn.setIcon(icon("copy", T.text, 14)); self.paste_btn.clicked.connect(self.paste)
        h.addWidget(self.title); h.addWidget(hint, 1); h.addWidget(self.paste_btn); h.addWidget(self.add_col); h.addWidget(self.add_row)
        lay.addWidget(head)
        self.table = QTableWidget(0, 0); self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setDefaultSectionSize(170); self.table.horizontalHeader().setMinimumSectionSize(90)
        self.table.horizontalHeader().setContextMenuPolicy(Qt.CustomContextMenu); self.table.horizontalHeader().customContextMenuRequested.connect(self._header_menu)
        self.table.horizontalHeader().setSectionsClickable(True)
        self.table.horizontalHeader().sectionDoubleClicked.connect(self._rename_column)
        lay.addWidget(self.table, 1)
        self.table.itemChanged.connect(self._item_changed)
        a = QAction(self); a.setShortcut(QKeySequence.Paste); a.setShortcutContext(Qt.WidgetWithChildrenShortcut); a.triggered.connect(self.paste); self.table.addAction(a)
        d = QAction(self); d.setShortcut(QKeySequence.Delete); d.setShortcutContext(Qt.WidgetWithChildrenShortcut); d.triggered.connect(self._clear_selection); self.table.addAction(d)
        doc.nodeChanged.connect(lambda nid: self.refill() if nid == self.nid and not self._timer.isActive() else None)
        doc.reloaded.connect(lambda: self.set_node(None))

    def set_node(self, nid: str | None) -> None:
        if self._timer.isActive():
            self._commit()
        self.nid = nid if nid in self.doc.pipeline.nodes else None
        self.refill()

    def _params(self) -> dict:
        return self.doc.pipeline.nodes[self.nid].params if self.nid else {"columns": [], "rows": []}

    def refill(self) -> None:
        if self.nid is None:
            return
        p = self._params()
        cols = p.get("columns") or []
        rows = p.get("rows") or []
        self._suppress = True
        self.title.setText(self.doc.pipeline.nodes[self.nid].title)
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels([f"{c.get('name', '')}  ({dict(TYPES).get(c.get('type', 'text'), 'text')})" for c in cols])
        self.table.setRowCount(max(len(rows) + 5, 12))
        for r in range(self.table.rowCount()):
            for c in range(len(cols)):
                v = rows[r][c] if r < len(rows) and c < len(rows[r]) and rows[r][c] is not None else ""
                self.table.setItem(r, c, QTableWidgetItem(str(v)))
        self._suppress = False

    def _ensure_rows(self, n: int) -> None:
        self._suppress = True
        while self.table.rowCount() < n:
            r = self.table.rowCount(); self.table.insertRow(r)
            for c in range(self.table.columnCount()):
                self.table.setItem(r, c, QTableWidgetItem(""))
        self._suppress = False

    def _item_changed(self, item: QTableWidgetItem) -> None:
        if self._suppress:
            return
        if item.row() >= self.table.rowCount() - 2:
            self._ensure_rows(self.table.rowCount() + 5)
        self._timer.start()

    def _collect_rows(self) -> list[list]:
        rows = []
        ncol = self.table.columnCount()
        for r in range(self.table.rowCount()):
            vals = [(self.table.item(r, c).text() if self.table.item(r, c) else "") for c in range(ncol)]
            rows.append(vals)
        while rows and all(v == "" for v in rows[-1]):
            rows.pop()
        return rows

    def _commit(self) -> None:
        if self.nid is None:
            return
        self.doc.set_params(self.nid, {"rows": self._collect_rows()})

    def _add_column(self) -> None:
        name, ok = QInputDialog.getText(self, "New column", "Column name:")
        if not ok or not name.strip():
            return
        cols = list(self._params().get("columns") or []) + [{"name": name.strip(), "type": "text"}]
        self._commit()
        self.doc.set_params(self.nid, {"columns": cols})
        self.refill()

    def _rename_column(self, idx: int) -> None:
        cols = [dict(c) for c in (self._params().get("columns") or [])]
        if idx >= len(cols):
            return
        name, ok = QInputDialog.getText(self, "Rename column", "Column name:", text=cols[idx].get("name", ""))
        if ok and name.strip():
            cols[idx]["name"] = name.strip()
            self._commit(); self.doc.set_params(self.nid, {"columns": cols}); self.refill()

    def _header_menu(self, pos) -> None:
        idx = self.table.horizontalHeader().logicalIndexAt(pos)
        cols = [dict(c) for c in (self._params().get("columns") or [])]
        if idx < 0 or idx >= len(cols):
            return
        m = QMenu(self)
        m.addAction("Rename…").triggered.connect(lambda: self._rename_column(idx))
        tm = m.addMenu("Type")
        for k, label in TYPES:
            a = tm.addAction(label); a.setCheckable(True); a.setChecked(cols[idx].get("type") == k)
            a.triggered.connect(lambda _=False, k=k: self._set_type(idx, k))
        m.addSeparator()
        m.addAction("Remove column").triggered.connect(lambda: self._remove_column(idx))
        m.exec(self.table.horizontalHeader().mapToGlobal(pos))

    def _set_type(self, idx: int, kind: str) -> None:
        cols = [dict(c) for c in (self._params().get("columns") or [])]
        if idx >= len(cols):
            return
        cols[idx]["type"] = kind
        self._commit(); self.doc.set_params(self.nid, {"columns": cols}); self.refill()

    def _remove_column(self, idx: int) -> None:
        cols = [dict(c) for c in (self._params().get("columns") or [])]
        if idx >= len(cols):
            return
        rows = [[v for j, v in enumerate(r) if j != idx] for r in self._collect_rows()]
        del cols[idx]
        self.doc.set_params(self.nid, {"columns": cols, "rows": rows}); self.refill()

    def _clear_selection(self) -> None:
        self._suppress = True
        for i in self.table.selectedIndexes():
            it = self.table.item(i.row(), i.column())
            if it:
                it.setText("")
        self._suppress = False
        self._timer.start()

    def paste(self) -> None:
        text = QGuiApplication.clipboard().text()
        if not text.strip() or self.nid is None:
            return
        lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        while lines and lines[0].strip() == "":
            lines.pop(0)
        while lines and lines[-1] == "":
            lines.pop()
        if not lines:
            return
        sep = "\t" if "\t" in text else ("," if any("," in l for l in lines) else None)
        grid = [l.split(sep) if sep else [l] for l in lines]
        cur = self.table.currentIndex()
        r0, c0 = (cur.row(), cur.column()) if cur.isValid() else (0, 0)
        cols = [dict(c) for c in (self._params().get("columns") or [])]
        need = c0 + max(len(g) for g in grid)
        # a header row pasted into an empty table becomes the column names
        header_like = len(grid) > 1 and all(not self._numberish(v) for v in grid[0]) and r0 == 0 and c0 == 0 and not any(v for row in self._collect_rows() for v in row)
        if header_like:
            names = [v.strip() or f"column_{i + 1}" for i, v in enumerate(grid[0])]
            cols = []
            for i, n in enumerate(names):
                vals = [g[i] for g in grid[1:] if i < len(g) and g[i].strip()]
                cols.append({"name": n, "type": "number" if vals and all(self._numberish(v) for v in vals) else "text"})
            grid = grid[1:]
        while len(cols) < need:
            cols.append({"name": f"column_{len(cols) + 1}", "type": "text"})
        self.doc.set_params(self.nid, {"columns": cols})
        self.refill()
        self._ensure_rows(r0 + len(grid) + 5)
        self._suppress = True
        for i, row in enumerate(grid):
            for j, v in enumerate(row):
                self.table.item(r0 + i, c0 + j).setText(v.strip())
        self._suppress = False
        self._commit()

    @staticmethod
    def _numberish(v: str) -> bool:
        try:
            float(v.strip().replace(",", "")); return True
        except ValueError:
            return False
