"""Centre page for Inputs: named values with units, used in formulas and limits."""
from __future__ import annotations


from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QTableWidget, QTableWidgetItem, QPushButton, QAbstractItemView, QHeaderView

from .document import Document
from .theme import T
from .common import page_header
from .icons import icon


class InputsView(QWidget):
    def __init__(self, doc: Document, parent=None) -> None:
        super().__init__(parent)
        self.doc = doc
        self._suppress = False
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(0)
        head, h = page_header(spacing=6)
        t = QLabel("Inputs"); t.setObjectName("heading")
        sub = QLabel("Values you can use by name in formulas, filters and limits. Change one and everything that uses it recomputes."); sub.setObjectName("muted"); sub.setWordWrap(True)
        self.add_btn = QPushButton("Add input"); self.add_btn.setIcon(icon("plus", T.text, 14)); self.add_btn.clicked.connect(self._add)
        self.del_btn = QPushButton("Remove"); self.del_btn.clicked.connect(self._remove)
        h.addWidget(t); h.addWidget(sub, 1); h.addWidget(self.add_btn); h.addWidget(self.del_btn)
        lay.addWidget(head)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Name", "Value", "Unit", "Note"])
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.table.setColumnWidth(0, 200); self.table.setColumnWidth(1, 140); self.table.setColumnWidth(2, 100)
        self.table.verticalHeader().hide(); self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        lay.addWidget(self.table, 1)
        example = QLabel("Example: an input called <b>maximum allowed</b> with value 100 can be typed as the limit in <i>Check against limits</i> "
                         "or used in a formula as <code>[value] / [maximum allowed]</code>. Change it here and every step that uses it updates."); example.setObjectName("muted"); example.setWordWrap(True); example.setContentsMargins(12, 8, 12, 10)
        lay.addWidget(example)
        self.table.itemChanged.connect(self._changed)
        doc.inputsChanged.connect(self.refill)
        doc.reloaded.connect(self.refill)
        self.refill()

    def refill(self) -> None:
        self._suppress = True
        self.table.setRowCount(0)
        for i in self.doc.pipeline.inputs:
            r = self.table.rowCount(); self.table.insertRow(r)
            for c, v in enumerate([i.name, "" if i.value is None else str(i.value), i.unit, i.note]):
                self.table.setItem(r, c, QTableWidgetItem(v))
        self._suppress = False

    def _add(self) -> None:
        n = 1
        names = {i.name.lower() for i in self.doc.pipeline.inputs}
        while f"value {n}" in names:
            n += 1
        self.doc.set_input(f"value {n}", 0, "", "")
        self.refill()
        row = self.table.rowCount() - 1
        if row >= 0:
            self.table.editItem(self.table.item(row, 0))

    def _remove(self) -> None:
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            name = self.table.item(r, 0).text()
            self.doc.remove_input(name)
        self.refill()

    def _changed(self, item: QTableWidgetItem) -> None:
        if self._suppress:
            return
        r = item.row()
        old_name = self.doc.pipeline.inputs[r].name if r < len(self.doc.pipeline.inputs) else None
        name = self.table.item(r, 0).text().strip()
        value = self.table.item(r, 1).text().strip()
        unit = self.table.item(r, 2).text().strip()
        note = self.table.item(r, 3).text().strip()
        try:
            if old_name and old_name != name:
                self.doc.undo.beginMacro("Rename input")
                self.doc.remove_input(old_name)
                self.doc.set_input(name, value, unit, note)
                self.doc.undo.endMacro()
            else:
                self.doc.set_input(name, value, unit, note)
        except Exception as e:
            self.doc.message.emit(str(e))
            self.refill()
