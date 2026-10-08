"""Column pickers, the formula editor and their parameter widgets."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import polars as pl
from PySide6.QtCore import Qt, Signal, QTimer, QStringListModel
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox,
                               QPushButton, QToolButton, QLabel, QListWidget, QListWidgetItem, QPlainTextEdit, QFileDialog,
                               QGridLayout, QRadioButton, QSizePolicy, QColorDialog, QCompleter)

from ...core.params import Param
from ...core.expr import kind_of_dtype, check_formula, function_docs, NUM, STR, TIME, BOOL
from ...core.conditions import OPS, ops_for_kind
from ...core.timeutil import parse_duration, parse_bucket
from ...core.nodes._common import STAT_CHOICES
from ..theme import T
from ...views.palette import series_color
from ..icons import icon
from ..openonclick import open_list_on_click

from ._base import KIND_ICON, ParamWidget, columns_for, kind_of, Schema  # noqa: E402

class ColumnCombo(QComboBox):
    """Editable combo listing columns with a type glyph. Blank allowed."""

    def __init__(self, allow_blank: bool = True, parent=None) -> None:
        super().__init__(parent); self.setMaxVisibleItems(25)
        self.setEditable(True)
        open_list_on_click(self)
        self.setInsertPolicy(QComboBox.NoInsert)
        self.allow_blank = allow_blank
        self._schema: Schema | None = None
        self.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.setMinimumContentsLength(6)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        # one completer for the widget's life: set_columns only swaps its model, so refreshes do not leak
        self._completer = QCompleter(self)
        self._completer.setCaseSensitivity(Qt.CaseInsensitive)
        self._completer.setFilterMode(Qt.MatchContains)
        self.setCompleter(self._completer)
        if allow_blank:
            self.lineEdit().setPlaceholderText("automatic")

    def set_columns(self, schema: Schema | None, group: str = "any") -> None:
        self._schema = schema
        cur = self.currentText()
        self.blockSignals(True)
        self.clear()
        if self.allow_blank:
            self.addItem("(automatic)", "")
        for c in columns_for(schema, group):
            self.addItem(f"{KIND_ICON.get(kind_of(schema, c), '?')}  {c}", c)
        self.setCurrentText(cur)
        self.blockSignals(False)
        # QCompleter.setModel takes ownership and frees the previous model, so refreshes do not leak
        self._completer.setModel(QStringListModel([c for c in columns_for(schema, group)], self._completer))

    def column(self) -> str:
        i = self.currentIndex()
        txt = self.currentText().strip()
        if i >= 0 and self.itemText(i) == self.currentText():
            return self.itemData(i) or ""
        # typed text: strip glyph prefix if present
        for k in range(self.count()):
            if self.itemText(k) == txt or self.itemData(k) == txt:
                return self.itemData(k) or ""
        return txt

    def set_column(self, name: str | None) -> None:
        name = name or ""
        for k in range(self.count()):
            if self.itemData(k) == name:
                self.setCurrentIndex(k)
                if name == "":
                    self.setEditText("")
                return
        self.setCurrentText(name)


class ColumnWidget(ParamWidget):
    def __init__(self, param: Param, parent=None) -> None:
        super().__init__(param, parent)
        lay = QHBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        self.combo = ColumnCombo(allow_blank=not param.required)
        lay.addWidget(self.combo)
        self.combo.currentIndexChanged.connect(lambda _: self.changed.emit())
        self.combo.lineEdit().editingFinished.connect(self.changed.emit)

    def set_schema(self, schema, schemas=None) -> None:
        super().set_schema(schema, schemas)
        s = schemas.get(self.param.port) if (schemas and self.param.port) else schema
        self.combo.set_columns(s, self.param.column_group)

    def value(self) -> Any:
        return self.combo.column()

    def set_value(self, v: Any) -> None:
        self.combo.set_column(v)


class ColumnsWidget(ParamWidget):
    """Searchable checklist of columns."""

    def __init__(self, param: Param, parent=None) -> None:
        super().__init__(param, parent)
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(3)
        top = QHBoxLayout()
        self.search = QLineEdit(); self.search.setPlaceholderText("filter…"); self.search.setClearButtonEnabled(True)
        self.search.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed); self.search.setMinimumWidth(40)
        all_b = QToolButton(); all_b.setObjectName("quiet"); all_b.setText("All"); none_b = QToolButton(); none_b.setObjectName("quiet"); none_b.setText("None")
        top.addWidget(self.search, 1); top.addWidget(all_b); top.addWidget(none_b)
        self.list = QListWidget(); self.list.setMaximumHeight(150); self.list.setStyleSheet(f"QListWidget {{ border: 1px solid {T.border}; border-radius: 4px; background: {T.panel}; }} QListWidget::item {{ padding: 2px 4px; }}")
        self.count = QLabel(""); self.count.setObjectName("muted")
        lay.addLayout(top); lay.addWidget(self.list); lay.addWidget(self.count)
        self._selected: list[str] = []
        self.search.textChanged.connect(self._refill)
        all_b.clicked.connect(lambda: self._set_all(True)); none_b.clicked.connect(lambda: self._set_all(False))
        self.list.itemChanged.connect(self._item_changed)

    def set_schema(self, schema, schemas=None) -> None:
        super().set_schema(schema, schemas)
        self._refill()

    def _cols(self) -> list[str]:
        s = self.schemas.get(self.param.port) if (self.schemas and self.param.port) else self.schema
        return columns_for(s, self.param.column_group)

    def _refill(self) -> None:
        q = self.search.text().lower()
        self.list.blockSignals(True)
        self.list.clear()
        cols = self._cols()
        shown = [c for c in cols if q in c.lower()] + [c for c in self._selected if c not in cols and q in c.lower()]
        for c in shown:
            it = QListWidgetItem(f"{KIND_ICON.get(kind_of(self.schema, c), '?')}  {c}")
            it.setData(Qt.UserRole, c)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if c in self._selected else Qt.Unchecked)
            if c not in cols:
                it.setForeground(QColor("#ef4444")); it.setToolTip("This column does not exist any more")
            self.list.addItem(it)
        self.list.blockSignals(False)
        self._update_count()

    def _update_count(self) -> None:
        n = len(self._selected)
        self.count.setText(f"{n} selected" if n else ("none selected" + (" = all columns" if not self.param.required else "")))

    def _set_all(self, on: bool) -> None:
        self._selected = list(self._cols()) if on else []
        self._refill(); self.changed.emit()

    def _item_changed(self, it: QListWidgetItem) -> None:
        c = it.data(Qt.UserRole)
        if it.checkState() == Qt.Checked and c not in self._selected:
            self._selected.append(c)
        elif it.checkState() != Qt.Checked and c in self._selected:
            self._selected.remove(c)
        self._update_count(); self.changed.emit()

    def value(self) -> Any:
        return list(self._selected)

    def set_value(self, v: Any) -> None:
        self._selected = list(v or [])
        self._refill()


# ---------------------------------------------------------------- formulas
class FormulaEditor(QWidget):
    changed = Signal()

    def __init__(self, parent=None, lines: int = 2) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(3)
        self.edit = QPlainTextEdit(); self.edit.setPlaceholderText("e.g. [Price] * [Quantity]")
        f = QFont("monospace"); f.setStyleHint(QFont.Monospace); self.edit.setFont(f)
        self.edit.setFixedHeight(24 * lines + 10)
        self.edit.setTabChangesFocus(True)
        helpers = QHBoxLayout(); helpers.setSpacing(4)
        self.col_combo = QComboBox(); self.col_combo.addItem("Insert column…")
        self.fn_combo = QComboBox(); self.fn_combo.addItem("Insert function…")
        for cb in (self.col_combo, self.fn_combo):
            cb.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon); cb.setMinimumContentsLength(8)
            cb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        for n, d in function_docs():
            self.fn_combo.addItem(n, n); self.fn_combo.setItemData(self.fn_combo.count() - 1, d, Qt.ToolTipRole)
        helpers.addWidget(self.col_combo, 1); helpers.addWidget(self.fn_combo, 1)
        self.msg = QLabel(""); self.msg.setWordWrap(True); self.msg.setObjectName("muted")
        lay.addWidget(self.edit); lay.addLayout(helpers); lay.addWidget(self.msg)
        self.schema: Schema | None = None
        self.edit.textChanged.connect(self._on_text)
        self.col_combo.activated.connect(self._insert_col)
        self.fn_combo.activated.connect(self._insert_fn)
        self._timer = QTimer(self); self._timer.setSingleShot(True); self._timer.setInterval(250); self._timer.timeout.connect(self.validate)

    def set_schema(self, schema: Schema | None) -> None:
        self.schema = schema
        self.col_combo.blockSignals(True)
        self.col_combo.clear(); self.col_combo.addItem("Insert column…")
        for c in (schema or {}):
            self.col_combo.addItem(f"{KIND_ICON.get(kind_of(schema, c), '?')}  {c}", c)
        self.col_combo.blockSignals(False)
        self.validate()

    def _insert_col(self, i: int) -> None:
        c = self.col_combo.itemData(i)
        if c:
            self.edit.insertPlainText(f"[{c}]" if not c.isidentifier() else c)
            self.edit.setFocus()
        self.col_combo.setCurrentIndex(0)

    def _insert_fn(self, i: int) -> None:
        n = self.fn_combo.itemData(i)
        if n:
            self.edit.insertPlainText(f"{n}(")
            self.edit.setFocus()
        self.fn_combo.setCurrentIndex(0)

    def _on_text(self) -> None:
        self._timer.start()
        self.changed.emit()

    def text(self) -> str:
        return self.edit.toPlainText()

    def set_text(self, t: str) -> None:
        if self.edit.toPlainText() != (t or ""):
            self.edit.blockSignals(True); self.edit.setPlainText(t or ""); self.edit.blockSignals(False)
        self.validate()

    def validate(self) -> str | None:
        t = self.text().strip()
        if not t:
            self.msg.setText(""); return None
        if self.schema is None:
            self.msg.setText("Connect an input to check this formula"); return None
        err = check_formula(t, self.schema)
        if err:
            self.msg.setText(f"⚠ {err}"); self.msg.setStyleSheet("color: #ef4444;")
            return err
        from ...core.expr import compile_formula
        try:
            _, kind, used = compile_formula(t, self.schema)
        except Exception as e:  # check_formula passed but compiling did not: show the real reason, not a tick
            err = str(e) or "Could not check this formula"
            self.msg.setText(f"⚠ {err}"); self.msg.setStyleSheet("color: #ef4444;")
            return err
        self.msg.setText(f"✓ gives a {kind}" + (f" · uses {', '.join(sorted(used))}" if used else ""))
        self.msg.setStyleSheet(f"color: {T.muted};")
        return None


class ExprWidget(ParamWidget):
    immediate = False

    def __init__(self, param: Param, parent=None) -> None:
        super().__init__(param, parent)
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        self.editor = FormulaEditor()
        lay.addWidget(self.editor)
        self.editor.changed.connect(self.changed.emit)

    def set_schema(self, schema, schemas=None) -> None:
        super().set_schema(schema, schemas)
        self.editor.set_schema(schema)

    def value(self) -> Any:
        return self.editor.text()

    def set_value(self, v: Any) -> None:
        self.editor.set_text(v or "")

    def problem(self) -> str | None:
        return self.editor.validate()

