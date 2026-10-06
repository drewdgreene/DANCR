"""The row-list family: formulas, conditions, aggregations, mappings, series, limits and fixes."""
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

from ._base import ParamWidget, columns_for, kind_of, Schema  # noqa: E402
from ._columns import ColumnCombo, ExprWidget, FormulaEditor  # noqa: E402

def remove_button(tip: str) -> QToolButton:
    rm = QToolButton(); rm.setObjectName("quiet"); rm.setIcon(icon("x", T.muted)); rm.setToolTip(tip)
    return rm


class RowListWidget(ParamWidget):
    """A vertical list of row widgets with an Add button. Each row is a QWidget with ``changed`` and
    ``removed(row)`` signals and an ``item()`` giving its value; subclasses build rows and shape the value."""
    add_text = "+ Add"
    spacing = 4
    keep_one_blank = False          # show one empty row when the value is empty
    can_add = True                  # False when rows come from elsewhere (the table's cell menu)

    def __init__(self, param: Param, parent=None) -> None:
        super().__init__(param, parent)
        self.lay = QVBoxLayout(self); self.lay.setContentsMargins(0, 0, 0, 0); self.lay.setSpacing(self.spacing)
        self.rows: list[QWidget] = []
        self.add_btn = QPushButton(self.add_text); self.add_btn.clicked.connect(lambda: (self.add_row(self.empty_item()), self.changed.emit()))
        self.add_btn.setVisible(self.can_add)
        self.lay.addWidget(self.add_btn)

    def empty_item(self) -> Any:
        return {}

    def build_row(self, item: Any) -> QWidget:
        raise NotImplementedError

    def items(self, v: Any) -> list:
        """The per-row items a stored value holds."""
        return list(v or [])

    def add_row(self, item: Any) -> QWidget:
        row = self.build_row(item)
        row.changed.connect(self.changed.emit)
        row.removed.connect(self.remove_row)
        self.lay.insertWidget(self.lay.count() - 1, row)
        self.rows.append(row)
        return row

    def remove_row(self, row: QWidget) -> None:
        self.rows = [r for r in self.rows if r is not row]
        row.setParent(None); row.deleteLater()
        self.changed.emit()

    def value(self) -> Any:
        return [r.item() for r in self.rows]

    def set_value(self, v: Any) -> None:
        items = self.items(v)
        if self.rows and self.value() == (v if v is not None else self.value()):
            return
        for r in self.rows:
            r.setParent(None); r.deleteLater()
        self.rows = []
        for it in items:
            self.add_row(it)
        if self.keep_one_blank and not self.rows:
            self.add_row(self.empty_item())

    def set_schema(self, schema, schemas=None) -> None:
        super().set_schema(schema, schemas)
        for r in self.rows:
            self.apply_schema(r)

    def apply_schema(self, row: QWidget) -> None:
        if hasattr(row, "col"):
            row.col.set_columns(self.schema, self.param.column_group)


class _Row(QWidget):
    changed = Signal()
    removed = Signal(object)

    def _remove_button(self, tip: str) -> QToolButton:
        rm = remove_button(tip); rm.clicked.connect(lambda: self.removed.emit(self))
        return rm


class FormulaRow(_Row):
    def __init__(self, item: dict, schema: Schema | None) -> None:
        super().__init__()
        self.setObjectName("row"); self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(f"QWidget#row {{ border-top: 1px solid {T.border_soft}; padding-top: 4px; }}")
        v = QVBoxLayout(self); v.setContentsMargins(0, 6, 0, 2); v.setSpacing(4)
        top = QHBoxLayout()
        self.name = QLineEdit(item.get("name", "")); self.name.setPlaceholderText("new column name")
        top.addWidget(QLabel("Name")); top.addWidget(self.name, 1); top.addWidget(self._remove_button("Remove this formula"))
        self.editor = FormulaEditor(); self.editor.set_text(item.get("expr", "")); self.editor.set_schema(schema)
        v.addLayout(top); v.addWidget(self.editor)
        self.name.textEdited.connect(lambda _: self.changed.emit())
        self.editor.changed.connect(self.changed.emit)

    def item(self) -> dict:
        return {"name": self.name.text().strip(), "expr": self.editor.text().strip()}


class FormulasWidget(RowListWidget):
    """List of (new column name, formula). Each formula sees the columns the ones above it create."""
    immediate = False
    add_text = "+ Add formula"
    spacing = 6
    keep_one_blank = True

    def empty_item(self) -> dict:
        return {"name": "", "expr": ""}

    def build_row(self, item: dict) -> FormulaRow:
        row = FormulaRow(item, self._chain_schema(len(self.rows)))
        row.editor.changed.connect(self._rechain)
        return row

    def _chain_schema(self, upto: int) -> Schema | None:
        if self.schema is None:
            return None
        s = dict(self.schema)
        for r in self.rows[:upto]:
            if r.name.text().strip():
                s[r.name.text().strip()] = pl.Float64
        return s

    def _rechain(self) -> None:
        for i, r in enumerate(self.rows):
            r.editor.schema = self._chain_schema(i)

    def apply_schema(self, row: FormulaRow) -> None:
        row.editor.set_schema(self._chain_schema(self.rows.index(row)))

    def problem(self) -> str | None:
        for r in self.rows:
            if r.editor.text().strip() and not r.name.text().strip():
                return "Give the new column a name"
            err = r.editor.validate()
            if err:
                return err
        return None


# ---------------------------------------------------------------- conditions
class ConditionRow(_Row):
    def __init__(self, schema: Schema | None, suggest: Callable[[str], list[str]] | None, parent=None) -> None:
        super().__init__(parent)
        self.schema = schema
        self.suggest = suggest
        g = QGridLayout(self); g.setContentsMargins(0, 0, 0, 0); g.setHorizontalSpacing(4); g.setVerticalSpacing(3)
        self.col = ColumnCombo(allow_blank=True); self.col.set_columns(schema)
        self.op = QComboBox(); self.op.setMaxVisibleItems(25)
        self.val = QComboBox(); self.val.setMaxVisibleItems(25); self.val.setEditable(True); open_list_on_click(self.val); self.val.setInsertPolicy(QComboBox.NoInsert); self.val.lineEdit().setPlaceholderText("value")
        self.val2 = QComboBox(); self.val2.setEditable(True); self.val2.lineEdit().setPlaceholderText("and"); self.val2.hide()
        for cb in (self.op, self.val, self.val2):
            cb.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon); cb.setMinimumContentsLength(6)
            cb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        g.addWidget(self.col, 0, 0, 1, 2); g.addWidget(self._remove_button("Remove this condition"), 0, 2)
        g.addWidget(self.op, 1, 0); g.addWidget(self.val, 1, 1, 1, 2)
        g.addWidget(self.val2, 2, 1, 1, 2)
        self.col.currentIndexChanged.connect(self._col_changed)
        self.col.lineEdit().editingFinished.connect(self._col_changed)
        self.op.currentIndexChanged.connect(self._op_changed)
        self.val.currentTextChanged.connect(lambda _: self.changed.emit())
        self.val2.currentTextChanged.connect(lambda _: self.changed.emit())
        self._fill_ops()

    def _fill_ops(self) -> None:
        cur = self.op.currentData()
        self.op.blockSignals(True)
        self.op.clear()
        for k, label in ops_for_kind(kind_of(self.schema, self.col.column())):
            self.op.addItem(label, k)
        i = self.op.findData(cur)
        self.op.setCurrentIndex(i if i >= 0 else 0)
        self.op.blockSignals(False)
        self._op_changed()

    def _col_changed(self, *_: Any) -> None:
        self._fill_ops()
        c = self.col.column()
        if self.suggest and c:
            cur = self.val.currentText()
            self.val.blockSignals(True)
            self.val.clear(); self.val.addItems(self.suggest(c)); self.val.setCurrentText(cur)
            self.val.blockSignals(False)
        self.changed.emit()

    def _op_changed(self, *_: Any) -> None:
        k = self.op.currentData()
        spec = OPS.get(k or "eq")
        self.val.setVisible(bool(spec and spec[1]))
        self.val2.setVisible(bool(spec and spec[2]))
        self.changed.emit()

    def item(self) -> dict[str, Any]:
        return {"column": self.col.column(), "op": self.op.currentData() or "eq",
                "value": self.val.currentText(), "value2": self.val2.currentText()}

    def set_rule(self, r: dict[str, Any]) -> None:
        for w in (self.col, self.op, self.val, self.val2):
            w.blockSignals(True)
        self.col.set_column(r.get("column"))
        self._fill_ops()
        i = self.op.findData(r.get("op", "eq"))
        self.op.setCurrentIndex(i if i >= 0 else 0)
        self.val.setCurrentText(str(r.get("value", "") or ""))
        self.val2.setCurrentText(str(r.get("value2", "") or ""))
        for w in (self.col, self.op, self.val, self.val2):
            w.blockSignals(False)
        self._op_changed()


class ConditionsWidget(RowListWidget):
    immediate = False
    add_text = "+ Add condition"
    spacing = 6
    keep_one_blank = True

    def __init__(self, param: Param, parent=None, suggest: Callable[[str], list[str]] | None = None) -> None:
        super().__init__(param, parent)
        self.suggest = suggest
        match_row = QHBoxLayout()
        self.all_b = QRadioButton("Match all conditions"); self.any_b = QRadioButton("Match any")
        self.all_b.setChecked(True)
        match_row.addWidget(self.all_b); match_row.addWidget(self.any_b); match_row.addStretch()
        self.lay.insertLayout(0, match_row)
        self.all_b.toggled.connect(lambda _: self.changed.emit())

    def is_blank(self) -> bool:
        from ...core.conditions import incomplete_rules
        v = self.value()
        rules = [r for r in (v.get("rules") or []) if r.get("column")]
        return not rules or len(incomplete_rules(v)) == len(rules)

    def focus_entry(self) -> None:
        for row in self.rows:
            if not row.val.isHidden() and not row.val.currentText().strip():
                row.val.setFocus(); row.val.lineEdit().selectAll(); return
        if self.rows:
            self.rows[0].col.setFocus()
        else:
            self.setFocus()

    def build_row(self, item: dict) -> ConditionRow:
        row = ConditionRow(self.schema, self.suggest)
        row.set_rule(item)
        return row

    def items(self, v: Any) -> list:
        return list((v or {}).get("rules") or [])

    def apply_schema(self, row: ConditionRow) -> None:
        row.schema = self.schema
        row.col.set_columns(self.schema)
        row._fill_ops()

    def value(self) -> Any:
        return {"match": "all" if self.all_b.isChecked() else "any", "rules": [r.item() for r in self.rows]}

    def set_value(self, v: Any) -> None:
        v = v or {"match": "all", "rules": []}
        (self.all_b if v.get("match", "all") == "all" else self.any_b).setChecked(True)
        super().set_value(v)


# ---------------------------------------------------------------- aggregations
class CheckCombo(QComboBox):
    """Multi-select combo with checkable items."""
    changed = Signal()

    def __init__(self, items: list[tuple[str, str]], parent=None) -> None:
        super().__init__(parent)
        from PySide6.QtGui import QStandardItemModel, QStandardItem
        self._model = QStandardItemModel(self)
        for v, label in items:
            it = QStandardItem(label); it.setData(v, Qt.UserRole); it.setCheckable(True); it.setCheckState(Qt.Unchecked)
            self._model.appendRow(it)
        self.setModel(self._model)
        self._model.itemChanged.connect(lambda _: (self._refresh_text(), self.changed.emit()))
        self.setEditable(True); self.lineEdit().setReadOnly(True)
        self._refresh_text()

    def values(self) -> list[str]:
        return [self._model.item(i).data(Qt.UserRole) for i in range(self._model.rowCount()) if self._model.item(i).checkState() == Qt.Checked]

    def set_values(self, vals: list[str]) -> None:
        self._model.blockSignals(True)
        for i in range(self._model.rowCount()):
            it = self._model.item(i)
            it.setCheckState(Qt.Checked if it.data(Qt.UserRole) in vals else Qt.Unchecked)
        self._model.blockSignals(False)
        self._refresh_text()

    def _refresh_text(self) -> None:
        self.lineEdit().setText(", ".join(self.values()) or "choose…")


class AggregationRow(_Row):
    def __init__(self, item: dict, schema: Schema | None, group: str) -> None:
        super().__init__()
        h = QHBoxLayout(self); h.setContentsMargins(0, 0, 0, 0); h.setSpacing(4)
        self.col = ColumnCombo(allow_blank=True); self.col.set_columns(schema, group); self.col.set_column(item.get("column"))
        self.stats = CheckCombo(STAT_CHOICES); self.stats.set_values(item.get("stats") or ["mean"])
        h.addWidget(self.col, 3); h.addWidget(self.stats, 2); h.addWidget(self._remove_button("Remove this column"))
        self.col.currentIndexChanged.connect(lambda _: self.changed.emit()); self.stats.changed.connect(self.changed.emit)

    def item(self) -> dict:
        return {"column": self.col.column(), "stats": self.stats.values()}


class AggregationsWidget(RowListWidget):
    add_text = "+ Add column"

    def build_row(self, item: dict) -> AggregationRow:
        return AggregationRow(item, self.schema, self.param.column_group)

    def value(self) -> Any:
        return [r.item() for r in self.rows if r.col.column()]


# ---------------------------------------------------------------- mapping / series
class MappingRow(_Row):
    def __init__(self, item: tuple[str, str], schema: Schema | None) -> None:
        super().__init__()
        old, new = item
        h = QHBoxLayout(self); h.setContentsMargins(0, 0, 0, 0); h.setSpacing(4)
        self.col = ColumnCombo(); self.col.set_columns(schema); self.col.set_column(old)
        self.edit = QLineEdit(new); self.edit.setPlaceholderText("new name")
        h.addWidget(self.col, 3); h.addWidget(QLabel("→")); h.addWidget(self.edit, 3); h.addWidget(self._remove_button("Remove this rename"))
        self.col.currentIndexChanged.connect(lambda _: self.changed.emit()); self.edit.textEdited.connect(lambda _: self.changed.emit())

    def item(self) -> tuple[str, str]:
        return self.col.column(), self.edit.text().strip()


class MappingWidget(RowListWidget):
    immediate = False
    add_text = "+ Rename a column"

    def empty_item(self) -> tuple[str, str]:
        return "", ""

    def build_row(self, item: tuple[str, str]) -> MappingRow:
        return MappingRow(item, self.schema)

    def items(self, v: Any) -> list:
        return list((v or {}).items())

    def apply_schema(self, row: MappingRow) -> None:
        row.col.set_columns(self.schema)

    def value(self) -> Any:
        return {c: n for c, n in (r.item() for r in self.rows) if c and n}


class SeriesRow(_Row):
    def __init__(self, item: dict, schema: Schema | None, group: str, default_color: str) -> None:
        super().__init__()
        g = QGridLayout(self); g.setContentsMargins(0, 0, 0, 4); g.setHorizontalSpacing(4); g.setVerticalSpacing(3)
        self.col = ColumnCombo(allow_blank=True); self.col.set_columns(schema, group); self.col.set_column(item.get("column"))
        self.color_b = QPushButton(); self.color_b.setFixedWidth(28); self.color_b.setToolTip("Colour")
        self._set_color(item.get("color") or default_color)
        self.label = QLineEdit(item.get("label") or ""); self.label.setPlaceholderText("name shown on the chart (optional)")
        g.addWidget(self.col, 0, 0); g.addWidget(self.color_b, 0, 1); g.addWidget(self._remove_button("Remove this series"), 0, 2)
        g.addWidget(self.label, 1, 0, 1, 3)
        self.col.currentIndexChanged.connect(lambda _: self.changed.emit())
        self.label.editingFinished.connect(self.changed.emit)
        self.color_b.clicked.connect(self._pick_color)

    def _set_color(self, color: str) -> None:
        self.color_b.setProperty("color", color); self.color_b.setStyleSheet(f"background: {color}; border-radius: 4px;")

    def _pick_color(self) -> None:
        c = QColorDialog.getColor(QColor(self.color_b.property("color")), self, "Series colour")
        if c.isValid():
            self._set_color(c.name()); self.changed.emit()

    def item(self) -> dict:
        d = {"column": self.col.column(), "color": self.color_b.property("color")}
        if self.label.text().strip():
            d["label"] = self.label.text().strip()
        return d


class SeriesWidget(RowListWidget):
    add_text = "+ Add series"
    keep_one_blank = True

    def build_row(self, item: dict) -> SeriesRow:
        return SeriesRow(item, self.schema, self.param.column_group, series_color(len(self.rows)))

    def value(self) -> Any:
        return [r.item() for r in self.rows if r.col.column()]


# ---------------------------------------------------------------- limits (chart limit lines)
class LimitRow(_Row):
    def __init__(self, item: dict) -> None:
        super().__init__()
        h = QHBoxLayout(self); h.setContentsMargins(0, 0, 0, 0); h.setSpacing(4)
        self.value_e = QLineEdit(str(item.get("value", "") if item.get("value") is not None else "")); self.value_e.setPlaceholderText("number or input name")
        self.label = QLineEdit(item.get("label") or ""); self.label.setPlaceholderText("label")
        h.addWidget(self.value_e, 2); h.addWidget(self.label, 2); h.addWidget(self._remove_button("Remove this limit line"))
        self.value_e.editingFinished.connect(self.changed.emit); self.label.editingFinished.connect(self.changed.emit)

    def item(self) -> dict:
        v = self.value_e.text().strip()
        return {"value": v, "label": self.label.text().strip() or v}


class LimitsWidget(RowListWidget):
    """Rows of value + label. Values may be numbers or the names of inputs."""
    add_text = "+ Add limit line"

    def build_row(self, item: dict) -> LimitRow:
        return LimitRow(item)

    def value(self) -> Any:
        return [r.item() for r in self.rows if r.value_e.text().strip()]


# ---------------------------------------------------------------- fixes (cell corrections)
class FixRow(_Row):
    """One correction: where, what it was, what it is now, and why."""

    def __init__(self, item: dict) -> None:
        super().__init__()
        self._item = dict(item)
        h = QHBoxLayout(self); h.setContentsMargins(0, 0, 0, 0); h.setSpacing(4)
        was = "empty" if item.get("was") in (None, "") else str(item.get("was"))
        new = "empty" if item.get("value") in (None, "") else str(item.get("value"))
        lab = QLabel(f"Row {int(item.get('row', 0)):,}, <b>{item.get('column')}</b>: {was} → {new}" + (f"<br><span style='color:{T.muted}'>{item['note']}</span>" if item.get("note") else ""))
        lab.setWordWrap(True); lab.setTextFormat(Qt.RichText)
        h.addWidget(lab, 1); h.addWidget(self._remove_button("Remove this correction"), 0, Qt.AlignTop)

    def item(self) -> dict:
        return dict(self._item)


class FixesWidget(RowListWidget):
    """The list of corrections made from the table's cell menu; each can be removed here."""
    spacing = 3
    can_add = False

    def __init__(self, param: Param, parent=None) -> None:
        super().__init__(param, parent)
        self.hint = QLabel("Right-click a cell in the table and choose Fix this value to add one."); self.hint.setObjectName("muted"); self.hint.setWordWrap(True)
        self.lay.insertWidget(0, self.hint)
        self.changed.connect(self._sync_hint)

    def build_row(self, item: Any) -> QWidget:
        return FixRow(item)

    def set_value(self, v: Any) -> None:
        super().set_value(v)
        self._sync_hint()

    def _sync_hint(self) -> None:
        self.hint.setVisible(not self.rows)


# ---------------------------------------------------------------- factory
