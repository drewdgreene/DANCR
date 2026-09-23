"""Parameter widgets generated from Param specs."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import polars as pl
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox,
                               QPushButton, QToolButton, QLabel, QListWidget, QListWidgetItem, QPlainTextEdit, QFileDialog,
                               QGridLayout, QRadioButton, QSizePolicy, QColorDialog, QCompleter)

from ..core.params import Param
from ..core.expr import _kind_of_dtype, check_formula, function_docs, NUM, STR, TIME, BOOL
from ..core.conditions import OPS, ops_for_kind
from ..core.timeutil import parse_duration
from ..core.nodes._common import STAT_CHOICES
from .theme import T, SERIES_COLORS
from .icons import icon
from .openonclick import open_list_on_click

Schema = dict[str, pl.DataType]
KIND_ICON = {NUM: "#", STR: "Aa", TIME: "◷", BOOL: "✓", "duration": "Δ", "any": "?"}


def kind_of(schema: Schema | None, col: str) -> str:
    if not schema or col not in schema:
        return "any"
    return _kind_of_dtype(schema[col])


def columns_for(schema: Schema | None, group: str) -> list[str]:
    if not schema:
        return []
    if group == "any":
        return list(schema)
    want = {"numeric": NUM, "temporal": TIME, "string": STR, "bool": BOOL}[group]
    return [c for c, dt in schema.items() if _kind_of_dtype(dt) == want]


class ParamWidget(QWidget):
    changed = Signal()
    immediate = True     # commit without debounce

    def __init__(self, param: Param, parent=None) -> None:
        super().__init__(parent)
        self.param = param
        self.schema: Schema | None = None
        self.schemas: dict[str, Schema] = {}

    def value(self) -> Any:
        raise NotImplementedError

    def set_value(self, v: Any) -> None:
        raise NotImplementedError

    def set_schema(self, schema: Schema | None, schemas: dict[str, Schema] | None = None) -> None:
        self.schema = schema
        self.schemas = schemas or {}

    def problem(self) -> str | None:
        return None

    def is_blank(self) -> bool:
        v = self.value()
        return v in (None, "", [], {}) or (isinstance(v, dict) and not v.get("rules"))

    def focus_entry(self) -> None:
        """Put the cursor where the person most likely wants to type."""
        self.setFocus()


# ---------------------------------------------------------------- simple kinds
class TextWidget(ParamWidget):
    immediate = False

    def __init__(self, param: Param, parent=None) -> None:
        super().__init__(param, parent)
        lay = QHBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit(); self.edit.setPlaceholderText(param.placeholder or "")
        lay.addWidget(self.edit)
        self.edit.textEdited.connect(lambda _: self.changed.emit())

    def value(self) -> Any:
        return self.edit.text()

    def set_value(self, v: Any) -> None:
        self.edit.setText("" if v is None else str(v))


class DurationWidget(TextWidget):
    def __init__(self, param: Param, parent=None) -> None:
        super().__init__(param, parent)
        self.edit.setPlaceholderText(param.placeholder or "e.g. 30s, 5m, 1h, 1d")
        self.edit.textChanged.connect(self._validate)

    def _validate(self, text: str) -> None:
        ok = True
        if text.strip():
            try:
                parse_duration(text)
            except ValueError:
                ok = False
        self.edit.setStyleSheet("" if ok else "QLineEdit { border: 1px solid #ef4444; }")

    def problem(self) -> str | None:
        t = self.edit.text().strip()
        if not t:
            return None
        try:
            parse_duration(t); return None
        except ValueError as e:
            return str(e)


class IntWidget(ParamWidget):
    def __init__(self, param: Param, parent=None) -> None:
        super().__init__(param, parent)
        lay = QHBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        self.spin = QSpinBox()
        self.spin.setRange(int(param.min) if param.min is not None else -10**9, int(param.max) if param.max is not None else 10**9)
        self.spin.setSingleStep(int(param.step or 1))
        lay.addWidget(self.spin); lay.addStretch()
        self.spin.valueChanged.connect(lambda _: self.changed.emit())

    def value(self) -> Any:
        return self.spin.value()

    def set_value(self, v: Any) -> None:
        try:
            self.spin.setValue(int(v))
        except (TypeError, ValueError):
            self.spin.setValue(int(self.param.default or 0))


class FloatWidget(ParamWidget):
    def __init__(self, param: Param, parent=None) -> None:
        super().__init__(param, parent)
        lay = QHBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        self.spin = QDoubleSpinBox()
        self.spin.setDecimals(4)
        self.spin.setRange(param.min if param.min is not None else -1e12, param.max if param.max is not None else 1e12)
        self.spin.setSingleStep(param.step or 0.5)
        lay.addWidget(self.spin); lay.addStretch()
        self.spin.valueChanged.connect(lambda _: self.changed.emit())

    def value(self) -> Any:
        return self.spin.value()

    def set_value(self, v: Any) -> None:
        try:
            self.spin.setValue(float(v))
        except (TypeError, ValueError):
            self.spin.setValue(float(self.param.default or 0))


class BoolWidget(ParamWidget):
    def __init__(self, param: Param, parent=None) -> None:
        super().__init__(param, parent)
        lay = QHBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        self.box = QCheckBox(param.label)
        lay.addWidget(self.box)
        self.box.toggled.connect(lambda _: self.changed.emit())

    def value(self) -> Any:
        return self.box.isChecked()

    def set_value(self, v: Any) -> None:
        self.box.setChecked(bool(v))


class ChoiceWidget(ParamWidget):
    def __init__(self, param: Param, parent=None) -> None:
        super().__init__(param, parent)
        lay = QHBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        self.combo = QComboBox(); self.combo.setMaxVisibleItems(25)
        self.combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.combo.setMinimumContentsLength(8)
        self.combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        for v, label in param.choices or []:
            self.combo.addItem(label, v)
            self.combo.setItemData(self.combo.count() - 1, label, Qt.ToolTipRole)
        lay.addWidget(self.combo)
        self.combo.currentIndexChanged.connect(lambda _: self.changed.emit())

    def value(self) -> Any:
        return self.combo.currentData()

    def set_value(self, v: Any) -> None:
        i = self.combo.findData(v)
        self.combo.setCurrentIndex(i if i >= 0 else 0)


class PathWidget(ParamWidget):
    immediate = False

    def __init__(self, param: Param, parent=None, save: bool = False) -> None:
        super().__init__(param, parent)
        self.save = save
        lay = QHBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit(); self.edit.setPlaceholderText("Choose a file…"); self.edit.setMinimumWidth(60)
        btn = QPushButton("Browse…"); btn.clicked.connect(self.browse)
        lay.addWidget(self.edit, 1); lay.addWidget(btn)
        self.edit.textEdited.connect(lambda _: self.changed.emit())
        self.base_dir: Path | None = None

    def browse(self) -> None:
        start = str(self.base_dir or Path.home())
        if self.save:
            filt = "Report (*.html)" if self.param.help and ".html" in self.param.help else "CSV (*.csv);;Excel (*.xlsx);;Parquet (*.parquet)"
            cur = self.edit.text().strip()
            if cur:
                start = str((self.base_dir / cur) if (self.base_dir and not Path(cur).is_absolute()) else cur)
            elif "html" in filt:
                start = str(Path(start) / "report.html")
            f, _ = QFileDialog.getSaveFileName(self, "Save as", start, filt)
            if f and "html" in filt and not f.lower().endswith((".html", ".htm")):
                f += ".html"
        else:
            f, _ = QFileDialog.getOpenFileName(self, "Choose a data file", start,
                                               "Data files (*.csv *.tsv *.txt *.dat *.xlsx *.xlsm *.xls *.parquet);;All files (*)")
        if f:
            p = Path(f)
            if self.base_dir:
                try:
                    p = p.relative_to(self.base_dir)
                except ValueError:
                    pass
            self.edit.setText(str(p))
            self.changed.emit()

    def value(self) -> Any:
        return self.edit.text()

    def set_value(self, v: Any) -> None:
        self.edit.setText("" if v is None else str(v))


class SheetWidget(ParamWidget):
    """Excel sheet chooser; fed by the sibling 'path' param."""

    def __init__(self, param: Param, parent=None) -> None:
        super().__init__(param, parent)
        lay = QHBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        self.combo = QComboBox(); self.combo.setMaxVisibleItems(25); self.combo.setEditable(True); open_list_on_click(self.combo)
        self.combo.lineEdit().setPlaceholderText("first sheet")
        lay.addWidget(self.combo)
        self.combo.currentTextChanged.connect(lambda _: self.changed.emit())

    def set_sheets(self, sheets: list[str]) -> None:
        cur = self.combo.currentText()
        self.combo.blockSignals(True)
        self.combo.clear(); self.combo.addItem("")
        for s in sheets:
            self.combo.addItem(s)
        self.combo.setCurrentText(cur)
        self.combo.blockSignals(False)

    def value(self) -> Any:
        return self.combo.currentText()

    def set_value(self, v: Any) -> None:
        self.combo.setCurrentText("" if v is None else str(v))


class TextListWidget(TextWidget):
    def __init__(self, param: Param, parent=None) -> None:
        super().__init__(param, parent)
        self.edit.setPlaceholderText(param.placeholder or "comma separated, e.g. mean, max")

    def value(self) -> Any:
        return [s.strip() for s in self.edit.text().split(",") if s.strip()]

    def set_value(self, v: Any) -> None:
        self.edit.setText(", ".join(v or []) if isinstance(v, list) else str(v or ""))


# ---------------------------------------------------------------- columns
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
        comp = QCompleter([c for c in columns_for(schema, group)], self)
        comp.setCaseSensitivity(Qt.CaseInsensitive)
        comp.setFilterMode(Qt.MatchContains)
        self.setCompleter(comp)

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
        else:
            from ..core.expr import compile_formula
            try:
                _, kind, used = compile_formula(t, self.schema)
                self.msg.setText(f"✓ gives a {kind}" + (f" · uses {', '.join(sorted(used))}" if used else ""))
            except Exception:
                self.msg.setText("✓")
            self.msg.setStyleSheet(f"color: {T.muted};")
        return err


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


def remove_button(tip: str) -> QToolButton:
    rm = QToolButton(); rm.setObjectName("quiet"); rm.setIcon(icon("x", T.muted)); rm.setToolTip(tip)
    return rm


class RowListWidget(ParamWidget):
    """A vertical list of row widgets with an Add button. Each row is a QWidget with ``changed`` and
    ``removed(row)`` signals and an ``item()`` giving its value; subclasses build rows and shape the value."""
    add_text = "+ Add"
    spacing = 4
    keep_one_blank = False          # show one empty row when the value is empty

    def __init__(self, param: Param, parent=None) -> None:
        super().__init__(param, parent)
        self.lay = QVBoxLayout(self); self.lay.setContentsMargins(0, 0, 0, 0); self.lay.setSpacing(self.spacing)
        self.rows: list[QWidget] = []
        self.add_btn = QPushButton(self.add_text); self.add_btn.clicked.connect(lambda: (self.add_row(self.empty_item()), self.changed.emit()))
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
        match = QHBoxLayout()
        self.all_b = QRadioButton("Match all conditions"); self.any_b = QRadioButton("Match any")
        self.all_b.setChecked(True)
        match.addWidget(self.all_b); match.addWidget(self.any_b); match.addStretch()
        self.lay.insertLayout(0, match)
        self.all_b.toggled.connect(lambda _: self.changed.emit())

    def is_blank(self) -> bool:
        from ..core.conditions import incomplete_rules
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
        return SeriesRow(item, self.schema, self.param.column_group, SERIES_COLORS[len(self.rows) % len(SERIES_COLORS)])

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
class FixesWidget(ParamWidget):
    """The list of corrections made from the table's cell menu; each can be removed here."""

    def __init__(self, param: Param, parent=None) -> None:
        super().__init__(param, parent)
        self.lay = QVBoxLayout(self); self.lay.setContentsMargins(0, 0, 0, 0); self.lay.setSpacing(3)
        self._fixes: list[dict[str, Any]] = []
        self.hint = QLabel("Right-click a cell in the table and choose Fix this value to add one."); self.hint.setObjectName("muted"); self.hint.setWordWrap(True)
        self.lay.addWidget(self.hint)

    def _rebuild(self) -> None:
        while self.lay.count() > 1:
            w = self.lay.takeAt(1).widget()
            if w:
                w.deleteLater()
        for i, f in enumerate(self._fixes):
            w = QWidget(); h = QHBoxLayout(w); h.setContentsMargins(0, 0, 0, 0); h.setSpacing(4)
            was = "empty" if f.get("was") in (None, "") else str(f.get("was"))
            new = "empty" if f.get("value") in (None, "") else str(f.get("value"))
            lab = QLabel(f"Row {int(f.get('row', 0)):,}, <b>{f.get('column')}</b>: {was} → {new}" + (f"<br><span style='color:{T.muted}'>{f['note']}</span>" if f.get("note") else ""))
            lab.setWordWrap(True); lab.setTextFormat(Qt.RichText)
            rm = QToolButton(); rm.setObjectName("quiet"); rm.setIcon(icon("x", T.muted)); rm.setToolTip("Remove this correction")
            rm.clicked.connect(lambda _=False, i=i: self._remove(i))
            h.addWidget(lab, 1); h.addWidget(rm, 0, Qt.AlignTop)
            self.lay.addWidget(w)
        self.hint.setVisible(not self._fixes)

    def _remove(self, i: int) -> None:
        del self._fixes[i]; self._rebuild(); self.changed.emit()

    def value(self) -> Any:
        return [dict(f) for f in self._fixes]

    def set_value(self, v: Any) -> None:
        v = [dict(f) for f in (v or [])]
        if v == self._fixes:
            return
        self._fixes = v; self._rebuild()


# ---------------------------------------------------------------- factory
def make_widget(param: Param, node_type_key: str, suggest: Callable[[str], list[str]] | None = None) -> ParamWidget:
    k = param.kind
    if node_type_key == "load_file" and param.name == "sheet":
        return SheetWidget(param)
    if k == "text":
        return TextWidget(param)
    if k == "int":
        return IntWidget(param)
    if k == "float":
        return FloatWidget(param)
    if k == "bool":
        return BoolWidget(param)
    if k == "choice":
        return ChoiceWidget(param)
    if k == "path":
        return PathWidget(param, save=(node_type_key in ("export", "report", "workbook")))
    if k == "duration":
        return DurationWidget(param)
    if k == "column":
        return ColumnWidget(param)
    if k == "columns":
        return ColumnsWidget(param)
    if k == "expr":
        return ExprWidget(param)
    if k == "formulas":
        return FormulasWidget(param)
    if k == "conditions":
        return ConditionsWidget(param, suggest=suggest)
    if k == "aggregations":
        return AggregationsWidget(param)
    if k == "mapping":
        return MappingWidget(param)
    if k == "series":
        return SeriesWidget(param)
    if k == "text_list":
        return TextListWidget(param)
    if k == "limits":
        return LimitsWidget(param)
    if k == "fixes":
        return FixesWidget(param)
    return TextWidget(param)
