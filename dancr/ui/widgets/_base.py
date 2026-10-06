"""Parameter-widget base and the simple scalar widgets."""
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


Schema = dict[str, pl.DataType]
KIND_ICON = {NUM: "#", STR: "Aa", TIME: "◷", BOOL: "✓", "duration": "Δ", "any": "?"}


def kind_of(schema: Schema | None, col: str) -> str:
    if not schema or col not in schema:
        return "any"
    return kind_of_dtype(schema[col])


def columns_for(schema: Schema | None, group: str) -> list[str]:
    if not schema:
        return []
    if group == "any":
        return list(schema)
    want = {"numeric": NUM, "temporal": TIME, "string": STR, "bool": BOOL}[group]
    return [c for c, dt in schema.items() if kind_of_dtype(dt) == want]


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
        self._parse = parse_bucket if param.kind == "bucket" else parse_duration
        self.edit.setPlaceholderText(param.placeholder or "e.g. 30s, 5m, 1h, 1d")
        self.edit.textChanged.connect(self._validate)

    def _validate(self, text: str) -> None:
        ok = True
        if text.strip():
            try:
                self._parse(text)
            except ValueError:
                ok = False
        self.edit.setStyleSheet("" if ok else "QLineEdit { border: 1px solid #ef4444; }")

    def problem(self) -> str | None:
        t = self.edit.text().strip()
        if not t:
            return None
        try:
            self._parse(t); return None
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

    def __init__(self, param: Param, parent=None, save: bool = False, directory: bool = False) -> None:
        super().__init__(param, parent)
        self.save = save
        self.directory = directory
        lay = QHBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit(); self.edit.setPlaceholderText("Choose a folder…" if directory else "Choose a file…"); self.edit.setMinimumWidth(60)
        btn = QPushButton("Browse…"); btn.clicked.connect(self.browse)
        lay.addWidget(self.edit, 1); lay.addWidget(btn)
        self.edit.textEdited.connect(lambda _: self.changed.emit())
        self.base_dir: Path | None = None

    def browse(self) -> None:
        start = str(self.base_dir or Path.home())
        if self.directory:
            cur = self.edit.text().strip()
            if cur:
                cand = (self.base_dir / cur) if (self.base_dir and not Path(cur).is_absolute()) else Path(cur)
                start = str(cand if cand.is_dir() else (self.base_dir or Path.home()))
            f = QFileDialog.getExistingDirectory(self, "Choose a folder", start)
        elif self.save:
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
