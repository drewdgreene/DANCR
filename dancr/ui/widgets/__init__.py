"""The parameter-widget library: one widget per setting kind, and the factory that builds them."""
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

from ._base import (  # noqa: E402,F401
    KIND_ICON, BoolWidget, ChoiceWidget, DurationWidget, FloatWidget, IntWidget, ParamWidget, PathWidget, Schema,
    SheetWidget, TextListWidget, TextWidget, columns_for, kind_of,
)
from ._columns import ColumnCombo, ColumnWidget, ColumnsWidget, ExprWidget, FormulaEditor  # noqa: E402,F401
from ._rows import (  # noqa: E402,F401
    AggregationRow, AggregationsWidget, CheckCombo, ConditionRow, ConditionsWidget, FixRow, FixesWidget,
    FormulaRow, FormulasWidget, LimitRow, LimitsWidget, MappingRow, MappingWidget, RowListWidget, SeriesRow,
    SeriesWidget, remove_button,
)

def make_widget(param: Param, node_type_key: str, suggest: Callable[[str], list[str]] | None = None) -> ParamWidget:
    k = param.kind
    if node_type_key == "load_file" and param.name == "sheet":
        return SheetWidget(param)
    if k == "text":
        w = TextWidget(param)
        if param.secret:
            w.edit.setEchoMode(QLineEdit.Password)   # a credential is not shown on screen
        return w
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
    if k == "dir":
        return PathWidget(param, directory=True)
    if k in ("duration", "bucket"):
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
