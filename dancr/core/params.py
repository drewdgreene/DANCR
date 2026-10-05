"""Declarative parameter specs.

Every node type declares its parameters as a list of :class:`Param`. The GUI
generates its inspector form from these, the CLI validates against them, the
MCP server publishes them as JSON schema, and serialization is automatic.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

# Kinds understood by the inspector, CLI and MCP layers.
KINDS = {
    "text",          # str
    "int",           # int
    "float",         # float
    "bool",          # bool
    "choice",        # one of `choices` values
    "column",        # a column name from the upstream schema
    "columns",       # list of column names
    "path",          # file path (str)
    "dir",           # a folder or glob path (str)
    "duration",      # str like "1m", "30s", "2h", "1d" (Polars duration syntax)
    "bucket",        # a duration, or calendar months, quarters and years ("1mo", "1q", "1y")
    "expr",          # formula text in the DANCR expression language
    "conditions",    # {"match": "all"|"any", "rules": [{"column","op","value","value2"}]}
    "aggregations",  # [{"column": str, "stats": [str], "alias": str|None}]
    "mapping",       # {old_name: new_name}
    "series",        # chart series: [{"column": str, "color": str|None, "label": str|None}]
    "formulas",      # [{"name": str, "expr": str, "unit": str|None}]
    "text_list",     # list[str]
    "table_columns", # [{"name": str, "type": number|text|datetime|bool}]
    "table_rows",    # [[cell, ...], ...]
    "fixes",         # [{"row": int, "column": str, "value": Any, "was": Any, "note": str}]
    "limits",        # [{"value": number|input name, "label": str}]
    "blocks",        # report layout: [{"type": text|heading|item, "text"?: str, "index"?: int}]
}

# Column type groups used to filter column pickers.
COLUMN_GROUPS = ("any", "numeric", "temporal", "string", "bool")


@dataclass
class Param:
    name: str
    label: str
    kind: str
    default: Any = None
    help: str = ""
    choices: list[tuple[Any, str]] | None = None   # (value, label)
    min: float | None = None
    max: float | None = None
    step: float | None = None
    column_group: str = "any"        # for column / columns / aggregations kinds
    port: str | None = None          # which input port supplies the schema (default: first)
    required: bool = False
    advanced: bool = False           # collapsed by default in the inspector
    visible_when: dict[str, Any] = field(default_factory=dict)  # {other_param: value or [values]}
    placeholder: str = ""
    secret: bool = False             # a credential: blanked wherever settings are shown (dancr.core.secrets)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"Unknown param kind {self.kind!r} for {self.name}")
        if self.column_group not in COLUMN_GROUPS:
            raise ValueError(f"Unknown column group {self.column_group!r}")

    def is_visible(self, params: dict[str, Any]) -> bool:
        for other, wanted in self.visible_when.items():
            current = params.get(other)
            if isinstance(wanted, (list, tuple, set)):
                if current not in wanted:
                    return False
            elif current != wanted:
                return False
        return True

    def default_value(self) -> Any:
        return copy.deepcopy(self.default)

    def _check_range(self, value: float) -> float:
        """Enforce the declared bounds here, not only in the GUI, so every interface is protected."""
        if self.min is not None and value < self.min:
            raise ValueError(f"{self.label}: must be at least {self.min:g}")
        if self.max is not None and value > self.max:
            raise ValueError(f"{self.label}: must be at most {self.max:g}")
        return value

    def coerce(self, value: Any) -> Any:
        """Coerce a loosely-typed value (from JSON, CLI or a form) to this param's type.
        None means "use the default"."""
        if value is None:
            return self.default_value()
        k = self.kind
        try:
            if k == "int":
                return self._check_range(_whole_number(value))
            if k == "float":
                return self._check_range(_finite_number(value))
            if k == "bool":
                if isinstance(value, str):
                    from .dtypes import text_to_bool
                    return text_to_bool(value)
                return bool(value)
            if k in ("duration", "bucket"):
                text = str(value).strip()
                if text:
                    from .timeutil import parse_duration, parse_bucket
                    (parse_bucket if k == "bucket" else parse_duration)(text)   # raises with a helpful message
                return text
            if k in ("text", "path", "dir", "expr", "column"):
                return str(value)
            if k == "choice":
                if self.choices is not None:
                    valid = [c[0] for c in self.choices]
                    if value not in valid:
                        # allow matching by label, case-insensitively
                        for v, label in self.choices:
                            if str(value).lower() in (str(v).lower(), str(label).lower()):
                                return v
                        raise ValueError(f"{self.label}: {value!r} is not one of {valid}")
                return value
            if k in ("columns", "text_list"):
                if isinstance(value, str):
                    return [s.strip() for s in value.split(",") if s.strip()]
                if not isinstance(value, (list, tuple)):
                    raise ValueError(f"{self.label}: expected a list of names")
                return [str(v) for v in value]
            return self._coerce_structured(value)
        except (TypeError, ValueError, OverflowError) as e:
            msg = str(e)
            raise ValueError(msg if msg.startswith(self.label + ":") else f"{self.label}: {msg}") from e

    def _coerce_structured(self, value: Any) -> Any:
        k = self.kind
        if k == "conditions":
            if not isinstance(value, dict):
                raise ValueError(f"{self.label}: expected {{'match': 'all'|'any', 'rules': [...]}}")
            rules = value.get("rules") or []
            if not isinstance(rules, list) or any(not isinstance(r, dict) for r in rules):
                raise ValueError(f"{self.label}: 'rules' must be a list of {{column, op, value}}")
            match = value.get("match") or "all"
            if match not in ("all", "any"):
                raise ValueError(f"{self.label}: 'match' must be 'all' or 'any'")
            return {"match": match, "rules": [dict(r) for r in rules]}
        if k in ("aggregations", "series", "formulas", "table_columns", "fixes", "limits", "blocks"):
            if not isinstance(value, list) or any(not isinstance(r, dict) for r in value):
                raise ValueError(f"{self.label}: expected a list of objects")
            return [dict(r) for r in value]
        if k == "table_rows":
            if not isinstance(value, list) or any(not isinstance(r, (list, tuple)) for r in value):
                raise ValueError(f"{self.label}: expected a list of rows")
            return [list(r) for r in value]
        if k == "mapping":
            if not isinstance(value, dict):
                raise ValueError(f"{self.label}: expected an object of old name → new name")
            return {str(a): str(b) for a, b in value.items()}
        return copy.deepcopy(value)

    def to_json(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name, "label": self.label, "kind": self.kind,
            "default": self.default, "help": self.help, "required": self.required,
        }
        if self.choices:
            d["choices"] = [{"value": v, "label": l} for v, l in self.choices]
        if self.min is not None:
            d["min"] = self.min
        if self.max is not None:
            d["max"] = self.max
        if self.kind in ("column", "columns", "aggregations"):
            d["column_group"] = self.column_group
        if self.visible_when:
            d["visible_when"] = self.visible_when
        if self.advanced:
            d["advanced"] = True
        if self.secret:
            d["secret"] = True
        return d


def _finite_number(value: Any) -> float:
    """A setting that is a number: typed text is read as everywhere else ('1,5' is 1.5); NaN and
    infinity are refused."""
    from .dtypes import number_from_text
    if isinstance(value, bool):
        raise ValueError(f"{value!r} is not a number")
    return number_from_text(value, "value") if not isinstance(value, (int, float)) or value != value or abs(value) == float("inf") \
        else float(value)


def _whole_number(value: Any) -> int:
    """A setting that is a whole number. 2.9 is refused rather than silently becoming 2."""
    if isinstance(value, bool):
        raise ValueError(f"{value!r} is not a whole number")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        from .dtypes import typed_value
        n = typed_value(value)
        if n is None:
            raise ValueError(f"{value!r} is not a whole number")
        value = n
        if isinstance(value, int):
            return value
    f = _finite_number(value)
    if not f.is_integer():
        raise ValueError(f"{value!r} is not a whole number")
    return int(f)



# ------------------------------------------------------------------ project inputs
# How a setting names a project input, one rule everywhere (formulas, filter values, limits, ranges, chart
# lines): the input's name, in any case, as a whole word of a setting's text. The executor finds the inputs a
# step names with ``inputs_named`` and hands the step only those, and only their values go into its cache
# key, so a step can never read an input its cache key leaves out.

def find_input(inputs: dict[str, Any] | None, text: Any) -> tuple[str, Any] | None:
    """(name, value) of the input a piece of settings text names, or None."""
    if not inputs or not isinstance(text, str):
        return None
    key = text.strip().lower()
    for name, value in inputs.items():
        if name.lower() == key:
            return name, value
    return None


def inputs_named(inputs: dict[str, Any] | None, *settings: Any) -> dict[str, Any]:
    """The inputs whose names appear as a whole word in the text values (not the keys) of ``settings``."""
    import re
    if not inputs:
        return {}
    texts = [t.lower() for s in settings for t in _string_values(s)]
    return {name: value for name, value in inputs.items()
            if any(re.search(r"(?<!\w)" + re.escape(name.lower()) + r"(?!\w)", t) for t in texts)}


def _string_values(obj: Any) -> list[str]:
    """Every text value in a settings value, however deeply nested (dict keys are names, not values)."""
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [t for v in obj.values() for t in _string_values(v)]
    if isinstance(obj, (list, tuple)):
        return [t for v in obj for t in _string_values(v)]
    return []
