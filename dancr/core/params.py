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
    "duration",      # str like "1m", "30s", "2h", "1d" (Polars duration syntax)
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

    def coerce(self, value: Any) -> Any:
        """Coerce a loosely-typed value (from JSON, CLI or a form) to this param's type.
        None means "use the default"."""
        if value is None:
            return self.default_value()
        k = self.kind
        try:
            if k == "int":
                return int(value)
            if k == "float":
                return float(value)
            if k == "bool":
                if isinstance(value, str):
                    return value.strip().lower() in ("1", "true", "yes", "on", "y")
                return bool(value)
            if k == "duration":
                text = str(value).strip()
                if text:
                    from .timeutil import parse_duration
                    parse_duration(text)   # raises with a helpful message
                return text
            if k in ("text", "path", "expr", "column"):
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
        except (TypeError, ValueError) as e:
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
        return d
