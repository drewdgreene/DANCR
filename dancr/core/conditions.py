"""Plain-English filter conditions -> Polars boolean masks."""
from __future__ import annotations

from typing import Any

import polars as pl

from .expr import _kind_of_dtype, NUM, STR, BOOL, TIME, DUR
from .dtypes import datetime_literal, is_date, number_from_text, text_to_bool
from .timeutil import parse_duration

# op key -> (label, needs_value, needs_second_value, applies to kinds)
OPS: dict[str, tuple[str, bool, bool, tuple[str, ...]]] = {
    "eq": ("equals", True, False, (NUM, STR, TIME, BOOL, DUR)),
    "ne": ("does not equal", True, False, (NUM, STR, TIME, BOOL, DUR)),
    "gt": ("is greater than", True, False, (NUM, TIME, DUR)),
    "lt": ("is less than", True, False, (NUM, TIME, DUR)),
    "ge": ("is at least", True, False, (NUM, TIME, DUR)),
    "le": ("is at most", True, False, (NUM, TIME, DUR)),
    "between": ("is between", True, True, (NUM, TIME, DUR)),
    "contains": ("contains", True, False, (STR,)),
    "not_contains": ("does not contain", True, False, (STR,)),
    "starts": ("starts with", True, False, (STR,)),
    "ends": ("ends with", True, False, (STR,)),
    "in": ("is one of", True, False, (NUM, STR)),
    "empty": ("is empty", False, False, (NUM, STR, TIME, BOOL, DUR)),
    "not_empty": ("is not empty", False, False, (NUM, STR, TIME, BOOL, DUR)),
    "true": ("is true", False, False, (BOOL,)),
    "false": ("is false", False, False, (BOOL,)),
}


def ops_for_kind(kind: str) -> list[tuple[str, str]]:
    return [(k, v[0]) for k, v in OPS.items() if kind in v[3] or kind == "any"]


def _literal(value: Any, kind: str, what: str, dtype: pl.DataType | None = None) -> pl.Expr:
    if kind == NUM:
        return pl.lit(number_from_text(value, what))
    if kind == TIME:
        return datetime_literal(value, dtype if dtype is not None else pl.Datetime("us"), what)
    if kind == DUR:
        try:
            _, secs = parse_duration(str(value))
        except ValueError as e:
            raise ValueError(f"{what}: {e}") from None
        return pl.duration(microseconds=int(round(secs * 1e6)))
    if kind == BOOL:
        return pl.lit(text_to_bool(value))
    return pl.lit(str(value))


def _resolve_input(value: Any, inputs: dict[str, Any] | None) -> Any:
    if inputs and isinstance(value, str):
        key = value.strip().lower()
        for k, v in inputs.items():
            if k.lower() == key:
                return v
    return value


def rule_mask(schema: dict[str, pl.DataType], rule: dict[str, Any], inputs: dict[str, Any] | None = None) -> pl.Expr:
    col = rule.get("column")
    op = rule.get("op", "eq")
    rule = {**rule, "value": _resolve_input(rule.get("value"), inputs), "value2": _resolve_input(rule.get("value2"), inputs)}
    if col not in schema:
        raise ValueError(f"Filter: there is no column called {col!r}")
    if op not in OPS:
        raise ValueError(f"Filter: unknown condition {op!r}")
    dtype = schema[col]
    kind = _kind_of_dtype(dtype)
    if kind not in OPS[op][3] and kind != "any":
        raise ValueError(f"Filter: '{OPS[op][0]}' does not apply to {col} (a {kind} column)")
    c = pl.col(col)
    if kind == TIME and is_date(dtype):
        c = c.cast(pl.Datetime("us"))
    label = OPS[op][0]
    what = f"{col} {label}"
    v = rule.get("value")
    v2 = rule.get("value2")
    if op == "empty":
        e = c.is_null()
        if kind == STR:
            e = e | (c.cast(pl.Utf8).str.strip_chars() == "")
        elif kind == NUM:
            e = e | c.cast(pl.Float64).is_nan()
        return e
    if op == "not_empty":
        return ~rule_mask(schema, {**rule, "op": "empty"})
    if op == "true":
        return c.cast(pl.Boolean) == True  # noqa: E712
    if op == "false":
        return c.cast(pl.Boolean) == False  # noqa: E712
    if op in ("contains", "not_contains", "starts", "ends"):
        s = c.cast(pl.Utf8)
        needle = str(v or "")
        if not rule.get("case_sensitive", False):
            s = s.str.to_lowercase()
            needle = needle.lower()
        if op == "contains":
            return s.str.contains(needle, literal=True)
        if op == "not_contains":
            return ~s.str.contains(needle, literal=True)
        if op == "starts":
            return s.str.starts_with(needle)
        return s.str.ends_with(needle)
    if op == "in":
        if isinstance(v, list):
            items = v
        else:
            text = str(v or "")
            sep = ";" if ";" in text else ("," if kind != NUM or not _looks_like_thousands(text) else " ")
            items = [x.strip() for x in text.split(sep) if x.strip()]
        if kind == NUM:
            nums = [number_from_text(x, what) for x in items]
            return c.cast(pl.Float64).is_in(nums)
        return c.cast(pl.Utf8).is_in([str(x) for x in items])
    if kind == STR and op in ("eq", "ne"):
        s = c.cast(pl.Utf8)
        needle = pl.lit(str(v if v is not None else ""))
        if not rule.get("case_sensitive", False):
            s = s.str.to_lowercase()
            needle = pl.lit(str(v if v is not None else "").lower())
        return (s == needle) if op == "eq" else (s != needle)
    if v in (None, ""):
        raise ValueError(f"{what}: enter a value")
    lit = _literal(v, kind, what, dtype)
    if kind == NUM:
        c = c.cast(pl.Float64)
    if op == "eq":
        return c == lit
    if op == "ne":
        return c != lit
    if op == "gt":
        return c > lit
    if op == "lt":
        return c < lit
    if op == "ge":
        return c >= lit
    if op == "le":
        return c <= lit
    if op == "between":
        if v2 in (None, ""):
            raise ValueError(f"{what}: enter both ends of the range")
        return (c >= lit) & (c <= _literal(v2, kind, what, dtype))
    raise ValueError(f"Filter: unknown condition {op!r}")


def _looks_like_thousands(text: str) -> bool:
    """'1,000' or '1,000 2,500' — commas used as thousands separators, not list separators."""
    import re
    return bool(re.fullmatch(r"\s*(-?\d{1,3}(,\d{3})+(\.\d+)?\s*)+", text))


def incomplete_rules(conditions: dict[str, Any]) -> list[dict[str, Any]]:
    """Rules that name a column and an operator but still lack the value they need."""
    out = []
    for r in (conditions or {}).get("rules", []):
        op = r.get("op", "eq")
        if not r.get("column") or op not in OPS:
            continue
        _, needs_v, needs_v2, _ = OPS[op]
        if (needs_v and str(r.get("value") if r.get("value") is not None else "").strip() == "") or \
                (needs_v2 and str(r.get("value2") if r.get("value2") is not None else "").strip() == ""):
            out.append(r)
    return out


def build_mask(schema: dict[str, pl.DataType], conditions: dict[str, Any], inputs: dict[str, Any] | None = None) -> pl.Expr | None:
    skip = [id(r) for r in incomplete_rules(conditions)]
    rules = [r for r in (conditions or {}).get("rules", []) if r.get("column") and id(r) not in skip]
    if not rules:
        return None
    masks = [rule_mask(schema, r, inputs) for r in rules]
    match = (conditions or {}).get("match", "all")
    out = masks[0]
    for m in masks[1:]:
        out = (out & m) if match == "all" else (out | m)
    return out


def describe(conditions: dict[str, Any]) -> str:
    rules = [r for r in (conditions or {}).get("rules", []) if r.get("column")]
    if not rules:
        return "no conditions"
    joiner = " and " if (conditions or {}).get("match", "all") == "all" else " or "
    parts = []
    for r in rules:
        spec = OPS.get(r.get("op", "eq"))
        if spec is None:
            parts.append(f"{r['column']} ?")
            continue
        s = f"{r['column']} {spec[0]}"
        if spec[1]:
            s += f" {r.get('value', '')}"
            if spec[2]:
                s += f" and {r.get('value2', '')}"
        parts.append(s)
    return joiner.join(parts)
