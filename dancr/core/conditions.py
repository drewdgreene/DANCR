"""Plain-English filter conditions -> Polars boolean masks."""
from __future__ import annotations

from typing import Any

import polars as pl

from .expr import kind_of_dtype, excel_compare, NUM, STR, BOOL, TIME, DUR
from .dtypes import datetime_literal, is_date, number_from_text, text_to_bool
from .params import find_input
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
    "year": ("is in the year", True, False, (TIME,)),
    "month": ("is in the month", True, False, (TIME,)),
}

MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
          "november", "december"]


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
    found = find_input(inputs, value)
    return value if found is None else found[1]


def _blank(v: Any) -> bool:
    return v is None or (isinstance(v, str) and not v.strip()) or (isinstance(v, list) and not v)


def rule_mask(schema: dict[str, pl.DataType], rule: dict[str, Any], inputs: dict[str, Any] | None = None) -> pl.Expr:
    """One rule as a true/false column. A condition that needs a value and has none is refused: blank cells
    are asked for with 'is empty' and 'is not empty'."""
    col = rule.get("column")
    op = rule.get("op", "eq")
    rule = {**rule, "value": _resolve_input(rule.get("value"), inputs), "value2": _resolve_input(rule.get("value2"), inputs)}
    if col not in schema:
        raise ValueError(f"Filter: there is no column called {col!r}")
    if op not in OPS:
        raise ValueError(f"Filter: unknown condition {op!r}")
    dtype = schema[col]
    kind = kind_of_dtype(dtype)
    if kind not in OPS[op][3] and kind != "any":
        raise ValueError(f"Filter: '{OPS[op][0]}' does not apply to {col} (a {kind} column)")
    c = pl.col(col)
    if kind == TIME and is_date(dtype):
        c = c.cast(pl.Datetime("us"))
    label = OPS[op][0]
    what = f"{col} {label}"
    v = rule.get("value")
    v2 = rule.get("value2")
    if OPS[op][1] and _blank(v):
        raise ValueError(f"{what}: enter a value. For blank cells, use 'is empty' or 'is not empty'")
    if OPS[op][2] and _blank(v2):
        raise ValueError(f"{what}: enter both ends of the range")
    if op == "empty":
        e = c.is_null()
        if kind == STR:
            e = e | (c.cast(pl.Utf8).str.strip_chars() == "")
        elif kind == NUM:
            e = e | c.cast(pl.Float64).is_nan()
        return e
    if op == "not_empty":
        return ~rule_mask(schema, {"column": col, "op": "empty"})
    if op in ("year", "month"):
        n = _month_number(v) if op == "month" else int(number_from_text(v, what))
        if op == "month" and not 1 <= n <= 12:
            raise ValueError(f"{what}: {v!r} is not a month (1–12 or its name)")
        return (c.dt.year() if op == "year" else c.dt.month()) == n
    if op == "true":
        return c.cast(pl.Boolean) == True  # noqa: E712
    if op == "false":
        return c.cast(pl.Boolean) == False  # noqa: E712
    if op in ("contains", "not_contains", "starts", "ends"):
        s = c.cast(pl.Utf8)
        needle = str(v)
        if not rule.get("case_sensitive", False):
            s = s.str.to_lowercase()
            needle = needle.lower()
        if op == "contains":
            return s.str.contains(needle, literal=True)
        if op == "not_contains":                    # a blank cell does not contain it (as in Excel)
            return (~s.str.contains(needle, literal=True)).fill_null(True)
        if op == "starts":
            return s.str.starts_with(needle)
        return s.str.ends_with(needle)
    if op == "in":
        if isinstance(v, list):
            items = v
        else:
            # items are separated by ";" or, when there is no ";", by ",": "100,200,300" is three numbers,
            # and "1,000; 2,500" is how to list numbers written with thousands separators
            text = str(v)
            items = [x.strip() for x in text.split(";" if ";" in text else ",") if x.strip()]
        if not items:
            raise ValueError(f"{what}: enter a value")
        if kind == NUM:
            nums = [number_from_text(x, what) for x in items]
            return c.cast(pl.Float64).is_in(nums)
        s = c.cast(pl.Utf8)
        wanted = [str(x) for x in items]
        if not rule.get("case_sensitive", False):     # match eq/ne, which are case-insensitive by default
            s = s.str.to_lowercase()
            wanted = [x.lower() for x in wanted]
        return s.is_in(wanted)
    if kind == NUM:
        c = c.cast(pl.Float64)
    lit = _literal(v, kind, what, dtype)
    case = bool(rule.get("case_sensitive", False))
    if op == "between":
        return excel_compare(">=", c, lit, kind) & excel_compare("<=", c, _literal(v2, kind, what, dtype), kind)
    sym = {"eq": "=", "ne": "!=", "gt": ">", "lt": "<", "ge": ">=", "le": "<="}.get(op)
    if sym is None:
        raise ValueError(f"Filter: unknown condition {op!r}")
    return excel_compare(sym, c, lit, kind, case_sensitive=case)     # the same rule as = and <> in formulas


def incomplete_rules(conditions: dict[str, Any]) -> list[dict[str, Any]]:
    """Rules that name a column and an operator but still lack the value they need (a rule just made from
    the window, before its value is typed). The Filter step leaves them out and says so; rule_mask refuses them."""
    out = []
    for r in (conditions or {}).get("rules", []):
        op = r.get("op", "eq")
        if not r.get("column") or op not in OPS:
            continue
        _, needs_v, needs_v2, _ = OPS[op]
        if (needs_v and _blank(r.get("value"))) or (needs_v2 and _blank(r.get("value2"))):
            out.append(r)
    return out


def build_mask(schema: dict[str, pl.DataType], conditions: dict[str, Any], inputs: dict[str, Any] | None = None) -> pl.Expr | None:
    rules = [r for r in (conditions or {}).get("rules", []) if r.get("column")]
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


def _month_number(v: Any) -> int:
    """3, '3', 'March' or 'mar' as 3."""
    text = str(v).strip().lower()
    if text.isdigit():
        return int(text)
    for i, name in enumerate(MONTHS, start=1):
        if name == text or (len(text) >= 3 and name.startswith(text)):
            return i
    return 0
