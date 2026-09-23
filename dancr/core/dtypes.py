"""Datetime/dtype normalisation shared by nodes, conditions, formulas and views."""
from __future__ import annotations

import math
from typing import Any

import polars as pl

from .timeutil import detect_datetime_format


def is_datetime(dt: pl.DataType) -> bool:
    return isinstance(dt, pl.Datetime) or dt == pl.Datetime


def is_date(dt: pl.DataType) -> bool:
    return isinstance(dt, pl.Date) or dt == pl.Date


def is_temporal(dt: pl.DataType) -> bool:
    return is_datetime(dt) or is_date(dt)


def time_unit(dt: pl.DataType) -> str:
    return dt.time_unit if isinstance(dt, pl.Datetime) and dt.time_unit else "us"


def time_zone(dt: pl.DataType) -> str | None:
    return dt.time_zone if isinstance(dt, pl.Datetime) else None


def datetime_literal(value: Any, dt: pl.DataType, what: str = "value") -> pl.Expr:
    """A literal comparable to a column of dtype `dt` (unit and time zone aligned).

    A naive literal is read in the column's zone; a literal with an offset keeps its instant and is
    converted to the column's zone (or to UTC wall time for a naive column)."""
    s = str(value).strip()
    fmt = detect_datetime_format(pl.Series([s]), 1.0)
    if fmt is None:
        raise ValueError(f"{what}: cannot read {value!r} as a date/time. Try 2024-06-01 or 2024-06-01 12:30")
    aware = time_zone(pl.Series([s]).str.to_datetime(fmt, strict=True).dtype) is not None
    lit = pl.lit(s).str.to_datetime(fmt, strict=True)
    unit, tz = ("us", None) if is_date(dt) else (time_unit(dt), time_zone(dt))
    if aware:
        lit = lit.dt.convert_time_zone(tz) if tz else lit.dt.convert_time_zone("UTC").dt.replace_time_zone(None)
    elif tz:
        lit = lit.dt.replace_time_zone(tz, ambiguous="earliest")
    return lit.cast(pl.Datetime(unit, tz))


def align_time_column(expr: pl.Expr, dt_from: pl.DataType, dt_to: pl.DataType) -> pl.Expr:
    """Cast a time column expression so it matches another time column's dtype."""
    if is_date(dt_from):
        expr = expr.cast(pl.Datetime("us"))
    if is_date(dt_to):
        return expr.cast(pl.Datetime("us")).dt.replace_time_zone(None)
    unit, tz = time_unit(dt_to), time_zone(dt_to)
    tz_from = time_zone(dt_from) if is_datetime(dt_from) else None
    expr = expr.cast(pl.Datetime(unit, tz_from))
    if tz and not tz_from:
        expr = expr.dt.replace_time_zone(tz, ambiguous="earliest")
    elif tz and tz_from:
        expr = expr.dt.convert_time_zone(tz)
    elif not tz and tz_from:
        expr = expr.dt.replace_time_zone(None)
    return expr


def strip_time_zones(df: pl.DataFrame) -> pl.DataFrame:
    """For Excel: tz-aware datetimes become naive local wall time."""
    casts = [pl.col(c).dt.replace_time_zone(None).alias(c) for c, dt in df.schema.items()
             if isinstance(dt, pl.Datetime) and dt.time_zone]
    return df.with_columns(casts) if casts else df


def json_safe(obj: Any) -> Any:
    """Replace NaN/inf floats with None recursively so json.dumps stays strict."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def temp_name(base: str, taken: set[str] | dict) -> str:
    """A temporary column name that does not clash with existing columns."""
    name = f"__dancr_{base}"
    i = 1
    while name in taken:
        name = f"__dancr_{base}{i}"
        i += 1
    return name


# ---------------------------------------------------------------- text -> value, one way everywhere
TRUE_WORDS = ("true", "1", "yes", "y", "t")


def text_to_bool(value: Any) -> bool:
    """'yes', 'true', '1', 'y', 't' (any case) are true; everything else is false."""
    return str(value).strip().lower() in TRUE_WORDS


def text_to_bool_expr(expr: pl.Expr) -> pl.Expr:
    return expr.cast(pl.Utf8).str.strip_chars().str.to_lowercase().is_in(list(TRUE_WORDS))


def text_to_number_expr(expr: pl.Expr) -> pl.Expr:
    """Text like ' 1,200.5 ' -> 1200.5; anything unreadable becomes null."""
    return expr.cast(pl.Utf8).str.strip_chars().str.replace_all(",", "").cast(pl.Float64, strict=False)


def number_from_text(value: Any, what: str) -> float:
    """A python float from a typed-in value ('1,200' works), or a plain-English error."""
    if isinstance(value, bool):
        raise ValueError(f"{what}: {value!r} is not a number")
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(",", "").strip())
    except ValueError:
        raise ValueError(f"{what}: {value!r} is not a number") from None
