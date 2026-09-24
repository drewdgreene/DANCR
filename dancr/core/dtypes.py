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

    A naive literal is read in the column's zone. A literal with an offset is the same instant in the
    column's zone; against a naive column (whose zone is unknown) its own wall time is used, so
    "12:00+02:00" means 12:00 whatever the column. A date that reads two ways (01/05/2024) is refused
    rather than guessed, and so is a local time that does not exist (inside a daylight-saving jump)."""
    from .timeutil import swap_day_month
    s = str(value).strip()
    fmt = detect_datetime_format(pl.Series([s]), 1.0)
    if fmt is None:
        raise ValueError(f"{what}: cannot read {value!r} as a date/time. Try 2024-06-01 or 2024-06-01 12:30")
    parsed = pl.Series([s]).str.to_datetime(fmt, strict=True)
    twin = swap_day_month(fmt)
    if twin is not None:
        other = pl.Series([s]).str.to_datetime(twin, strict=False)
        if other[0] is not None and other[0] != parsed[0]:
            raise ValueError(f"{what}: {value!r} could be day/month or month/day. Write it as year-month-day, "
                             f"e.g. {parsed[0]:%Y-%m-%d} or {other[0]:%Y-%m-%d}")
    aware = time_zone(parsed.dtype) is not None
    lit = pl.lit(s).str.to_datetime(fmt, strict=True)
    unit, tz = ("us", None) if is_date(dt) else (time_unit(dt), time_zone(dt))
    if aware:
        if tz:
            lit = lit.dt.convert_time_zone(tz)
        else:                                       # its own wall time: the UTC instant moved back by its offset
            import re
            m = re.search(r"([+-])(\d{2}):?(\d{2})$", s)
            minutes = 0 if m is None else (1 if m.group(1) == "+" else -1) * (int(m.group(2)) * 60 + int(m.group(3)))
            lit = lit.dt.replace_time_zone(None) + pl.duration(minutes=minutes)
    elif tz:
        if parsed.dt.replace_time_zone(tz, ambiguous="earliest", non_existent="null")[0] is None:
            raise ValueError(f"{what}: {value!r} does not exist in {tz}: the clocks jumped forward over it")
        lit = lit.dt.replace_time_zone(tz, ambiguous="earliest")
    return lit.cast(pl.Datetime(unit, tz))


def align_time_column(expr: pl.Expr, dt_from: pl.DataType, dt_to: pl.DataType) -> pl.Expr:
    """Cast a time column expression so it matches another time column's dtype. Between a zoned time and one
    without a zone (or a date), the local wall-clock time is what is compared: 00:20 in Oslo is 00:20 on that
    date. A wall time the clocks skipped (inside a daylight-saving jump) has no moment, so it becomes blank."""
    tz_from = time_zone(dt_from) if is_datetime(dt_from) else None
    if is_date(dt_to):
        if tz_from:
            expr = expr.dt.replace_time_zone(None)
        return expr.cast(pl.Datetime("us"))
    if is_date(dt_from):
        expr = expr.cast(pl.Datetime("us"))
    unit, tz = time_unit(dt_to), time_zone(dt_to)
    expr = expr.cast(pl.Datetime(unit, tz_from))
    if tz and not tz_from:
        expr = expr.dt.replace_time_zone(tz, ambiguous="earliest", non_existent="null")
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
TRUE_WORDS = ("true", "1", "yes", "y", "t", "on")


def text_to_bool(value: Any) -> bool:
    """'yes', 'true', '1', 'y', 't', 'on' (any case) are true; everything else is false."""
    return str(value).strip().lower() in TRUE_WORDS


def text_to_bool_expr(expr: pl.Expr) -> pl.Expr:
    return expr.cast(pl.Utf8).str.strip_chars().str.to_lowercase().is_in(list(TRUE_WORDS))


# How typed numbers are read, one rule everywhere (columns of text, filter values, inputs, entered data):
#   1,234,567.5   commas that group thousands are dropped
#   1.234.567,5   dots that group thousands, with a decimal comma
#   1,5  12,25    a single comma followed by anything but exactly three digits is a decimal comma
#   1,500         three digits after the comma: thousands, as Excel reads it (set 'decimal comma' on a
#                 file that means 1.5)
_THOUSANDS = r"^[+-]?\d{1,3}(,\d{3})+(\.\d*)?$"
_EU_THOUSANDS = r"^[+-]?\d{1,3}(\.\d{3})+,\d+$"
_DECIMAL_COMMA = r"^[+-]?\d*,\d+$"


def normalise_number_text(s: str) -> str:
    """Typed text rewritten so ``float()`` reads it as a person meant it (see the rules above)."""
    import re
    s = s.strip()
    if re.fullmatch(_THOUSANDS, s):
        return s.replace(",", "")
    if re.fullmatch(_EU_THOUSANDS, s):
        return s.replace(".", "").replace(",", ".")
    if re.fullmatch(_DECIMAL_COMMA, s):
        return s.replace(",", ".")
    return s


def text_to_number_expr(expr: pl.Expr) -> pl.Expr:
    """Text like ' 1,200.5 ' -> 1200.5 and '1,5' -> 1.5; anything unreadable becomes null."""
    s = expr.cast(pl.Utf8).str.strip_chars()
    s = (pl.when(s.str.contains(_THOUSANDS)).then(s.str.replace_all(",", ""))
         .when(s.str.contains(_EU_THOUSANDS)).then(s.str.replace_all(".", "", literal=True).str.replace(",", ".", literal=True))
         .when(s.str.contains(_DECIMAL_COMMA)).then(s.str.replace(",", ".", literal=True))
         .otherwise(s))
    return s.cast(pl.Float64, strict=False)


def number_from_text(value: Any, what: str) -> float:
    """A python float from a typed-in value ('1,200' and '1,5' work), or a plain-English error."""
    if isinstance(value, bool):
        raise ValueError(f"{what}: {value!r} is not a number")
    if isinstance(value, (int, float)):
        return float(value)
    try:
        f = float(normalise_number_text(str(value)))
    except ValueError:
        raise ValueError(f"{what}: {value!r} is not a number") from None
    if f != f or f in (float("inf"), float("-inf")):
        raise ValueError(f"{what}: {value!r} is not a number")
    return f


def resolve_number(text: Any, inputs: dict[str, Any] | None, what: str) -> float | None:
    """A typed number, or the name of an input that holds one (any case); blank -> None. Used wherever a
    person types a threshold: limits, ranges, chart reference lines."""
    if text is None or (isinstance(text, str) and not text.strip()):
        return None
    try:
        return number_from_text(text, what)
    except ValueError:
        pass
    s = str(text).strip().lower()
    for k, v in (inputs or {}).items():
        if k.lower() == s and isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
    raise ValueError(f"{what}: {text!r} is not a number or the name of an input")


def typed_value(text: str) -> int | float | None:
    """A typed number kept exact: whole numbers stay int (a 20-digit id keeps every digit), others float.
    None when the text is not a finite number."""
    import re
    s = normalise_number_text(text)
    if re.fullmatch(r"[+-]?\d+", s):
        return int(s)
    if not re.fullmatch(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?", s):
        return None                          # 'nan', 'inf', '1_000' and other things float() would accept
    f = float(s)
    return f if f == f and f not in (float("inf"), float("-inf")) else None
