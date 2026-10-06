"""Inquiry nodes: steps whose job is to *tell* a person something.

Where the other nodes prepare a table, these answer a question: what changed, what drives it, what
relates to what, is the data trustworthy, and where is it heading. Each returns a small summary table
plus a finding — a plain sentence — so the answer engine, the report and the result card can all say
what the step found.

Everything here is deterministic: no model, no randomness; ties fall back to column order.
"""
from __future__ import annotations

import math
import re
from typing import Any

import polars as pl

from ...params import Param
from ...registry import NodeType, Ctx, NodeResult, registry
from ...findings import finding, fmt_number, fmt_pct
from ...expr import NUM, TIME, STR, kind_of_dtype
from ...dtypes import resolve_number, temp_name
from ...timeutil import parse_bucket
from .._common import first_input, schema_of, require_column, number_param
STAT_CHOICES = [("sum", "total"), ("mean", "average"), ("median", "median"), ("min", "minimum"),
                ("max", "maximum"), ("count", "count of rows")]
ASSOC_ROWS = 100_000            # rows sampled to estimate associations (reported as a sample)
ASSOC_MAX_COLS = 15             # bounds the O(n²) pair count
CATEGORY_CAP = 20               # categories compared when measuring a categorical association


def _looks_like_key(name: str) -> bool:
    """A column name that reads like a key/id (customer_id, OrderNo, sku), for the duplicate-value check."""
    from ...names import looks_like_key
    return looks_like_key(name)


def _label(ctx: Ctx, name: str | None) -> str:
    """How a column is shown: its display label (with unit when it has one), else its name."""
    if not name:
        return ""
    meta = (ctx.columns or {}).get(name) or {}
    lab = meta.get("label") or name
    unit = meta.get("unit") or ""
    return f"{lab} ({unit})" if unit and unit not in lab else lab


def _stat(e: pl.Expr, stat: str) -> pl.Expr:
    if stat in ("mean", "median", "min", "max", "sum"):
        e = e.fill_nan(None)         # NaN is a blank, as everywhere else: it must not poison this figure
    if stat == "mean":
        return e.cast(pl.Float64).mean()
    if stat == "median":
        return e.cast(pl.Float64).median()
    if stat == "min":
        return e.min()
    if stat == "max":
        return e.max()
    if stat == "sum":
        return e.sum()
    raise ValueError(f"Unknown statistic {stat!r}")


def _value_expr(measure: str | None, stat: str, cond: pl.Expr | None = None) -> pl.Expr:
    """The number a question is about: a statistic of a column, or the number of rows. ``cond`` restricts the
    rows first (a period, a group), so the aggregation stays a scalar."""
    base = pl.col("__one") if (not measure or stat == "count") else pl.col(measure)
    if cond is not None:
        base = base.filter(cond)
    return base.sum() if (not measure or stat == "count") else _stat(base, stat)


def _stat_label(stat: str) -> str:
    return {"sum": "total", "mean": "average", "median": "median", "min": "lowest", "max": "highest",
            "count": "number of rows"}.get(stat, stat)


def _direction(delta: float | None) -> str:
    if delta is None or delta == 0:
        return "flat"
    return "up" if delta > 0 else "down"


def _verb(delta: float | None) -> str:
    if delta is None or delta == 0:
        return "was flat"
    return "rose" if delta > 0 else "fell"


# ---------------------------------------------------------------- periods
def _every(text: Any) -> str:
    """A step for comparing periods: a normalised bucket Polars can truncate to (no quarters)."""
    unit = parse_bucket(str(text or "1mo"))[0]
    if unit.endswith("q"):
        raise ValueError("Quarter steps aren't supported here. Use months (1mo), weeks (1w), days (1d), hours (1h) or years (1y)")
    if not re.fullmatch(r"\d+(us|ms|s|m|h|d|w|mo|y)", unit):
        raise ValueError(f"Cannot use {text!r} as a period. Try 1h, 1d, 1w, 1mo or 1y")
    return unit


def _truncate(t: str, every: str) -> pl.Expr:
    return pl.col(t).dt.truncate(every)


def _latest_two(lf: pl.LazyFrame, texpr: pl.Expr, every: str) -> tuple[Any, Any]:
    vals = lf.select(texpr.drop_nulls().unique().sort(descending=True).head(2).alias("__b")) \
             .collect(engine="streaming")["__b"].to_list()
    if len(vals) < 2:
        raise ValueError(f"Need readings in at least two {every} periods to compare, and this table has "
                         f"{len(vals)}. Try a shorter period or a bigger file")
    return vals[0], vals[1]


def _period_label(v: Any, every: str) -> str:
    if v is None:
        return "?"
    try:
        if every.endswith("y"):
            return f"{v.year}"
        if every.endswith("mo"):
            return f"{v:%B %Y}"
        if every.endswith("w") or every.endswith("d"):
            return f"{v:%Y-%m-%d}"
        return f"{v:%Y-%m-%d %H:%M}"
    except AttributeError:
        return str(v)


def _signed(v: Any) -> str:
    if v is None:
        return "–"
    return ("+" if float(v) >= 0 else "") + fmt_number(v)
