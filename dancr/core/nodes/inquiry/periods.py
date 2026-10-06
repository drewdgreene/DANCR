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
from ._shared import (
    STAT_CHOICES,
    ASSOC_ROWS,
    ASSOC_MAX_COLS,
    CATEGORY_CAP,
    _looks_like_key,
    _label,
    _stat,
    _value_expr,
    _stat_label,
    _direction,
    _verb,
    _every,
    _truncate,
    _latest_two,
    _period_label,
    _signed,
)

# =================================================================== compare periods
def _compare_periods(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any]) -> NodeResult:
    lf = first_input(inputs)
    schema = schema_of(lf)
    t = require_column(schema, params.get("time_column"), "time column", TIME)
    by = [require_column(schema, c, "group column") for c in (params.get("by") or [])]
    measure = (params.get("measure") or "").strip()
    measure = require_column(schema, measure, "column to compare", NUM) if measure else ""
    stat = params.get("stat") or ("sum" if measure else "count")
    if not measure:
        stat = "count"
    every = _every(params.get("every"))
    cur, prev = _latest_two(lf, _truncate(t, every), every)
    sub = lf.with_columns(_truncate(t, every).alias("__period"), pl.lit(1).alias("__one")) \
            .filter(pl.col("__period").is_in([prev, cur]))
    # the latest period may have only begun (data to a Wednesday, weekly): then it is compared with the same stretch
    # of the period before, not with the whole of it
    partial = None
    last = lf.select(pl.col(t).max()).collect(engine="streaming").item()
    end = pl.select(pl.lit(cur).dt.offset_by(every)).item()
    try:
        covered = (last - cur) / (end - cur)
    except (TypeError, ZeroDivisionError):
        covered = 1.0
    if covered < 0.9:
        prev_end = prev + (last - cur)
        sub = sub.filter((pl.col("__period") == cur) | (pl.col(t) <= prev_end))
        partial = (last, prev_end)
    val = lambda cond: _value_expr(measure or None, stat, cond)  # noqa: E731
    aggs = [val(pl.col("__period") == prev).alias("previous"),
            val(pl.col("__period") == cur).alias("current")]
    out = sub.group_by(by).agg(aggs) if by else sub.select(aggs)
    out = out.with_columns([
        (pl.col("current") - pl.col("previous")).alias("change"),
        pl.when(pl.col("previous").is_null() | (pl.col("previous") == 0)).then(None)
          .otherwise((pl.col("current") - pl.col("previous")) / pl.col("previous").abs() * 100).alias("change_percent"),
    ]).sort("change", descending=True, nulls_last=True)
    df = out.collect(engine="streaming")
    cur_label, prev_label = _period_label(cur, every), _period_label(prev, every)
    what = _label(ctx, measure) if measure else "rows"
    if df.height == 0:
        return NodeResult(df.lazy(), messages=["Nothing to compare in those two periods"])
    # the whole table's figure for each period, not a sum of the groups' figures: an average (or a median, a
    # minimum) of the whole period is not the sum of the groups' averages
    whole = sub.select(aggs).collect(engine="streaming").row(0, named=True) if by else df.row(0, named=True)
    prev_tot = float(whole["previous"] or 0.0)
    cur_tot = float(whole["current"] or 0.0)
    delta = cur_tot - prev_tot
    pct = (100.0 * delta / abs(prev_tot)) if prev_tot else None
    leader = ""
    if by:
        rows = df.to_dicts()
        top = max(rows, key=lambda r: abs(float(r["change"] or 0.0)))
        leader = f", led by {top.get(by[0])} ({_signed(top.get('change'))})"
    if partial is not None:                            # say which stretch of each period was compared
        upto = lambda v: f"{v:%Y-%m-%d}" if hasattr(v, "year") else str(v)   # noqa: E731
        cur_label = f"{cur_label} so far (to {upto(partial[0])})"
        prev_label = f"the same part of {prev_label} (to {upto(partial[1])})"
    statement = (f"{what[:1].upper()}{what[1:]} {_verb(delta)} {fmt_pct(abs(pct)) if pct is not None else ''} "
                 f"in {cur_label} vs {prev_label} ({fmt_number(prev_tot)} → {fmt_number(cur_tot)}){leader}").replace("  ", " ")
    report = {"current": cur_label, "previous": prev_label, "current_total": cur_tot, "previous_total": prev_tot,
              "change": delta, "change_percent": pct, "groups": df.height,
              "finding": finding("change", statement, magnitude=(abs(pct) if pct is not None else abs(delta)),
                                 direction=_direction(delta), exact=True, current=cur_label, previous=prev_label)}
    msgs = [f"{cur_label}: {fmt_number(cur_tot)} · {prev_label}: {fmt_number(prev_tot)} · change {_signed(delta)}"
            + (f" ({fmt_pct(pct)})" if pct is not None else "")]
    return NodeResult(df.lazy(), report=report, messages=msgs)


registry.register(NodeType(
    key="compare_periods", uses_labels=True, label="Compare two periods", category="Analyse & model", icon="⇄",
    description="Compare this period with the one before it, for the whole table or for each group: what rose, "
                "what fell, and by how much.",
    apply=_compare_periods,
    summary=lambda p: f"{p.get('measure') or 'rows'} per {p.get('every') or '1mo'}",
    params=[
        Param("time_column", "Time column", "column", column_group="temporal", required=True),
        Param("every", "Each period is", "bucket", default="1mo"),
        Param("measure", "Number to compare", "column", column_group="numeric", help="Blank = count the rows"),
        Param("stat", "How to combine it", "choice", default="sum", choices=STAT_CHOICES),
        Param("by", "Compare within each", "columns", default=[], help="Optional groups (region, product…)"),
    ],
))


# =================================================================== contribution
