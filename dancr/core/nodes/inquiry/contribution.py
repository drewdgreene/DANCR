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

def _contribution(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any]) -> NodeResult:
    lf = first_input(inputs)
    schema = schema_of(lf)
    by = [require_column(schema, c, "group column") for c in (params.get("by") or [])]
    if not by:
        raise ValueError("Choose what to break the total down by")
    measure = (params.get("measure") or "").strip()
    measure = require_column(schema, measure, "column", NUM) if measure else ""
    stat = params.get("stat") or ("sum" if measure else "count")
    if not measure:
        stat = "count"
    time = (params.get("time_column") or "").strip()
    every = _every(params.get("every")) if params.get("every") else ""
    what = _label(ctx, measure) if measure else "rows"
    val = lambda cond=None: _value_expr(measure or None, stat, cond)  # noqa: E731
    if time and every:                                        # contribution to a change
        time = require_column(schema, time, "time column", TIME)
        cur, prev = _latest_two(lf, _truncate(time, every), every)
        sub = lf.with_columns(_truncate(time, every).alias("__period"), pl.lit(1).alias("__one")) \
                .filter(pl.col("__period").is_in([prev, cur]))
        out = sub.group_by(by).agg([
            val(pl.col("__period") == prev).alias("previous"),
            val(pl.col("__period") == cur).alias("current"),
        ]).with_columns((pl.col("current") - pl.col("previous")).alias("change"))
        net = float(out.select(pl.col("change").sum()).collect(engine="streaming")[0, 0] or 0.0)
        out = out.with_columns(
            pl.when(pl.lit(abs(net) > 0)).then(pl.col("change") / abs(pl.lit(net)) * 100).otherwise(None)
            .alias("contribution_percent")
        ).with_columns(pl.col("change").abs().alias("__abs")).sort("__abs", descending=True, nulls_last=True).drop("__abs")
        df = out.collect(engine="streaming")
        if df.height:
            top = df.row(0, named=True)
            direction = _direction(net)
            statement = (f"{what[:1].upper()}{what[1:]} {_verb(net)} by {fmt_number(abs(net))} in {_period_label(cur, every)} "
                         f"vs {_period_label(prev, every)}; {fmt_pct(abs(float(top['contribution_percent'] or 0)))} of the "
                         f"move came from {top.get(by[0])} ({_signed(top.get('change'))})")
        else:
            statement = "Nothing changed between those two periods"
        report = {"total_change": net, "groups": df.height,
                  "finding": finding("change", statement, magnitude=abs(net), direction=direction, exact=True,
                                     mode="change")}
        return NodeResult(df.lazy(), report=report,
                          messages=[f"Net change {_signed(net)} between {_period_label(prev, every)} and {_period_label(cur, every)}"])
    # contribution to a total: value, share and cumulative share (Pareto)
    out = lf.with_columns(pl.lit(1).alias("__one")).group_by(by).agg(val().alias("value")) \
            .sort("value", descending=True, nulls_last=True)      # a group with no values is not the largest
    total = pl.col("value").sum()
    share = lambda e: pl.when(total != 0).then(e / total * 100)  # noqa: E731 - a total of 0 has no shares
    out = out.with_columns([
        share(pl.col("value")).alias("share_percent"),
        share(pl.col("value").cum_sum()).alias("cumulative_percent"),
    ])
    df = out.collect(engine="streaming")
    if df.height == 0:
        raise ValueError("That table has no rows to break down")
    top = df.row(0, named=True)
    top3 = float(df["share_percent"].head(3).sum() or 0.0)
    if stat not in ("sum", "count"):
        # averages (medians, extremes) are not parts of a whole: say which group is highest, not a share
        statement = f"{top.get(by[0])} has the highest {what.lower()} ({fmt_number(top.get('value'))})"
        report = {"groups": df.height, "top": top.get(by[0]),
                  "finding": finding("share", statement, magnitude=None, direction="flat", exact=True)}
        return NodeResult(df.drop("share_percent", "cumulative_percent").lazy(), report=report,
                          messages=[f"{df.height} {by[0]} groups; the highest is {top.get(by[0])}"])
    statement = f"{top.get(by[0])} is {fmt_pct(top.get('share_percent'))} of {what.lower()} ({fmt_number(top.get('value'))})"
    if df.height > 3:
        statement += f"; the top three are {fmt_pct(top3)}"
        if top3 >= 90:
            statement += f", so {what.lower()} is concentrated in a few {by[0]}s"
    report = {"groups": df.height, "top": top.get(by[0]), "top_share": top.get("share_percent"),
              "top3_share": top3,
              "finding": finding("share", statement, magnitude=top.get("share_percent"), direction="flat", exact=True)}
    return NodeResult(df.lazy(), report=report,
                      messages=[f"{df.height} {by[0]} groups; the largest is {top.get(by[0])}"])


registry.register(NodeType(
    key="contribution", uses_labels=True, label="What drives it", category="Analyse & model", icon="◐",
    description="Break a total (or a change) down into what each group contributes: the biggest contributor, "
                "its share, and how concentrated the whole is.",
    apply=_contribution,
    summary=lambda p: f"{'change in ' if p.get('every') else ''}{p.get('measure') or 'rows'} by {', '.join(p.get('by') or [])}",
    params=[
        Param("by", "Break it down by", "columns", default=[], required=True),
        Param("measure", "Number", "column", column_group="numeric", help="Blank = count the rows"),
        Param("stat", "How to combine it", "choice", default="sum", choices=STAT_CHOICES),
        Param("time_column", "Time column (to explain a change)", "column", column_group="temporal", advanced=True),
        Param("every", "Compare period", "bucket", default="", advanced=True, help="With a time column: contribution to the change since that period"),
    ],
))


# =================================================================== associations
