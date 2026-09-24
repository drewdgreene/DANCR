"""Analysis nodes: compare columns, outliers, summaries, grouping."""
from __future__ import annotations

from typing import Any

import polars as pl

from ..params import Param
from ..registry import NodeType, Ctx, NodeResult, registry
from ._common import first_input, schema_of, require_column, build_aggregations, STAT_HELP
from ..expr import NUM
from ..dtypes import temp_name


# ---------------------------------------------------------- remove outliers
def _outliers(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any]) -> NodeResult:
    lf = first_input(inputs)
    schema = schema_of(lf)
    cols = params.get("columns") or []
    if not cols:
        raise ValueError("Choose the column(s) to check")
    cols = [require_column(schema, c, "column", NUM) for c in cols]
    method = params.get("method") or "rolling"
    action = params.get("action") or "remove"
    flag_name = (params.get("flag_column") or "is_outlier").strip() or "is_outlier"
    if action == "flag" and flag_name in schema:
        raise ValueError(f"There is already a column called {flag_name!r}; choose another flag column name")
    bad_col = {c: temp_name(f"bad_{c}", schema) for c in cols}
    flags = []
    msgs = []
    for c in cols:
        e = pl.col(c).cast(pl.Float64)
        if method == "zscore":
            k = float(params.get("threshold") or 3)
            bad = ((e - e.mean()) / e.std()).abs() > k
            msgs.append(f"{c}: more than {k:g} standard deviations from the mean")
        elif method == "iqr":
            k = float(params.get("iqr_factor") or 1.5)
            q1, q3 = e.quantile(0.25), e.quantile(0.75)
            iqr = q3 - q1
            bad = (e < q1 - k * iqr) | (e > q3 + k * iqr)
            msgs.append(f"{c}: outside {k:g}× the interquartile range")
        elif method == "rolling":
            n = int(params.get("window") or 51)
            if n < 3:
                raise ValueError("The rolling window must be at least 3 rows")
            k = float(params.get("threshold") or 5)
            med = e.rolling_median(window_size=n, min_samples=1, center=True)
            dev = (e - med).abs()
            if params.get("local_spread"):
                scale = dev.rolling_median(window_size=n, min_samples=1, center=True) * 1.4826
                msgs.append(f"{c}: more than {k:g}× the local spread from the rolling median over {n} rows")
            else:
                scale = dev.median() * 1.4826     # robust noise level of the whole column (one pass, fast)
                msgs.append(f"{c}: more than {k:g}× the typical noise from the rolling median over {n} rows")
            bad = dev > k * pl.max_horizontal(scale, pl.lit(1e-12))
        elif method == "range":
            lo, hi = params.get("min"), params.get("max")
            if lo in (None, "") and hi in (None, ""):
                raise ValueError("Enter a minimum and/or maximum")
            bad = pl.lit(False)
            if lo not in (None, ""):
                bad = bad | (e < float(lo))
            if hi not in (None, ""):
                bad = bad | (e > float(hi))
            msgs.append(f"{c}: outside {lo} .. {hi}")
        else:
            raise ValueError(f"Unknown method {method!r}")
        flags.append(bad.fill_null(False).alias(bad_col[c]))
    lf2 = lf.with_columns(flags)
    any_bad = pl.any_horizontal([pl.col(bad_col[c]) for c in cols])
    temp = list(bad_col.values())
    if action == "remove":
        out = lf2.filter(~any_bad).drop(temp)
    elif action == "blank":
        out = lf2.with_columns([pl.when(pl.col(bad_col[c])).then(None).otherwise(pl.col(c)).alias(c) for c in cols]).drop(temp)
    elif action == "flag":
        out = lf2.with_columns(any_bad.alias(flag_name)).drop(temp)
    elif action == "clip":
        if method != "range":
            raise ValueError("'Clip to range' only works with the 'Outside a fixed range' method")
        lo, hi = params.get("min"), params.get("max")
        out = lf.with_columns([pl.col(c).clip(float(lo) if lo not in (None, "") else None, float(hi) if hi not in (None, "") else None).alias(c) for c in cols])
    else:
        raise ValueError(f"Unknown action {action!r}")
    return NodeResult(out, messages=msgs)


registry.register(NodeType(
    key="remove_outliers", label="Remove spikes", category="Clean up", icon="↯",
    description="Find spikes and impossible values and remove, blank out or flag them.",
    apply=_outliers,
    summary=lambda p: f"{p.get('method', 'rolling')} → {p.get('action', 'remove')}",
    params=[
        Param("columns", "Columns", "columns", column_group="numeric", default=[], required=True),
        Param("method", "How to spot them", "choice", default="rolling", choices=[
            ("rolling", "Spikes: far from the rolling median (best for values that change smoothly)"),
            ("zscore", "Far from the overall average (z-score)"),
            ("iqr", "Outside the interquartile range"),
            ("range", "Outside a fixed range")]),
        Param("window", "Rolling window (rows)", "int", default=51, min=3, visible_when={"method": "rolling"}),
        Param("threshold", "How far counts as an outlier", "float", default=5.0, min=0.1, visible_when={"method": ["rolling", "zscore"]},
              help="In multiples of the spread. 3 is strict, 5 is moderate, 10 is lenient"),
        Param("iqr_factor", "Multiplier", "float", default=1.5, min=0.1, visible_when={"method": "iqr"}),
        Param("local_spread", "Estimate the noise level locally (slower)", "bool", default=False, advanced=True, visible_when={"method": "rolling"}),
        Param("min", "Minimum allowed", "text", default="", visible_when={"method": "range"}),
        Param("max", "Maximum allowed", "text", default="", visible_when={"method": "range"}),
        Param("action", "What to do with them", "choice", default="remove", choices=[
            ("remove", "Remove the rows"), ("blank", "Blank out the value"), ("flag", "Add a true/false column"), ("clip", "Clip to the range")]),
        Param("flag_column", "Flag column name", "text", default="is_outlier", visible_when={"action": "flag"}),
    ],
))


# ---------------------------------------------------------------- summarize
def _summarize(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any]) -> NodeResult:
    from ...views.stats import column_summary
    lf = first_input(inputs)
    return NodeResult(column_summary(lf, params.get("columns") or None).lazy(),
                      messages=["Quartiles are exact: each number column is sorted once, which takes a while on very large tables"])


registry.register(NodeType(
    key="summarize", label="Describe the columns", category="Analyse & model", icon="Σ",
    description="One row per column: count, missing, average, spread, minimum, quartiles, maximum.",
    apply=_summarize,
    summary=lambda p: f"{len(p['columns'])} columns" if p.get("columns") else "all columns",
    params=[Param("columns", "Columns", "columns", default=[], help="Empty = all")],
))


# ------------------------------------------------------------ group summary
def _group_summary(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any]) -> NodeResult:
    lf = first_input(inputs)
    schema = schema_of(lf)
    by = [require_column(schema, c, "group column") for c in (params.get("by") or [])]
    if len(set(by)) != len(by):
        raise ValueError("The same column is listed twice under 'Group by'")
    only = [require_column(schema, c, "column", NUM) for c in (params.get("columns") or [])]
    aggs = build_aggregations(schema, params.get("aggregations"), exclude=by,
                              default_stats=tuple(params.get("default_stats") or ["mean"]), only=only or None)
    count_col = (params.get("count_column") or "").strip()
    if count_col:
        if count_col in by or count_col in {a.meta.output_name() for a in aggs}:
            raise ValueError(f"The count column cannot be called {count_col!r}: that name is already used in the output")
        aggs.append(pl.len().alias(count_col))
    if not aggs:
        raise ValueError("Nothing to summarise: add a statistic")
    return NodeResult(lf.group_by(by).agg(aggs).sort(by) if by else lf.select(aggs))


registry.register(NodeType(
    key="group_summary", label="Totals by group", category="Analyse & model", icon="⊞",
    description="Like a pivot table: one row per group with averages, totals, counts...",
    apply=_group_summary,
    summary=lambda p: f"by {', '.join(p.get('by') or [])}" if p.get("by") else "whole table",
    params=[
        Param("by", "Group by", "columns", default=[]),
        Param("columns", "Columns to summarise", "columns", column_group="numeric", default=[], help="Empty = every number column"),
        Param("default_stats", "Statistics", "text_list", default=["mean"], help=STAT_HELP),
        Param("aggregations", "Choose statistics column by column", "aggregations", default=[], column_group="numeric", advanced=True),
        Param("count_column", "Add a count column", "text", default="", placeholder="e.g. rows"),
    ],
))
