"""Analysis nodes: compare columns, outliers, summaries, grouping."""
from __future__ import annotations

from typing import Any

import polars as pl

from ..params import Param
from ..registry import NodeType, Ctx, NodeResult, registry
from ._common import first_input, schema_of, require_column, build_aggregations, number_param, STAT_HELP, column_title
from ..expr import NUM
from ..dtypes import temp_name, resolve_number
from ..findings import finding, fmt_number, fmt_pct


# ---------------------------------------------------------- remove outliers
def _range(params: dict[str, Any], ctx: Ctx) -> tuple[float | None, float | None]:
    """The fixed range's ends: numbers ('1,000' works) or input names, as in 'Check against limits'."""
    lo = resolve_number(params.get("min"), ctx.inputs, "Minimum")
    hi = resolve_number(params.get("max"), ctx.inputs, "Maximum")
    if lo is None and hi is None:
        raise ValueError("Enter a minimum and/or maximum")
    if lo is not None and hi is not None and lo > hi:
        raise ValueError(f"The minimum ({lo:g}) is above the maximum ({hi:g})")
    return lo, hi


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
        raise ValueError(f"There is already a column called {flag_name!r}. Choose another name for the flag column")
    bad_col = {c: temp_name(f"bad_{c}", schema) for c in cols}
    by = [require_column(schema, c, "group column") for c in (params.get("by") or [])]
    if by and method not in ("zscore", "iqr"):
        raise ValueError("'Within each group' works with the z-score and interquartile-range methods")
    within = (lambda x: x.over(by)) if by else (lambda x: x)          # noqa: E731 - each group against its own values
    where = f" within each {', '.join(by)}" if by else ""
    flags = []
    msgs = []
    for c in cols:
        e = pl.col(c).cast(pl.Float64).fill_nan(None)      # NaN (0/0) is a blank: it neither moves the statistics nor is flagged
        if method == "zscore":
            k = number_param(params, "threshold", 3, "The threshold", above=0)
            # a spread of (almost) nothing, as in a constant column, flags nothing rather than dividing by zero
            spread = pl.max_horizontal(within(e.std()), within(e.mean()).abs() * 1e-12)
            bad = (e - within(e.mean())).abs() > k * spread
            msgs.append(f"{c}: more than {k:g} standard deviations from the mean{where}")
        elif method == "iqr":
            k = number_param(params, "iqr_factor", 1.5, "The range factor", at_least=0)
            q1 = within(e.quantile(0.25, interpolation="linear"))       # QUARTILE.INC
            q3 = within(e.quantile(0.75, interpolation="linear"))
            iqr = q3 - q1
            bad = (e < q1 - k * iqr) | (e > q3 + k * iqr)
            msgs.append(f"{c}: outside {k:g}× the interquartile range{where}")
        elif method == "rolling":
            n = int(number_param(params, "window", 51, "The rolling window", whole=True))
            if n < 3:
                raise ValueError("The rolling window must be at least 3 rows")
            k = number_param(params, "threshold", 5, "The threshold", above=0)
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
            lo, hi = _range(params, ctx)
            bad = pl.lit(False)
            if lo is not None:
                bad = bad | (e < lo)
            if hi is not None:
                bad = bad | (e > hi)
            msgs.append(f"{c}: outside {params.get('min') or '…'} .. {params.get('max') or '…'}")
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
        lo, hi = _range(params, ctx)
        fractional = any(v is not None and float(v) != int(v) for v in (lo, hi))

        def clipped(c: str) -> pl.Expr:
            col = pl.col(c)
            if fractional and schema[c].is_integer():
                col = col.cast(pl.Float64)                 # 1.5 .. 3.5 on whole numbers: the ends stay as given
            return col.clip(lo, hi).alias(c)
        out = lf.with_columns([clipped(c) for c in cols])
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
        Param("by", "Within each", "columns", default=[], advanced=True, visible_when={"method": ["zscore", "iqr"]},
              help="Judge each value against its own group (sun leaves against sun leaves), not the whole table"),
    ],
))


# ---------------------------------------------------------------- summarize
def _summarize(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any]) -> NodeResult:
    from ...views.stats import column_summary
    lf = first_input(inputs)
    df = column_summary(lf, params.get("columns") or None)
    report: dict[str, Any] = {}
    try:
        rows = int(df["rows"][0]) if df.height else 0
        gaps = [(r["column"], int(r["missing"] or 0)) for r in df.iter_rows(named=True) if int(r["missing"] or 0) > 0]
        report.update({"columns": df.height, "rows": rows, "columns_with_missing": len(gaps)})
        if gaps:
            worst = max(gaps, key=lambda x: x[1])
            pct = (100.0 * worst[1] / rows) if rows else 0.0
            report.update({"worst_column": worst[0], "worst_missing": worst[1]})
            said = (f"{len(gaps)} of {df.height} columns have blanks; {column_title(ctx, worst[0])} is missing "
                    f"{fmt_pct(pct)} of its values")
            report["finding"] = finding("summary", said, magnitude=pct, exact=True)
        else:
            report["finding"] = finding("summary", f"{df.height} columns over {rows:,} rows, with no blanks", exact=True)
    except Exception:  # noqa: BLE001 - a finding is a bonus; the summary table is the result
        pass
    return NodeResult(df.lazy(), report=report,
                      messages=["Quartiles are exact. Each number column is sorted to get them, which takes a while on very large tables"])


registry.register(NodeType(
    key="summarize", uses_labels=True, label="Describe the columns", category="Analyse & model", icon="Σ",
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
            raise ValueError(f"The count column can't be called {count_col!r} because that name is already used in the output")
        aggs.append(pl.len().alias(count_col))
    if not aggs:
        raise ValueError("Nothing to summarise. Add a statistic")
    out = lf.group_by(by).agg(aggs).sort(by) if by else lf.select(aggs)
    report: dict[str, Any] = {}
    if by:
        try:
            name = aggs[0].meta.output_name()
            rows = int(out.select(pl.len()).collect(engine="streaming")[0, 0])
            if 0 < rows <= 500:
                df = out.select([by[0], name]).collect(engine="streaming").drop_nulls(name)
                first = (params.get("aggregations") or [{}])[0].get("stats") or params.get("default_stats") or ["mean"]
                stat = "rows" if count_col and name == count_col else (first[0] if first else "mean")
                total = df[name].sum()
                top = df.sort(name, descending=True, nulls_last=True).row(0, named=True)
                # a share of the whole only means something for what adds up: not for an average or a minimum
                share = (100.0 * float(top[name]) / float(total)) if total and stat in ("sum", "count", "rows") else None
                said = f"{top[by[0]]} is the largest {column_title(ctx, by[0])} by {name}"
                if share is not None:
                    said += f" ({fmt_number(top[name])}, {fmt_pct(share)} of the total)"
                report["finding"] = finding("share", said, magnitude=share, direction="flat", exact=True)
        except Exception:  # noqa: BLE001 - a finding is a bonus; the totals themselves are the result
            pass
    return NodeResult(out, report=report)


registry.register(NodeType(
    key="group_summary", uses_labels=True, label="Totals by group", category="Analyse & model", icon="⊞",
    description="Like a pivot table, with one row per group and its averages, totals or counts.",
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
