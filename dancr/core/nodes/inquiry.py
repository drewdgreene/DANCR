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

from ..params import Param
from ..registry import NodeType, Ctx, NodeResult, registry
from ..findings import finding, fmt_number, fmt_pct
from ..expr import NUM, TIME, STR, kind_of_dtype
from ..dtypes import resolve_number, temp_name
from ..timeutil import parse_bucket
from ._common import first_input, schema_of, require_column, number_param

STAT_CHOICES = [("sum", "total"), ("mean", "average"), ("median", "median"), ("min", "minimum"),
                ("max", "maximum"), ("count", "count of rows")]
ASSOC_ROWS = 100_000            # rows sampled to estimate associations (reported as a sample)
ASSOC_MAX_COLS = 15             # bounds the O(n²) pair count
CATEGORY_CAP = 20               # categories compared when measuring a categorical association


def _label(ctx: Ctx, name: str | None) -> str:
    """How a column is shown: its display label (with unit when it has one), else its name."""
    if not name:
        return ""
    meta = (ctx.columns or {}).get(name) or {}
    lab = meta.get("label") or name
    unit = meta.get("unit") or ""
    return f"{lab} ({unit})" if unit and unit not in lab else lab


def _stat(e: pl.Expr, stat: str) -> pl.Expr:
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
    prev_tot = float(df["previous"].sum() or 0.0)
    cur_tot = float(df["current"].sum() or 0.0)
    delta = cur_tot - prev_tot
    pct = (100.0 * delta / abs(prev_tot)) if prev_tot else None
    leader = ""
    if by:
        rows = df.to_dicts()
        top = max(rows, key=lambda r: abs(float(r["change"] or 0.0)))
        leader = f", led by {top.get(by[0])} ({_signed(top.get('change'))})"
    statement = (f"{what.capitalize()} {_verb(delta)} {fmt_pct(abs(pct)) if pct is not None else ''} "
                 f"in {cur_label} vs {prev_label} ({fmt_number(prev_tot)} → {fmt_number(cur_tot)}){leader}").replace("  ", " ")
    report = {"current": cur_label, "previous": prev_label, "current_total": cur_tot, "previous_total": prev_tot,
              "change": delta, "change_percent": pct, "groups": df.height,
              "finding": finding("change", statement, magnitude=(abs(pct) if pct is not None else abs(delta)),
                                 direction=_direction(delta), exact=True, current=cur_label, previous=prev_label)}
    msgs = [f"{cur_label}: {fmt_number(cur_tot)} · {prev_label}: {fmt_number(prev_tot)} · change {_signed(delta)}"
            + (f" ({fmt_pct(pct)})" if pct is not None else "")]
    return NodeResult(df.lazy(), report=report, messages=msgs)


registry.register(NodeType(
    key="compare_periods", label="Compare two periods", category="Analyse & model", icon="⇄",
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
        ).with_columns(pl.col("change").abs().alias("__abs")).sort("__abs", descending=True).drop("__abs")
        df = out.collect(engine="streaming")
        if df.height:
            top = df.row(0, named=True)
            direction = _direction(net)
            statement = (f"{what.capitalize()} {_verb(net)} by {fmt_number(abs(net))} in {_period_label(cur, every)} "
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
    out = lf.with_columns(pl.lit(1).alias("__one")).group_by(by).agg(val().alias("value")).sort("value", descending=True)
    out = out.with_columns([
        (pl.col("value") / pl.col("value").sum() * 100).alias("share_percent"),
        (pl.col("value").cum_sum() / pl.col("value").sum() * 100).alias("cumulative_percent"),
    ])
    df = out.collect(engine="streaming")
    if df.height == 0:
        raise ValueError("That table has no rows to break down")
    top = df.row(0, named=True)
    top3 = float(df["share_percent"].head(3).sum() or 0.0)
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
    key="contribution", label="What drives it", category="Analyse & model", icon="◐",
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
def _pearson(df: pl.DataFrame, a: str, b: str) -> float | None:
    d = df.select(pl.col(a).cast(pl.Float64), pl.col(b).cast(pl.Float64)).drop_nulls().drop_nans()
    if d.height < 10:
        return None
    try:
        r = d.select(pl.corr(a, b)).item()
    except Exception:  # noqa: BLE001
        return None
    return float(r) if r is not None and r == r else None


def _eta_squared(df: pl.DataFrame, cat: str, num: str) -> float | None:
    d = df.select(pl.col(cat).cast(pl.Utf8).alias("g"), pl.col(num).cast(pl.Float64).alias("y")).drop_nulls().drop_nans()
    if d.height < 10 or d["g"].n_unique() < 2:
        return None
    grand = d["y"].mean()
    g = d.group_by("g").agg([pl.col("y").mean().alias("m"), pl.len().alias("n")])
    ss_between = float((((g["m"] - grand) ** 2) * g["n"]).sum() or 0.0)
    ss_total = float(((d["y"] - grand) ** 2).sum() or 0.0)
    if ss_total <= 0:
        return None
    return max(0.0, min(1.0, ss_between / ss_total))


def _top_categories(df: pl.DataFrame, col: str) -> list[str]:
    vc = df[col].cast(pl.Utf8).value_counts(sort=True)
    name = vc.columns[0]
    return [v for v in vc[name].head(CATEGORY_CAP).to_list() if v is not None]


def _cramers_v(df: pl.DataFrame, a: str, b: str) -> float | None:
    va, vb = _top_categories(df, a), _top_categories(df, b)
    if len(va) < 2 or len(vb) < 2:
        return None
    d = df.select(pl.col(a).cast(pl.Utf8).alias("a"), pl.col(b).cast(pl.Utf8).alias("b")).drop_nulls() \
          .filter(pl.col("a").is_in(va) & pl.col("b").is_in(vb))
    n = d.height
    if n < 10:
        return None
    tab = d.group_by(["a", "b"]).len()
    rt = tab.group_by("a").agg(pl.col("len").sum().alias("rt"))
    ct = tab.group_by("b").agg(pl.col("len").sum().alias("ct"))
    t = tab.join(rt, on="a").join(ct, on="b").with_columns((pl.col("rt") * pl.col("ct") / n).alias("e"))
    chi2 = float((((t["len"] - t["e"]) ** 2) / t["e"]).sum() or 0.0)
    denom = n * (min(len(va), len(vb)) - 1)
    return max(0.0, min(1.0, math.sqrt(chi2 / denom))) if denom > 0 else None


def _associations(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any]) -> NodeResult:
    lf = first_input(inputs)
    schema = schema_of(lf)
    target = (params.get("target") or "").strip()
    target = require_column(schema, target, "target column") if target else ""
    given = [require_column(schema, c, "column") for c in (params.get("columns") or [])]
    pool = given or [c for c in schema if kind_of_dtype(schema[c]) in (NUM, STR)]
    pool = [c for c in pool if c != target]
    measures = [c for c in pool if kind_of_dtype(schema[c]) == NUM]
    others = [c for c in pool if c != target and kind_of_dtype(schema[c]) != NUM]
    chosen = (([target] if target else []) + (measures + others)[:ASSOC_MAX_COLS])
    if len(chosen) < 2:
        raise ValueError("Need at least two columns to look for relationships between")
    df = lf.select(chosen).head(ASSOC_ROWS).collect(engine="streaming")
    rows: list[dict[str, Any]] = []
    for i, a in enumerate(chosen):
        for b in chosen[i + 1:]:
            ka, kb = kind_of_dtype(schema[a]), kind_of_dtype(schema[b])
            value = kind = None
            if ka == NUM and kb == NUM:
                value, kind = _pearson(df, a, b), "straight line"
                label = "r"
            elif ka == NUM or kb == NUM:
                cat, num = (b, a) if ka == NUM else (a, b)
                value, kind, label = _eta_squared(df, cat, num), "group difference", "eta²"
            else:
                value, kind, label = _cramers_v(df, a, b), "association", "V"
            if value is None:
                continue
            rows.append({"left": _label(ctx, a), "right": _label(ctx, b), "kind": kind, "measure": label,
                         "strength": round(abs(float(value)), 4), "value": round(float(value), 4)})
    if not rows:
        raise ValueError("No two columns had enough values to compare")
    out = pl.DataFrame(rows).sort("strength", descending=True)
    best = out.row(0, named=True)
    strong = int((out["strength"] >= 0.5).sum())
    movers = {"straight line": "move together", "group difference": "differ between groups",
              "association": "go together"}[best["kind"]]
    direction = "flat"
    if best["kind"] == "straight line":
        direction = _direction(best["value"])
    statement = (f"{best['left']} and {best['right']} {movers} most strongly "
                 f"({best['measure']} = {fmt_number(best['value'], 3)})"
                 + (f"; {strong} of {out.height} pairs are strong" if strong > 1 else ""))
    report = {"pairs": out.height, "strong": strong, "sample_rows": df.height, "target": _label(ctx, target) or None,
              "finding": finding("association", statement, magnitude=best["strength"], direction=direction,
                                 exact=False, sample_rows=df.height)}
    msgs = [f"From the first {df.height:,} rows. The strongest is {best['left']} ↔ {best['right']}"]
    return NodeResult(out.lazy(), report=report, messages=msgs)


registry.register(NodeType(
    key="associations", label="What relates to what", category="Analyse & model", icon="∞",
    description="Rank how strongly the columns move together: straight-line relationships between numbers, "
                "differences between groups, and associations between categories.",
    apply=_associations,
    summary=lambda p: f"relating to {p.get('target')}" if p.get("target") else "every pair of columns",
    params=[
        Param("target", "Related to", "column", help="Blank = every pair of columns"),
        Param("columns", "Columns to compare", "columns", default=[], help="Blank = all"),
    ],
))


# =================================================================== check data
def _check_data(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any]) -> NodeResult:
    from ..dtypes import text_to_number_expr
    lf = first_input(inputs)
    schema = schema_of(lf)
    cols = [require_column(schema, c, "column") for c in (params.get("columns") or list(schema))]
    exprs: list[pl.Expr] = [pl.len().alias("__n")]
    for i, c in enumerate(cols):
        dt = schema[c]
        exprs.append(pl.col(c).null_count().alias(f"miss{i}"))
        exprs.append(pl.col(c).approx_n_unique().alias(f"u{i}"))
        if dt.is_numeric():
            exprs += [pl.col(c).min().alias(f"lo{i}"), pl.col(c).max().alias(f"hi{i}")]
        if kind_of_dtype(dt) == STR:
            exprs.append(text_to_number_expr(pl.col(c)).is_not_null().sum().alias(f"num{i}"))
            s = pl.col(c).cast(pl.Utf8)
            exprs.append((s != s.str.strip_chars()).sum().alias(f"ws{i}"))
    row = lf.select(exprs).collect(engine="streaming").row(0, named=True)
    rows = int(row["__n"])
    out_rows: list[dict[str, Any]] = []
    problems: list[str] = []
    for i, c in enumerate(cols):
        dt = schema[c]
        miss = int(row[f"miss{i}"] or 0)
        distinct = int(row[f"u{i}"] or 0)
        pct = (100.0 * miss / rows) if rows else 0.0
        issues: list[tuple[int, str]] = []
        if rows and miss == rows:
            issues.append((3, "empty: every cell is blank"))
        elif pct >= 50:
            issues.append((3, f"mostly blank ({fmt_pct(pct)} missing)"))
        elif pct >= 5:
            issues.append((2, f"{fmt_pct(pct)} missing"))
        if rows > 1 and distinct <= 1:
            issues.append((2, "one value only"))
        if kind_of_dtype(dt) == STR:
            num_like = int(row[f"num{i}"] or 0)
            filled = max(rows - miss, 1)
            if num_like >= max(3, 0.5 * filled) and distinct > num_like:
                issues.append((2, f"numbers stored as text ({num_like:,} look like numbers)"))
            if int(row[f"ws{i}"] or 0) > 0:
                issues.append((1, f"{int(row[f'ws{i}']):,} values have spaces around them"))
        issues.sort(reverse=True)
        sev, issue = issues[0] if issues else (0, "")
        if sev >= 2:
            problems.append(f"{_label(ctx, c)} ({issue})")
        entry = {"column": c, "type": kind_of_dtype(dt), "missing": miss, "missing_percent": round(pct, 2),
                 "distinct": distinct, "issue": issue, "severity": sev}
        if dt.is_numeric():
            entry["minimum"], entry["maximum"] = row.get(f"lo{i}"), row.get(f"hi{i}")
        out_rows.append(entry)
    out = pl.DataFrame(out_rows).sort(["severity", "missing_percent"], descending=[True, True])
    dup = None
    if rows and len(cols) <= 30 and rows <= 5_000_000:
        try:
            distinct = int(lf.select(pl.struct(cols).n_unique()).collect(engine="streaming")[0, 0])
            dup = rows - distinct
        except Exception:  # noqa: BLE001 - duplicate detection is a bonus, never a failure
            dup = None
    if problems:
        statement = (f"{len(problems)} of {len(cols)} columns need a look: " + "; ".join(problems[:3])
                     + (f" and {len(problems) - 3} more" if len(problems) > 3 else ""))
    elif dup:
        statement = f"The values look clean, but {dup:,} of {rows:,} rows are exact duplicates"
    else:
        statement = f"The data looks clean: no blanks, duplicates or misread numbers ({len(cols)} columns, {rows:,} rows)"
    report = {"rows": rows, "columns": len(cols), "problems": len(problems), "duplicate_rows": dup,
              "finding": finding("quality", statement, magnitude=len(problems), direction="flat", exact=True,
                                 duplicate_rows=dup)}
    return NodeResult(out.lazy(), report=report,
                      messages=[f"{rows:,} rows, {len(cols)} columns; {len(problems)} need a look"])


registry.register(NodeType(
    key="check_data", label="Check the data", category="Clean up", icon="⚑",
    description="One pass over every column: blanks, duplicates, values that are really numbers stored as text, "
                "columns with a single value. Says what is worth a look.",
    apply=_check_data,
    summary=lambda p: f"{len(p['columns'])} columns" if p.get("columns") else "every column",
    params=[Param("columns", "Columns", "columns", default=[], help="Empty = all")],
))


# =================================================================== forecast
def _cycle_expr(cycle: str, t: str) -> pl.Expr:
    c = pl.col(t).dt
    if cycle == "hour":
        return c.hour()
    if cycle == "month":
        return c.month()
    return c.weekday()                     # 1 = Monday … 7 = Sunday


def _py_key(v: Any, cycle: str) -> int:
    if cycle == "hour":
        return int(v.hour)
    if cycle == "month":
        return int(v.month)
    return int(v.isoweekday())


_NICE_STEPS = [(1, "1s"), (5, "5s"), (10, "10s"), (30, "30s"), (60, "1m"), (300, "5m"), (900, "15m"), (1800, "30m"),
               (3600, "1h"), (6 * 3600, "6h"), (86400, "1d"), (7 * 86400, "1w"), (30.436875 * 86400, "1mo"),
               (91.310625 * 86400, "3mo"), (365.2425 * 86400, "1y")]
_STEP_WORDS = {"s": "second", "m": "minute", "h": "hour", "d": "day", "w": "week", "mo": "month", "y": "year"}


def _step_words(step: str) -> str:
    m = re.fullmatch(r"(\d+)(us|ms|s|m|h|d|w|mo|y)", step)
    if not m:
        return step
    n, unit = int(m.group(1)), m.group(2)
    word = _STEP_WORDS.get(unit, unit)
    return word if n == 1 else f"{n} {word}s"


def _nice_step(secs: float) -> str:
    """The time step to project on when none is given: the plainest interval near how often the rows arrive."""
    if secs <= 0:
        return "1d"
    best = min(_NICE_STEPS, key=lambda s: abs(math.log(max(s[0], 1e-9) / secs)))
    return best[1]


def _forecast(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any]) -> NodeResult:
    lf = first_input(inputs)
    schema = schema_of(lf)
    t = require_column(schema, params.get("time_column"), "time column", TIME)
    col = require_column(schema, params.get("column"), "column to project", NUM)
    method = params.get("method") or "linear"
    cycle = params.get("cycle") or "weekday"
    horizon = int(number_param(params, "horizon", 10, "The horizon", whole=True, at_least=1, at_most=1000))
    prep = (lf.select([pl.col(t), pl.col(col).cast(pl.Float64).alias("__y")])
              .with_columns(pl.col(t).dt.epoch("s").cast(pl.Float64).alias("__t"))
              .filter(pl.col("__t").is_finite() & pl.col("__y").is_finite()))
    head = prep.select([pl.len().alias("n"), pl.col(t).min().alias("lo"), pl.col(t).max().alias("hi")]) \
               .collect(engine="streaming").row(0, named=True)
    n = int(head["n"])
    if n < 3:
        raise ValueError(f"{_label(ctx, col)} has too few values to project")
    last = head["hi"]
    every = (params.get("every") or "").strip()
    if every:
        step = parse_bucket(every)[0]
    else:
        span = max((head["hi"] - head["lo"]).total_seconds(), 0.0)
        step = _nice_step(span / max(n - 1, 1))
    from ..fits import fit_frame
    fit = fit_frame(prep, "__t", "__y", "linear")[0]
    a, b = float(fit.params[0]), float(fit.params[1])
    rmse = float(fit.rmse or 0.0)
    vals = [pl.select(pl.lit(last).dt.offset_by(f"{i}{step}")).item() for i in range(1, horizon + 1)]
    future_times = pl.Series("__t2", vals)
    epochs = future_times.dt.epoch("s").cast(pl.Float64).to_list()
    preds = [a * e + b for e in epochs]
    if method == "seasonal":                     # add the usual offset for each time of day/week/month
        resid = prep.select([(pl.col("__y") - (a * pl.col("__t") + b)).alias("r"),
                             _cycle_expr(cycle, t).alias("k")]).group_by("k").agg(pl.col("r").mean().alias("off")) \
                    .collect(engine="streaming")
        offsets = {int(k): float(o) for k, o in zip(resid["k"].to_list(), resid["off"].to_list()) if o is not None}
        preds = [p + offsets.get(_py_key(v, cycle), 0.0) for p, v in zip(preds, vals)]
    z = 1.96
    lower = [p - z * rmse for p in preds]
    upper = [p + z * rmse for p in preds]
    lower_name = temp_name(f"{col}_lower", schema)
    upper_name = temp_name(f"{col}_upper", schema)
    future = pl.DataFrame({t: vals, col: preds, lower_name: lower, upper_name: upper})
    step_secs = parse_bucket(step)[1]
    per_step = a * step_secs
    step_name = _step_words(step)
    at = future[t][-1]
    threshold = resolve_number(params.get("threshold"), ctx.inputs, "Threshold")
    crossing = ""
    if threshold is not None and a != 0:
        hit = next((i for i, p in enumerate(preds) if (a > 0 and p >= threshold) or (a < 0 and p <= threshold)), None)
        if hit is not None:
            crossing = f"; at that rate it crosses {fmt_number(threshold)} around {future[t][hit]:%Y-%m-%d}"
    trend = "rise" if per_step > 0 else "fall" if per_step < 0 else "stay flat"
    statement = (f"{_label(ctx, col)} is projected to {trend} about {fmt_number(abs(per_step))} per {step_name} "
                 f"(±{fmt_number(rmse)}); by {at:%Y-%m-%d} it is expected around {fmt_number(preds[-1])}{crossing}")
    report = {"method": method, "time_column": t, "column": col, "slope_per_step": per_step, "rmse": rmse,
              "points": n, "horizon": horizon, "step": step, "at": str(at),
              "finding": finding("forecast", statement, magnitude=abs(per_step), direction=_direction(per_step),
                                 exact=False, method=method, horizon=horizon, last=str(at))}
    msgs = [f"Fitted on {n:,} points; the band is ±{fmt_number(z * rmse)} (about 95%)"]
    return NodeResult(future.lazy(), report=report, messages=msgs)


registry.register(NodeType(
    key="forecast", label="Project forward", category="Time", icon="↗",
    description="Continue a value forward in time from its trend (and its usual day/week/month pattern), with a "
                "band that says how sure the projection is. Says when a threshold would be crossed.",
    apply=_forecast,
    summary=lambda p: f"{p.get('column') or '?'} for {p.get('horizon') or 10} more",
    params=[
        Param("time_column", "Time column", "column", column_group="temporal", required=True),
        Param("column", "Column to project", "column", column_group="numeric", required=True),
        Param("horizon", "How many steps ahead", "int", default=10, min=1, max=1000),
        Param("method", "Method", "choice", default="linear", choices=[
            ("linear", "Follow the straight-line trend"),
            ("seasonal", "Trend plus the usual hour/day/week/month pattern")]),
        Param("cycle", "Pattern repeats each", "choice", default="weekday", visible_when={"method": "seasonal"},
              choices=[("weekday", "Week"), ("hour", "Day"), ("month", "Year")]),
        Param("every", "One step is", "bucket", default="", advanced=True, help="Blank = as often as the readings arrive"),
        Param("threshold", "Crossing to look for", "text", default="", advanced=True, placeholder="a number or an input name"),
    ],
))
