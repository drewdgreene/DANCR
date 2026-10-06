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
    if method == "seasonal" and cycle == "hour" and isinstance(schema[t], pl.Date):
        raise ValueError(f"A pattern through the day needs times of day, but {t} holds dates only. "
                         "Choose a pattern that repeats each week or each year")
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
        if isinstance(schema[t], pl.Date) and parse_bucket(step)[1] < 86400:
            step = "1d"                                # dates hold no time of day: 6 hours on from a date is the same date
    from ...fits import fit_frame
    fit = fit_frame(prep, "__t", "__y", "linear")[0]
    a, b = float(fit.params[0]), float(fit.params[1])
    rmse = float(fit.rmse or 0.0)
    parts = re.findall(r"(\d+)([a-z]+)", step)
    # step i is i whole steps on from the last reading ("1d" three times is "3d", not "31d"); each counted from
    # the last reading, so month steps from the 31st stay at month ends instead of drifting
    vals = [pl.select(pl.lit(last).dt.offset_by("".join(f"{int(k) * i}{u}" for k, u in parts))).item()
            for i in range(1, horizon + 1)]
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
    lower_name = f"{col}_lower" if f"{col}_lower" not in (t, col) else temp_name(f"{col}_lower", schema)
    upper_name = f"{col}_upper" if f"{col}_upper" not in (t, col) else temp_name(f"{col}_upper", schema)
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
    key="forecast", uses_labels=True, label="Project forward", category="Time", icon="↗",
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
