"""Time-series operations for data with a date/time column."""
from __future__ import annotations

from typing import Any

import polars as pl

from ..params import Param
from ..registry import NodeType, InputSpec, Ctx, NodeResult, registry
from ..timeutil import parse_duration, format_seconds
from ._common import first_input, schema_of, require_column, temporal_columns, build_aggregations, stat_expr, STAT_HELP, STAT_CHOICES
from ..expr import TIME, NUM
from ..dtypes import is_date, align_time_column, temp_name


def _time_col(schema: dict[str, pl.DataType], params: dict[str, Any], key: str = "time_column") -> str:
    name = params.get(key) or next(iter(temporal_columns(schema)), None)
    if not name:
        raise ValueError("This table has no date/time column. Use 'Change type' to turn a column into a date/time first.")
    return require_column(schema, name, "time column", TIME)


def _with_time(lf: pl.LazyFrame, schema: dict[str, pl.DataType], t: str) -> pl.LazyFrame:
    """Rows with a blank time are dropped (they cannot be placed in time); Date columns become Datetime."""
    lf = lf.filter(pl.col(t).is_not_null())
    if is_date(schema[t]):
        lf = lf.with_columns(pl.col(t).cast(pl.Datetime("us")))
    return lf


# ------------------------------------------------------------ time buckets
def _time_buckets(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any]) -> NodeResult:
    lf = first_input(inputs)
    schema = schema_of(lf)
    t = _time_col(schema, params)
    lf = _with_time(lf, schema, t)
    every, secs = parse_duration(params.get("every") or "1m")
    only = [require_column(schema, c, "column", NUM) for c in (params.get("columns") or [])]
    aggs = build_aggregations(schema, params.get("aggregations"), exclude=[t],
                              default_stats=tuple(params.get("default_stats") or ["mean"]), only=only or None)
    if not aggs:
        raise ValueError("There are no number columns to summarise")
    count_col = (params.get("count_column") or "").strip()
    if count_col:
        if count_col == t or count_col in {a.meta.output_name() for a in aggs}:
            raise ValueError(f"The count column cannot be called {count_col!r}: that name is already used in the output")
        aggs.append(pl.len().alias(count_col))
    bucket = pl.col(t).dt.truncate(every).alias(t)
    out = lf.group_by(bucket).agg(aggs).sort(t)
    return NodeResult(out, messages=[f"Grouped rows into {every} buckets by {t}"])


registry.register(NodeType(
    key="time_buckets", label="Average over time", category="Time", icon="◷",
    description="Group rows into time buckets (every second, minute, hour...) and summarise each bucket. "
                "The fastest way to shrink millions of rows into something you can chart and reason about.",
    apply=_time_buckets,
    summary=lambda p: f"every {p.get('every') or '1m'}",
    params=[
        Param("every", "Bucket size", "duration", default="1m", required=True, placeholder="e.g. 1s, 1m, 15m, 1h, 1d"),
        Param("columns", "Columns to summarise", "columns", column_group="numeric", default=[],
              help="Empty = every number column"),
        Param("default_stats", "Statistics", "text_list", default=["mean"], help=STAT_HELP),
        Param("time_column", "Time column", "column", column_group="temporal", help="Blank = first date/time column", advanced=True),
        Param("aggregations", "Choose statistics column by column", "aggregations", default=[], column_group="numeric", advanced=True),
        Param("count_column", "Add a column counting rows per bucket", "text", default="", advanced=True, placeholder="e.g. count"),
    ],
))


# ------------------------------------------------------------------ rolling
def _rolling(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any]) -> NodeResult:
    lf = first_input(inputs)
    schema = schema_of(lf)
    cols = params.get("columns") or []
    if not cols:
        raise ValueError("Choose the column(s) to smooth")
    cols = [require_column(schema, c, "column", NUM) for c in cols]
    stat = params.get("stat") or "mean"
    window = str(params.get("window") or "20").strip()
    replace = bool(params.get("replace") or False)
    centered = bool(params.get("centered", True))
    name = lambda c: c if replace else f"{c}_{stat}_{window}"
    if not replace:
        clash = [name(c) for c in cols if name(c) in schema]
        if clash:
            raise ValueError(f"There is already a column called {clash[0]!r}; tick 'Replace the original columns' or rename it first")
    exprs = []
    if window.isdigit():
        n = int(window)
        if n < 1:
            raise ValueError("The window must be at least 1 row")
        for c in cols:
            exprs.append(getattr(pl.col(c), f"rolling_{stat}")(window_size=n, min_samples=1, center=centered).alias(name(c)))
        return NodeResult(lf.with_columns(exprs))
    t = _time_col(schema, params)
    every, secs = parse_duration(window)
    lf = _with_time(lf, schema, t).sort(t)
    by = t
    if centered:
        # a window ending half a span later is centred on each row
        by = temp_name("centre", schema)
        lf = lf.with_columns((pl.col(t) + pl.duration(microseconds=int(secs * 1e6 / 2))).alias(by))
    for c in cols:
        exprs.append(getattr(pl.col(c), f"rolling_{stat}_by")(by=by, window_size=every, closed="right").alias(name(c)))
    out = lf.with_columns(exprs)
    return NodeResult(out.drop(by) if by != t else out)


registry.register(NodeType(
    key="rolling", label="Smooth out noise", category="Time", icon="〰",
    description="Rolling average, median, min, max or standard deviation over a number of rows or a time span.",
    apply=_rolling,
    summary=lambda p: f"rolling {p.get('stat', 'mean')} over {p.get('window', 20)}",
    params=[
        Param("columns", "Columns", "columns", column_group="numeric", default=[], required=True),
        Param("stat", "Statistic", "choice", default="mean", choices=[
            ("mean", "average"), ("median", "median (robust to spikes)"), ("min", "minimum"), ("max", "maximum"), ("std", "standard deviation"), ("sum", "total")]),
        Param("window", "Window", "text", default="20", required=True, help="A number of rows (e.g. 20) or a time span (e.g. 30s, 5m)"),
        Param("time_column", "Time column", "column", column_group="temporal", help="Used when the window is a time span"),
        Param("centered", "Centre the window on each row", "bool", default=True, advanced=True),
        Param("replace", "Replace the original columns", "bool", default=False),
    ],
))


# ----------------------------------------------------------- rate of change
def _rate(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any]) -> pl.LazyFrame:
    lf = first_input(inputs)
    schema = schema_of(lf)
    cols = params.get("columns") or []
    if not cols:
        raise ValueError("Choose the column(s)")
    cols = [require_column(schema, c, "column", NUM) for c in cols]
    t = _time_col(schema, params)
    per = params.get("per") or "s"
    if per not in ("s", "m", "h", "d"):
        raise ValueError("'Per' must be second, minute, hour or day")
    div = {"s": 1e6, "m": 60e6, "h": 3600e6, "d": 86400e6}[per]
    n = max(1, int(params.get("span") or 1))
    lf = _with_time(lf, schema, t).sort(t)
    dt = (pl.col(t) - pl.col(t).shift(n)).dt.total_microseconds().cast(pl.Float64) / div
    exprs = []
    for c in cols:
        dy = pl.col(c).cast(pl.Float64) - pl.col(c).cast(pl.Float64).shift(n)
        rate = pl.when(dt > 0).then(dy / dt).otherwise(None)
        exprs.append(rate.alias(f"{c}_per_{ {'s': 'second', 'm': 'minute', 'h': 'hour', 'd': 'day'}[per] }"))
    return lf.with_columns(exprs)


registry.register(NodeType(
    key="rate_of_change", label="Rate of change", category="Time", icon="∂",
    description="How fast a value is changing per second, minute, hour or day.",
    apply=_rate,
    summary=lambda p: f"{', '.join(p.get('columns') or [])} per {p.get('per', 's')}",
    params=[
        Param("columns", "Columns", "columns", column_group="numeric", default=[], required=True),
        Param("time_column", "Time column", "column", column_group="temporal"),
        Param("per", "Per", "choice", default="s", choices=[("s", "second"), ("m", "minute"), ("h", "hour"), ("d", "day")]),
        Param("span", "Compare with N rows earlier", "int", default=1, min=1, advanced=True,
              help="Larger spans give a smoother rate"),
    ],
))


# ------------------------------------------------------------------ gaps
def _gaps(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any]) -> NodeResult:
    lf = first_input(inputs)
    schema = schema_of(lf)
    t = _time_col(schema, params)
    lf = _with_time(lf, schema, t).sort(t)
    expected = (params.get("expected") or "").strip()
    if expected:
        _, exp_s = parse_duration(expected)
    else:
        # median spacing over a sample (ignoring duplicate timestamps)
        sample = lf.select(pl.col(t)).head(200_000).collect(engine="streaming")[t]
        d = sample.diff().dt.total_microseconds().drop_nulls()
        d = d.filter(d > 0)
        if len(d) == 0:
            raise ValueError("Not enough distinct timestamps to work out the normal spacing. Set 'Normal spacing' by hand.")
        exp_s = float(d.median()) / 1e6
    factor = float(params.get("factor") or 1.5)
    thresh_us = int(exp_s * factor * 1e6)
    dt = (pl.col(t) - pl.col(t).shift(1)).dt.total_microseconds()
    out = (lf.select([
                pl.col(t).shift(1).alias("gap_start"),
                pl.col(t).alias("gap_end"),
                dt.alias("__dt")])                       # a fresh 3-column frame: no clash possible
             .filter(pl.col("__dt") > thresh_us)
             .with_columns([
                 (pl.col("__dt").cast(pl.Float64) / 1e6).alias("gap_seconds"),
                 pl.max_horizontal((pl.col("__dt").cast(pl.Float64) / 1e6 / exp_s).round(0) - 1, pl.lit(1)).cast(pl.Int64).alias("missing_readings")])
             .drop("__dt"))
    return NodeResult(out, report={"expected_spacing_s": exp_s, "threshold_s": exp_s * factor},
                      messages=[f"Normal spacing is {format_seconds(exp_s)}; a gap is anything over {format_seconds(exp_s * factor)}"])


registry.register(NodeType(
    key="find_gaps", label="Find gaps", category="Time", icon="⌷",
    description="List the places where the data stops and restarts. Output is one row per gap.",
    apply=_gaps,
    summary=lambda p: f"gaps > {p.get('factor', 1.5)}× normal spacing" if not p.get("expected") else f"spacing {p['expected']}",
    params=[
        Param("time_column", "Time column", "column", column_group="temporal"),
        Param("expected", "Normal spacing between rows", "duration", default="", placeholder="blank = detect (e.g. 50ms, 1s)"),
        Param("factor", "Count as a gap when spacing exceeds", "float", default=1.5, min=1.0, help="× the normal spacing"),
    ],
))


# ---------------------------------------------------------------- resample to grid
def _regular_grid(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any]) -> NodeResult:
    lf = first_input(inputs)
    schema = schema_of(lf)
    t = _time_col(schema, params)
    lf = _with_time(lf, schema, t).sort(t)
    tdt = pl.Datetime("us") if is_date(schema[t]) else schema[t]
    every, secs = parse_duration(params.get("every") or "1s")
    bounds = lf.select(pl.col(t).min().alias("lo"), pl.col(t).max().alias("hi")).collect(engine="streaming")
    lo, hi = bounds["lo"][0], bounds["hi"][0]
    if lo is None:
        raise ValueError("The time column is empty")
    n_ticks = int((hi - lo).total_seconds() / secs) + 1 if secs > 0 else float("inf")
    if n_ticks > 200_000_000:
        raise ValueError(f"A {every} spacing over this time span would create {n_ticks:,.0f} rows. Use a larger spacing.")
    idx = temp_name("tick", schema)
    step_us = int(round(secs * 1e6))
    start = pl.lit(lo).cast(pl.Datetime("us"))
    grid = (pl.LazyFrame().select(pl.int_range(0, n_ticks, dtype=pl.Int64).alias(idx))
              .select((start + pl.duration(microseconds=pl.col(idx) * step_us)).alias(t)))
    if isinstance(tdt, pl.Datetime) and tdt.time_zone:
        grid = grid.with_columns(align_time_column(pl.col(t), pl.Datetime("us"), tdt).alias(t))
    grid = grid.with_columns(pl.col(t).cast(tdt))
    method = params.get("method") or "nearest"
    if method in ("nearest", "backward", "forward"):
        out = grid.join_asof(lf, on=t, strategy=method)
    else:  # interpolate by time; real readings on a tick win over the empty grid row
        num = [c for c, dt in schema.items() if dt.is_numeric() and c != t]
        others = [c for c in schema if c != t]
        empty_grid = grid.with_columns([pl.lit(None).cast(schema[c]).alias(c) for c in others]).select(list(schema))
        merged = pl.concat([lf.select(list(schema)), empty_grid], how="vertical").sort(t, maintain_order=True)
        merged = merged.with_columns([pl.col(c).interpolate_by(pl.col(t)) for c in num])
        out = grid.join(merged.unique(subset=[t], keep="first", maintain_order=True), on=t, how="left", maintain_order="left").sort(t)
    return NodeResult(out, messages=[f"Resampled onto a regular {every} grid from {lo} to {hi}"])


registry.register(NodeType(
    key="regular_grid", label="Even out the timing", category="Time", icon="▦",
    description="Resample onto evenly spaced timestamps (every second, minute...). Useful before comparing two series recorded at different times.",
    apply=_regular_grid,
    summary=lambda p: f"every {p.get('every') or '1s'}, {p.get('method', 'nearest')}",
    params=[
        Param("time_column", "Time column", "column", column_group="temporal"),
        Param("every", "Spacing", "duration", default="1s", required=True),
        Param("method", "Value at each tick", "choice", default="nearest", choices=[
            ("nearest", "nearest row"), ("backward", "last row before"), ("forward", "first row after"), ("interpolate", "estimated between the readings either side")]),
    ],
))


# ------------------------------------------------------- summarise around each sample
MAX_OVERLAP = 500


def _around(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any]) -> NodeResult:
    """Each sample gets the statistics of every log row inside its window. A log row can belong to several
    samples when samples are closer together than the window, so the log is matched to the nearest sample
    (asof) and then to the next K samples along, K being the most samples any one window can hold."""
    samples = (inputs.get("samples") or [None])[0]
    log = (inputs.get("log") or [None])[0]
    if samples is None or log is None:
        raise ValueError("Connect the short table to 'Samples' and the continuous log to 'Log'")
    ss, ls = schema_of(samples), schema_of(log)
    st = _time_col(ss, params, "sample_time")
    lt = _time_col(ls, params, "log_time")
    window, secs = parse_duration(params.get("window") or "1d")
    w_us = int(round(secs * 1e6))
    side = params.get("side") or "before"
    if side not in ("before", "after", "around"):
        raise ValueError("'Which side' must be before, after or around")
    cols = [require_column(ls, c, "column", NUM) for c in (params.get("columns") or [])] or [c for c in ls if ls[c].is_numeric() and c != lt]
    stats = params.get("stats") or ["mean"]
    valid = {v for v, _ in STAT_CHOICES}
    for stt in stats:
        if stt not in valid:
            raise ValueError(f"Unknown statistic {stt!r}. Use: {', '.join(sorted(valid))}")
    taken = {**ss, **ls}
    key, s_time, anchor = temp_name("sample", taken), temp_name("sample_time", taken), temp_name("anchor", taken)
    s_sorted = _with_time(samples, ss, st).sort(st).with_row_index(key).with_columns([pl.col(key).cast(pl.Int64), pl.col(st).alias(s_time)])
    l_sorted = _with_time(log, ls, lt).sort(lt)
    if ls[lt] != ss[st]:
        l_sorted = l_sorted.with_columns(align_time_column(pl.col(lt), ls[lt], ss[st]).alias(lt))
    tdt = pl.Datetime("us") if is_date(ss[st]) else ss[st]
    # window (relative to the sample time) and which way to look for the nearest sample
    if side == "before":
        inside = (pl.col(lt) > pl.col(s_time) - pl.duration(microseconds=w_us)) & (pl.col(lt) <= pl.col(s_time))
        strategy, step, shift_us = "forward", 1, 0
    elif side == "after":
        inside = (pl.col(lt) >= pl.col(s_time)) & (pl.col(lt) < pl.col(s_time) + pl.duration(microseconds=w_us))
        strategy, step, shift_us = "backward", -1, 0
    else:
        half = pl.duration(microseconds=w_us // 2)
        inside = (pl.col(lt) >= pl.col(s_time) - half) & (pl.col(lt) <= pl.col(s_time) + half)
        strategy, step, shift_us = "forward", 1, -(w_us // 2)
    k = _max_overlap(s_sorted.select(pl.col(s_time).cast(pl.Datetime("us"))).collect(engine="streaming")[s_time], w_us)
    if k > MAX_OVERLAP:
        raise ValueError(f"Up to {k:,} samples fall inside one {window} window. Use a shorter window, or 'Average over time' on the log instead.")
    l_anch = l_sorted.with_columns((pl.col(lt) + pl.duration(microseconds=shift_us)).cast(tdt).alias(anchor))
    nearest = l_anch.join_asof(s_sorted.select([pl.col(s_time).cast(tdt).alias(anchor), pl.col(key)]), on=anchor, strategy=strategy).drop(anchor)
    nearest = nearest.filter(pl.col(key).is_not_null())
    lookup = s_sorted.select([pl.col(key), pl.col(s_time)])
    pieces = []
    for j in range(k):
        cand = nearest.with_columns((pl.col(key) + step * j).alias(key)) if j else nearest   # a missing neighbour simply does not join
        pieces.append(cand.join(lookup, on=key, how="inner").filter(inside).select([pl.col(key), *cols]))
    joined = pl.concat(pieces) if len(pieces) > 1 else pieces[0]
    aggs = [stat_expr(c, stt).alias(f"{c}_{stt}" if len(stats) > 1 else c) for c in cols for stt in stats]
    summary = joined.group_by(key).agg(aggs)
    out = s_sorted.join(summary, on=key, how="left", maintain_order="left").drop([key, s_time])
    where = {"before": "before it", "after": "after it", "around": "around it"}[side]
    return NodeResult(out, messages=[f"For each sample, {', '.join(stats)} of {', '.join(cols)} over the {window} {where}"])


def _max_overlap(times: pl.Series, w_us: int) -> int:
    """The most samples any window of length w can contain (two pointers over the sorted sample times)."""
    ts = times.drop_nulls().dt.timestamp("us").to_list()
    best, lo = 1, 0
    for hi in range(len(ts)):
        while ts[hi] - ts[lo] >= w_us:
            lo += 1
        best = max(best, hi - lo + 1)
    return best


registry.register(NodeType(
    key="summarise_around", label="Summarise around each sample", category="Time", icon="⧖",
    description="For each row of a short table (weekly samples, say), average a continuous log over the window before, after or around it. Joins sparse samples to dense logs.",
    apply=_around,
    inputs=[InputSpec("samples", "Samples"), InputSpec("log", "Log")],
    summary=lambda p: f"{', '.join(p.get('stats') or ['mean'])} over {p.get('window') or '1d'} {p.get('side') or 'before'} each sample",
    params=[
        Param("window", "Window", "duration", default="1d", required=True, placeholder="e.g. 6h, 1d, 7d"),
        Param("side", "Which side of each sample", "choice", default="before", choices=[
            ("before", "the window before the sample"), ("after", "the window after the sample"), ("around", "centred on the sample")]),
        Param("columns", "Log columns", "columns", column_group="numeric", port="log", help="Empty = every number column"),
        Param("stats", "Statistics", "text_list", default=["mean"], help=STAT_HELP),
        Param("sample_time", "Time column in samples", "column", column_group="temporal", port="samples", advanced=True),
        Param("log_time", "Time column in the log", "column", column_group="temporal", port="log", advanced=True),
    ],
))
