"""Level-of-detail queries for charts.

All functions take a LazyFrame (usually ``pl.scan_parquet`` of a node output)
and return small numpy arrays sized to the pixels on screen, never the raw
rows. Every heavy query runs on Polars' streaming engine so memory stays
roughly flat regardless of table size. Time axes are float seconds since the
Unix epoch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from ..core.expr import _kind_of_dtype, NUM, TIME

MAX_RAW = 6000
STREAM = "streaming"


def _collect(lf: pl.LazyFrame) -> pl.DataFrame:
    return lf.collect(engine=STREAM)


@dataclass
class Axis:
    column: str | None          # None = row number
    kind: str                   # "time" | "number" | "index"
    lo: float | None = None
    hi: float | None = None


@dataclass
class Series:
    name: str
    x: np.ndarray
    y: np.ndarray
    mode: str = "raw"           # raw | envelope
    color: str | None = None


@dataclass
class LineData:
    axis: Axis
    series: list[Series] = field(default_factory=list)
    total_rows: int = 0
    rows_in_range: int = 0
    mode: str = "raw"


def break_gaps(x: np.ndarray, y: np.ndarray, factor: float = 8.0) -> tuple[np.ndarray, np.ndarray]:
    """Insert a NaN point wherever consecutive x values are more than `factor` × the median
    spacing of the points given apart, so a line chart shows dropouts as breaks instead of
    bridging them. On an envelope the points are the per-pixel extremes, so the median is the
    typical pixel spacing; on raw points it is the typical sample spacing."""
    if len(x) < 4:
        return x, y
    dx = np.diff(x)
    pos = dx[dx > 0]
    if len(pos) == 0:
        return x, y
    typical = np.median(pos)
    big = np.where(dx > factor * typical)[0]
    if len(big) == 0:
        return x, y
    xs = np.insert(x.astype(float), big + 1, x[big] + dx[big] / 2)
    ys = np.insert(y.astype(float), big + 1, np.nan)
    return xs, ys


def _x_expr(schema: dict[str, pl.DataType], x: str | None) -> tuple[pl.Expr, str]:
    if x is None:
        return pl.col("__row").cast(pl.Float64), "index"
    if x not in schema:
        raise ValueError(f"There is no column called {x!r}")
    dt = schema[x]
    kind = _kind_of_dtype(dt)
    if kind == TIME:
        if isinstance(dt, pl.Date):
            return pl.col(x).cast(pl.Datetime("us")).dt.timestamp("us").cast(pl.Float64) / 1e6, "time"
        return pl.col(x).dt.timestamp("us").cast(pl.Float64) / 1e6, "time"
    if kind == NUM:
        return pl.col(x).cast(pl.Float64), "number"
    raise ValueError(f"{x} cannot be used as an axis (it is {kind}). Pick a number or date/time column.")


def _range_filter(schema: dict[str, pl.DataType], x: str | None, r0: float, r1: float) -> pl.Expr:
    """Filter on the ORIGINAL column so Parquet row-group statistics can skip data."""
    if x is None:
        return (pl.col("__row") >= int(r0)) & (pl.col("__row") <= int(r1))
    dt = schema[x]
    if _kind_of_dtype(dt) == TIME:
        unit = dt.time_unit if isinstance(dt, pl.Datetime) else "us"
        tz = dt.time_zone if isinstance(dt, pl.Datetime) else None
        mult = {"ns": 1e9, "us": 1e6, "ms": 1e3}[unit]
        lo = pl.lit(int(r0 * mult)).cast(pl.Datetime(unit))
        hi = pl.lit(int(r1 * mult) + 1).cast(pl.Datetime(unit))
        if tz:
            lo = lo.dt.replace_time_zone("UTC").dt.convert_time_zone(tz)
            hi = hi.dt.replace_time_zone("UTC").dt.convert_time_zone(tz)
        if isinstance(dt, pl.Date):
            return (pl.col(x).cast(pl.Datetime("us")) >= lo) & (pl.col(x).cast(pl.Datetime("us")) <= hi)
        return (pl.col(x) >= lo) & (pl.col(x) <= hi)
    return (pl.col(x) >= r0) & (pl.col(x) <= r1)


def x_bounds(lf: pl.LazyFrame, x: str | None) -> tuple[float, float, int]:
    schema = dict(lf.collect_schema())
    if x is None:
        n = int(_collect(lf.select(pl.len()))[0, 0])
        return 0.0, float(max(n - 1, 0)), n
    xe, _ = _x_expr(schema, x)
    r = _collect(lf.select(xe.min().alias("lo"), xe.max().alias("hi"), pl.len().alias("n")))
    lo, hi, n = r["lo"][0], r["hi"][0], int(r["n"][0])
    return (float(lo) if lo is not None else 0.0), (float(hi) if hi is not None else 0.0), n


def line_data(lf: pl.LazyFrame, x: str | None, ys: list[str], x_range: tuple[float, float] | None = None,
              width_px: int = 1200, max_raw: int = MAX_RAW, bounds: tuple[float, float, int] | None = None) -> LineData:
    """Per pixel column, the M4 points of each series: first, lowest, highest and last finite value, in x order.
    Rows without an x, and values that are NaN or infinite, are left out of the drawing (and of the counts)."""
    schema = dict(lf.collect_schema())
    for y in ys:
        if y not in schema:
            raise ValueError(f"There is no column called {y!r}")
        if _kind_of_dtype(schema[y]) != NUM:
            raise ValueError(f"{y} is not a number column")
    base = lf if x is not None else lf.with_row_index("__row")
    xe, xkind = _x_expr(schema, x)
    lo, hi, total = bounds if bounds is not None else x_bounds(lf, x)
    axis = Axis(x, xkind, lo, hi)
    if x_range is not None:
        r0, r1 = x_range
        base = base.filter(_range_filter(schema, x, r0, r1))
        lo, hi = max(lo, r0), min(hi, r1)
    base = base.with_columns(xe.alias("__x")).filter(pl.col("__x").is_not_null() & pl.col("__x").is_finite())
    n_in = int(_collect(base.select(pl.len()))[0, 0])
    out = LineData(axis, total_rows=total, rows_in_range=n_in)
    if n_in == 0:
        for y in ys:
            out.series.append(Series(y, np.empty(0), np.empty(0)))
        return out
    if n_in <= max_raw:
        df = _collect(base.select(["__x", *ys]).sort("__x"))
        xs = df["__x"].to_numpy()
        for y in ys:
            yv = df[y].cast(pl.Float64).to_numpy()
            m = np.isfinite(yv)
            out.series.append(Series(y, xs[m], yv[m], "raw"))
        out.mode = "raw"
        return out
    width_px = max(50, int(width_px))
    span = (hi - lo) or 1.0
    bucket = ((pl.col("__x") - lo) / span * width_px).floor().clip(0, width_px - 1).cast(pl.Int32).alias("__b")
    aggs: list[pl.Expr] = []
    xc = pl.col("__x")
    for y in ys:
        yc = pl.col(y).cast(pl.Float64)
        ok = yc.is_finite()
        yf, xf = yc.filter(ok), xc.filter(ok)
        aggs += [xf.first().alias(f"{y}__x0"), yf.first().alias(f"{y}__y0"),
                 xf.sort_by(yf).first().alias(f"{y}__xmin"), yf.min().alias(f"{y}__ymin"),
                 xf.sort_by(yf).last().alias(f"{y}__xmax"), yf.max().alias(f"{y}__ymax"),
                 xf.last().alias(f"{y}__x1"), yf.last().alias(f"{y}__y1")]
    df = _collect(base.with_columns(bucket).group_by("__b").agg(aggs).sort("__b"))
    for y in ys:
        pts = [df[f"{y}__{k}"].to_numpy().astype(float) for k in ("x0", "y0", "xmin", "ymin", "xmax", "ymax", "x1", "y1")]
        x0, y0, xa, ya, xb, yb, x1, y1 = pts
        # min and max in x order, so the line never doubles back inside a pixel column
        swap = xb < xa
        xa, xb = np.where(swap, xb, xa), np.where(swap, xa, xb)
        ya, yb = np.where(swap, yb, ya), np.where(swap, ya, yb)
        xs = np.stack([x0, xa, xb, x1], 1).ravel()
        yv = np.stack([y0, ya, yb, y1], 1).ravel()
        m = np.isfinite(yv) & np.isfinite(xs)
        out.series.append(Series(y, xs[m], yv[m], "envelope"))
    out.mode = "envelope"
    return out


@dataclass
class ScatterData:
    mode: str                          # raw | density
    x: np.ndarray | None = None
    y: np.ndarray | None = None
    density: np.ndarray | None = None  # shape (W, H)
    extent: tuple[float, float, float, float] | None = None  # x0, x1, y0, y1
    rows: int = 0
    x_kind: str = "number"


def scatter_data(lf: pl.LazyFrame, x: str, y: str, x_range: tuple[float, float] | None = None,
                 y_range: tuple[float, float] | None = None, width_px: int = 600, height_px: int = 400,
                 max_raw: int = MAX_RAW) -> ScatterData:
    schema = dict(lf.collect_schema())
    xe, xkind = _x_expr(schema, x)
    if _kind_of_dtype(schema[y]) != NUM:
        raise ValueError(f"{y} is not a number column")
    base = lf
    if x_range and xkind != "index":
        base = base.filter(_range_filter(schema, x, x_range[0], x_range[1]))
    base = base.select(xe.alias("__x"), pl.col(y).cast(pl.Float64).alias("__y")).filter(pl.col("__x").is_finite() & pl.col("__y").is_finite())
    if y_range:
        base = base.filter((pl.col("__y") >= y_range[0]) & (pl.col("__y") <= y_range[1]))
    n = int(_collect(base.select(pl.len()))[0, 0])
    if n <= max_raw:
        df = _collect(base)
        return ScatterData("raw", df["__x"].to_numpy(), df["__y"].to_numpy(), rows=n, x_kind=xkind)
    b = _collect(base.select(pl.col("__x").min().alias("x0"), pl.col("__x").max().alias("x1"), pl.col("__y").min().alias("y0"), pl.col("__y").max().alias("y1")))
    x0, x1, y0, y1 = (float(b[c][0]) for c in ("x0", "x1", "y0", "y1"))
    sx, sy = (x1 - x0) or 1.0, (y1 - y0) or 1.0
    W, H = max(10, int(width_px)), max(10, int(height_px))
    df = _collect(base.with_columns([
            ((pl.col("__x") - x0) / sx * W).floor().clip(0, W - 1).cast(pl.Int32).alias("__bx"),
            ((pl.col("__y") - y0) / sy * H).floor().clip(0, H - 1).cast(pl.Int32).alias("__by")])
          .group_by(["__bx", "__by"]).agg(pl.len().alias("n")))
    grid = np.zeros((W, H), dtype=np.float64)
    grid[df["__bx"].to_numpy(), df["__by"].to_numpy()] = df["n"].to_numpy()
    return ScatterData("density", density=grid, extent=(x0, x1, y0, y1), rows=n, x_kind=xkind)


@dataclass
class HistData:
    edges: np.ndarray
    counts: np.ndarray
    rows: int
    lo: float
    hi: float


def histogram_data(lf: pl.LazyFrame, column: str, bins: int = 50, x_range: tuple[float, float] | None = None) -> HistData:
    schema = dict(lf.collect_schema())
    if _kind_of_dtype(schema[column]) != NUM:
        raise ValueError(f"{column} is not a number column")
    base = lf.select(pl.col(column).cast(pl.Float64).alias("__v")).filter(pl.col("__v").is_finite())
    if x_range:
        base = base.filter((pl.col("__v") >= x_range[0]) & (pl.col("__v") <= x_range[1]))
        lo, hi = x_range
    else:
        b = _collect(base.select(pl.col("__v").min().alias("lo"), pl.col("__v").max().alias("hi")))
        lo, hi = b["lo"][0], b["hi"][0]
        if lo is None:
            return HistData(np.array([0.0, 1.0]), np.array([0]), 0, 0.0, 1.0)
        lo, hi = float(lo), float(hi)
    span = (hi - lo) or 1.0
    bins = max(2, int(bins))
    df = _collect(base.with_columns(((pl.col("__v") - lo) / span * bins).floor().clip(0, bins - 1).cast(pl.Int32).alias("__b"))
                  .group_by("__b").agg(pl.len().alias("n")))
    counts = np.zeros(bins, dtype=np.int64)
    counts[df["__b"].to_numpy()] = df["n"].to_numpy()
    edges = lo + np.arange(bins + 1) * span / bins
    return HistData(edges, counts, int(counts.sum()), lo, hi)


@dataclass
class BarData:
    labels: list[str]
    values: np.ndarray
    stat: str


def bar_data(lf: pl.LazyFrame, category: str, value: str | None, stat: str = "mean", top: int = 60) -> BarData:
    from ..core.nodes._common import stat_expr
    from .table import format_value
    if value and stat != "count":
        agg = stat_expr(value, stat).alias("__v")
    else:
        agg = pl.len().alias("__v")
        stat = "count"
    df = _collect(lf.group_by(category).agg(agg).sort("__v", descending=True).head(top))
    return BarData([format_value(v) for v in df[category].to_list()], df["__v"].cast(pl.Float64).to_numpy(), stat)


def group_values(lf: pl.LazyFrame, column: str, limit: int = 12) -> list[Any]:
    """Distinct values of a colour-by column, most frequent first (at most `limit`)."""
    df = _collect(lf.group_by(column).agg(pl.len().alias("n")).sort("n", descending=True).head(limit))
    return [v for v in df[column].to_list() if v is not None]
