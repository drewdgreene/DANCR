"""One query layer for both map renderers: the pyqtgraph view in the window and the matplotlib PNG for
reports, the CLI and MCP. It resolves the map settings against a schema, reads the points (sampling when
there are very many) and returns plain data; the renderers only draw.

Like the chart layer, nothing here touches Qt, so it runs on a worker thread and in tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from ..core.expr import NUM, kind_of_dtype
from ..core.geo import grid_degrees, world_bounds
from .palette import SERIES_COLORS

MAX_POINTS = 80_000            # above this the map samples, evenly and deterministically, and says so
MAX_CATEGORIES = 12
OTHER_COLOR = "#c3c8d0"


class MapError(ValueError):
    """The map settings cannot be drawn; the message says what to pick."""


@dataclass
class MapData:
    """What one map shows."""
    kind: str = "points"                     # "points" or "cells"
    xs: np.ndarray = field(default_factory=lambda: np.empty(0))     # longitudes
    ys: np.ndarray = field(default_factory=lambda: np.empty(0))     # latitudes
    color: np.ndarray | None = None          # continuous colour values (numeric color-by)
    codes: np.ndarray | None = None          # category codes (category color-by)
    legend: list[tuple[str, str]] = field(default_factory=list)     # (label, colour) for categories
    size: np.ndarray | None = None           # numeric values scaling point area
    labels: list[str] | None = None
    cell_size: float | None = None           # degrees, when drawing grid squares
    extent: tuple[float, float, float, float] = (-180.0, 180.0, -90.0, 90.0)   # lon0, lon1, lat0, lat1
    total: int = 0
    plotted: int = 0
    stride: int = 1
    basemap: bool = True
    color_column: str | None = None          # the column colour comes from, for the colourbar label
    note: str = ""

    def summary(self) -> str:
        n = f"{self.plotted:,} point{'s' if self.plotted != 1 else ''}"
        if self.cell_size is not None:
            n = f"{self.plotted:,} cell{'s' if self.plotted != 1 else ''} of {self.cell_size:g}°"
        bits = [n]
        if self.legend:
            bits.append(f"{len(self.legend)} group{'s' if len(self.legend) != 1 else ''}")
        if self.stride > 1:
            bits.append(f"1 in {self.stride} shown ({self.total:,} rows)")
        if self.note:
            bits.append(self.note)
        return " · ".join(bits)


def _numeric(schema: dict[str, pl.DataType], name: str | None) -> bool:
    return bool(name) and name in schema and kind_of_dtype(schema[name]) == NUM


def resolve_settings(schema: dict[str, pl.DataType], params: dict[str, Any]) -> dict[str, Any]:
    """Check the map settings against the schema and return a plain dict of what to read."""
    lat = params.get("lat")
    lon = params.get("lon")
    if not lat or lat not in schema:
        raise MapError("Choose the latitude column")
    if not lon or lon not in schema:
        raise MapError("Choose the longitude column")
    if kind_of_dtype(schema[lat]) != NUM or kind_of_dtype(schema[lon]) != NUM:
        raise MapError("The latitude and longitude must be number columns. Use 'Fix numbers and dates' first")
    color_by = params.get("color_by") or None
    if color_by and color_by not in schema:
        raise MapError(f"There is no column called {color_by!r} to colour by")
    size_by = params.get("size_by") or None
    if size_by:
        if size_by not in schema:
            raise MapError(f"There is no column called {size_by!r} to size by")
        if kind_of_dtype(schema[size_by]) != NUM:
            raise MapError("The column to size by must be a number column")
    label = params.get("label") or None
    if label and label not in schema:
        raise MapError(f"There is no column called {label!r} to label by")
    cell_size = None
    raw = str(params.get("cell_size") or "").strip()
    if raw:
        cell_size = grid_degrees(raw)
    return {"lat": lat, "lon": lon, "color_by": color_by, "size_by": size_by, "label": label,
            "cell_size": cell_size,
            "basemap": bool(params.get("basemap", True)),
            "extent_mode": params.get("extent") or "auto",
            "projection": params.get("projection") or "equirectangular"}


def query_map(lf: pl.LazyFrame, schema: dict[str, pl.DataType], params: dict[str, Any], *,
              cap: int = MAX_POINTS) -> MapData:
    """Read the points (or cells) a map shows, sampling evenly when there are more than ``cap``."""
    r = resolve_settings(schema, params)
    lat, lon = r["lat"], r["lon"]
    total = int(lf.select(pl.len()).collect(engine="streaming")[0, 0])

    keep = [lat, lon]
    for extra in (r["color_by"], r["size_by"], r["label"]):
        if extra and extra not in keep:
            keep.append(extra)

    stride = 1
    work = lf
    if total > cap:
        stride = int(np.ceil(total / cap))
        work = work.with_row_index("__dancr_row").filter(pl.col("__dancr_row") % stride == 0)
    df = work.select(keep).drop_nulls(subset=[lat, lon]).collect(engine="streaming")

    xs = df[lon].cast(pl.Float64).to_numpy()
    ys = df[lat].cast(pl.Float64).to_numpy()
    good = np.isfinite(xs) & np.isfinite(ys)
    if not good.all():
        xs, ys = xs[good], ys[good]
        df = df.filter(pl.Series(good))

    md = MapData(xs=xs, ys=ys, total=total, plotted=len(xs), stride=stride,
                 basemap=r["basemap"], cell_size=r["cell_size"], color_column=r["color_by"])

    if r["color_by"]:
        col = df[r["color_by"]]
        if kind_of_dtype(schema[r["color_by"]]) == NUM:
            md.color = col.cast(pl.Float64).to_numpy()
        else:
            vals = col.cast(pl.Utf8).fill_null("(blank)")
            counts = vals.value_counts(sort=True, name="_n")
            top = [row[0] for row in counts.head(MAX_CATEGORIES).iter_rows()]
            lookup = {v: i for i, v in enumerate(top)}
            md.codes = np.array([lookup.get(v, -1) for v in vals.to_list()], dtype=np.int64)
            md.legend = [(v, SERIES_COLORS[i % len(SERIES_COLORS)]) for i, v in enumerate(top)]
            if len(counts) > len(top):
                md.note = f"{len(counts) - len(top):,} smaller groups in grey"

    if r["size_by"]:
        md.size = df[r["size_by"]].cast(pl.Float64).to_numpy()

    if r["label"]:
        labels = df[r["label"]].cast(pl.Utf8).to_list()
        if len(labels) <= 400:                  # labels only when there are few enough to read
            md.labels = ["" if v is None else v for v in labels]

    if len(xs) and r["extent_mode"] != "world":
        md.extent = _padded_extent(float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max()),
                                   cell_size=md.cell_size)
    else:
        md.extent = world_bounds()
    return md


def _padded_extent(x0: float, x1: float, y0: float, y1: float, *, cell_size: float | None) -> tuple[float, float, float, float]:
    """A small margin so points are never on the frame; a single point or line gets a sensible window."""
    dx = x1 - x0
    dy = y1 - y0
    pad_x = max(dx * 0.06, (cell_size or 0) or 0.02)
    pad_y = max(dy * 0.06, (cell_size or 0) or 0.02)
    if dx == 0:
        pad_x = max(pad_x, 0.05)
    if dy == 0:
        pad_y = max(pad_y, 0.05)
    x0, x1 = max(-180.0, x0 - pad_x), min(180.0, x1 + pad_x)
    y0, y1 = max(-90.0, y0 - pad_y), min(90.0, y1 + pad_y)
    return (x0, x1, y0, y1)
