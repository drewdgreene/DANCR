"""One query layer for both chart renderers: the pyqtgraph view in the window and the matplotlib
PNG for reports, the CLI and MCP. It resolves the chart settings against a schema, runs the
level-of-detail queries in ``lod`` and returns plain data; the renderers only draw."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from . import lod
from ..core.expr import kind_of_dtype, TIME, NUM
from ..core.dtypes import resolve_number

MAX_PANELS = 8


class ChartError(ValueError):
    """The chart settings cannot be drawn; the message says what to pick."""


@dataclass
class ChartData:
    """What one panel shows. ``kind`` is line | scatter | hist | bar."""
    kind: str
    x: str | None = None                    # axis column for line/scatter (None = row number)
    ys: list[str] = field(default_factory=list)
    series_specs: list[dict[str, Any]] = field(default_factory=list)   # the chosen series, aligned to ``ys`` (a
    #                                     column dropped from the schema must not shift the others' colour/label)
    line: lod.LineData | None = None
    scatter: lod.ScatterData | None = None
    groups: list[tuple[str, Any]] | None = None   # colour-by: (value, LineData | ScatterData)
    fits: list[tuple[Any, np.ndarray | None, np.ndarray | None]] = field(default_factory=list)   # (Fit, xs, ys) or (message, None, None)
    note: str = ""                          # what is not shown (colour-by values beyond the most common ones)
    mean: float | None = None
    hist: lod.HistData | None = None
    bar: lod.BarData | None = None

    @property
    def axis(self) -> lod.Axis | None:
        if self.line is not None:
            return self.line.axis
        if self.groups and self.kind == "line":
            return self.groups[0][1].axis
        return None

    def summary(self) -> str:
        if self.kind == "line":
            datas = [g for _, g in self.groups] if self.groups else ([self.line] if self.line else [])
            rows_in = sum(d.rows_in_range for d in datas)
            total = max((d.total_rows for d in datas), default=0) if self.groups else sum(d.total_rows for d in datas)   # every group sees the whole table's count
            mode = "envelope" if any(d.mode == "envelope" for d in datas) else "raw"
            return (f"{rows_in:,} of {total:,} rows in view · {'min/max per pixel' if mode == 'envelope' else 'every point'}"
                    + (f" · {self.note}" if self.note else ""))
        if self.kind == "scatter":
            return f"{self.scatter.rows:,} points" + (" as density" if self.scatter.mode == "density" else "")
        if self.kind == "hist":
            return f"{self.hist.rows:,} values · {len(self.hist.counts)} bins"
        shown = len(self.bar.labels)
        return f"{shown} categories" + (f" (the top {shown} of {self.bar.total:,})" if self.bar.total > shown else "")


def _group_label(value: Any) -> str:
    return "(blank)" if value is None else str(value)


def _others_note(column: str, others: int, rows: int) -> str:
    if not others:
        return ""
    return f"{others:,} less common value{'s' if others != 1 else ''} of {column} not shown ({rows:,} rows)"


def limit_values(params: dict[str, Any], inputs: dict[str, Any] | None) -> list[tuple[float, str]]:
    """Limit lines as (value, label); a value may be a number or the name of an input."""
    out = []
    for l in params.get("limits") or []:
        v = l.get("value")
        if v in (None, ""):
            continue
        try:
            fv = resolve_number(v, inputs, "limit")
        except ValueError:
            continue                        # a limit naming an input that no longer exists is not drawn
        if fv is None:
            continue
        out.append((fv, l.get("label") or str(v)))
    return out


def resolve_columns(schema: dict[str, pl.DataType], spec: dict[str, Any]) -> tuple[str | None, list[str]]:
    """The x column and the y columns a line or scatter chart uses: what was chosen if it exists,
    else the first date/time column for x and up to four number columns for y."""
    kind = spec.get("kind", "line")
    x = spec.get("x") or None
    if x is not None and x not in schema:
        x = None
    if x is None and kind == "line":
        x = next((c for c, dt in schema.items() if kind_of_dtype(dt) == TIME), None)
    ys = [s["column"] for s in (spec.get("series") or []) if s.get("column") and s["column"] in schema]
    if not ys:
        ys = [c for c, dt in schema.items() if kind_of_dtype(dt) == NUM and c != x][:4]
    return x, ys


def resolve_series_specs(spec: dict[str, Any], ys: list[str]) -> list[dict[str, Any]]:
    """The chosen series entries, in the same order as the resolved ``ys``: a series whose column is no longer in
    the table is left out here too, so it cannot shift the other series' colour and label."""
    chosen = [s for s in (spec.get("series") or []) if s.get("column") in ys]
    return chosen if chosen else [{"column": y} for y in ys]


def _mean(lf: pl.LazyFrame, col: str) -> float | None:
    v = lf.select(pl.col(col).cast(pl.Float64).mean()).collect(engine="streaming")[0, 0]
    return None if v is None or v != v else float(v)


def _filter_group(lf: pl.LazyFrame, col: str, value: Any) -> pl.LazyFrame:
    """The rows of one group, matched as values (as text, a true/false column reads "true" in Polars but
    "True" in Python, and dates and numbers have several spellings)."""
    return lf.filter(pl.col(col).is_null() if value is None else pl.col(col) == pl.lit(value))


def query_one(lf: pl.LazyFrame, schema: dict[str, pl.DataType], spec: dict[str, Any], *, x_range: tuple[float, float] | None = None,
              width_px: int = 1200, height_px: int = 600, bounds: dict[str, tuple[float, float, int]] | None = None,
              whole: pl.LazyFrame | None = None) -> ChartData:
    """Query one panel. ``bounds`` caches the x extent per axis column across calls (zooming); it is
    measured on ``whole`` (the unsplit frame) so split panels share one x axis."""
    kind = spec.get("kind", "line")
    bounds = {} if bounds is None else bounds
    whole = lf if whole is None else whole
    color_by = spec.get("color_by") or None
    if color_by and color_by not in schema:
        color_by = None
    if kind == "line":
        x, ys = resolve_columns(schema, spec)
        if not ys:
            raise ChartError("Choose a Y column for the chart")
        key = x or "__row"
        if key not in bounds:
            bounds[key] = lod.x_bounds(whole, x)
        cd = ChartData("line", x, ys, series_specs=resolve_series_specs(spec, ys))
        values, others, other_rows = lod.group_values_info(lf, color_by) if color_by else ([], 0, 0)
        if values:
            # with no x column the rows are numbered before the groups are taken apart, so each group keeps its rows
            base = lf if x else lf.with_row_index("__row")
            cd.groups = [(_group_label(g), lod.line_data(_filter_group(base, color_by, g), x, ys[:1], x_range=x_range, width_px=width_px, bounds=bounds[key]))
                         for g in values]
            cd.note = _others_note(color_by, others, other_rows)
        else:                                          # no colour-by, or no rows to colour
            cd.line = lod.line_data(lf, x, ys, x_range=x_range, width_px=width_px, bounds=bounds[key])
            if spec.get("mean_line"):
                cd.mean = _mean(lf, ys[0])
        return cd
    if kind == "scatter":
        x, ys = resolve_columns(schema, spec)
        if not x or not ys:
            raise ChartError("A scatter chart needs an X column and a Y column")
        w, h = max(1, width_px // 2), max(1, height_px // 2)
        cd = ChartData("scatter", x, ys[:1], series_specs=resolve_series_specs(spec, ys[:1]))
        cd.scatter = lod.scatter_data(lf, x, ys[0], x_range=x_range, width_px=w, height_px=h)
        values, others, other_rows = lod.group_values_info(lf, color_by) if color_by else ([], 0, 0)
        if values:
            cd.groups = [(_group_label(g), lod.scatter_data(_filter_group(lf, color_by, g), x, ys[0], x_range=x_range, width_px=w, height_px=h, max_raw=20000))
                         for g in values]
            cd.note = _others_note(color_by, others, other_rows)
        if spec.get("fit") and cd.scatter.x_kind != "time":
            from ..core.fits import fit_frame, curve_points
            try:
                for f in fit_frame(lf, x, ys[0], spec["fit"], int(spec.get("degree") or 2), color_by):
                    cx, cy = curve_points(f)
                    cd.fits.append((f, cx, cy))
            except ValueError as e:
                cd.fits.append((str(e), None, None))
        if spec.get("mean_line"):
            cd.mean = _mean(lf, ys[0])
        return cd
    if kind == "histogram":
        col = spec.get("column") or next((s["column"] for s in (spec.get("series") or []) if s.get("column")), None)
        if not col:
            raise ChartError("A histogram needs a column")
        return ChartData("hist", ys=[col], hist=lod.histogram_data(lf, col, int(spec.get("bins") or 50), x_range=x_range))
    if kind == "bar":
        cat = spec.get("category") or spec.get("x")
        if not cat:
            raise ChartError("A bar chart needs a category column for X")
        return ChartData("bar", x=cat, ys=[spec["value"]] if spec.get("value") else [],
                         bar=lod.bar_data(lf, cat, spec.get("value"), spec.get("stat", "mean"), error=spec.get("error") or ""))
    raise ChartError(f"Unknown chart type {kind!r}")


def query_panels(lf: pl.LazyFrame, schema: dict[str, pl.DataType], spec: dict[str, Any], *, x_range: tuple[float, float] | None = None,
                 width_px: int = 1200, height_px: int = 600, bounds: dict[str, tuple[float, float, int]] | None = None,
                 max_panels: int = MAX_PANELS) -> list[tuple[str | None, ChartData]]:
    """One (label, data) per panel: a single unlabelled panel, or one per value of ``split_by``."""
    split_by = spec.get("split_by") or None
    if split_by and (split_by not in schema or spec.get("kind", "line") == "bar"):
        split_by = None
    kw = dict(x_range=x_range, width_px=width_px, height_px=height_px, bounds={} if bounds is None else bounds, whole=lf)
    groups = lod.group_values(lf, split_by, limit=max_panels) if split_by else []
    if not groups:
        return [(None, query_one(lf, schema, spec, **kw))]
    kind = spec.get("kind", "line")
    if x_range is None and kind in ("histogram", "scatter"):
        # every panel on one x scale, measured over the whole table: each panel's own extent would put the other
        # panels' values off screen (the panels share their x axis) and bin each histogram differently
        col = (spec.get("column") or next((s["column"] for s in (spec.get("series") or []) if s.get("column")), None)) \
            if kind == "histogram" else resolve_columns(schema, spec)[0]
        if col and col in schema:
            lo, hi, _ = lod.x_bounds(lf, col)
            if hi > lo:
                kw["x_range"] = (lo, hi)
    return [(_group_label(g), query_one(_filter_group(lf, split_by, g), schema, spec, **kw)) for g in groups]
