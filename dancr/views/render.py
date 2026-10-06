"""Headless chart rendering to PNG (matplotlib, Agg). Used by the CLI, MCP and reports.

Each chart is its own Figure on its own Agg canvas, never pyplot's shared state, so charts can be drawn
on several threads at once (the MCP server runs tools in parallel)."""
from __future__ import annotations

import functools

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from .chartquery import ChartData, query_panels, limit_values
from .palette import SERIES_COLORS as PALETTE, series_color
from .table import column_title


def _mpl():
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import matplotlib.dates as mdates
    return Figure, FigureCanvasAgg, mdates


def _title(c: str | None, columns: dict[str, dict] | None) -> str:
    return column_title(c, columns)


def render_chart(lf: pl.LazyFrame, params: dict[str, Any], out: Path | str, width: int = 1400, height: int = 700,
                 x_range: tuple[float, float] | None = None, dpi: int = 100, columns: dict[str, dict] | None = None,
                 inputs: dict[str, Any] | None = None) -> Path:
    from ..core.nodes.outputs import validate_chart
    schema = dict(lf.collect_schema())
    validate_chart(schema, params)
    Figure, FigureCanvasAgg, mdates = _mpl()
    out = Path(out)
    title = params.get("title") or ""
    panels = query_panels(lf, schema, params, x_range=x_range, width_px=width, height_px=height)
    fig = Figure(figsize=(width / dpi, (height if len(panels) == 1 else height * 0.55 * len(panels)) / dpi), dpi=dpi)
    FigureCanvasAgg(fig)
    axes = fig.subplots(len(panels), 1, sharex=(len(panels) > 1), squeeze=False)
    axes = [a[0] for a in axes]
    for i, ((label, cd), ax) in enumerate(zip(panels, axes)):
        x_dt = schema.get(cd.x) if cd.x else None
        tz = x_dt.time_zone if isinstance(x_dt, pl.Datetime) else None
        sub = _draw_panel(ax, cd, params, columns, inputs, mdates, tz)
        if label is not None:
            ax.set_title(label, fontsize=9, color="#555", loc="left")
        if i < len(panels) - 1:
            ax.set_xlabel("")
        ax.grid(True, alpha=0.25)
    if title:
        fig.suptitle(title, fontsize=11)
        axes[-1].text(0.99, 0.01, sub, transform=axes[-1].transAxes, ha="right", va="bottom", fontsize=7, color="#888")
    elif len(panels) == 1:
        axes[0].set_title(sub, fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(out)
    finally:
        fig.clear()          # break the Figure<->canvas cycle: the GUI collects garbage rarely (gc is off)
    return out


def _draw_panel(ax, cd: ChartData, params: dict[str, Any], columns, inputs, mdates, tz: str | None = None) -> str:
    """Draw one queried panel into `ax`; returns the small info text. Times are drawn as wall times in the
    column's zone (``tz``), as the table shows them."""
    _to_dates = functools.partial(_wall_times, tz=tz)
    specs = params.get("series") or []
    breaks = params.get("break_gaps", True)
    if cd.kind == "line":
        if cd.groups:
            for gi, (g, data) in enumerate(cd.groups):
                for s in data.series:
                    xs, yv = lod_break(s.x, s.y, breaks)
                    ax.plot(_to_dates(xs) if data.axis.kind == "time" else xs, yv, lw=0.8, color=PALETTE[gi % len(PALETTE)], label=g)
        else:
            for i, s in enumerate(cd.line.series):
                spec = specs[i] if i < len(specs) else {}
                xs, yv = lod_break(s.x, s.y, breaks)
                ax.plot(_to_dates(xs) if cd.line.axis.kind == "time" else xs, yv, lw=0.8, color=series_color(i, spec),
                        label=spec.get("label") or _title(s.name, columns))
        if cd.mean is not None:
            ax.axhline(cd.mean, color="#555", lw=0.9, ls="--", label=f"average {cd.mean:.4g}")
        if cd.axis.kind == "time":
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        ax.set_xlabel(_title(cd.x, columns) if cd.x else "row")
        if len(cd.ys) == 1:
            ax.set_ylabel(_title(cd.ys[0], columns))
    elif cd.kind == "scatter":
        d = cd.scatter
        if cd.groups:
            # Groups too big to plot raw fall back to a density grid; draw the combined cloud as a
            # background so an all-density grouped scatter is not a blank panel.
            if d.mode == "density" and any(gd.mode == "density" for _, gd in cd.groups):
                _draw_density(ax, d, tz)
            for gi, (g, gd) in enumerate(cd.groups):
                if gd.mode == "raw":
                    ax.scatter(_to_dates(gd.x) if gd.x_kind == "time" else gd.x, gd.y, s=5, alpha=0.6, color=PALETTE[gi % len(PALETTE)], label=g)
        elif d.mode == "raw":
            ax.scatter(_to_dates(d.x) if d.x_kind == "time" else d.x, d.y, s=4, alpha=0.6, color=PALETTE[0])
        else:
            _draw_density(ax, d, tz)
        for gi, (f, cx, cy) in enumerate(cd.fits):
            if cx is None:
                ax.text(0.01, 0.99, f"fit: {f}", transform=ax.transAxes, va="top", fontsize=8, color="#b00")
            else:
                lab = f.equation + (f"  R²={f.r2:.3f}" if f.r2 is not None else "")
                ax.plot(cx, cy, lw=1.6, color=PALETTE[gi % len(PALETTE)] if cd.groups else "#111", label=(f"{f.group}: " if f.group is not None else "") + lab)
        if cd.mean is not None:
            ax.axhline(cd.mean, color="#555", lw=0.9, ls="--", label=f"average {cd.mean:.4g}")
        ax.set_xlabel(_title(cd.x, columns)); ax.set_ylabel(_title(cd.ys[0], columns))
    elif cd.kind == "hist":
        h = cd.hist
        ax.bar(h.edges[:-1], h.counts, width=np.diff(h.edges), align="edge", color=PALETTE[0], edgecolor="white", linewidth=0.3)
        ax.set_xlabel(_title(cd.ys[0], columns)); ax.set_ylabel("count")
        from matplotlib.ticker import MaxNLocator
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))       # counts are whole numbers
        for lv, lab in limit_values(params, inputs):
            ax.axvline(lv, color="#dc2626", lw=1.1, ls=":", label=lab)
    else:
        b = cd.bar
        ax.bar(range(len(b.labels)), b.values, color=PALETTE[0], yerr=b.errors, capsize=5 if b.errors is not None else 0,
               error_kw={"elinewidth": 1.2, "ecolor": "#333"})
        ax.set_xticks(range(len(b.labels)))
        ax.set_xticklabels(b.labels, rotation=45 if len(b.labels) > 4 else 0, ha="right" if len(b.labels) > 4 else "center", fontsize=8)
        ax.set_ylabel(f"{b.stat} of {cd.ys[0] if cd.ys else 'rows'}")
        if b.error:
            from .lod import ERROR_WORDS
            ax.text(0.99, 0.99, f"error bars: {ERROR_WORDS[b.error]}", transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#555")
    if cd.kind in ("line", "scatter"):
        for lv, lab in limit_values(params, inputs):
            ax.axhline(lv, color="#dc2626", lw=1.1, ls=":", label=lab)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best", fontsize=8)
    if params.get("y_label"):
        ax.set_ylabel(params["y_label"])
    if params.get("log_y"):
        ax.set_yscale("log")
    return cd.summary()


def lod_break(x: np.ndarray, y: np.ndarray, breaks: bool) -> tuple[np.ndarray, np.ndarray]:
    from . import lod
    return lod.break_gaps(x, y) if breaks else (x, y)


def _draw_density(ax, d, tz: str | None = None) -> None:
    """Draw a per-pixel density grid as a log-coloured image; a time x in the same date units the raw points use."""
    x0, x1, y0, y1 = d.extent
    if getattr(d, "x_kind", None) == "time":
        import matplotlib.dates as mdates
        x0, x1 = (float(v) for v in mdates.date2num(_wall_times(np.array([x0, x1]), tz)))
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.imshow(np.log1p(d.density.T), origin="lower", aspect="auto", extent=(x0, x1, y0, y1), cmap="viridis")


def _wall_times(x: np.ndarray, tz: str | None = None) -> np.ndarray:
    """Epoch seconds as datetimes: UTC for a naive column, the wall time in ``tz`` (with the offset of each
    moment, across daylight-saving changes) for a zoned one."""
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.datetime64("NaT", "us"), dtype="datetime64[us]")
    ok = np.isfinite(x)
    out[ok] = (x[ok] * 1e6).astype("datetime64[us]")
    if tz:
        out = (pl.Series(out).dt.replace_time_zone("UTC").dt.convert_time_zone(tz).dt.replace_time_zone(None)
               .to_numpy().astype("datetime64[us]"))
    return out


# =================================================================== maps
def render_map(lf: pl.LazyFrame, params: dict[str, Any], out: Path | str, width: int = 1400, height: int = 800,
               dpi: int = 100, columns: dict[str, dict] | None = None, inputs: dict[str, Any] | None = None) -> Path:
    """Draw a map of a node's output to PNG (matplotlib, Agg). Country outlines come from the bundled offline
    basemap; points or grid cells are drawn from the data. Thread-safe: its own Figure, never pyplot state."""
    from ..core.nodes.geo import validate_map
    from .mapquery import query_map
    from .geo_draw import draw_map
    schema = dict(lf.collect_schema())
    validate_map(schema, params)
    md = query_map(lf, schema, params)
    Figure, FigureCanvasAgg, _mdates = _mpl()
    out = Path(out)
    title = params.get("title") or ""
    fig = Figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1)
    sub = draw_map(ax, md, params, columns)
    ax.set_xlabel(""); ax.set_ylabel("")
    if title:
        fig.suptitle(title, fontsize=11)
        ax.text(0.99, 0.01, sub, transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="#888")
    else:
        ax.set_title(sub, fontsize=10)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(out)
    finally:
        fig.clear()          # break the Figure<->canvas cycle: the GUI collects garbage rarely (gc is off)
    return out
