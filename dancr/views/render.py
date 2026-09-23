"""Headless chart rendering to PNG (matplotlib, Agg). Used by the CLI, MCP and reports."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from .chartquery import ChartData, query_panels, limit_values

PALETTE = ["#2f80ed", "#eb5757", "#27ae60", "#f2994a", "#9b51e0", "#00a3bf", "#e91e63", "#795548"]


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    return plt, mdates


def _title(c: str | None, columns: dict[str, dict] | None) -> str:
    if not c:
        return ""
    m = (columns or {}).get(c) or {}
    label = m.get("label") or c
    return f"{label} ({m['unit']})" if m.get("unit") else label


def render_chart(lf: pl.LazyFrame, params: dict[str, Any], out: Path | str, width: int = 1400, height: int = 700,
                 x_range: tuple[float, float] | None = None, dpi: int = 100, columns: dict[str, dict] | None = None,
                 inputs: dict[str, Any] | None = None) -> Path:
    plt, mdates = _mpl()
    out = Path(out)
    title = params.get("title") or ""
    schema = dict(lf.collect_schema())
    panels = query_panels(lf, schema, params, x_range=x_range, width_px=width, height_px=height)
    fig, axes = plt.subplots(len(panels), 1, figsize=(width / dpi, (height if len(panels) == 1 else height * 0.55 * len(panels)) / dpi),
                             dpi=dpi, sharex=(len(panels) > 1), squeeze=False)
    axes = [a[0] for a in axes]
    for i, ((label, cd), ax) in enumerate(zip(panels, axes)):
        sub = _draw_panel(ax, cd, params, columns, inputs, mdates)
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
    fig.savefig(out)
    plt.close(fig)
    return out


def _draw_panel(ax, cd: ChartData, params: dict[str, Any], columns, inputs, mdates) -> str:
    """Draw one queried panel into `ax`; returns the small info text."""
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
                ax.plot(_to_dates(xs) if cd.line.axis.kind == "time" else xs, yv, lw=0.8, color=spec.get("color") or PALETTE[i % len(PALETTE)],
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
            for gi, (g, gd) in enumerate(cd.groups):
                if gd.mode == "raw":
                    ax.scatter(_to_dates(gd.x) if gd.x_kind == "time" else gd.x, gd.y, s=5, alpha=0.6, color=PALETTE[gi % len(PALETTE)], label=g)
        elif d.mode == "raw":
            ax.scatter(_to_dates(d.x) if d.x_kind == "time" else d.x, d.y, s=4, alpha=0.6, color=PALETTE[0])
        else:
            x0, x1, y0, y1 = d.extent
            ax.imshow(np.log1p(d.density.T), origin="lower", aspect="auto", extent=(x0, x1, y0, y1), cmap="viridis")
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
        for lv, lab in limit_values(params, inputs):
            ax.axvline(lv, color="#dc2626", lw=1.1, ls=":", label=lab)
    else:
        b = cd.bar
        ax.bar(range(len(b.labels)), b.values, color=PALETTE[0])
        ax.set_xticks(range(len(b.labels)))
        ax.set_xticklabels(b.labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(f"{b.stat} of {cd.ys[0] if cd.ys else 'rows'}")
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


def _to_dates(x: np.ndarray) -> np.ndarray:
    out = np.full(len(x), np.datetime64("NaT", "us"), dtype="datetime64[us]")
    ok = np.isfinite(x)
    out[ok] = (x[ok] * 1e6).astype("datetime64[us]")
    return out
