"""Draw a queried map onto a matplotlib Axes. Kept apart from ``render.py`` so the drawing rules
(outlines, point/cell layers, legend, colourbar, aspect) live in one place, shared by every headless
renderer. Pure matplotlib: no Qt, safe on a worker thread.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..core.geo import world_outlines
from .mapquery import OTHER_COLOR, MapData
from .palette import SERIES_COLORS
from .table import column_title


def project(lon, lat, projection: str):
    """Project (lon, lat) degrees to plotting coordinates. Equirectangular keeps lon/lat; Mercator stretches
    latitude by ``ln(tan(pi/4 + lat/2))``, clamped near the poles where it runs to infinity. Longitude is
    unchanged either way, so both render deterministically with no projection library."""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    if projection == "mercator":
        lat = np.clip(lat, -85.05112878, 85.05112878)
        y = np.degrees(np.log(np.tan(np.pi / 4 + np.radians(lat) / 2)))
        return lon, y
    return lon, lat


def _title(name: str | None, columns: dict[str, dict] | None) -> str:
    return column_title(name, columns)


def _overlaps(ring, extent) -> bool:
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    x0, x1, y0, y1 = extent
    return not (max(xs) < x0 or min(xs) > x1 or max(ys) < y0 or min(ys) > y1)


def draw_map(ax, md: MapData, params: dict[str, Any], columns: dict[str, dict] | None = None) -> str:
    """Draw the basemap and the data layers into ``ax``; return the small info text."""
    import matplotlib.patches as mpatches

    projection = params.get("projection") or "equirectangular"
    x0, x1, y0, y1 = md.extent
    px0, py_hi = project(x0, y1, projection)
    px1, py_lo = project(x1, y0, projection)
    ax.set_xlim(px0, px1)
    ax.set_ylim(py_lo, py_hi)

    if projection == "mercator":
        ax.set_aspect("equal")
    else:
        mid = max(-80.0, min(80.0, (y0 + y1) / 2.0))
        ax.set_aspect(1.0 / max(0.05, float(np.cos(np.radians(mid)))))

    if md.basemap:
        for ring in world_outlines():
            if not _overlaps(ring, md.extent):
                continue
            lons = np.array([p[0] for p in ring])
            lats = np.array([p[1] for p in ring])
            rx, ry = project(lons, lats, projection)
            ax.plot(rx, ry, color="#cbd5e1", linewidth=0.5, zorder=1)

    handled_legend = False
    if md.plotted:
        handled_legend = _draw_layers(ax, md, projection, columns)
    if md.legend and not handled_legend:
        ax.legend(handles=[mpatches.Patch(color=c, label=str(l)) for l, c in md.legend],
                  loc="best", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.15, zorder=0)
    ax.tick_params(labelsize=7)
    return md.summary()


def _color_array(md: MapData) -> np.ndarray | None:
    """A colour per point from category codes, or None when the colour is continuous (or absent)."""
    if md.codes is None:
        return None
    palette = [c for _, c in md.legend]
    return np.array([palette[c] if 0 <= c < len(palette) else OTHER_COLOR for c in md.codes], dtype=object)


def _marker_sizes(md: MapData) -> float | np.ndarray:
    if md.size is None or not len(md.size):
        return 14.0
    v = np.asarray(md.size, dtype=float)
    finite = np.isfinite(v)
    lo = float(np.nanmin(v[finite])) if finite.any() else 0.0
    hi = float(np.nanmax(v[finite])) if finite.any() else 1.0
    span = (hi - lo) or 1.0
    out = 10.0 + 90.0 * np.clip((v - lo) / span, 0.0, 1.0)
    return np.where(finite, out, 10.0)


def _draw_layers(ax, md: MapData, projection: str, columns) -> bool:
    """Draw points or cells; return True when a category legend was drawn here."""
    x, y = project(md.xs, md.ys, projection)
    sizes = _marker_sizes(md)
    colors = _color_array(md)
    marker = "s" if (md.kind == "cells" and md.cell_size) else "o"
    if marker == "s" and np.isscalar(sizes):
        # size the square to the cell on screen: fraction of the extent times the axes width in points
        fig_w_pt = ax.figure.get_size_inches()[0] * 72.0 * ax.get_position().width
        frac = md.cell_size / max(1e-9, md.extent[1] - md.extent[0])
        sizes = max(1.5, min(160.0, frac * fig_w_pt)) ** 2

    if md.color is not None and not md.legend:
        sc = ax.scatter(x, y, c=md.color, s=sizes, marker=marker, cmap="viridis", linewidths=0, alpha=0.88, zorder=3)
        cb = ax.figure.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
        cb.set_label(_title(md.color_column, columns), fontsize=8)
        if md.labels:
            _labels(ax, x, y, md.labels)
        return False
    if colors is not None:
        ax.scatter(x, y, c=list(colors), s=sizes, marker=marker, linewidths=0, alpha=0.88, zorder=3)
        if md.legend:
            import matplotlib.patches as mpatches
            ax.legend(handles=[mpatches.Patch(color=c, label=str(l)) for l, c in md.legend],
                      loc="best", fontsize=8, framealpha=0.9)
        if md.labels:
            _labels(ax, x, y, md.labels)
        return bool(md.legend)
    ax.scatter(x, y, c=SERIES_COLORS[0], s=sizes, marker=marker, linewidths=0, alpha=0.8, zorder=3)
    if md.labels:
        _labels(ax, x, y, md.labels)
    return False


def _labels(ax, x, y, labels) -> None:
    for xi, yi, lab in zip(x, y, labels):
        if lab:
            ax.annotate(str(lab), (xi, yi), xytext=(3, 3), textcoords="offset points", fontsize=7, zorder=4)
