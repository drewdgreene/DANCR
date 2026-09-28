"""Chart series colours.

One list for the live chart view (pyqtgraph) and the headless renderer (matplotlib), so a chart
looks the same in the window, in a saved PNG and in a report. No heavy imports here on purpose.
"""
from __future__ import annotations

SERIES_COLORS = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed", "#0891b2", "#db2777", "#78716c"]


def series_color(index: int, spec: dict | None = None) -> str:
    """The colour for a series: an explicit colour from its spec, else the palette in order."""
    if spec and spec.get("color"):
        return str(spec["color"])
    return SERIES_COLORS[index % len(SERIES_COLORS)]
