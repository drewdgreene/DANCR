"""Phosphor icons (MIT, vendored under dancr/assets/icons) tinted for the current theme."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from PySide6.QtCore import Qt, QByteArray, QRectF
from PySide6.QtGui import QIcon, QPixmap, QPainter, QImage
from PySide6.QtSvg import QSvgRenderer

ICON_DIR = Path(__file__).resolve().parent.parent / "assets" / "icons"

NODE_ICONS = {
    "load_file": "file-csv", "keep_rows": "funnel", "choose_columns": "columns", "sort": "sort-ascending",
    "calculate": "function", "take_sample": "scissors", "fix_missing": "bandaids", "change_type": "text-aa",
    "remove_duplicates": "copy-simple", "remove_outliers": "lightning-slash", "stack": "rows",
    "combine": "arrows-merge", "time_buckets": "clock", "rolling": "wave-sine", "rate_of_change": "trend-up",
    "find_gaps": "arrows-out-line-horizontal", "regular_grid": "ruler", "compare_columns": "chart-scatter",
    "summarize": "sigma", "group_summary": "table", "chart": "chart-line", "export": "download-simple",
    "report": "article", "fit_curve": "chart-scatter", "predict": "trend-up", "check_limits": "check-circle",
    "enter_data": "note-pencil", "fix_values": "note-pencil", "summarise_around": "clock", "workbook": "table",
}


@lru_cache(maxsize=512)
def svg_bytes(name: str, color: str) -> bytes:
    path = ICON_DIR / f"{name}.svg"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        text = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256"><circle cx="128" cy="128" r="60" fill="currentColor"/></svg>'
    return text.replace("currentColor", color).encode()


@lru_cache(maxsize=512)
def renderer(name: str, color: str) -> QSvgRenderer:
    r = QSvgRenderer(QByteArray(svg_bytes(name, color)))
    r.setAspectRatioMode(Qt.KeepAspectRatio)
    return r


@lru_cache(maxsize=1024)
def pixmap(name: str, color: str, size: int) -> QPixmap:
    img = QImage(size, size, QImage.Format_ARGB32_Premultiplied)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    renderer(name, color).render(p, QRectF(0, 0, size, size))
    p.end()
    return QPixmap.fromImage(img)


def icon(name: str, color: str, size: int = 20) -> QIcon:
    ic = QIcon()
    for s in (size, size * 2):          # a double-size pixmap covers high-DPI screens
        ic.addPixmap(pixmap(name, color, s))
    return ic


def paint(painter: QPainter, name: str, color: str, rect: QRectF) -> None:
    renderer(name, color).render(painter, rect)


def node_icon_name(type_key: str) -> str:
    return NODE_ICONS.get(type_key, "circle-dashed")
