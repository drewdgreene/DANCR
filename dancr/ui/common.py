"""Small pieces every centre page shares: the header strip and the coloured status line."""
from __future__ import annotations

from PySide6.QtWidgets import QFrame, QHBoxLayout

from .theme import T, STATUS_COLORS


def page_header(spacing: int = 10) -> tuple[QFrame, QHBoxLayout]:
    """The strip at the top of a centre page; returns the frame and its row layout."""
    head = QFrame(); head.setStyleSheet(f"QFrame {{ background: {T.bg}; border-bottom: 1px solid {T.border}; }}")
    h = QHBoxLayout(head); h.setContentsMargins(12, 6, 12, 6); h.setSpacing(spacing)
    return head, h


def status_dot(status: str, text: str) -> str:
    """'● text' with the dot in the colour of a step status (idle, stale, running, done, failed, preview)."""
    return f"<span style='color:{STATUS_COLORS.get(status, T.faint)}'>●</span> {text}"
