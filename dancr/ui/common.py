"""Small pieces every centre page shares: the header strip, the coloured status line, and listening to
the Document for as long as a widget lives."""
from __future__ import annotations

from typing import Any, Callable

from PySide6.QtCore import QObject
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


def listen(owner: QObject, signal: Any, slot: Callable) -> None:
    """Connect a Document signal to ``slot`` for as long as ``owner`` lives.

    Qt drops a connection when the receiving widget is deleted only for its own methods, not for lambdas.
    The Document outlives its windows (a theme switch rebuilds the window around the same Document), so
    every view connects through here and nothing calls into a deleted widget afterwards. The signal is
    connected to a guard, never to ``slot`` itself, so dropping it never has to touch a deleted object."""
    n = _positional_count(slot)
    alive = [True]

    def call(*args: Any) -> None:
        if alive[0]:
            slot(*(args if n is None else args[:n]))

    def drop(*_: Any) -> None:
        alive[0] = False
        try:
            signal.disconnect(call)
        except (RuntimeError, TypeError, SystemError):
            pass
    signal.connect(call)
    owner.destroyed.connect(drop)


def _positional_count(fn: Callable) -> int | None:
    """How many signal arguments ``fn`` takes (Qt passes a slot only as many as it accepts); None = all."""
    import inspect
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return None
    if any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in params):
        return None
    return sum(1 for p in params if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD))
