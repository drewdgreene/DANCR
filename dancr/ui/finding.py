"""The result card: what a step found, set like a result rather than a footnote."""
from __future__ import annotations

import html
import re
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QLabel, QWidget

from ..core.fits import equation as fit_equation
from .theme import T


def pretty_equation(kind: str, params: list[float], x: str, y: str) -> str:
    """Equation as HTML: real multiplication sign, superscript powers, display names for the columns."""
    s = html.escape(fit_equation(kind, params, x, y))
    s = s.replace("·", " × ")
    s = re.sub(r"\^\(([^)]*)\)", r"<sup>\1</sup>", s)          # e^(0.3·x)
    s = re.sub(r"\^(-?[\d.]+)", r"<sup>\1</sup>", s)            # x^2, x^0.7
    return s


def _fmt(v: Any, digits: int = 4) -> str:
    if v is None:
        return "–"
    if isinstance(v, int):
        return f"{v:,}"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if abs(f) >= 1e5 or (abs(f) < 1e-3 and f != 0):
        return f"{f:.{digits}g}"
    return f"{f:,.{digits}g}"


class Stat(QWidget):
    def __init__(self, value: str, label: str) -> None:
        super().__init__()
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(0)
        v = QLabel(value); f = QFont(T.ui_font); f.setPointSizeF(13); f.setBold(True); v.setFont(f)
        v.setTextInteractionFlags(Qt.TextSelectableByMouse)
        l = QLabel(label); l.setObjectName("muted"); lf = QFont(T.ui_font); lf.setPointSizeF(8.5); l.setFont(lf)
        lay.addWidget(v); lay.addWidget(l)


class FindingCard(QFrame):
    """Shows the finding of a step (fit equation and quality, pass/fail verdict, gaps, saved files, notes)."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("finding")
        self.setStyleSheet(f"QFrame#finding {{ background: {T.panel}; border: 1px solid {T.border}; border-radius: 8px; }}")
        self.lay = QVBoxLayout(self); self.lay.setContentsMargins(14, 10, 14, 12); self.lay.setSpacing(6)
        self.hide()

    def _clear(self) -> None:
        while self.lay.count():
            it = self.lay.takeAt(0)
            w = it.widget()
            if w is not None:
                w.deleteLater()
            elif it.layout() is not None:
                sub = it.layout()
                while sub.count():
                    w2 = sub.takeAt(0).widget()
                    if w2 is not None:
                        w2.deleteLater()
                sub.deleteLater()

    def _heading(self, text: str, color: str | None = None) -> None:
        h = QLabel(text.upper()); f = QFont(T.ui_font); f.setPointSizeF(8); f.setBold(True); h.setFont(f)
        h.setStyleSheet(f"color: {color or T.muted}; letter-spacing: 1px;")
        self.lay.addWidget(h)

    def _big(self, html_text: str, size: float = 11.5, color: str | None = None) -> None:
        l = QLabel(html_text); l.setWordWrap(True); l.setTextFormat(Qt.RichText); l.setTextInteractionFlags(Qt.TextSelectableByMouse)
        f = QFont(T.ui_font); f.setPointSizeF(size); f.setWeight(QFont.DemiBold); l.setFont(f)
        if color:
            l.setStyleSheet(f"color: {color};")
        self.lay.addWidget(l)

    def _stats(self, items: list[tuple[str, str]]) -> None:
        row = QHBoxLayout(); row.setSpacing(18)
        for value, label in items:
            row.addWidget(Stat(value, label))
        row.addStretch()
        self.lay.addLayout(row)

    def _text(self, text: str, muted: bool = True, rich: bool = False) -> None:
        l = QLabel(text if rich else html.escape(text)); l.setWordWrap(True); l.setTextFormat(Qt.RichText)
        l.setTextInteractionFlags(Qt.TextSelectableByMouse)
        if muted:
            l.setObjectName("muted")
        self.lay.addWidget(l)

    # ------------------------------------------------------------ content
    def set_finding(self, node_type: str, st, title) -> None:
        """`title` maps a column name to its display name (with unit)."""
        self._clear()
        rep = st.report or {}
        msgs = list(st.messages or [])
        shown = False
        if st.status == "done" and node_type == "fit_curve" and rep.get("fits"):
            self._heading("Result · " + {"linear": "straight line", "polynomial": "curve", "saturating": "levels off", "exponential": "exponential",
                                         "power": "power law", "logarithmic": "logarithmic"}.get(rep.get("kind", ""), rep.get("kind", "")))
            for f in rep["fits"]:
                if f.get("group") is not None:
                    self._text(f"<b>{html.escape(str(f['group']))}</b>", muted=True, rich=True)
                self._big(pretty_equation(f["kind"], f["params"], title(f["x"]), title(f["y"])))
                r2 = f.get("r2")
                self._stats([(_fmt(r2, 4), "R²"), (_fmt(f.get("rmse"), 3), "typical error"), (f"{int(f.get('n') or 0):,}", "points")])
                if r2 is not None:
                    self._text(f"The curve explains {r2 * 100:.2f}% of the variation in {title(f['y'])}.")
            shown = True
        elif st.status == "done" and node_type == "check_limits" and rep.get("verdict"):
            ok = rep["verdict"] == "PASS"
            color = T.ok if ok else T.danger
            self._heading("Result", color)
            self._big(rep["verdict"], size=15, color=color)
            n, bad = int(rep.get("rows") or 0), int(rep.get("outside") or 0)
            self._stats([(f"{bad:,}", "outside"), (f"{rep.get('outside_percent', 0):.2f}%", "of rows"), (f"{n:,}", "checked")])
            self._text(f"Limit: {rep.get('limit', '')}." + (" Every row is within it." if ok else (f" About 1 in {round(n / bad)} rows is outside." if bad and n / bad >= 2 else "")))
            shown = True
        elif st.status == "done" and node_type == "find_gaps":
            self._heading("Result")
            n = int(st.rows or 0)
            self._big(f"{n:,} gap{'s' if n != 1 else ''} found", size=13)
            for m in msgs:
                self._text(m)
            shown = True
        elif st.status == "done" and node_type == "predict" and rep.get("equation"):
            self._heading("Result")
            self._text("Predicted with:")
            self._big(html.escape(rep["equation"]).replace("·", " × "))
            for m in msgs[1:]:
                self._text(m, muted=False)
            shown = True
        elif msgs:
            self._heading("Notes" if st.status != "failed" else "Problem")
            for m in msgs:
                self._text(m)
            shown = True
        if not shown:
            for k, v in rep.items():
                if isinstance(v, (list, dict)):
                    continue
                self._text(f"{k.replace('_', ' ')}: {_fmt(v)}")
                shown = True
            if shown:
                self.lay.insertWidget(0, QLabel(""))
        self.setVisible(shown)
