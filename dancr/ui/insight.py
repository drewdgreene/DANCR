"""The insight bar: the one plain sentence worth reading right now, pinned above the canvas.

It shows the finding of the selected step (or the best finding of the last run), and clicking it takes you to
the step that produced it. A quiet strip, not a banner: it appears only when there is something to say."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel

from .theme import T
from .icons import icon


class InsightBar(QFrame):
    jumped = Signal(str)              # the node id that produced the finding

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._node: str | None = None
        self.setObjectName("insightbar")
        self.setStyleSheet(f"QFrame#insightbar {{ background: {T.panel}; border-bottom: 1px solid {T.border}; }}")
        h = QHBoxLayout(self); h.setContentsMargins(12, 6, 12, 6); h.setSpacing(8)
        self.icon = QLabel(); self.icon.setPixmap(icon("lightbulb", T.accent, 16).pixmap(16, 16))
        self.label = QLabel(""); self.label.setWordWrap(False)
        f = self.label.font(); f.setPointSizeF(f.pointSizeF()); self.label.setFont(f)
        self.label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        h.addWidget(self.icon); h.addWidget(self.label, 1)
        self.hide()

    def set_insight(self, text: str, node: str | None = None, quiet: bool = False) -> None:
        """Show ``text``; ``node`` is where a click goes. Blank hides the strip. ``quiet`` is a muted tone for
        an informational line (no finding yet) rather than a discovered one."""
        self._node = node if node else None
        if not text:
            self.hide(); return
        self.label.setText(text)
        self.label.setStyleSheet(f"color: {T.muted if quiet else T.text};")
        self.icon.setPixmap(icon("lightbulb", T.muted if quiet else T.accent, 16).pixmap(16, 16))
        self.setCursor(Qt.PointingHandCursor if self._node else Qt.ArrowCursor)
        self.show()

    def clear(self) -> None:
        self._node = None
        self.hide()

    def mouseReleaseEvent(self, e) -> None:  # noqa: N802
        if e.button() == Qt.LeftButton and self._node:
            self.jumped.emit(self._node)
        super().mouseReleaseEvent(e)
