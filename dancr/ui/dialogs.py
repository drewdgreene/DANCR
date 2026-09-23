"""Small dialogs and overlays used by the main window."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QFrame, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QDialog, QListWidget, QListWidgetItem,
                               QDialogButtonBox)

from .document import Document
from .theme import T
from .icons import icon


class Toast(QFrame):
    """A small message at the bottom of the centre with an optional action (e.g. Undo)."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setStyleSheet(f"QFrame {{ background: {T.text}; border-radius: 6px; }} QLabel {{ color: {T.panel}; }} QPushButton {{ color: #93c5fd; background: transparent; border: none; font-weight: 600; padding: 2px 6px; }}")
        h = QHBoxLayout(self); h.setContentsMargins(14, 8, 10, 8); h.setSpacing(12)
        self.label = QLabel(""); self.button = QPushButton("Undo"); self.button.setCursor(Qt.PointingHandCursor)
        h.addWidget(self.label); h.addWidget(self.button)
        self._timer = QTimer(self); self._timer.setSingleShot(True); self._timer.timeout.connect(self.hide)
        self._slot: Callable[[], None] | None = None
        self.button.clicked.connect(self._fire)
        self.hide()

    def show_message(self, text: str, action: str | None = None, slot: Callable[[], None] | None = None, ms: int = 6000) -> None:
        self.label.setText(text); self.button.setVisible(bool(action)); self.button.setText(action or "")
        self._slot = slot
        self.adjustSize(); self.reposition(); self.show(); self.raise_(); self._timer.start(ms)

    def _fire(self) -> None:
        self.hide()
        if self._slot:
            self._slot()

    def reposition(self) -> None:
        p = self.parentWidget()
        if p:
            self.move((p.width() - self.width()) // 2, p.height() - self.height() - 18)


class VersionsDialog(QDialog):
    def __init__(self, parent, doc: Document) -> None:
        super().__init__(parent)
        self.setWindowTitle("Earlier versions"); self.resize(520, 380)
        self.doc = doc
        lay = QVBoxLayout(self)
        lab = QLabel("DANCR keeps a copy of the project every time you save. Pick one to go back to it (your current version is kept too)."); lab.setWordWrap(True)
        self.list = QListWidget()
        for p in doc.versions():
            stamp = p.stem[:10] + "  " + p.stem[11:].replace("-", ":")
            try:
                n = len(json.loads(p.read_text(encoding="utf-8")).get("nodes") or [])
            except Exception:  # noqa: BLE001
                n = 0
            it = QListWidgetItem(icon("clock-counter-clockwise", T.muted, 16), f"{stamp}   ·   {n} steps"); it.setData(Qt.UserRole, str(p)); self.list.addItem(it)
        if not self.list.count():
            self.list.addItem("No earlier versions yet — they appear after you save.")
        bb = QDialogButtonBox(QDialogButtonBox.Cancel)
        self.restore = bb.addButton("Restore this version", QDialogButtonBox.AcceptRole)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        lay.addWidget(lab); lay.addWidget(self.list, 1); lay.addWidget(bb)

    def chosen(self) -> Path | None:
        it = self.list.currentItem()
        return Path(it.data(Qt.UserRole)) if it and it.data(Qt.UserRole) else None
