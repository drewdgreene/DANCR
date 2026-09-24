"""Small dialogs and overlays used by the main window."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QFrame, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QDialog, QListWidget, QListWidgetItem,
                               QDialogButtonBox)

from .document import Document
from .workers import Serial
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
        lay = QVBoxLayout(self)
        lab = QLabel("DANCR keeps a copy of the project every time it is saved. Pick one to go back to it (your current version is kept too)."); lab.setWordWrap(True)
        self.list = QListWidget()
        versions = doc.versions()
        for p in versions:
            m = re.match(r"(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})", p.name)
            when = f"{m.group(1)}  {m.group(2)}:{m.group(3)}:{m.group(4)}" if m else p.stem
            stamp = when + ("   ·   autosave" if p.name.endswith(".auto.json") else "")
            it = QListWidgetItem(icon("clock-counter-clockwise", T.muted, 16), stamp); it.setData(Qt.UserRole, str(p)); self.list.addItem(it)
        if versions:                                  # step counts read on a worker: the files may be on a slow drive
            self._counts = Serial(self, waits_for_run=False)
            self._counts.submit(lambda: [_step_count(p) for p in versions], self._show_counts)
        if not self.list.count():
            self.list.addItem("No earlier versions yet — they appear after you save.")
        bb = QDialogButtonBox(QDialogButtonBox.Cancel)
        self.restore = bb.addButton("Restore this version", QDialogButtonBox.AcceptRole)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        lay.addWidget(lab); lay.addWidget(self.list, 1); lay.addWidget(bb)

    def _show_counts(self, counts: list[int | None]) -> None:
        for i, n in enumerate(counts):
            it = self.list.item(i)
            if it is not None and n is not None:
                it.setText(f"{it.text()}   ·   {n} steps")

    def chosen(self) -> Path | None:
        it = self.list.currentItem()
        return Path(it.data(Qt.UserRole)) if it and it.data(Qt.UserRole) else None


def _step_count(path: Path) -> int | None:
    try:
        return len(json.loads(path.read_text(encoding="utf-8")).get("nodes") or [])
    except (OSError, ValueError, AttributeError):
        return None
