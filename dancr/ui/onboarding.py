"""A short first-run tour: what DANCR is for, how to read a file, and the two optional hookups.

Skippable, remembered, and reopened from Help. It is not setup: the window, the engine and the command line
all work without it. The optional slides only offer a button into the real dialog.
"""
from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QStackedWidget, QVBoxLayout, QWidget

from .theme import T
from .icons import icon

SETTING_SEEN = "onboarding/seen"


def _slide(name: str, title: str, body: str, bullets: list[str] | None = None) -> QWidget:
    w = QWidget()
    lay = QVBoxLayout(w); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(12)
    ic = QLabel(); ic.setPixmap(icon(name, T.accent, 36).pixmap(36, 36))
    t = QLabel(title); t.setStyleSheet("font-size: 16pt; font-weight: 600;")
    b = QLabel(body); b.setWordWrap(True)
    lay.addWidget(ic); lay.addWidget(t); lay.addWidget(b)
    if bullets:
        for line in bullets:
            row = QHBoxLayout(); row.setSpacing(8)
            dot = QLabel("•"); dot.setStyleSheet(f"color: {T.accent}; font-weight: 700;")
            lab = QLabel(line); lab.setObjectName("muted"); lab.setWordWrap(True)
            row.addWidget(dot, 0, Qt.AlignTop); row.addWidget(lab, 1)
            lay.addLayout(row)
    lay.addStretch()
    return w


class OnboardingDialog(QDialog):
    """The tour. ``on_assistant`` and ``on_agents`` open the setup dialogs from the last two slides."""

    def __init__(self, parent: QWidget | None = None, *, on_assistant: Callable[[], None] | None = None,
                 on_agents: Callable[[], None] | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle("Getting started")
        self.resize(620, 460)
        self._on_assistant, self._on_agents = on_assistant, on_agents
        self.setStyleSheet(f"QDialog {{ background: {T.panel}; }}")

        lay = QVBoxLayout(self); lay.setContentsMargins(28, 24, 28, 18); lay.setSpacing(16)
        self.stack = QStackedWidget()
        self.stack.addWidget(_slide("hand", "Welcome to DANCR",
            "DANCR is for people who have the data but not the time. Point it at your spreadsheets, build the "
            "answer in steps, and run the same steps again on next month's files.",
            ["No SQL and nothing hidden behind code.",
             "Your files, read the way they were made.",
             "Every number comes from a calculation you can see."]))
        self.stack.addWidget(_slide("file-csv", "Messy files are fine",
            "A title row, two tables on one sheet, a total row at the bottom: DANCR reads it as a table anyway. "
            "Drop a file, or a whole folder of them, to start.",
            ["CSV, Excel, Parquet, GeoJSON, shapefiles, NetCDF, HDF5, PDF and Office.",
             "It flags a duplicated ID, a gap, or a value outside its range."]))
        self.stack.addWidget(_slide("sparkle", "Ask, or build it by hand",
            "Type a question like “total sales by region” and DANCR builds the steps for you. Or add the steps "
            "yourself: pick columns, filter rows, join two tables, chart the result.",
            ["The words are read by rules, not a model, so the same question builds the same answer.",
             "The engine runs the steps and holds the numbers; a chart or a report comes from them."]))
        w4 = _slide("magic-wand", "The Assistant is optional",
            "A chat that proposes steps and asks when a choice matters. It needs a model key, and it can suggest "
            "but never compute: a figure no calculation produced is flagged, not trusted.")
        self._add_action(w4, "Add a key…", self._assistant)
        self.stack.addWidget(w4)
        w5 = _slide("list-bullets", "Use it from your coding agent",
            "One line wires DANCR into Claude, opencode or Cursor, so an agent can build and run pipelines, read "
            "results and export, with every call logged and a policy you control.",
            ["Nothing leaves this machine by default: no account and no telemetry.",
             "The only network features are the ones you turn on: a URL or database source, a document endpoint, "
             "and the Assistant."])
        self._add_action(w5, "Set it up…", self._agents)
        self.stack.addWidget(w5)
        lay.addWidget(self.stack, 1)

        foot = QHBoxLayout()
        skip = QPushButton("Skip"); skip.setObjectName("quiet"); skip.clicked.connect(self._done)
        self.pos = QLabel(""); self.pos.setObjectName("faint")
        self.back = QPushButton("Back"); self.back.clicked.connect(self._back)
        self.next = QPushButton("Next"); self.next.setObjectName("primary"); self.next.clicked.connect(self._next)
        foot.addWidget(skip); foot.addStretch(); foot.addWidget(self.pos); foot.addWidget(self.back); foot.addWidget(self.next)
        lay.addLayout(foot)
        self._sync()

    def _add_action(self, slide: QWidget, label: str, slot: Callable[[], None]) -> None:
        row = QHBoxLayout(); row.setContentsMargins(0, 0, 0, 0)
        b = QPushButton(label); b.clicked.connect(slot)
        row.addWidget(b); row.addStretch()
        slide.layout().insertLayout(slide.layout().count() - 1, row)

    def _assistant(self) -> None:
        if self._on_assistant:
            self._on_assistant()

    def _agents(self) -> None:
        if self._on_agents:
            self._on_agents()

    def _back(self) -> None:
        self.stack.setCurrentIndex(self.stack.currentIndex() - 1); self._sync()

    def _next(self) -> None:
        if self.stack.currentIndex() >= self.stack.count() - 1:
            self._done(); return
        self.stack.setCurrentIndex(self.stack.currentIndex() + 1); self._sync()

    def _sync(self) -> None:
        i, n = self.stack.currentIndex(), self.stack.count()
        self.pos.setText(f"{i + 1} / {n}")
        self.back.setEnabled(i > 0)
        self.next.setText("Done" if i == n - 1 else "Next")

    def _done(self) -> None:
        QSettings().setValue(SETTING_SEEN, True)
        self.accept()
