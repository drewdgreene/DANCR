"""Start screen: open a data file, pick up a recent project, or start from a template on sample data."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame, QGridLayout, QScrollArea

from ..core.samples import TEMPLATES
from .theme import T
from .icons import icon


class Card(QFrame):
    clicked = Signal()

    def __init__(self, title: str, blurb: str, icon_name: str) -> None:
        super().__init__()
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet(f"Card {{ background: {T.panel}; border: 1px solid {T.border}; border-radius: 8px; }} Card:hover {{ border-color: {T.accent}; }}")
        lay = QVBoxLayout(self); lay.setContentsMargins(14, 12, 14, 12); lay.setSpacing(4)
        top = QHBoxLayout(); ic = QLabel(); ic.setPixmap(icon(icon_name, T.accent, 20).pixmap(20, 20))
        t = QLabel(title); t.setStyleSheet("font-weight: 600;")
        top.addWidget(ic); top.addWidget(t); top.addStretch()
        b = QLabel(blurb); b.setObjectName("muted"); b.setWordWrap(True)
        lay.addLayout(top); lay.addWidget(b)
        self.setMinimumWidth(240)

    def mouseReleaseEvent(self, e) -> None:
        if e.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mouseReleaseEvent(e)


class StartPage(QWidget):
    openData = Signal()
    openProject = Signal()
    openRecent = Signal(str)
    template = Signal(str)
    blank = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet(f"StartPage {{ background: {T.bg}; }}")
        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QFrame.NoFrame); outer.addWidget(scroll)
        body = QWidget(); scroll.setWidget(body)
        lay = QVBoxLayout(body); lay.setContentsMargins(48, 40, 48, 40); lay.setSpacing(18)
        title = QLabel("Start with your data"); title.setStyleSheet("font-size: 20pt; font-weight: 600;")
        sub = QLabel("Open a CSV, Excel or Parquet file. You will see it as a table straight away; every change you make is a step you can undo, rerun and share.")
        sub.setObjectName("muted"); sub.setWordWrap(True)
        lay.addWidget(title); lay.addWidget(sub)
        row = QHBoxLayout(); row.setSpacing(10)
        b1 = QPushButton("Open a data file…"); b1.setObjectName("primary"); b1.setIcon(icon("folder-open", "#ffffff", 16)); b1.clicked.connect(self.openData.emit)
        b2 = QPushButton("Open a project…"); b2.clicked.connect(self.openProject.emit)
        b3 = QPushButton("Type in a table"); b3.clicked.connect(self.blank.emit)
        row.addWidget(b1); row.addWidget(b2); row.addWidget(b3); row.addStretch()
        lay.addLayout(row)
        self.recent_lab = QLabel("Recent"); self.recent_lab.setObjectName("section")
        self.recent_box = QVBoxLayout(); self.recent_box.setSpacing(2)
        lay.addWidget(self.recent_lab); lay.addLayout(self.recent_box)
        tl = QLabel("Try a template on sample data"); tl.setObjectName("section")
        th = QLabel("Each one builds a small project on a generated sample file (two values recorded every few seconds for a few days) so you can see how the pieces fit. Swap in your own file afterwards."); th.setObjectName("muted"); th.setWordWrap(True)
        lay.addWidget(tl); lay.addWidget(th)
        grid = QGridLayout(); grid.setSpacing(10)
        icons = {"compare": "chart-line", "limits": "check-circle", "fit": "chart-scatter", "report": "article"}
        for i, t in enumerate(TEMPLATES):
            c = Card(t["title"], t["blurb"], icons.get(t["key"], "table"))
            c.clicked.connect(lambda k=t["key"]: self.template.emit(k))
            grid.addWidget(c, i // 2, i % 2)
        lay.addLayout(grid)
        drop = QLabel("Tip: you can also drop a file anywhere on this window."); drop.setObjectName("faint")
        lay.addWidget(drop); lay.addStretch()

    def set_recent(self, paths: list[str]) -> None:
        while self.recent_box.count():
            w = self.recent_box.takeAt(0).widget()
            if w:
                w.deleteLater()
        shown = [p for p in paths if Path(p).exists()][:6]
        self.recent_lab.setVisible(bool(shown))
        for p in shown:
            b = QPushButton(f"{Path(p).stem}   —   {Path(p).parent}"); b.setObjectName("quiet"); b.setStyleSheet("text-align: left;")
            b.setIcon(icon("file", T.muted, 14)); b.clicked.connect(lambda _=False, p=p: self.openRecent.emit(p))
            self.recent_box.addWidget(b)
