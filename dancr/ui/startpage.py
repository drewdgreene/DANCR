"""Start screen: open data files, go back to a recent project, or open one of the example projects."""
from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton, QFrame, QScrollArea, QMenu

from ..core.samples import EXAMPLES

from .theme import T
from .icons import icon

MAX_RECENT = 8
COLUMN_WIDTH = 680


def _folder(path: Path) -> str:
    """The folder a project is in, with the home folder written as ~."""
    folder = path.parent
    try:
        return "~/" + str(folder.relative_to(Path.home())) if folder != Path.home() else "~"
    except ValueError:
        return str(folder)


def _ago(seconds: float) -> str:
    minutes = seconds / 60
    if minutes < 1:
        return "just now"
    if minutes < 60:
        return f"{int(minutes)} min ago"
    hours = minutes / 60
    if hours < 24:
        return f"{int(hours)} h ago"
    days = int(hours / 24)
    if days == 1:
        return "yesterday"
    if days < 30:
        return f"{days} days ago"
    return time.strftime("%d %b %Y", time.localtime(time.time() - seconds))


class RecentRow(QFrame):
    """One recent project: its name, its folder and when it was last changed. Click to open it; right-click to
    take it off the list."""
    clicked = Signal()
    remove = Signal()

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path
        self.setObjectName("recentRow")
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip(str(path))
        self.setStyleSheet(f"QFrame#recentRow {{ border-radius: 6px; }} QFrame#recentRow:hover {{ background: {T.hover}; }}")
        lay = QHBoxLayout(self); lay.setContentsMargins(10, 6, 10, 6); lay.setSpacing(10)
        ic = QLabel(); ic.setPixmap(icon("file", T.muted, 16).pixmap(16, 16))
        name = QLabel(path.stem); name.setStyleSheet("font-weight: 600;")
        self.folder = QLabel(); self.folder.setObjectName("muted")
        self.folder.setMinimumWidth(1)                 # a long folder is shortened, never widens the page
        self._folder_text = _folder(path)
        try:
            when = _ago(max(0.0, time.time() - path.stat().st_mtime))
        except OSError:
            when = ""
        changed = QLabel(when); changed.setObjectName("faint")
        lay.addWidget(ic); lay.addWidget(name); lay.addWidget(self.folder, 1); lay.addWidget(changed)

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self.folder.setText(self.folder.fontMetrics().elidedText(self._folder_text, Qt.ElideMiddle, max(1, self.folder.width())))

    def mouseReleaseEvent(self, e) -> None:
        if e.button() == Qt.LeftButton and self.rect().contains(e.position().toPoint()):
            self.clicked.emit()
        super().mouseReleaseEvent(e)

    def contextMenuEvent(self, e) -> None:
        m = QMenu(self)
        m.addAction("Remove from this list", self.remove.emit)
        m.exec(e.globalPos())


EXAMPLE_ICONS = {"shop": "trend-up", "loggers": "wave-sine", "budget": "columns", "batches": "check-circle"}


class ExampleCard(QFrame):
    """One example project: what it is and what it shows. Click to open it."""
    clicked = Signal()

    def __init__(self, title: str, blurb: str, icon_name: str) -> None:
        super().__init__()
        self.setObjectName("exampleCard")
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet(f"QFrame#exampleCard {{ background: {T.panel}; border: 1px solid {T.border}; border-radius: 8px; }}"
                           f"QFrame#exampleCard:hover {{ border-color: {T.accent}; }}")
        lay = QVBoxLayout(self); lay.setContentsMargins(14, 12, 14, 12); lay.setSpacing(4)
        top = QHBoxLayout(); top.setSpacing(8)
        ic = QLabel(); ic.setPixmap(icon(icon_name, T.accent, 18).pixmap(18, 18))
        t = QLabel(title); t.setStyleSheet("font-weight: 600;")
        top.addWidget(ic); top.addWidget(t); top.addStretch()
        b = QLabel(blurb); b.setObjectName("muted"); b.setWordWrap(True)
        lay.addLayout(top); lay.addWidget(b); lay.addStretch()

    def mouseReleaseEvent(self, e) -> None:
        if e.button() == Qt.LeftButton and self.rect().contains(e.position().toPoint()):
            self.clicked.emit()
        super().mouseReleaseEvent(e)


class StartPage(QWidget):
    openProject = Signal()
    openRecent = Signal(str)
    removeRecent = Signal(str)
    blank = Signal()
    openFiles = Signal()
    example = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet(f"StartPage {{ background: {T.bg}; }}")
        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer.addWidget(scroll)
        body = QWidget(); scroll.setWidget(body)
        row = QHBoxLayout(body); row.setContentsMargins(32, 48, 32, 40)
        col = QWidget(); col.setMaximumWidth(COLUMN_WIDTH)
        row.addStretch(); row.addWidget(col, 100); row.addStretch()
        lay = QVBoxLayout(col); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(14)

        title = QLabel("DANCR"); title.setStyleSheet("font-size: 20pt; font-weight: 600;")
        sub = QLabel("Open a CSV, Excel or Parquet file to start. You can drop several files on the window at once.")
        sub.setObjectName("muted"); sub.setWordWrap(True)
        lay.addWidget(title); lay.addWidget(sub)
        buttons = QHBoxLayout(); buttons.setSpacing(10)
        b_files = QPushButton("Open data files…"); b_files.setObjectName("primary"); b_files.setIcon(icon("folder-open", "#ffffff", 16))
        b_files.clicked.connect(self.openFiles.emit)
        b_project = QPushButton("Open a project…"); b_project.clicked.connect(self.openProject.emit)
        b_table = QPushButton("Type in a table"); b_table.clicked.connect(self.blank.emit)
        for b in (b_files, b_project, b_table):
            buttons.addWidget(b)
        buttons.addStretch()
        lay.addLayout(buttons)

        lay.addSpacing(18)
        self.recent_lab = QLabel("Recent projects"); self.recent_lab.setObjectName("section")
        self.recent_box = QVBoxLayout(); self.recent_box.setSpacing(0)
        lay.addWidget(self.recent_lab); lay.addLayout(self.recent_box)

        lay.addSpacing(18)
        ex_lab = QLabel("Examples"); ex_lab.setObjectName("section")
        ex_text = QLabel("Finished projects on sample data. Open one to see how the steps, answers and report fit together.")
        ex_text.setObjectName("muted"); ex_text.setWordWrap(True)
        lay.addWidget(ex_lab); lay.addWidget(ex_text)
        grid = QGridLayout(); grid.setSpacing(10)
        grid.setColumnStretch(0, 1); grid.setColumnStretch(1, 1)
        self.cards: dict[str, ExampleCard] = {}
        for i, e in enumerate(EXAMPLES):
            card = ExampleCard(e["title"], e["blurb"], EXAMPLE_ICONS.get(e["key"], "table"))
            card.clicked.connect(lambda k=e["key"]: self.example.emit(k))
            grid.addWidget(card, i // 2, i % 2)
            self.cards[e["key"]] = card
        lay.addLayout(grid)
        lay.addStretch()

    def set_recent(self, paths: list[str]) -> None:
        while self.recent_box.count():
            w = self.recent_box.takeAt(0).widget()
            if w:
                w.deleteLater()
        shown = [Path(p) for p in paths if Path(p).is_file()][:MAX_RECENT]
        self.recent_lab.setVisible(bool(shown))
        for p in shown:
            r = RecentRow(p)
            r.clicked.connect(lambda p=p: self.openRecent.emit(str(p)))
            r.remove.connect(lambda p=p: self.removeRecent.emit(str(p)))
            self.recent_box.addWidget(r)
