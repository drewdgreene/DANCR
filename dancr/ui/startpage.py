"""Start screen: start something new, open a recent project, or open a worked example.

Three ways in (new, open, import), then templates and examples as cards, then the optional setup. Simple and
one click deep: no screen is a dead end."""
from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QFrame, QGridLayout, QHBoxLayout, QLabel, QMenu, QPushButton, QScrollArea, QVBoxLayout,
                               QWidget)

from ..core.samples import EXAMPLES, TEMPLATES

from .theme import T
from .icons import icon

MAX_RECENT = 8
COLUMN_WIDTH = 880

TEMPLATE_ICONS = {"compare": "chart-line", "limits": "check-circle", "fit": "chart-scatter", "report": "article"}
EXAMPLE_ICONS = {"trial": "grid-four", "shop": "trend-up", "budget": "columns", "batches": "check-circle"}


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


class Card(QFrame):
    """A clickable card: an icon, a title and one line of why. ``action`` cards lead with a larger icon."""
    clicked = Signal()

    def __init__(self, title: str, sub: str, icon_name: str, *, action: bool = False) -> None:
        super().__init__()
        self.setObjectName("card")
        self.setCursor(Qt.PointingHandCursor)
        pad = (18, 16, 18, 16) if action else (14, 12, 14, 12)
        icon_size = 26 if action else 18
        self.setStyleSheet(f"QFrame#card {{ background: {T.panel}; border: 1px solid {T.border}; border-radius: 8px; }}"
                           f"QFrame#card:hover {{ border-color: {T.accent}; }}")
        lay = QVBoxLayout(self); lay.setContentsMargins(*pad); lay.setSpacing(6)
        ic = QLabel(); ic.setPixmap(icon(icon_name, T.accent, icon_size).pixmap(icon_size, icon_size))
        t = QLabel(title); t.setStyleSheet(f"font-weight: 600;{' font-size: 12pt;' if action else ''}")
        s = QLabel(sub); s.setObjectName("muted"); s.setWordWrap(True)
        if action:
            lay.addWidget(ic); lay.addWidget(t); lay.addWidget(s); lay.addStretch()
        else:
            top = QHBoxLayout(); top.setSpacing(8)
            top.addWidget(ic); top.addWidget(t); top.addStretch()
            lay.addLayout(top); lay.addWidget(s); lay.addStretch()

    def mouseReleaseEvent(self, e) -> None:
        if e.button() == Qt.LeftButton and self.rect().contains(e.position().toPoint()):
            self.clicked.emit()
        super().mouseReleaseEvent(e)


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
        m.deleteLater()


class StartPage(QWidget):
    openProject = Signal()
    openRecent = Signal(str)
    removeRecent = Signal(str)
    blank = Signal()
    openFiles = Signal()
    example = Signal(str)
    template = Signal(str)
    formatsHelp = Signal()
    guide = Signal()
    assistantSetup = Signal()
    agentSetup = Signal()
    diagnostics = Signal()
    tour = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet(f"StartPage {{ background: {T.bg}; }}")
        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer.addWidget(scroll)
        body = QWidget(); scroll.setWidget(body)
        row = QHBoxLayout(body); row.setContentsMargins(32, 40, 32, 40)
        col = QWidget(); col.setMaximumWidth(COLUMN_WIDTH)
        row.addStretch(); row.addWidget(col, 100); row.addStretch()
        lay = QVBoxLayout(col); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(12)

        title = QLabel("DANCR"); title.setStyleSheet("font-size: 20pt; font-weight: 600;")
        sub = QLabel("Analyse spreadsheets without SQL. Build the answer in steps, then run the same steps again "
                     "on next month's files.")
        sub.setObjectName("muted"); sub.setWordWrap(True)
        lay.addWidget(title); lay.addWidget(sub)
        from .. import __version__
        local = QLabel(f"Local only, no account, no telemetry · version {__version__}")
        local.setObjectName("faint")
        lay.addWidget(local)

        # First-run banner: a nudge into the tour or an example, gone once the tour is seen.
        self.welcome = QFrame()
        self.welcome.setStyleSheet(f"QFrame {{ background: {T.panel}; border: 1px solid {T.accent}; border-radius: 8px; }}")
        wl = QHBoxLayout(self.welcome); wl.setContentsMargins(16, 12, 12, 12); wl.setSpacing(12)
        wtext = QVBoxLayout(); wtext.setSpacing(2)
        wt = QLabel("New here?"); wt.setStyleSheet("font-weight: 600;")
        ws = QLabel("Take a one-minute tour, or open a worked example and poke at it.")
        ws.setObjectName("muted"); ws.setWordWrap(True)
        wtext.addWidget(wt); wtext.addWidget(ws)
        wl.addLayout(wtext, 1)
        b_tour = QPushButton("Take the tour"); b_tour.setObjectName("primary"); b_tour.clicked.connect(self.tour.emit)
        b_ex = QPushButton("Open an example"); b_ex.clicked.connect(lambda: self.example.emit(EXAMPLES[0]["key"]))
        wl.addWidget(b_tour); wl.addWidget(b_ex)
        lay.addWidget(self.welcome)

        self.cards: dict[str, Card] = {}
        start = QHBoxLayout(); start.setSpacing(10)
        for ident, t, s, name, sig in (("new", "New project", "An empty canvas. Drop a file or type a table.", "plus", self.blank.emit),
                                       ("open", "Open project…", "A .json project you saved before.", "folder-open", self.openProject.emit),
                                       ("import", "Import data", "A file, a folder, a URL or a database.", "download-simple", self.openFiles.emit)):
            c = Card(t, s, name, action=True)
            c.clicked.connect(sig)
            self.cards[ident] = c
            start.addWidget(c, 1)
        lay.addLayout(start)

        self.recent_lab, self.recent_box = self._section(lay, "Recent")
        lay.addLayout(self.recent_box)

        self._cards_grid(lay, "Start from a template",
                         "A ready-made set of steps on sample data. Bring your own file later.",
                         [(t["key"], t["title"], t["blurb"], TEMPLATE_ICONS.get(t["key"], "table"), lambda k=t["key"]: self.template.emit(k))
                          for t in TEMPLATES])
        self._cards_grid(lay, "Examples",
                         "Finished projects on realistic sample data. Open one to see how the pieces fit.",
                         [(e["key"], e["title"], e["blurb"], EXAMPLE_ICONS.get(e["key"], "table"), lambda k=e["key"]: self.example.emit(k))
                          for e in EXAMPLES])

        self._cards_grid(lay, "Get set up",
                         "Optional. DANCR works without any of these.",
                         [("assistant", "Connect the Assistant", "Add a model key so the chat can propose steps.", "magic-wand", self.assistantSetup.emit),
                          ("agents", "Use a coding agent", "Wire DANCR into Claude, opencode or Cursor over MCP.", "list-bullets", self.agentSetup.emit),
                          ("diagnostics", "Check this build", "See which readers are present.", "check-circle", self.diagnostics.emit),
                          ("guide", "Read the guide", "Every step, setting and formula.", "question", self.guide.emit)])
        lay.addStretch()

        b_formats = QPushButton("What can DANCR read?")
        b_formats.setObjectName("quiet")
        b_formats.setToolTip("The file formats DANCR opens, and what to do with the ones it can't")
        b_formats.clicked.connect(self.formatsHelp.emit)
        frow = QHBoxLayout(); frow.addWidget(b_formats); frow.addStretch()
        lay.addLayout(frow)

    def _section(self, lay: QVBoxLayout, title: str) -> tuple[QLabel, QVBoxLayout]:
        lab = QLabel(title); lab.setObjectName("section")
        box = QVBoxLayout(); box.setSpacing(0)
        lay.addSpacing(10); lay.addWidget(lab)
        return lab, box

    def _cards_grid(self, lay: QVBoxLayout, title: str, blurb: str, cards: list[tuple]) -> None:
        lab = QLabel(title); lab.setObjectName("section")
        text = QLabel(blurb); text.setObjectName("muted"); text.setWordWrap(True)
        lay.addSpacing(10); lay.addWidget(lab); lay.addWidget(text)
        grid = QGridLayout(); grid.setSpacing(10)
        grid.setColumnStretch(0, 1); grid.setColumnStretch(1, 1)
        for i, (ident, t, s, ic, slot) in enumerate(cards):
            card = Card(t, s, ic)
            card.clicked.connect(slot)
            grid.addWidget(card, i // 2, i % 2)
            self.cards[ident] = card
        lay.addLayout(grid)

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

    def set_first_run(self, first_run: bool) -> None:
        self.welcome.setVisible(first_run)
