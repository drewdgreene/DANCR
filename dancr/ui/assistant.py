"""The Assistant in the window: a chat in the right dock, beside the deterministic ask bar.

A single vertical thread of plain cards, in the same visual language as the rest of DANCR: flat surfaces, a
hairline border, one accent used only for the primary action, and no decoration for its own sake. The reply
may carry a plan or a batch of project changes; the person approves it, and nothing here touches the project
itself. While the model works, a small card shows what it is doing.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable

from PySide6.QtCore import Qt, QEvent, QSettings, Signal, QTimer, QObject
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPlainTextEdit, QPushButton,
                               QToolButton, QFrame, QScrollArea, QStackedWidget, QLineEdit, QMenu,
                               QInputDialog, QMessageBox, QSizePolicy, QApplication)

from ..core.assistant import AssistantSession, Thread, load_thread
from ..core.assistant.client import ModelSettings, Provider, provider_for
from .common import listen
from .icons import icon
from .theme import T
from .workers import Task, alive, view_pool

log = logging.getLogger("dancr.ui")

SETTING_KEY = "assistant/api_key"
SETTING_BASE = "assistant/base_url"
SETTING_MODEL = "assistant/model"
SETTING_SAMPLES = "assistant/allow_samples"
SETTING_CONSENT = "assistant/consent"

RADIUS = 4

# DeepSeek V4.1 Flash on Fireworks, per million tokens. Only a rough session estimate; the token counts are exact.
PRICE_IN_PER_M = 0.22
PRICE_CACHED_PER_M = 0.007
PRICE_OUT_PER_M = 0.66

# What a tool call means, in plain words, for the working card.
TOOL_TEXT = {
    "list_tables": "Looking at the tables",
    "list_steps": "Reading the project's steps",
    "describe_table": "Reading the columns",
    "get_stats": "Running the numbers",
    "get_sample": "Reading a few rows",
    "read_question": "Reading the question",
    "suggest_answers": "Checking available answers",
    "list_connections": "Checking how the tables connect",
    "answer_reference": "Checking the answer reference",
    "list_node_types": "Checking the step catalogue",
    "formula_reference": "Checking the formula reference",
    "node_status": "Reading a step's result",
    "ask_choice": "Preparing a question",
    "propose": "Drafting the plan",
    "propose_edits": "Drafting the changes",
}


# =================================================================== small style helpers
def _card(border: str | None = None) -> QFrame:
    f = QFrame()
    f.setStyleSheet(f"QFrame {{ background:{T.panel}; border:1px solid {border or T.border_soft}; border-radius:{RADIUS}px; }}")
    f.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
    return f


def _plain(label: QLabel, color: str, size: int | None = None, bold: bool = False) -> QLabel:
    css = f"color:{color};border:none;background:transparent;"
    if size:
        css += f"font-size:{size}pt;"
    if bold:
        css += "font-weight:600;"
    label.setStyleSheet(css)
    label.setWordWrap(True)
    return label


def _primary(text: str) -> QPushButton:
    b = QPushButton(text); b.setObjectName("primary"); b.setCursor(Qt.PointingHandCursor)
    return b


def _quiet(text: str) -> QToolButton:
    b = QToolButton(); b.setText(text); b.setObjectName("quiet"); b.setCursor(Qt.PointingHandCursor)
    return b


def _copy_button(text: str) -> QToolButton:
    b = QToolButton(); b.setIcon(icon("copy", T.muted, 13)); b.setObjectName("quiet")
    b.setAutoRaise(True); b.setToolTip("Copy"); b.setCursor(Qt.PointingHandCursor)
    b.clicked.connect(lambda: QApplication.clipboard().setText(text or ""))
    return b


# =================================================================== the dock (Settings / Assistant)
class SideDock(QWidget):
    """The right column: node settings, or the Assistant, chosen by a small header."""

    tabChanged = Signal(bool)               # the person switched tabs here; the window keeps its action in step

    def __init__(self, inspector: QWidget, assistant: QWidget, parent=None) -> None:
        super().__init__(parent)
        self.inspector = inspector
        self.assistant = assistant
        head = QFrame(); head.setObjectName("sideTabs")
        head.setStyleSheet(f"QFrame#sideTabs {{ background: {T.bg}; border-bottom: 1px solid {T.border}; }}")
        h = QHBoxLayout(head); h.setContentsMargins(8, 3, 8, 3); h.setSpacing(2)
        self.settings_btn = QToolButton(); self.settings_btn.setText("Settings"); self.settings_btn.setCheckable(True)
        self.assistant_btn = QToolButton(); self.assistant_btn.setText("Assistant"); self.assistant_btn.setCheckable(True)
        for b in (self.settings_btn, self.assistant_btn):
            b.setObjectName("sideTab"); b.setCursor(Qt.PointingHandCursor)
            b.setStyleSheet(
                f"QToolButton#sideTab {{ border: none; padding: 4px 10px; color: {T.muted}; }}"
                f"QToolButton#sideTab:hover {{ color: {T.text}; }}"
                f"QToolButton#sideTab:checked {{ color: {T.text}; border-bottom: 2px solid {T.accent}; }}")
        self.settings_btn.clicked.connect(lambda: self.set_assistant(False))
        self.assistant_btn.clicked.connect(lambda: self.set_assistant(True))
        h.addWidget(self.settings_btn); h.addWidget(self.assistant_btn); h.addStretch(1)
        self.stack = QStackedWidget()
        self.stack.addWidget(inspector); self.stack.addWidget(assistant)
        v = QVBoxLayout(self); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(0)
        v.addWidget(head); v.addWidget(self.stack, 1)
        self.set_assistant(False)

    def set_assistant(self, on: bool) -> None:
        changed = self.stack.currentWidget() is not (self.assistant if on else self.inspector)
        self.stack.setCurrentWidget(self.assistant if on else self.inspector)
        self.assistant_btn.setChecked(on); self.settings_btn.setChecked(not on)
        if changed:
            self.tabChanged.emit(on)


# =================================================================== cards
class Chip(QFrame):
    """The table a question is about, removable."""

    def __init__(self, text: str, on_remove=None, tooltip: str = "", parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet(f"QFrame {{ border:1px solid {T.border}; border-radius:{RADIUS}px; background:transparent; }}")
        self.setToolTip(tooltip)
        h = QHBoxLayout(self); h.setContentsMargins(7, 0, 3, 0); h.setSpacing(2)
        h.addWidget(_plain(QLabel(text), T.muted, size=9))
        if on_remove:
            x = _quiet("✕"); x.setToolTip("Detach"); x.clicked.connect(lambda: on_remove())
            h.addWidget(x)


class _Row(QWidget):
    """One message in the thread, left (assistant) or right (you)."""

    def __init__(self, card: QWidget, right: bool = False, parent=None) -> None:
        super().__init__(parent)
        h = QHBoxLayout(self); h.setContentsMargins(0, 0, 0, 0); h.setSpacing(0)
        if right:
            h.addStretch(1); h.addWidget(card, 0)
        else:
            h.addWidget(card, 1)


class UserCard(QFrame):
    def __init__(self, text: str, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet(f"QFrame {{ background:{T.panel}; border:1px solid {T.border_soft}; border-radius:{RADIUS}px; }}")
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        v = QVBoxLayout(self); v.setContentsMargins(10, 6, 10, 6)
        lab = QLabel(text); lab.setWordWrap(True); lab.setTextInteractionFlags(Qt.TextSelectableByMouse)
        v.addWidget(lab)


class SuggestionRow(QFrame):
    """A clickable next question."""

    clicked = Signal()

    def __init__(self, text: str, parent=None) -> None:
        super().__init__(parent)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet(f"QFrame {{ border:none; border-radius:{RADIUS}px; background:transparent; }}"
                           f"QFrame:hover {{ background:{T.hover}; }}")
        h = QHBoxLayout(self); h.setContentsMargins(4, 2, 4, 2)
        lab = _plain(QLabel(text), T.muted)
        lab.setAttribute(Qt.WA_TransparentForMouseEvents)
        h.addWidget(lab, 1)

    def mouseReleaseEvent(self, e) -> None:  # noqa: N802
        if e.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mouseReleaseEvent(e)


class AssistantCard(QFrame):
    """A reply: the model's words, with a note when it named a number the engine never produced."""

    def __init__(self, text: str, flags: list[str] | None = None, next_questions: list[str] | None = None,
                 on_question=None, unverified: list[str] | None = None, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet(f"QFrame {{ background:{T.panel}; border:1px solid {T.border_soft}; border-radius:{RADIUS}px; }}")
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        v = QVBoxLayout(self); v.setContentsMargins(10, 7, 10, 8); v.setSpacing(6)
        head = QHBoxLayout(); head.setSpacing(4)
        head.addWidget(_plain(QLabel("Assistant"), T.muted, size=9))
        head.addStretch(1); head.addWidget(_copy_button(text))
        v.addLayout(head)
        body = _plain(QLabel(text or "(no words)"), T.text)
        body.setTextInteractionFlags(Qt.TextSelectableByMouse)
        v.addWidget(body)
        flags = flags or []
        if "unverified-figure" in flags:
            tokens = ", ".join(str(t) for t in (unverified or [])[:8])
            msg = f"Not backed by a run: {tokens}" if tokens else "A figure here was not in a run."
            v.addWidget(_plain(QLabel(msg), T.warn))
        if "truncated" in flags:
            v.addWidget(_plain(QLabel("The reply was cut off. Ask me to carry on."), T.warn))
        if next_questions:
            v.addWidget(_plain(QLabel("You could ask"), T.faint, size=9))
            for q in next_questions[:3]:
                row = SuggestionRow(q)
                if on_question:
                    row.clicked.connect(lambda t=q: on_question(t))
                v.addWidget(row)


class ProposalCard(QFrame):
    """A plan: what the engine would build. Nothing runs until the person says so."""

    def __init__(self, proposal: dict[str, Any], on_build, on_discard, parent=None) -> None:
        super().__init__(parent)
        self.proposal = proposal
        self.setStyleSheet(f"QFrame {{ background:{T.panel}; border:1px solid {T.border}; border-radius:{RADIUS}px; }}")
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        v = QVBoxLayout(self); v.setContentsMargins(10, 8, 10, 9); v.setSpacing(7)
        title = proposal.get("title") or ("Steps to build" if proposal.get("kind") == "steps" else "Plan")
        v.addWidget(_plain(QLabel(title), T.text, bold=True))
        names = [str(s.get("title") or s.get("type")) for s in (proposal.get("steps") or [])]
        if names and proposal.get("title") and names[-1] == proposal["title"]:
            names = names[:-1]
        if names:
            self.steps_label = _plain(QLabel(" → ".join(names)), T.muted)
            v.addWidget(self.steps_label)
        if proposal.get("why"):
            v.addWidget(_plain(QLabel(str(proposal["why"])), T.muted))
        assumptions = [str(a) for a in (proposal.get("assumptions") or []) if str(a).strip()]
        if assumptions:
            holder = QWidget(); hl = QVBoxLayout(holder); hl.setContentsMargins(0, 0, 0, 0); hl.setSpacing(2)
            for a in assumptions[:6]:
                hl.addWidget(_plain(QLabel("•  " + a), T.muted, size=9))
            holder.setVisible(False)
            more = _quiet(f"{len(assumptions)} assumption{'s' if len(assumptions) != 1 else ''}")
            more.setCheckable(True)
            more.toggled.connect(holder.setVisible)
            v.addWidget(more); v.addWidget(holder)
        row = QHBoxLayout(); row.setSpacing(6)
        self.build_btn = _primary("Build & run"); self.build_btn.clicked.connect(lambda: on_build(self))
        self.discard_btn = _quiet("Discard"); self.discard_btn.clicked.connect(lambda: on_discard(self))
        row.addWidget(self.build_btn); row.addWidget(self.discard_btn); row.addStretch(1)
        v.addLayout(row)

    def built(self, finding: str = "", on_reveal=None, on_save=None, on_replace=None) -> None:
        self.build_btn.setVisible(False); self.discard_btn.setVisible(False)
        self._show_finding(finding)
        row = QHBoxLayout(); row.setSpacing(2)
        for text, tip, cb in (("Canvas", "Show these steps on the canvas", on_reveal),
                              ("Save…", "Save these steps as a new project", on_save),
                              ("Replace", "Replace the canvas with these steps", on_replace)):
            if cb is None:
                continue
            b = _quiet(text); b.setToolTip(tip); b.clicked.connect(lambda _=False, f=cb: f())
            row.addWidget(b)
        row.addStretch(1)
        self.layout().addLayout(row)

    def mark_built(self, finding: str = "") -> None:
        self.build_btn.setVisible(False); self.discard_btn.setVisible(False)
        self._show_finding(finding)

    def _show_finding(self, finding: str) -> None:
        if not finding:
            return
        box = QWidget(); v = QVBoxLayout(box); v.setContentsMargins(0, 2, 0, 2); v.setSpacing(1)
        v.addWidget(_plain(QLabel("Result"), T.faint, size=9))
        body = _plain(QLabel(finding), T.text); body.setTextInteractionFlags(Qt.TextSelectableByMouse)
        v.addWidget(body)
        self.layout().addWidget(box)


class EditsCard(QFrame):
    """Changes to the project itself (rename steps, label columns, set an Input). One click applies them."""

    def __init__(self, proposal: dict[str, Any], on_apply, on_discard, parent=None) -> None:
        super().__init__(parent)
        self.proposal = proposal
        self.setStyleSheet(f"QFrame {{ background:{T.panel}; border:1px solid {T.border}; border-radius:{RADIUS}px; }}")
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        v = QVBoxLayout(self); v.setContentsMargins(10, 8, 10, 9); v.setSpacing(7)
        v.addWidget(_plain(QLabel(proposal.get("title") or "Changes to the project"), T.text, bold=True))
        if proposal.get("reply"):
            v.addWidget(_plain(QLabel(str(proposal["reply"])), T.muted))
        for e in (proposal.get("edits") or [])[:20]:
            v.addWidget(_plain(QLabel("•  " + str(e.get("summary") or e.get("op"))), T.text, size=9))
        row = QHBoxLayout(); row.setSpacing(6)
        self.apply_btn = _primary("Apply"); self.apply_btn.clicked.connect(lambda: on_apply(self))
        self.discard_btn = _quiet("Discard"); self.discard_btn.clicked.connect(lambda: on_discard(self))
        row.addWidget(self.apply_btn); row.addWidget(self.discard_btn); row.addStretch(1)
        v.addLayout(row)

    def applied(self) -> None:
        self.apply_btn.setVisible(False); self.discard_btn.setVisible(False)
        self.layout().addWidget(_plain(QLabel("Applied. Ctrl+Z undoes it."), T.faint, size=9))


class ChoiceRow(QFrame):
    clicked = Signal()

    def __init__(self, text: str, parent=None) -> None:
        super().__init__(parent)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet(f"QFrame {{ border:1px solid {T.border}; border-radius:{RADIUS}px; background:transparent; }}"
                           f"QFrame:hover {{ border-color:{T.accent}; background:{T.hover}; }}")
        h = QHBoxLayout(self); h.setContentsMargins(9, 5, 9, 5)
        lab = _plain(QLabel(text), T.text)
        lab.setAttribute(Qt.WA_TransparentForMouseEvents)
        h.addWidget(lab, 1)

    def mouseReleaseEvent(self, e) -> None:  # noqa: N802
        if e.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mouseReleaseEvent(e)


class ChoiceCard(QFrame):
    def __init__(self, proposal: dict[str, Any], on_choice, parent=None) -> None:
        super().__init__(parent)
        self.proposal = proposal
        self.setStyleSheet(f"QFrame {{ background:{T.panel}; border:1px solid {T.border}; border-radius:{RADIUS}px; }}")
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        v = QVBoxLayout(self); v.setContentsMargins(10, 8, 10, 9); v.setSpacing(7)
        v.addWidget(_plain(QLabel(proposal.get("question") or "Which is right?"), T.text, bold=True))
        for opt in proposal.get("options") or []:
            row = ChoiceRow(str(opt))
            row.clicked.connect(lambda t=str(opt): on_choice(t))
            v.addWidget(row)


def _action_card(title: str, text: str, actions: list[tuple[str, Callable[[], None]]] | None = None,
                 tone: str = "muted") -> QFrame:
    """A state that needs the person (no key, a failure), with the action right there."""
    card = _card(T.border)
    v = QVBoxLayout(card); v.setContentsMargins(10, 8, 10, 9); v.setSpacing(7)
    if title:
        color = T.danger if tone == "danger" else T.warn if tone == "warn" else T.text
        v.addWidget(_plain(QLabel(title), color, bold=True))
    if text:
        v.addWidget(_plain(QLabel(text), T.muted))
    if actions:
        row = QHBoxLayout(); row.setSpacing(6)
        for i, (label, fn) in enumerate(actions):
            b = _primary(label) if i == 0 else _quiet(label)
            b.clicked.connect(lambda _=False, f=fn: f())
            row.addWidget(b)
        row.addStretch(1)
        v.addLayout(row)
    return card


class NoteCard(QFrame):
    def __init__(self, text: str, error: bool = False, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet(f"QFrame {{ background:transparent; border:1px solid {T.border_soft}; border-radius:{RADIUS}px; }}")
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        v = QVBoxLayout(self); v.setContentsMargins(10, 6, 10, 6)
        v.addWidget(_plain(QLabel(text), T.danger if error else T.muted))


class WorkingCard(QFrame):
    """While the model works: a line of what it is doing, so the panel is never a blank pause."""

    FRAMES = ("·", "··", "···")

    def __init__(self, status: str = "Thinking", parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet(f"QFrame {{ background:{T.panel}; border:1px solid {T.border_soft}; border-radius:{RADIUS}px; }}")
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        v = QVBoxLayout(self); v.setContentsMargins(10, 7, 10, 8); v.setSpacing(4)
        top = QHBoxLayout(); top.setSpacing(6)
        self.dots = _plain(QLabel("·"), T.muted)
        self.status = _plain(QLabel(f"{status}…"), T.text)
        self.elapsed = _plain(QLabel(""), T.faint, size=9)
        top.addWidget(self.dots, 0, Qt.AlignTop); top.addWidget(self.status, 1); top.addWidget(self.elapsed, 0, Qt.AlignTop)
        v.addLayout(top)
        self.log = _plain(QLabel(""), T.muted, size=9); self.log.setVisible(False)
        v.addWidget(self.log)
        self._steps: list[str] = []
        self._frame = 0
        self._t0 = time.monotonic()
        self._timer = QTimer(self); self._timer.setInterval(420); self._timer.timeout.connect(self._tick)
        self._timer.start()

    def _tick(self) -> None:
        self._frame = (self._frame + 1) % len(self.FRAMES)
        self.dots.setText(self.FRAMES[self._frame])
        secs = int(time.monotonic() - self._t0)
        self.elapsed.setText(f"{secs}s" if secs >= 2 else "")

    def set_status(self, text: str) -> None:
        self.status.setText(text if text.endswith("…") else f"{text}…")

    def add_tool(self, phrase: str) -> None:
        if self._steps and self._steps[-1] == phrase:
            return
        self._steps.append(phrase)
        self.log.setText("\n".join(self._steps[-5:]))
        self.log.setVisible(True)

    def stop(self) -> None:
        self._timer.stop()


class EmptyState(QFrame):
    def __init__(self, on_ask, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet("QFrame { background:transparent; border:none; }")
        v = QVBoxLayout(self); v.setContentsMargins(14, 14, 14, 10); v.setSpacing(7)
        v.addStretch(1)
        v.addWidget(_plain(QLabel("Ask about this project"), T.text, bold=True))
        v.addWidget(_plain(QLabel("Type a question below, or pick a starting point."), T.muted))
        row = QHBoxLayout(); row.setSpacing(6)
        for label, q in (("Profile the tables", "Profile these tables"),
                         ("What can I ask?", "What can I ask?")):
            b = _quiet(label); b.clicked.connect(lambda _=False, text=q: on_ask(text))
            row.addWidget(b)
        row.addStretch(1)
        v.addLayout(row)
        v.addWidget(_plain(QLabel("Commands:  /profile  /connections  /explain  /clean  /report"), T.faint, size=9))
        v.addWidget(_plain(QLabel("First time? Add a model key from the ⋮ menu above."), T.faint, size=9))
        v.addStretch(2)


# =================================================================== the panel
class _Progress(QObject):
    """Carries a worker thread's progress onto the GUI thread (a queued signal)."""
    event = Signal(object)


class AssistantPanel(QFrame):
    """The Assistant chat: a wrapping thread, a working card, context chips, and a composer."""

    buildRequested = Signal(dict)
    applyEditsRequested = Signal(list)
    revealRequested = Signal(str)
    saveProjectRequested = Signal(str)
    replaceCanvasRequested = Signal(str)
    proposalPreview = Signal(dict)      # a proposed plan, to ghost onto the canvas
    proposalCleared = Signal()          # the proposal was dismissed or the project changed

    def __init__(self, doc, understanding, parent=None) -> None:
        super().__init__(parent)
        self.doc = doc
        self.understanding = understanding
        self.settings = QSettings()
        self._thread: Thread = load_thread(doc.pipeline)
        self._provider: Provider | None = None
        self._task: Task | None = None
        self._session: AssistantSession | None = None
        self._pending_card: ProposalCard | None = None
        self._pending_edits_card: EditsCard | None = None
        self._working: WorkingCard | None = None
        self._progress = _Progress(); self._progress.event.connect(self._on_progress)
        self._busy = False
        self._focus: str | None = None
        self._announced: set[str] = set()
        self._right_cards: list[QWidget] = []
        self._stick = True
        self._history: list[str] = []
        self._hist = -1
        self._last_text = ""
        self._build_ui()
        listen(self, doc.reloaded, self._project_replaced)
        listen(self, doc.nodeAdded, self._on_node_added)
        listen(self, doc.undo.indexChanged, self._maybe_thread_changed)
        self._reload_cards()

    # ---------------------------------------------------------------- ui
    def _build_ui(self) -> None:
        v = QVBoxLayout(self); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(0)
        head = QFrame(); head.setStyleSheet(f"QFrame {{ background: {T.bg}; border-bottom: 1px solid {T.border}; }}")
        h = QHBoxLayout(head); h.setContentsMargins(12, 6, 8, 6); h.setSpacing(7)
        self.dot = QLabel("○"); self.dot.setToolTip("No model key set")
        self.title = QLabel("Assistant"); self.title.setStyleSheet("font-weight: 600;")
        self.status_line = _plain(QLabel("Ready"), T.faint, size=9)
        self.menu_btn = _quiet("⋮"); self.menu_btn.setToolTip("Assistant settings and key")
        self.menu_btn.clicked.connect(self._show_menu)
        h.addWidget(self.dot); h.addWidget(self.title); h.addWidget(self.status_line, 1); h.addWidget(self.menu_btn)
        v.addWidget(head)

        self.chip_bar = QWidget(); self.chip_bar.setStyleSheet("background:transparent;")
        self.chip_layout = QHBoxLayout(self.chip_bar); self.chip_layout.setContentsMargins(12, 5, 12, 3); self.chip_layout.setSpacing(6)
        v.addWidget(self.chip_bar)
        self._rebuild_chips()

        self.scroll = QScrollArea(); self.scroll.setWidgetResizable(True); self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.body = QWidget(); self.body.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        self.thread_layout = QVBoxLayout(self.body)
        self.thread_layout.setContentsMargins(12, 10, 12, 10); self.thread_layout.setSpacing(8)
        self.thread_layout.addStretch(1)
        self.scroll.setWidget(self.body)
        sb = self.scroll.verticalScrollBar()
        sb.rangeChanged.connect(lambda *_: self._scroll_bottom() if self._stick else None)
        sb.valueChanged.connect(self._on_scroll_moved)
        v.addWidget(self.scroll, 1)

        self.empty = EmptyState(self._ask_again)
        v.addWidget(self.empty, 1)

        self.cost = _plain(QLabel(""), T.faint, size=9)
        self.cost.setContentsMargins(12, 0, 12, 3)
        v.addWidget(self.cost)

        comp = QFrame(); comp.setStyleSheet(f"QFrame {{ border-top: 1px solid {T.border}; }}")
        cl = QHBoxLayout(comp); cl.setContentsMargins(10, 7, 10, 7); cl.setSpacing(7)
        self.edit = QPlainTextEdit(); self.edit.setPlaceholderText("Ask, or tell me to build…")
        self.edit.setFixedHeight(54); self.edit.setTabChangesFocus(True)
        self.edit.installEventFilter(self); self.edit.textChanged.connect(self._grow)
        self.send_btn = _primary("Send"); self.send_btn.setMinimumWidth(62)
        self.send_btn.clicked.connect(self._send_or_stop)
        cl.addWidget(self.edit, 1); cl.addWidget(self.send_btn, 0, Qt.AlignBottom)
        v.addWidget(comp)
        self._refresh_dot()

    def _grow(self) -> None:
        lines = self.edit.toPlainText().count("\n") + 1
        self.edit.setFixedHeight(max(54, min(54 + (lines - 1) * 17, 150)))

    # ---------------------------------------------------------------- settings / key
    def _settings_obj(self) -> ModelSettings:
        s = ModelSettings.from_env()
        key = str(self.settings.value(SETTING_KEY, "") or "")
        base = str(self.settings.value(SETTING_BASE, "") or "")
        model = str(self.settings.value(SETTING_MODEL, "") or "")
        if key:
            s.api_key = key
        if base:
            s.base_url = base
        if model:
            s.model = model
        return s

    def allow_samples(self) -> bool:
        return bool(self.settings.value(SETTING_SAMPLES, False, type=bool))

    def set_provider(self, provider: Provider | None) -> None:
        """Rig the provider (tests, or the window with DANCR_ASSISTANT_FAKE)."""
        self._provider = provider
        self._refresh_dot()

    def _configured(self) -> bool:
        return bool(self._provider) or self._settings_obj().configured

    def _refresh_dot(self) -> None:
        ok = self._configured()
        self.dot.setText("●" if ok else "○")
        self.dot.setToolTip("Connected" if ok else "No model key set — open the menu to add one")
        if not self._busy:
            self.status_line.setText("Ready" if ok else "No key")

    def _show_menu(self) -> None:
        m = QMenu(self)
        s = self._settings_obj()
        m.addAction(f"Model: {s.model}" if s.model else "Model").setEnabled(False)
        m.addAction("Set API key…", self._set_key)
        m.addAction("Set endpoint URL…", self._set_base)
        m.addAction("Set model id…", self._set_model)
        m.addSeparator()
        samples = m.addAction("Send sample rows to the model")
        samples.setCheckable(True); samples.setChecked(self.allow_samples())
        samples.toggled.connect(lambda on: self.settings.setValue(SETTING_SAMPLES, bool(on)))
        consent = m.addAction("Allow sending data to the model")
        consent.setCheckable(True); consent.setChecked(self.allow_egress())
        consent.toggled.connect(self.set_allow_egress)
        m.addSeparator()
        m.addAction("Show the connection map", self.show_connections)
        m.addAction("Clear this conversation", self.clear)
        m.exec(self.menu_btn.mapToGlobal(self.menu_btn.rect().bottomLeft()))

    def _set_key(self) -> None:
        s = self._settings_obj()
        text, ok = QInputDialog.getText(self, "Model API key", "API key (stored on this machine, never in the project):",
                                        QLineEdit.Password, s.api_key)
        if ok:
            self.settings.setValue(SETTING_KEY, text.strip()); self._refresh_dot()

    def _set_base(self) -> None:
        s = self._settings_obj()
        text, ok = QInputDialog.getText(self, "Endpoint URL", "OpenAI-style endpoint (…/v1):", text=s.base_url)
        if ok:
            self.settings.setValue(SETTING_BASE, text.strip()); self._refresh_dot()

    def _set_model(self) -> None:
        s = self._settings_obj()
        text, ok = QInputDialog.getText(self, "Model id", "Model id:", text=s.model)
        if ok:
            self.settings.setValue(SETTING_MODEL, text.strip()); self._refresh_dot()

    # ---------------------------------------------------------------- context chips
    def _rebuild_chips(self) -> None:
        while self.chip_layout.count():
            item = self.chip_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()          # not just detached: a chip holds signal closures, so let Qt free it
        if self._focus and self._focus in self.doc.pipeline.nodes:
            node = self.doc.pipeline.nodes[self._focus]
            self.chip_layout.addWidget(Chip(node.title, on_remove=lambda: self.set_focus(None),
                                            tooltip="Questions are answered about this table first"))
        add = _quiet("Add context")
        add.setToolTip("Choose the table the question is about")
        add.clicked.connect(self._pick_context)
        self.chip_layout.addWidget(add)
        self.chip_layout.addStretch(1)

    def _pick_context(self) -> None:
        m = QMenu(self)
        m.addAction("No table (ask about all)", lambda: self.set_focus(None))
        from ..core.understand import default_tables
        for nid in default_tables(self.doc.pipeline):
            m.addAction(self.doc.pipeline.nodes[nid].title, lambda n=nid: self.set_focus(n))
        m.exec(self.chip_bar.mapToGlobal(self.chip_bar.rect().bottomLeft()))

    def set_focus(self, nid: str | None) -> None:
        self._focus = nid if (nid and nid in self.doc.pipeline.nodes) else None
        self._rebuild_chips()

    # ---------------------------------------------------------------- thread rendering
    def clear(self) -> None:
        self.doc.set_thread(None)
        self._thread = Thread(model=self._settings_obj().model)
        self._reload_cards()

    def allow_egress(self) -> bool:
        """Consent is per endpoint: changing the model server asks again rather than silently sending there."""
        return bool(self.settings.value(self._consent_key(), False, type=bool)) or \
            bool(self.settings.value(SETTING_CONSENT, False, type=bool))

    def set_allow_egress(self, on: bool) -> None:
        self.settings.setValue(self._consent_key(), bool(on))

    def _consent_key(self) -> str:
        return f"{SETTING_CONSENT}/{self._settings_obj().base_url.strip()}"

    def _ensure_consent(self) -> bool:
        if self._provider is not None or self.allow_egress():
            return True
        if not self._settings_obj().configured:
            return True
        s = self._settings_obj()
        box = QMessageBox(self)
        box.setWindowTitle("Send project data to the model?")
        box.setText("The Assistant sends a profile of your project — column names, types, ranges, counts and "
                    "category values — and your questions. It does not send your rows unless you allow sample "
                    "rows.\n\nThis is the only DANCR feature that uses the internet.")
        box.setInformativeText(f"Endpoint: {s.base_url}\nModel: {s.model}")
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        box.setDefaultButton(QMessageBox.Yes)
        if box.exec() != QMessageBox.Yes:
            self._add_widget(_action_card("Not sent", "Allow it again from the ⋮ menu when you want to.",
                                          [("Open menu", self._show_menu)]))
            return False
        self.set_allow_egress(True)
        return True

    def _project_replaced(self) -> None:
        self._pending_card = None
        self._pending_edits_card = None
        self._finish_working()
        self._busy = False
        self._announced = set()
        self._focus = None
        self.proposalCleared.emit()
        self._reload_cards()

    def _on_node_added(self, nid: str) -> None:
        if not alive(self) or not self.isVisible():
            return
        node = self.doc.pipeline.nodes.get(nid)
        if node is None or nid in self._announced:
            return
        from ..core.registry import registry
        try:
            if registry.get(node.type).kind != "source":
                return
        except KeyError:
            return
        self._announced.add(nid)
        self._add_widget(AssistantCard(
            f"Table ready: {node.title}. Profile it, or see how it connects to the others?",
            None, ["Profile these tables", "How do these tables connect?"], self._ask_again))

    def _reload_cards(self) -> None:
        self._clear_thread()
        self._thread = load_thread(self.doc.pipeline)
        for turn in self._thread.turns[-20:]:
            self._add_turn(turn)
        empty = not self._thread.turns
        self.empty.setVisible(empty); self.scroll.setVisible(not empty)
        self._rebuild_chips()
        self._update_cost()
        self._set_busy(False, "")
        QTimer.singleShot(0, self._scroll_bottom)

    def _clear_thread(self) -> None:
        self._finish_working()             # stop the "Thinking" card's timer and drop it, not just detach it
        while self.thread_layout.count():
            item = self.thread_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()            # threads carry signal closures and child layouts: free them now
            else:
                self.thread_layout.removeItem(item)
        self.thread_layout.addStretch(1)
        self._right_cards = []
        self._pending_card = None
        self._pending_edits_card = None

    def _thread_key(self) -> str:
        return json.dumps(self._thread.to_dict(), sort_keys=True, default=str)

    def _maybe_thread_changed(self) -> None:
        if not alive(self):
            return
        other = load_thread(self.doc.pipeline)
        if json.dumps(other.to_dict(), sort_keys=True, default=str) != self._thread_key():
            self._thread = other
            self._reload_cards()

    def _update_cost(self) -> None:
        p = c = o = 0
        for t in self._thread.turns:
            u = t.usage or {}
            p += int(u.get("prompt_tokens") or 0)
            c += int(u.get("cached_tokens") or 0)
            o += int(u.get("completion_tokens") or 0)
        if not (p or o):
            self.cost.setText(""); return
        s = self._settings_obj()
        if "fireworks" in s.base_url.lower() or "deepseek" in s.model.lower():
            est = max(0, p - c) / 1e6 * PRICE_IN_PER_M + c / 1e6 * PRICE_CACHED_PER_M + o / 1e6 * PRICE_OUT_PER_M
            self.cost.setText(f"{p + o:,} tokens · about ${est:.4f}")
        else:
            self.cost.setText(f"{p + o:,} tokens · cost varies by provider")

    def _add_turn(self, turn) -> None:
        if turn.role == "user":
            self._add_user(turn.text)
            return
        if turn.kind in ("paused", "error"):
            self._add_widget(NoteCard(turn.text or "The model could not be reached", error=turn.kind == "error"))
            return
        self._add_widget(AssistantCard(turn.text, turn.flags, turn.next_questions, self._ask_again,
                                       unverified=turn.unverified))
        prop = turn.proposal
        if prop and prop.get("kind") in ("answer", "steps"):
            card = ProposalCard(prop, self._build, self._discard)
            if turn.answer or turn.node:
                card.mark_built(turn.finding)
            self._add_widget(card)
        elif prop and prop.get("kind") == "edits":
            self._add_widget(EditsCard(prop, self._apply_edits, self._discard_edits))
        elif prop and prop.get("kind") == "choice":
            self._add_widget(NoteCard("You were asked: " + str(prop.get("question") or "")))

    # ---------------------------------------------------------------- adding to the thread
    def _add_widget(self, w: QWidget, right: bool = False) -> None:
        if right:
            w.setMaximumWidth(self._bubble_width())
            self._right_cards.append(w)
            self.thread_layout.insertWidget(self.thread_layout.count() - 1, _Row(w, right=True))
        else:
            self.thread_layout.insertWidget(self.thread_layout.count() - 1, w)
        self.empty.setVisible(False); self.scroll.setVisible(True)
        if self._stick:
            QTimer.singleShot(0, self._scroll_bottom)
            QTimer.singleShot(60, self._scroll_bottom)

    def _add_user(self, text: str) -> None:
        self._add_widget(UserCard(text), right=True)

    def _remove_widget(self, w: QWidget) -> None:
        self.thread_layout.removeWidget(w); w.setParent(None); w.deleteLater()

    def _on_scroll_moved(self, value: int) -> None:
        if not alive(self):
            return
        sb = self.scroll.verticalScrollBar()
        self._stick = value >= sb.maximum() - 24

    def _scroll_bottom(self) -> None:
        if not alive(self):
            return
        sb = self.scroll.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _bubble_width(self) -> int:
        return max(180, int(self.scroll.viewport().width() * 0.82))

    def resizeEvent(self, e) -> None:  # noqa: N802
        super().resizeEvent(e)
        width = self._bubble_width()
        for card in self._right_cards:
            card.setMaximumWidth(width)

    # ---------------------------------------------------------------- progress
    def _on_progress(self, event: dict[str, Any]) -> None:
        if self._working is None or not alive(self._working):
            return
        if "tool" in event:
            self._working.add_tool(TOOL_TEXT.get(str(event["tool"]), str(event["tool"])))
        elif "status" in event:
            self._working.set_status(str(event["status"]))

    # ---------------------------------------------------------------- sending
    def keyPressEvent(self, e) -> None:  # noqa: N802
        if e.key() == Qt.Key_Escape and self._busy:
            self.stop(); return
        super().keyPressEvent(e)

    def eventFilter(self, obj, e):  # noqa: N802
        if obj is self.edit and e.type() == QEvent.Type.KeyPress:
            if e.key() == Qt.Key_Escape and self._busy:
                self.stop(); return True
            if e.key() in (Qt.Key_Return, Qt.Key_Enter) and not (e.modifiers() & Qt.ShiftModifier):
                self._send_or_stop(); return True
            if e.key() == Qt.Key_Up and (e.modifiers() & Qt.ControlModifier or not self.edit.toPlainText()):
                self._history_step(-1); return True
            if e.key() == Qt.Key_Down and (e.modifiers() & Qt.ControlModifier):
                self._history_step(1); return True
        return super().eventFilter(obj, e)

    def _history_step(self, delta: int) -> None:
        if not self._history or self._busy:
            return
        index = self._hist if self._hist >= 0 else len(self._history)
        index = max(0, min(len(self._history), index + delta))
        self._hist = index if index < len(self._history) else -1
        self.edit.setPlainText(self._history[index] if self._hist >= 0 else "")
        self.edit.moveCursor(QTextCursor.MoveOperation.End)

    def show_connections(self) -> None:
        conns = self._thread.connections
        if not conns:
            self._add_widget(_action_card("No connection map yet",
                                          "Ask “how do these tables connect?”, or type /connections.",
                                          [("Ask now", lambda: self._ask_again("How do these tables connect?"))]))
            return
        lines = []
        for c in conns:
            tables = " / ".join(str(t) for t in (c.get("tables") or []))
            bits = [str(c.get("kind") or "")]
            if c.get("left_on"):
                bits.append(f"{c['left_on']} = {c.get('right_on')}")
            if c.get("match_pct") is not None:
                bits.append(f"{c['match_pct']}% match")
            if c.get("cardinality"):
                bits.append(str(c["cardinality"]))
            lines.append(f"•  {tables}: " + ", ".join(b for b in bits if b))
        self._add_widget(AssistantCard("How the tables connect:\n" + "\n".join(lines)))

    def _ask_again(self, text: str) -> None:
        self.edit.setPlainText(text); self._send_or_stop()

    def _send_or_stop(self) -> None:
        if self._busy:
            self.stop()
        else:
            self.send()

    def stop(self) -> None:
        if self._task is not None:
            self._task.cancelled = True
            view_pool().take(self._task)
        if self._session is not None:
            try:
                self._session.provider.cancel()
            except Exception:  # noqa: BLE001
                pass
        self._finish_working()
        self._set_busy(False, "")

    COMMANDS = {
        "profile": "Profile every table in this project: what each one is, its columns and their roles, how many "
                   "rows, and how the tables connect. Then list the questions these tables can answer. Do not build yet.",
        "connections": "Look at how these tables relate. List every link with its match percentage and cardinality, "
                       "and every stack or time alignment. Where a pair has more than one possible link, ask me to "
                       "choose with ask_choice. Do not build yet.",
        "explain": "Explain the last result in one or two plain sentences, using only what a run produced.",
        "clean": "Check the data for problems (blanks, duplicates, values read wrong) and, only if something is wrong, "
                 "propose the cleanup.",
        "report": "Build a one-page report from the most important thing in this project.",
    }

    def _command(self, text: str) -> tuple[str, str | None]:
        if not text.startswith("/"):
            return text, None
        name, _, rest = text[1:].partition(" ")
        cmd = name.strip().lower()
        if cmd == "undo":
            if self.doc.undo.canUndo():
                label = self.doc.undo.undoText()
                self.doc.undo.undo()
                self._add_widget(NoteCard(f"Undid: {label}." if label else "Undid the last change."))
            else:
                self._add_widget(NoteCard("Nothing to undo."))
            return "", "done"
        if cmd == "build":
            return (rest.strip() or "What can you build from these tables?"), None
        if cmd in self.COMMANDS:
            return (f"{self.COMMANDS[cmd]}{(' ' + rest.strip()) if rest.strip() else ''}"), None
        self._add_widget(NoteCard(f"I don't know the /{cmd} command. Try /profile, /connections, /explain, /clean or /report."))
        return "", "done"

    def send(self) -> None:
        text, done = self._command(self.edit.toPlainText().strip())
        if done == "done":
            self.edit.clear(); return
        if not text or self._busy:
            return
        if not self._ensure_consent():
            return
        self._last_text = text
        self._history.append(text); self._hist = -1
        if len(self._history) > 100:                     # a long session must not grow the input history forever
            del self._history[:-100]
        self.edit.clear(); self._grow()
        self._add_user(text)
        self._set_busy(True, "Reading the tables first" if not self.understanding.full else "Thinking")
        if self.understanding.full and self.understanding.model is not None:
            self._start(text, self.understanding.model)
        else:
            self.understanding.when_full(lambda model: self._start(text, model))

    def _start(self, text: str, model) -> None:
        ex = self.doc.snapshot_executor()
        settings = self._settings_obj()
        provider = self._provider or provider_for(settings)
        self._session = AssistantSession(ex.pipeline, ex, model, provider, settings,
                                         allow_samples=self.allow_samples(), focus=self._focus,
                                         thread=self._thread, run=True)
        self._working = WorkingCard("Thinking")
        self._add_widget(self._working)
        task = Task(self._run_turn, text)
        task.waits_for_run = True
        # Capture the task and drop a reply that arrives after Stop or after a newer turn: a queued delivery can
        # outlive a cancel, and must not record the turn or add a proposal card.
        task.signals.done.connect(lambda reply, t=task: self._got_reply(text, reply) if t is self._task and not t.cancelled else None)
        task.signals.failed.connect(lambda msg, t=task: self._failed(msg) if t is self._task and not t.cancelled else None)
        task.signals.finished.connect(lambda t=task: self._set_busy(False, "") if t is self._task else None)
        self._task = task
        view_pool().start(task)

    def _run_turn(self, text: str):
        return self._session.turn(text, on_event=self._progress.event.emit)

    def _finish_working(self) -> None:
        if self._working is not None:
            self._working.stop()
            self._remove_widget(self._working)
            self._working = None

    def _failed(self, msg: str) -> None:
        if not alive(self):                  # the window was disposed (a theme switch) while the turn ran
            return
        self._finish_working()
        self._add_widget(_action_card("The Assistant could not finish", str(msg),
                                      [("Try again", lambda: self._ask_again(self._last_text))], tone="danger"))

    def _got_reply(self, text: str, reply) -> None:
        if self._session is None or not alive(self):
            return
        self._finish_working()
        self._session.record(text, reply)
        self.doc.set_thread(self._session.thread.to_dict())
        self._thread = self._session.thread
        self._update_cost()
        if reply.kind == "paused":
            self._add_widget(_action_card("Paused", reply.text or "The model is not available.",
                                          [("Add a key…", self._set_key), ("Dismiss", lambda: None)], tone="warn"))
            return
        if reply.kind == "error":
            self._add_widget(_action_card("The model could not be reached", reply.error or reply.text,
                                          [("Try again", lambda: self._ask_again(text))], tone="danger"))
            return
        if "budget" in reply.flags or "stuck" in reply.flags:
            self._add_widget(_action_card("Stopped early", reply.text,
                                          [("Carry on", lambda: self._ask_again("Please carry on and finish what you started.")),
                                           ("Try a different way", lambda: self._ask_again("Try a different approach."))], tone="warn"))
            return
        nxt = (reply.proposal or {}).get("next_questions") or []
        self._add_widget(AssistantCard(reply.text, reply.flags, nxt, self._ask_again, unverified=reply.unverified))
        prop = reply.proposal
        if prop and prop.get("kind") in ("answer", "steps"):
            card = ProposalCard(prop, self._build, self._discard)
            self._pending_card = card
            self._add_widget(card)
            self.proposalPreview.emit(prop)
        elif prop and prop.get("kind") == "edits":
            self._pending_edits_card = EditsCard(prop, self._apply_edits, self._discard_edits)
            self._add_widget(self._pending_edits_card)
        elif prop and prop.get("kind") == "choice":
            self._add_widget(ChoiceCard(prop, self._ask_again))

    def _build(self, card: ProposalCard) -> None:
        card.build_btn.setText("Building…"); card.build_btn.setEnabled(False)
        self._pending_card = card
        self.buildRequested.emit(card.proposal)

    def _discard(self, card: ProposalCard) -> None:
        card.setVisible(False)
        self.proposalCleared.emit()

    def _apply_edits(self, card: EditsCard) -> None:
        card.apply_btn.setText("Applying…"); card.apply_btn.setEnabled(False)
        self._pending_edits_card = card
        self.applyEditsRequested.emit(card.proposal.get("edits") or [])

    def _discard_edits(self, card: EditsCard) -> None:
        card.setVisible(False)

    def edits_applied(self, count: int = 0) -> None:
        card = self._pending_edits_card
        if card is not None:
            card.applied()
            self._pending_edits_card = None

    def built(self, finding: str = "", message: str = "", terminal: str | None = None, answer_id: str | None = None) -> None:
        card = self._pending_card
        if card is not None:
            card.built(finding,
                       on_reveal=(lambda t=terminal: self.revealRequested.emit(t)) if terminal else None,
                       on_save=(lambda t=terminal: self.saveProjectRequested.emit(t)) if terminal else None,
                       on_replace=(lambda t=terminal: self.replaceCanvasRequested.emit(t)) if terminal else None)
            self._pending_card = None
        self._note_built(finding=finding, terminal=terminal, answer_id=answer_id)
        if message:
            self._add_widget(NoteCard(message))

    def _note_built(self, finding: str = "", terminal: str | None = None, answer_id: str | None = None) -> None:
        """Fold the run's outcome into the turn that proposed it, so a reopened project shows the plan built —
        with the engine's Result — instead of offering to build it a second time."""
        if not alive(self) or not (finding or terminal or answer_id):
            return
        for turn in reversed(self._thread.turns):
            if turn.role == "assistant" and (turn.proposal or {}).get("kind") in ("answer", "steps"):
                if terminal:
                    turn.node = terminal
                if answer_id:
                    turn.answer = answer_id
                if finding:
                    turn.finding = finding
                self.doc.set_thread(self._thread.to_dict())
                return

    def _show_note(self, text: str, error: bool = False) -> None:
        self._add_widget(NoteCard(text, error=error))
        self._set_busy(False, "")

    def _set_busy(self, busy: bool, status: str) -> None:
        if not alive(self):                  # a queued result must not touch a widget Qt has deleted
            return
        self._busy = busy
        self.send_btn.setEnabled(True)
        self.send_btn.setText("Stop" if busy else "Send")
        if busy:
            self.status_line.setText(status or "Thinking")
        else:
            self.status_line.setText("Ready" if self._configured() else "No key")
