"""Answers in the window: the ask bar with its tray of suggested answers, and the panel of an answer's chips.

- :class:`Understanding` keeps a data model of the project's tables up to date in the background (a quick
  one from samples first, then one that has read every row) and hands the full one to anything that builds.
- :class:`AskBar` sits above the page: type a question, or pick one of the answers DANCR offers for the
  tables (or for the step selected). Each card shows a small live preview.
- :class:`AnswerPanel` shows the selected answer's choices as chips and what it assumed, each changeable.

Nothing here decides what an answer is: that is ``core/recipes/`` and ``core/ask/``.
"""
from __future__ import annotations

import copy
import json
import logging
from typing import Any, Callable

from PySide6.QtCore import Qt, Signal, QTimer, QRectF, QPointF, QStringListModel
from PySide6.QtGui import QPainter, QColor, QPen, QPainterPath
from PySide6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QLabel, QLineEdit, QPushButton, QToolButton, QFrame,
                               QScrollArea, QMenu, QCompleter, QSizePolicy)

from ..core import Pipeline
from ..core.executor import Executor
from ..core.understand import DataModel, understand, deepen, default_tables
from ..core.recipes import suggest, plan as make_plan, chips as make_chips, Suggestion
from ..core.ask import ask, vocabulary
from .common import listen
from .icons import icon
from .theme import T
from .workers import Serial

log = logging.getLogger("dancr.ui")

RECIPE_ICONS = {"compare": "arrows-merge", "trend": "trend-up", "breakdown": "sigma", "top": "sort-ascending",
                "relationship": "chart-scatter", "gaps": "circle-dashed", "outliers": "warning-circle",
                "single": "function", "distribution": "wave-sine", "linked": "columns", "stacked": "rows",
                "rows": "funnel", "describe": "list-bullets", "groups": "columns", "quality": "warning-circle",
                "toprows": "sort-ascending", "change": "clock-counter-clockwise", "explain": "sigma", "drivers": "chart-scatter",
                "forecast": "trend-up"}
PREVIEW_ROWS = 20_000


# =================================================================== the data model, kept current
class Understanding:
    """The project's data model, recomputed off the GUI thread whenever the tables could have changed.

    ``model`` is the latest one (quick or full); ``when_full(fn)`` calls ``fn(model)`` with a model that has
    read every row of the current tables, as soon as there is one. Answers are only ever built from that."""

    def __init__(self, doc, owner: QWidget) -> None:
        self.doc = doc
        self.model: DataModel | None = None
        self.full = False
        self.focus: str | None = None
        self._key: str | None = None
        self._waiters: list[Callable[[DataModel], None]] = []
        self._listeners: list[Callable[[], None]] = []
        self._quick = Serial(owner, waits_for_run=False)
        self._deep = Serial(owner)
        self._timer = QTimer(owner); self._timer.setSingleShot(True); self._timer.setInterval(350)
        self._timer.timeout.connect(self.refresh)
        d = doc
        for sig in (d.nodeAdded, d.nodeRemoved, d.nodeChanged, d.edgeAdded, d.edgeRemoved, d.statesChanged, d.columnsChanged):
            listen(owner, sig, lambda *_: self._timer.start())
        listen(owner, d.reloaded, self._project_replaced)
        self._timer.start()

    def _project_replaced(self) -> None:
        """Another project is open: nothing asked about the old one may be built into this one."""
        self._waiters = []
        self.model, self.full, self._key = None, False, None
        self.focus = None
        self._timer.start()

    def on_change(self, fn: Callable[[], None]) -> None:
        self._listeners.append(fn)

    def set_focus(self, nid: str | None) -> None:
        if nid == self.focus:
            return
        self.focus = nid
        if self.model is not None and (nid is None or nid in self.model.tables):
            self._notify()                        # the model already covers it: only the suggestions change
        else:
            self.refresh()

    def _nodes(self) -> list[str]:
        p = self.doc.pipeline
        nodes = default_tables(p)
        if self.focus and self.focus in p.nodes and self.focus not in nodes:
            nodes.append(self.focus)
        return nodes

    def key(self) -> str:
        """What the model depends on: which tables, and what is in them (their settings, their inputs and the files
        they read — a step's plan hash). Not whether a step has run yet: that changes where rows are read from,
        not what they are, so a loader finishing its run never throws a finished model away."""
        p = self.doc.pipeline
        ex = self.doc.executor
        memo: dict[str, str] = {}
        parts = []
        for nid in self._nodes():
            n = p.nodes[nid]
            try:
                h = ex.plan_hash(nid, memo)
            except Exception:  # noqa: BLE001 - a step that cannot be hashed yet: its settings stand in
                h = json.dumps(n.params, sort_keys=True, default=str)
            parts.append([nid, n.title, h])
        return json.dumps([parts, p.columns], sort_keys=True, default=str)

    def refresh(self) -> None:
        key = self.key()
        if key == self._key and self.model is not None:
            return
        self._key, self.full = key, False
        if not self._nodes():
            self.model = DataModel(); self.full = True
            self._notify(); self._serve(); return
        quick_ex = self.doc.snapshot_executor()
        full_ex = Executor(quick_ex.pipeline, quick_ex.cache_dir)     # one copy, read by both; an executor each
        nodes = self._nodes()

        def quick() -> DataModel:
            return understand(quick_ex.pipeline, quick_ex, nodes)

        def full() -> DataModel:
            return deepen(full_ex.pipeline, full_ex, understand(full_ex.pipeline, full_ex, nodes))

        self._quick.submit(quick, lambda m, k=key: self._got(m, False, k), lambda msg: log.warning("Could not read the tables: %s", msg))
        self._deep.submit(full, lambda m, k=key: self._got(m, True, k), lambda msg: log.warning("Could not read every row: %s", msg))

    def _got(self, model: DataModel, full: bool, key: str) -> None:
        if key != self._key or (self.full and not full):
            return
        if key != self.key():                  # the project changed while this was read: a newer one is coming
            self._timer.start()
            return
        self.model, self.full = model, full
        self._notify()
        if full:
            self._serve()

    def _notify(self) -> None:
        for fn in list(self._listeners):
            try:
                fn()
            except RuntimeError:
                pass

    def _serve(self) -> None:
        waiters, self._waiters = self._waiters, []
        for fn in waiters:
            fn(self.model)

    def when_full(self, fn: Callable[[DataModel], None]) -> None:
        self.refresh()
        if self.full and self.model is not None and self._key == self.key():
            fn(self.model)
        else:
            self._waiters.append(fn)
            if len(self._waiters) > 32:              # a read that never completes must not pin the window forever
                del self._waiters[:-32]

    @property
    def waiting(self) -> bool:
        return bool(self._waiters)


# =================================================================== previews for the cards
def preview_of(snapshot: Pipeline, cache, model: DataModel, spec: dict) -> dict[str, Any]:
    """A few numbers to draw a card's thumbnail: the answer built on a copy of the project and computed on a
    sample of its tables. Never shown as a result; the answer itself is computed on every row."""
    from ..core.planner import instantiate
    scratch = Pipeline.from_dict(snapshot.to_dict(), snapshot.path)
    p = make_plan(model, spec)
    res = instantiate(scratch, p)
    term = res[p.terminal]
    node = scratch.nodes[term]
    ex = Executor(scratch, cache)
    if node.type == "chart":
        src = scratch.inputs_of(term)["in"][0]
        df = ex.preview(src, PREVIEW_ROWS)[0]
        prm = node.params
        kind = prm.get("kind")
        try:
            if kind == "line":
                col = (prm.get("series") or [{}])[0].get("column")
                if prm.get("color_by") and prm["color_by"] in df.columns:
                    groups = []
                    for g in df[prm["color_by"]].unique(maintain_order=True).to_list()[:4]:
                        groups.append(df.filter(df[prm["color_by"]] == g)[col].cast(float).to_list())
                    return {"kind": "lines", "ys": groups}
                return {"kind": "lines", "ys": [df[col].cast(float).to_list()]}
            if kind == "bar":
                return {"kind": "bar", "ys": df[prm["value"]].cast(float).to_list()[:12]}
            if kind == "scatter":
                y = (prm.get("series") or [{}])[0].get("column")
                d = df.select(prm["x"], y).drop_nulls().head(400)
                return {"kind": "scatter", "xs": d[prm["x"]].cast(float).to_list(), "ys": d[y].cast(float).to_list()}
            if kind == "histogram":
                s = df[prm["column"]].drop_nulls().cast(float)
                if s.len() > 1:
                    lo, hi = s.min(), s.max()
                    if hi > lo:
                        bins = [0] * 24
                        for v in s.to_list():
                            bins[min(23, int((v - lo) / (hi - lo) * 24))] += 1
                        return {"kind": "bar", "ys": bins}
        except Exception:  # noqa: BLE001 - a card without a picture is fine
            pass
        return {"kind": "none"}
    df, _res, _k = ex.preview(term, 200)
    return {"kind": "table", "rows": df.height, "cols": df.width}


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and v == v and v not in (float("inf"), float("-inf"))


class Thumb(QWidget):
    """A tiny chart drawn from a preview (no axes: it only shows the shape)."""

    def __init__(self, recipe: str, parent=None) -> None:
        super().__init__(parent)
        self.recipe = recipe
        self.data: dict[str, Any] | None = None
        self.setFixedHeight(34)

    def set_data(self, data: dict[str, Any] | None) -> None:
        self.data = data
        self.update()

    def paintEvent(self, e) -> None:
        p = QPainter(self)
        try:
            self._paint(p)
        except Exception:  # noqa: BLE001 - a thumbnail must never break painting the window
            log.exception("Could not draw a suggestion's preview")
        finally:
            p.end()

    def _paint(self, p: QPainter) -> None:
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(2, 4, -2, -4)
        d = self.data or {}
        kind = d.get("kind")
        accent = QColor(T.accent)
        colors = [accent, QColor(T.warn), QColor(T.ok), QColor(T.danger)]
        if kind == "lines":
            series = [[v for v in ys if _num(v)] for ys in d.get("ys") or []]
            vals = [v for ys in series for v in ys]
            if len(vals) >= 2:
                lo, hi = min(vals), max(vals)
                span = (hi - lo) or 1.0
                for i, ys in enumerate(series):
                    if len(ys) < 2:
                        continue
                    path = QPainterPath()
                    for k, v in enumerate(ys):
                        pt = QPointF(r.left() + r.width() * k / (len(ys) - 1), r.bottom() - r.height() * (v - lo) / span)
                        path.moveTo(pt) if k == 0 else path.lineTo(pt)
                    p.setPen(QPen(colors[i % len(colors)], 1.4)); p.drawPath(path)
                return
        if kind == "bar":
            ys = [v for v in d.get("ys") or [] if _num(v)]
            if ys:
                top = max(max(ys), 0) or 1.0
                w = r.width() / len(ys)
                p.setPen(Qt.NoPen)
                for i, v in enumerate(ys):
                    h = r.height() * max(v, 0) / top
                    c = QColor(accent); c.setAlpha(200 - int(120 * i / max(1, len(ys))))
                    p.setBrush(c); p.drawRoundedRect(QRectF(r.left() + i * w + 1, r.bottom() - h, max(1.0, w - 2), h), 1.5, 1.5)
                return
        if kind == "scatter":
            xs, ys = d.get("xs") or [], d.get("ys") or []
            pts = [(x, y) for x, y in zip(xs, ys) if _num(x) and _num(y)]
            if len(pts) >= 2:
                xl, xh = min(x for x, _ in pts), max(x for x, _ in pts)
                yl, yh = min(y for _, y in pts), max(y for _, y in pts)
                c = QColor(accent); c.setAlpha(140); p.setPen(Qt.NoPen); p.setBrush(c)
                for x, y in pts:
                    p.drawEllipse(QPointF(r.left() + r.width() * (x - xl) / ((xh - xl) or 1), r.bottom() - r.height() * (y - yl) / ((yh - yl) or 1)), 1.6, 1.6)
                return
        # a table, or nothing to draw yet: the recipe's icon
        ic = icon(RECIPE_ICONS.get(self.recipe, "sparkle"), T.faint, 26)
        ic.paint(p, self.rect().adjusted(0, 6, 0, -6), Qt.AlignLeft | Qt.AlignVCenter)
        if kind == "table" and d.get("rows") is not None:
            p.setPen(QColor(T.muted)); f = p.font(); f.setPointSizeF(8.5); p.setFont(f)
            p.drawText(self.rect().adjusted(34, 0, 0, 0), Qt.AlignLeft | Qt.AlignVCenter, "a table of results")


class SuggestionCard(QFrame):
    chosen = Signal(object)            # the Suggestion

    def __init__(self, s: Suggestion, parent=None) -> None:
        super().__init__(parent)
        self.s = s
        self.setObjectName("suggestion")
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(212, 100)
        self.setToolTip(f"{s.title}\n{s.why}\nClick to build this answer")
        self.setStyleSheet(f"QFrame#suggestion {{ background: {T.panel}; border: 1px solid {T.border}; border-radius: 8px; }}"
                           f"QFrame#suggestion:hover {{ border-color: {T.accent}; }}")
        lay = QVBoxLayout(self); lay.setContentsMargins(10, 8, 10, 8); lay.setSpacing(2)
        self.thumb = Thumb(s.recipe)
        title = QLabel(s.title); title.setWordWrap(True); title.setStyleSheet("font-weight: 600; border: none;")
        title.setMaximumHeight(34)
        why = QLabel(s.why); why.setObjectName("faint"); why.setWordWrap(False); why.setStyleSheet("border: none;")
        why.setMaximumWidth(192)
        lay.addWidget(self.thumb); lay.addWidget(title); lay.addWidget(why); lay.addStretch()

    def mouseReleaseEvent(self, e) -> None:
        if e.button() == Qt.LeftButton and self.rect().contains(e.position().toPoint()):
            self.chosen.emit(self.s)
        super().mouseReleaseEvent(e)


# =================================================================== the ask bar
class AskBar(QFrame):
    """Ask in words, or pick an answer DANCR offers. Emits ``build(spec)``; the window builds it."""

    build = Signal(object)
    closed = Signal()

    def __init__(self, doc, understanding: Understanding, parent=None) -> None:
        super().__init__(parent)
        self.doc, self.u = doc, understanding
        self.setObjectName("askbar")
        self.setStyleSheet(f"QFrame#askbar {{ background: {T.bg}; border-bottom: 1px solid {T.border}; }}")
        outer = QVBoxLayout(self); outer.setContentsMargins(12, 6, 12, 6); outer.setSpacing(4)
        row = QHBoxLayout(); row.setSpacing(6)
        ic = QLabel(); ic.setPixmap(icon("sparkle", T.accent, 18).pixmap(18, 18))
        self.edit = QLineEdit(); self.edit.setClearButtonEnabled(True)
        self.edit.setPlaceholderText("Ask about your data, like “total sales by region”, “average pressure per hour” or “compare A and B”")
        self.edit.returnPressed.connect(self._ask)
        self.edit.textEdited.connect(lambda _: self._set_message(""))
        self._completer_model = QStringListModel(self)
        comp = QCompleter(self._completer_model, self); comp.setCaseSensitivity(Qt.CaseInsensitive)
        comp.setFilterMode(Qt.MatchContains); comp.setCompletionMode(QCompleter.PopupCompletion)
        self._completer = comp
        self.ask_btn = QPushButton("Answer"); self.ask_btn.setObjectName("primary"); self.ask_btn.clicked.connect(self._ask)
        self.toggle = QToolButton(); self.toggle.setObjectName("quiet"); self.toggle.setCheckable(True); self.toggle.setChecked(True)
        self.toggle.setToolTip("Show or hide the suggested answers"); self.toggle.toggled.connect(self._toggle_tray)
        self.toggle.setIcon(icon("list-bullets", T.muted, 14))
        close = QToolButton(); close.setObjectName("quiet"); close.setIcon(icon("x", T.muted, 14))
        close.setToolTip("Close (Esc)"); close.clicked.connect(self.closed.emit)
        row.addWidget(ic); row.addWidget(self.edit, 1); row.addWidget(self.ask_btn); row.addWidget(close)
        outer.addLayout(row)
        self.message = QLabel(""); self.message.setObjectName("muted"); self.message.setWordWrap(True)
        self.message.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.message.setTextInteractionFlags(Qt.TextBrowserInteraction); self.message.linkActivated.connect(self._hint_clicked)
        self.message.hide()
        outer.addWidget(self.message)
        self.heading = QLabel(""); self.heading.setObjectName("section")
        self.heading.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)     # never widens the window
        head = QHBoxLayout(); head.setSpacing(4); head.addWidget(self.heading, 1); head.addWidget(self.toggle)
        outer.addLayout(head)
        self.scroll = QScrollArea(); self.scroll.setFrameShape(QFrame.NoFrame); self.scroll.setWidgetResizable(True)
        self.scroll.setMinimumWidth(60)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff); self.scroll.setFixedHeight(124)
        self.tray = QWidget(); self.tray_lay = QHBoxLayout(self.tray); self.tray_lay.setContentsMargins(0, 0, 0, 12); self.tray_lay.setSpacing(8)   # room above the scroll bar
        self.tray_lay.addStretch()
        self.scroll.setWidget(self.tray)
        outer.addWidget(self.scroll)
        self.cards: list[SuggestionCard] = []
        self.suggestions: list[Suggestion] = []
        self._previews = Serial(self)
        self._suggesting = Serial(self, waits_for_run=False)
        self._shown_key: str | None = None
        self.u.on_change(self._model_changed)

    # -- the tray
    def _toggle_tray(self, on: bool) -> None:
        self.scroll.setVisible(on and bool(self.cards))
        self.toggle.setToolTip("Hide the suggested answers" if on else "Show the suggested answers")

    def _model_changed(self) -> None:
        m = self.u.model
        if m is None:
            return
        focus = self.u.focus if self.u.focus in m.tables else None

        def work():                               # off the GUI thread: planning every candidate takes a moment
            return sorted({" ".join(k) for k in vocabulary(m)}), suggest(m, focus)

        def done(result) -> None:
            if m is not self.u.model:
                return
            words, sugs = result
            self._completer_model.setStringList(words)
            self.edit.setCompleter(self._completer)
            key = json.dumps([focus, [x.spec for x in sugs]], sort_keys=True, default=str)
            if key == self._shown_key and self.cards:
                return
            self._shown_key = key
            self._fill(sugs, focus)
        self._suggesting.submit(work, done, lambda msg: log.warning("Could not suggest answers: %s", msg))

    def _fill(self, sugs: list[Suggestion], focus: str | None) -> None:
        for c in self.cards:
            c.setParent(None); c.deleteLater()
        self.cards = []
        self.suggestions = sugs
        m = self.u.model
        if not sugs:
            self.heading.setText("Add a data file and DANCR will suggest answers it can give")
            self.scroll.hide(); return
        where = m.tables[focus].title if focus and m and focus in m.tables else None
        self.heading.setText(f"Answers I can give about {where}" if where else "Answers I can give from your tables")
        for s in sugs:
            card = SuggestionCard(s)
            card.chosen.connect(lambda s: self.build.emit(copy.deepcopy(s.spec)))
            self.tray_lay.insertWidget(self.tray_lay.count() - 1, card)
            self.cards.append(card)
        self.scroll.setVisible(self.toggle.isChecked())
        self._start_previews()

    def _start_previews(self) -> None:
        m = self.u.model
        if m is None:
            return
        ex = self.doc.snapshot_executor()
        snapshot, cache = ex.pipeline, ex.cache_dir
        specs = [s.spec for s in self.suggestions]
        cards = list(self.cards)

        def work() -> list[dict | None]:
            out = []
            for spec in specs:
                try:
                    out.append(preview_of(snapshot, cache, m, spec))
                except Exception:  # noqa: BLE001 - no picture for this one
                    out.append(None)
            return out

        def done(datas: list[dict | None]) -> None:
            for card, d in zip(cards, datas):
                try:
                    card.thumb.set_data(d)
                except RuntimeError:
                    pass
        self._previews.submit(work, done)

    # -- asking
    def keyPressEvent(self, e) -> None:
        if e.key() == Qt.Key_Escape:
            self.closed.emit(); return
        super().keyPressEvent(e)

    def focus_edit(self) -> None:
        self.toggle.setChecked(True)
        self.edit.setFocus(); self.edit.selectAll()

    def _set_message(self, html: str, error: bool = False) -> None:
        self.message.setText(html)
        self.message.setStyleSheet(f"color: {T.danger if error else T.muted};")
        self.message.setVisible(bool(html))

    def _ask(self) -> None:
        text = self.edit.text().strip()
        if not text:
            self._set_message("Type a question, or pick one of the answers below."); return
        m = self.u.model
        if m is None:
            self._set_message("Still reading your tables. Ask again in a moment."); return
        from ..core import memory
        a = ask(m, text, aliases=memory.aliases(self.u.doc.pipeline))      # the words this project has learned
        if not a.ok:
            hints = " ".join(f'<a href="{h}">{h}</a>' for h in a.hints)
            self._set_message(a.message.split(" Did you mean")[0] + (f" Did you mean: {hints}" if hints else ""), error=True)
            return
        self._set_message("")
        self.build.emit(a.spec)

    def _hint_clicked(self, phrase: str) -> None:
        text = self.edit.text()
        # swap the word that was not understood for the phrase picked
        m = self.u.model
        if m is not None:
            a = ask(m, text)
            for u in a.unknown:
                import re
                text = re.sub(rf"\b{re.escape(u)}\b", phrase, text, count=1, flags=re.IGNORECASE)
        self.edit.setText(text)
        self._ask()

    def show_status(self, text: str, error: bool = False) -> None:
        self._set_message(text, error)


# =================================================================== the ask surface
class AskOverlay(QFrame):
    """A centred Ask card floating over the workspace, holding the ask bar.

    Asking is a transient surface: it never resizes the canvas or the result panel, so the window does not
    collapse into a squished strip every time a question is typed. Click the backdrop or press Esc to dismiss."""

    closed = Signal()

    def __init__(self, parent, askbar: "AskBar") -> None:
        super().__init__(parent)
        self._askbar = askbar
        self.setObjectName("askoverlay")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("QFrame#askoverlay { background: rgba(20, 22, 28, 110); }")
        outer = QVBoxLayout(self); outer.setContentsMargins(24, 24, 24, 24); outer.addStretch(1)
        row = QHBoxLayout(); row.addStretch(1)
        self._card = QFrame(); self._card.setObjectName("askcard")
        self._card.setAttribute(Qt.WA_StyledBackground, True)
        self._card.setStyleSheet(f"QFrame#askcard {{ background: {T.panel}; border: 1px solid {T.border};"
                                 " border-radius: 12px; }}")
        self._card.setMinimumWidth(460); self._card.setMaximumWidth(880)
        cl = QVBoxLayout(self._card); cl.setContentsMargins(0, 0, 0, 0); cl.setSpacing(0)
        cl.addWidget(askbar)
        row.addWidget(self._card); row.addStretch(1)
        outer.addLayout(row); outer.addStretch(1)
        self.hide()

    def open(self, focus: bool = True) -> None:
        parent = self.parentWidget()
        if parent is not None:
            self.setGeometry(parent.rect())            # cover the workspace; recentre on every open
        self._askbar.setVisible(True)
        self.show(); self.raise_()
        if focus:
            self._askbar.focus_edit()

    def fit(self) -> None:
        """Keep the overlay covering the workspace when the window is resized while it is open."""
        parent = self.parentWidget()
        if parent is not None and self.isVisible():
            self.setGeometry(parent.rect())

    def mousePressEvent(self, e) -> None:
        self.closed.emit()                             # only the backdrop reaches here; the card consumes its clicks
        super().mousePressEvent(e)


# =================================================================== the selected answer
class AnswerPanel(QFrame):
    """Title, chips and assumptions of the selected answer. Emits ``change(answer_id, key, value)`` (key "set" takes
    a dict of spec changes), ``showSteps``, ``delete(answer_id)`` and ``rename(answer_id, title)``."""

    change = Signal(str, str, object)
    showSteps = Signal()
    delete = Signal(str)
    rename = Signal(str, str)

    def __init__(self, doc, understanding: Understanding, parent=None) -> None:
        super().__init__(parent)
        self.doc, self.u = doc, understanding
        self.answer_id: str | None = None
        self.setObjectName("answerbar")
        self.setStyleSheet(f"QFrame#answerbar {{ background: {T.panel}; border-bottom: 1px solid {T.border}; }}")
        lay = QHBoxLayout(self); lay.setContentsMargins(12, 6, 10, 6); lay.setSpacing(6)
        star = QLabel(); star.setPixmap(icon("sparkle", T.accent, 16).pixmap(16, 16))
        self.title = QLineEdit(); self.title.setFrame(False); self.title.setStyleSheet("font-weight: 600; background: transparent;")
        self.title.setToolTip("The answer's name. Click to rename it"); self.title.setMinimumWidth(160)
        self.title.editingFinished.connect(self._renamed)
        self.chip_box = QHBoxLayout(); self.chip_box.setSpacing(4)
        self.assume_btn = QToolButton(); self.assume_btn.setObjectName("quiet"); self.assume_btn.setPopupMode(QToolButton.InstantPopup)
        self.assume_btn.setIcon(icon("lightbulb", T.muted, 14)); self.assume_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.kept = QLabel(""); self.kept.setObjectName("faint")
        show = QPushButton("Show the steps"); show.setObjectName("quiet"); show.clicked.connect(self.showSteps.emit)
        delete = QPushButton("Delete"); delete.setObjectName("quiet"); delete.clicked.connect(lambda: self.answer_id and self.delete.emit(self.answer_id))
        lay.addWidget(star); lay.addWidget(self.title); lay.addLayout(self.chip_box); lay.addWidget(self.assume_btn)
        lay.addWidget(self.kept); lay.addStretch(); lay.addWidget(show); lay.addWidget(delete)
        self.u.on_change(self.refresh)
        listen(self, doc.answerChanged, lambda aid: self.refresh() if aid == self.answer_id else None)

    def set_answer(self, aid: str | None) -> None:
        if aid != self.answer_id and self.title.hasFocus():
            self._renamed()                           # a name being typed belongs to the answer it was typed for
        self.answer_id = aid
        self._title_for = aid
        self.setVisible(aid is not None)
        a = self.doc.pipeline.answer(aid) if aid else None
        self._set_title(a.title if a is not None else "")
        self.refresh()

    def _set_title(self, text: str) -> None:
        self.title.blockSignals(True)
        self.title.setText(text)
        self.title.setCursorPosition(0)
        self.title.setFixedWidth(min(420, max(160, self.title.fontMetrics().horizontalAdvance(text) + 24)))
        self.title.blockSignals(False)

    def _renamed(self) -> None:
        aid = getattr(self, "_title_for", None)
        a = self.doc.pipeline.answer(aid) if aid else None
        if a is not None and self.title.text().strip() and self.title.text().strip() != a.title:
            self.rename.emit(a.id, self.title.text().strip())

    def refresh(self) -> None:
        a = self.doc.pipeline.answer(self.answer_id) if self.answer_id else None
        while self.chip_box.count():
            w = self.chip_box.takeAt(0).widget()
            if w is not None:
                w.setParent(None); w.deleteLater()
        if a is None:
            return
        if not self.title.hasFocus():
            self._set_title(a.title)
        m = self.u.model
        chips = []
        if m is not None and a.spec.get("table") in m.tables:
            try:
                chips = make_chips(m, a.spec)
            except Exception:  # noqa: BLE001
                log.exception("Could not list the choices of %s", a.id)
        for c in chips:
            b = QToolButton(); b.setObjectName("chip"); b.setText(c["text"]); b.setPopupMode(QToolButton.InstantPopup)
            b.setStyleSheet(f"QToolButton#chip {{ border: 1px solid {T.border}; border-radius: 10px; padding: 2px 8px; background: {T.bg}; }}"
                            f"QToolButton#chip:hover {{ border-color: {T.accent}; }}")
            menu = QMenu(b)
            for ch in c["choices"]:
                act = menu.addAction(ch["label"])
                act.setCheckable(True); act.setChecked(ch["value"] == c["value"])
                act.triggered.connect(lambda _=False, k=c["key"], v=ch["value"]: self.change.emit(self.answer_id, k, v))
            b.setMenu(menu)
            self.chip_box.addWidget(b)
        n = len(a.assumptions)
        self.assume_btn.setText(f"{n} assumption{'s' if n != 1 else ''}" if n else "")
        self.assume_btn.setVisible(bool(n))
        menu = getattr(self, "_assume_menu", None)
        if menu is None:                               # one menu, refilled: a new one per refresh was never freed
            menu = self._assume_menu = QMenu(self.assume_btn)
        menu.clear()
        for x in a.assumptions:
            head = menu.addAction(x["text"]); head.setEnabled(False)
            for ch in x.get("choices") or []:
                act = menu.addAction("    → " + ch["label"])
                act.triggered.connect(lambda _=False, v=ch["set"]: self.change.emit(self.answer_id, "set", v))
            menu.addSeparator()
        self.assume_btn.setMenu(menu)
        from ..core.answers import kept_by_hand
        kept = kept_by_hand(self.doc.pipeline, a)
        self.kept.setText(("Kept your changes to " + ", ".join(kept)) if kept else "")
