"""The guided build ("Build it for me"): a short interview that turns a pile of spreadsheets into a
tidy, connected pipeline and an Answer.

Deterministic, not AI: the questions have sensible defaults drawn from profiling the files, so the
user mostly confirms. Skipping every question still produces something reasonable.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame, QStackedWidget,
                               QCheckBox, QComboBox, QLineEdit, QFormLayout, QScrollArea, QFileDialog, QToolButton,
                               QSizePolicy, QDialog, QDialogButtonBox)

from ..core.profile import profile_files, suggest_links, suggest_stacks, match_percent, TableProfile, LinkSuggestion, StackSuggestion
from ..core.planner import plan
from ..core.nodes.load import CSV_EXT, EXCEL_EXT, PARQUET_EXT
from .document import Document
from .theme import T
from .icons import icon
from .workers import Serial

DATA_EXT = CSV_EXT | EXCEL_EXT | PARQUET_EXT
DATA_FILTER = "Data files (*.csv *.tsv *.txt *.dat *.xlsx *.xlsm *.xls *.parquet);;All files (*)"
# intent keys the planner understands, with the plain-English card text
GOALS = [
    ("total", "Total things up", "Group rows and add up a number — like a pivot table.", "sigma"),
    ("over_time", "See change over time", "Average a number over days, weeks or months, and chart it.", "trend-up"),
    ("describe", "Describe this data", "One row per column: count, missing, average, range.", "list-bullets"),
]
EVERY = [("1h", "every hour"), ("1d", "every day"), ("1w", "every week"), ("1mo", "every month")]


def _clear(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        w = item.widget()
        if w is not None:
            w.setParent(None); w.deleteLater()


def _reset_layout(page: QWidget) -> QVBoxLayout:
    """Give a page a single empty top-level layout, reusing it if it already has one."""
    lay = page.layout()
    if lay is None:
        lay = QVBoxLayout(page); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(0)
    else:
        _clear(lay)
    return lay


class WizardPage(QWidget):
    """The interview page. Emits ``requestBuild(plan, answer_id)`` (answer_id None = brand new)."""

    requestBuild = Signal(object, object)
    cancelled = Signal()

    def __init__(self, doc: Document, parent=None) -> None:
        super().__init__(parent)
        self.doc = doc
        self.paths: list[str] = []
        self.profiles: list[TableProfile] = []
        self.links: list[LinkSuggestion] = []
        self.stacks: list[StackSuggestion] = []
        self.skipped: list[tuple[str, str]] = []
        self.intent = "total"
        self.answer_id: str | None = None
        self._fixed_assembly: dict | None = None
        self._prefill: dict | None = None
        self._link_rows: list[dict] = []
        self._stack_checks: dict[int, QCheckBox] = {}
        self._serial = Serial(self)
        self._step = 0
        self._build_ui()

    # ------------------------------------------------------------ chrome
    def _build_ui(self) -> None:
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(0)
        head = QFrame(); head.setStyleSheet(f"QFrame {{ background: {T.bg}; border-bottom: 1px solid {T.border}; }}")
        h = QHBoxLayout(head); h.setContentsMargins(14, 8, 10, 8); h.setSpacing(8)
        ic = QLabel(); ic.setPixmap(icon("magic-wand", T.accent, 20).pixmap(20, 20))
        title = QLabel("Build it for me"); title.setObjectName("heading")
        h.addWidget(ic); h.addWidget(title); h.addSpacing(10)
        self.step_buttons: list[QToolButton] = []
        for name in ("Your data", "What you want", "Details", "Review"):
            b = QToolButton(); b.setObjectName("quiet"); b.setText(name); b.setCheckable(True)
            b.clicked.connect(lambda _=False, i=len(self.step_buttons): self._goto(i))
            self.step_buttons.append(b); h.addWidget(b)
            if len(self.step_buttons) < 4:
                sep = QLabel("›"); sep.setObjectName("faint"); h.addWidget(sep)
        h.addStretch()
        close = QToolButton(); close.setObjectName("quiet"); close.setIcon(icon("x", T.muted, 14)); close.setToolTip("Close")
        close.clicked.connect(self._cancel); h.addWidget(close)
        lay.addWidget(head)
        self.body = QStackedWidget()
        self.data_page = QWidget(); self.goal_page = QWidget(); self.detail_page = QWidget(); self.review_page = QWidget()
        for p in (self.data_page, self.goal_page, self.detail_page, self.review_page):
            self.body.addWidget(p)
        lay.addWidget(self.body, 1)
        foot = QFrame(); foot.setStyleSheet(f"QFrame {{ background: {T.bg}; border-top: 1px solid {T.border}; }}")
        f = QHBoxLayout(foot); f.setContentsMargins(14, 8, 14, 8); f.setSpacing(8)
        self.hint = QLabel(""); self.hint.setObjectName("muted")
        self.back_btn = QPushButton("Back"); self.back_btn.clicked.connect(self._back)
        self.skip_btn = QPushButton("Skip"); self.skip_btn.setObjectName("quiet"); self.skip_btn.clicked.connect(self._skip)
        self.next_btn = QPushButton("Next"); self.next_btn.setObjectName("primary"); self.next_btn.clicked.connect(self._next)
        f.addWidget(self.hint, 1); f.addWidget(self.back_btn); f.addWidget(self.skip_btn); f.addWidget(self.next_btn)
        lay.addWidget(foot)

    def _set_hint(self, text: str) -> None:
        self.hint.setText(text)

    # ------------------------------------------------------------ lifecycle
    def begin(self, paths: list[str] | None = None, answer_id: str | None = None) -> None:
        self.answer_id = answer_id
        self._fixed_assembly = None
        self._prefill = None
        self.paths = []                     # never the files of an earlier build (another project, perhaps)
        if answer_id is not None:
            answer = self.doc.pipeline.answer(answer_id)
            if answer is not None:
                self._prefill = dict(answer.config)
                self._fixed_assembly = self._prefill.get("assembly")
                self.intent = self._prefill.get("intent", "total")
                self.paths = self._paths_from_assembly(self._fixed_assembly)
        if paths:
            self.paths = [str(p) for p in paths]
        if not self.paths:
            # Use the files already on the map, so an open project does not have to be re-fed.
            self.paths = [str(p) for p in self.doc.source_paths()]
        if not self.paths:
            self._no_files()
            return
        self._profile()

    @staticmethod
    def _paths_from_assembly(assembly: dict | None) -> list[str]:
        if not assembly:
            return []
        if assembly.get("kind") == "stack":
            return list(assembly.get("paths") or [])
        if assembly.get("kind") == "join":
            return [assembly["primary"]] + [l["right"] for l in assembly.get("links") or []]
        if assembly.get("kind") == "single":
            return [assembly["path"]]
        return []

    def _no_files(self) -> None:
        self.profiles = []; self.links = []; self.stacks = []; self.skipped = []
        self._build_data_page()
        self._goto(0)
        self._update_nav()

    def _profile(self) -> None:
        outer = _reset_layout(self.data_page)
        holder = QWidget(); body = QVBoxLayout(holder); body.setContentsMargins(40, 40, 40, 40)
        lab = QLabel("Reading your files…"); lab.setObjectName("heading"); body.addWidget(lab)
        body.addStretch(); outer.addWidget(holder)
        self._goto(0)
        paths = list(self.paths)
        self._serial.submit(lambda: profile_files(paths), self._profiled, self._profile_failed)

    def _profiled(self, result) -> None:
        profs, skipped = result
        self.profiles = profs
        self.skipped = skipped
        self.links = suggest_links(profs)
        self.stacks = suggest_stacks(profs)
        self._build_data_page()
        self._build_goal_page()
        self._build_review_page()
        if self._prefill:                       # changing an existing answer: jump to the goal step
            self._apply_prefill()
            self._goto(1)
        else:
            self._goto(0)
        self._update_nav()

    def _profile_failed(self, message: str) -> None:
        outer = _reset_layout(self.data_page)
        holder = QWidget(); body = QVBoxLayout(holder); body.setContentsMargins(40, 40, 40, 40); body.setSpacing(10)
        lab = QLabel("I could not read those files"); lab.setObjectName("heading")
        msg = QLabel(message); msg.setObjectName("muted"); msg.setWordWrap(True)
        choose = QPushButton("Choose files…"); choose.setObjectName("primary"); choose.clicked.connect(self._choose)
        body.addWidget(lab); body.addWidget(msg); body.addWidget(choose, 0, Qt.AlignLeft); body.addStretch()
        outer.addWidget(holder)
        self._goto(0)

    # ------------------------------------------------------------ pages
    def _page_body(self, page: QWidget) -> QVBoxLayout:
        outer = _reset_layout(page)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QFrame.NoFrame)
        holder = QWidget(); inner = QVBoxLayout(holder); inner.setContentsMargins(28, 24, 28, 24); inner.setSpacing(12)
        scroll.setWidget(holder)
        outer.addWidget(scroll)
        return inner

    def _build_data_page(self) -> None:
        body = self._page_body(self.data_page)
        body.addWidget(self._heading("Here is what I found"))
        if not self.profiles:
            body.addWidget(self._muted("Drop a few spreadsheets here, or choose them."))
            b = QPushButton("Choose files…"); b.setObjectName("primary"); b.setIcon(icon("folder-open", "#ffffff", 16))
            b.clicked.connect(self._choose); body.addWidget(b, 0, Qt.AlignLeft)
            body.addStretch(); return
        for p in self.profiles:
            body.addWidget(self._muted(f"<b>{Path(p.path).name}</b> — {p.rows:,} rows × {len(p.columns)} columns"
                                       + (f" · dates in {p.time_column}" if p.time_column else "")))
        self._link_rows = []
        if self.links:
            body.addWidget(self._small("These look like they can be linked (joined on a shared key). Change the columns if I guessed wrong:"))
            for i, l in enumerate(self.links):
                body.addWidget(self._link_row(i, l))
        if len(self.profiles) >= 2:
            add = QPushButton("Link two more columns…"); add.setObjectName("quiet"); add.setIcon(icon("plus", T.muted, 14))
            add.clicked.connect(self._add_link)
            body.addWidget(add, 0, Qt.AlignLeft)
            if not self.links:
                body.addWidget(self._muted("I did not find a shared key, but you can join two tables on columns that match."))
        self._stack_checks = {}
        if self.stacks:
            body.addWidget(self._small("These files have the same columns, so they can be stacked:"))
            for i, s in enumerate(self.stacks):
                cb = QCheckBox(" + ".join(Path(p).name for p in s.paths))
                cb.setChecked(self._stack_prefilled(s))
                self._stack_checks[i] = cb
                body.addWidget(cb)
        if self.skipped:
            body.addWidget(self._small("Could not read these:"))
            for path, reason in self.skipped:
                body.addWidget(self._muted(f"<b>{Path(path).name}</b> — {reason}"))
        b = QPushButton("Add more files…"); b.setObjectName("quiet"); b.setIcon(icon("plus", T.muted, 14))
        b.clicked.connect(self._choose); body.addWidget(b, 0, Qt.AlignLeft)
        body.addStretch()

    # -- editable link rows
    def _table_label(self, path: str) -> QLabel:
        lab = QLabel(Path(path).name); lab.setObjectName("muted"); lab.setToolTip(path)
        return lab

    def _col_combo(self, path: str, current: str) -> QComboBox:
        prof = next((p for p in self.profiles if p.path == path), None)
        cols = [c.name for c in (prof.columns if prof else [])]
        cols.sort(key=lambda n: (not (prof.column(n).join_candidate if prof and prof.column(n) else False), n))
        combo = QComboBox(); combo.setMinimumWidth(130); combo.addItems(cols)
        if current in cols:
            combo.setCurrentText(current)
        return combo

    def _link_row(self, i: int, link: LinkSuggestion) -> QWidget:
        row = QWidget(); h = QHBoxLayout(row); h.setContentsMargins(0, 0, 0, 0); h.setSpacing(6)
        cb = QCheckBox(); cb.setChecked(self._link_prefilled(link))
        left = self._col_combo(link.left, link.left_col)
        right = self._col_combo(link.right, link.right_col)
        arrow = QLabel("↔"); arrow.setObjectName("muted")
        match = QLabel(""); match.setObjectName("faint")
        h.addWidget(cb); h.addWidget(self._table_label(link.left)); h.addWidget(left)
        h.addWidget(arrow); h.addWidget(self._table_label(link.right)); h.addWidget(right)
        h.addWidget(match); h.addStretch()
        self._link_rows.append({"check": cb, "left": left, "right": right, "match": match})
        left.currentIndexChanged.connect(lambda _: self._link_edited(i))
        right.currentIndexChanged.connect(lambda _: self._link_edited(i))
        self._link_edited(i)
        return row

    def _link_edited(self, i: int) -> None:
        if i >= len(self._link_rows) or i >= len(self.links):
            return
        row, link = self._link_rows[i], self.links[i]
        link.left_col, link.right_col = row["left"].currentText(), row["right"].currentText()
        link.match_pct = self._match(link.left, link.left_col, link.right, link.right_col)
        row["match"].setText(f"{link.match_pct:g}% match" if link.match_pct else "no values in common")

    def _match(self, left_path: str, left_col: str, right_path: str, right_col: str) -> float:
        lp = next((p for p in self.profiles if p.path == left_path), None)
        rp = next((p for p in self.profiles if p.path == right_path), None)
        lc = lp.column(left_col) if lp else None
        rc = rp.column(right_col) if rp else None
        if not lc or not rc:
            return 0.0
        return round(match_percent(lc, rc), 1)

    def _add_link(self) -> None:
        chosen = self._ask_link()
        if not chosen:
            return
        left_path, left_col, right_path, right_col = chosen
        self.links.append(LinkSuggestion(left_path, left_col, right_path, right_col, 1.0,
                                         self._match(left_path, left_col, right_path, right_col)))
        self._build_data_page()

    def _ask_link(self):
        if len(self.profiles) < 2:
            return None
        dlg = QDialog(self); dlg.setWindowTitle("Link two tables")
        form = QFormLayout(dlg)
        names = [Path(p.path).name for p in self.profiles]
        lt = QComboBox(); lt.addItems(names)
        rt = QComboBox(); rt.addItems(names)
        if len(names) > 1:
            rt.setCurrentIndex(1)
        lc, rc = QComboBox(), QComboBox()

        def fill(combo, path):
            prof = next((p for p in self.profiles if p.path == path), None)
            combo.clear()
            if prof:
                combo.addItems([c.name for c in prof.columns if c.join_candidate] + [c.name for c in prof.columns if not c.join_candidate])

        fill(lc, self.profiles[lt.currentIndex()].path)
        fill(rc, self.profiles[rt.currentIndex()].path)
        lt.currentIndexChanged.connect(lambda i: fill(lc, self.profiles[i].path))
        rt.currentIndexChanged.connect(lambda i: fill(rc, self.profiles[i].path))
        form.addRow("First table", lt); form.addRow("on column", lc)
        form.addRow("Second table", rt); form.addRow("on column", rc)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept); buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)
        if dlg.exec() != QDialog.Accepted:
            return None
        return (self.profiles[lt.currentIndex()].path, lc.currentText(),
                self.profiles[rt.currentIndex()].path, rc.currentText())

    def _build_goal_page(self) -> None:
        body = self._page_body(self.goal_page)
        body.addWidget(self._heading("What do you want to find out?"))
        body.addWidget(self._muted("Pick one. You can change everything afterwards, and build more than one answer from the same files."))
        self.goal_buttons: dict[str, QPushButton] = {}
        row = QHBoxLayout(); row.setSpacing(10)
        measures, cats, times = self._available()
        for key, title, blurb, ic in GOALS:
            disabled = (key == "over_time" and not times) or (key == "total" and not cats)
            b = QPushButton(f"{title}\n{blurb}")
            b.setCheckable(True); b.setEnabled(not disabled)
            b.setIcon(icon(ic, T.accent if not disabled else T.faint, 22)); b.setMinimumHeight(76)
            b.setStyleSheet(f"QPushButton {{ text-align: left; padding: 10px 12px; border: 1px solid {T.border}; border-radius: 8px; }}"
                            f"QPushButton:checked {{ border-color: {T.accent}; background: {T.hover}; }}")
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            b.clicked.connect(lambda _=False, k=key: self._set_intent(k))
            self.goal_buttons[key] = b; row.addWidget(b)
        body.addLayout(row)
        if not times:
            body.addWidget(self._muted("<i>“Change over time” needs a date column; none was found.</i>"))
        if not cats:
            body.addWidget(self._muted("<i>“Total things up” needs a category column; none was found.</i>"))
        body.addStretch()
        self._set_intent(self.intent)

    def _build_detail_page(self) -> None:
        body = self._page_body(self.detail_page)
        measures, cats, times = self._available()
        self._detail_inputs: dict[str, QComboBox | QLineEdit] = {}
        form = QFormLayout(); form.setSpacing(10); form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        if self.intent != "describe":
            if measures:
                m = QComboBox(); m.addItems(measures); form.addRow("Number to add up", m)
                self._detail_inputs["measure"] = m
            if self.intent == "over_time":
                t = QComboBox(); t.addItems(times)
                every = QComboBox()
                for v, label in EVERY:
                    every.addItem(label, v)
                form.addRow("Date column", t); form.addRow("How often", every)
                self._detail_inputs["time_column"] = t; self._detail_inputs["every"] = every
            else:
                g = QComboBox(); g.addItems(cats); form.addRow("Group by", g)
                self._detail_inputs["group"] = g
        body.addWidget(self._heading("A couple of details"))
        body.addLayout(form)
        self.title_edit = QLineEdit(); self.title_edit.setPlaceholderText("Name this answer")
        body.addWidget(self._small("Name this answer")); body.addWidget(self.title_edit)
        body.addWidget(self._muted("These are the only choices needed — everything else was worked out from your files."))
        body.addStretch()
        self._apply_prefill_slots()
        self.title_edit.editingFinished.connect(self._build_review_page)

    def _build_review_page(self) -> None:
        body = self._page_body(self.review_page)
        body.addWidget(self._heading("Here is what I will build"))
        self.review_sentence = QLabel(""); self.review_sentence.setWordWrap(True)
        self.review_sentence.setStyleSheet("font-size: 12pt;")
        body.addWidget(self.review_sentence)
        self.review_steps = QLabel(""); self.review_steps.setObjectName("muted"); self.review_steps.setWordWrap(True)
        body.addWidget(self.review_steps)
        self.review_warning = QLabel(""); self.review_warning.setObjectName("error"); self.review_warning.setWordWrap(True)
        body.addWidget(self.review_warning)
        body.addStretch()
        self._refresh_review()

    # ------------------------------------------------------------ small helpers
    def _heading(self, text: str) -> QLabel:
        lab = QLabel(text); lab.setStyleSheet("font-size: 15pt; font-weight: 600;"); return lab

    def _muted(self, text: str) -> QLabel:
        lab = QLabel(text); lab.setObjectName("muted"); lab.setWordWrap(True); return lab

    def _small(self, text: str) -> QLabel:
        lab = QLabel(text); lab.setObjectName("section"); return lab

    # ------------------------------------------------------------ navigation
    def _goto(self, i: int) -> None:
        if i == 2 and self.profiles:
            self._build_detail_page()
        if i == 3 and self.profiles:
            self._build_review_page()
        self._step = max(0, min(i, 3))
        self.body.setCurrentIndex(self._step)
        for k, b in enumerate(self.step_buttons):
            b.setChecked(k == self._step)
        self._update_nav()

    def _update_nav(self) -> None:
        has = bool(self.profiles)
        self.back_btn.setVisible(self._step > 0)
        self.skip_btn.setVisible(self._step in (1, 2) and has)
        self.next_btn.setText("Build the answer" if self._step == 3 else "Next")
        self.next_btn.setEnabled(has and (self._step < 3 or self._can_build()))
        for k, b in enumerate(self.step_buttons):
            b.setEnabled(has)
        hint = {"0": "Confirm how your files relate, then continue.",
                "1": "Choose the outcome closest to what you need.",
                "2": "Sensible choices are already made; change any.",
                "3": "Build it, then explore the result in the table or chart."}.get(str(self._step), "")
        self._set_hint(hint)

    def _next(self) -> None:
        if self._step >= 3:
            self._build()
            return
        self._goto(self._step + 1)

    def _back(self) -> None:
        self._goto(self._step - 1)

    def _skip(self) -> None:
        if self._step == 1:
            self._set_intent(self._first_viable_intent())
        self._goto(self._step + 1)

    def _cancel(self) -> None:
        self.cancelled.emit()

    def _choose(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self.window(), "Choose data files", str(Path.home()), DATA_FILTER)
        if files:
            self.paths = list(dict.fromkeys(self.paths + files))
            self._profile()

    # ------------------------------------------------------------ intent / slots
    def _first_viable_intent(self) -> str:
        measures, cats, times = self._available()
        if times:
            return "over_time"
        return "total" if cats else "total"

    def _set_intent(self, key: str) -> None:
        self.intent = key
        for k, b in getattr(self, "goal_buttons", {}).items():
            b.setChecked(k == key)

    def _apply_prefill(self) -> None:
        if self._prefill:
            self.intent = self._prefill.get("intent", self.intent)
            self._set_intent(self.intent)

    def _apply_prefill_slots(self) -> None:
        cfg = self._prefill or {}
        inp = getattr(self, "_detail_inputs", {})
        for key, widget in inp.items():
            val = cfg.get(key)
            if val and isinstance(widget, QComboBox):
                i = widget.findText(str(val))
                if i >= 0:
                    widget.setCurrentIndex(i)
        if "title" in cfg and cfg["title"] and hasattr(self, "title_edit"):
            self.title_edit.setText(cfg["title"])

    def _link_prefilled(self, link: LinkSuggestion) -> bool:
        asm = self._fixed_assembly
        if not asm or asm.get("kind") != "join":
            return True
        return any((l["right"] == link.right and l["left_on"] == link.left_col and l["right_on"] == link.right_col)
                   or (l["right"] == link.left and l["left_on"] == link.right_col and l["right_on"] == link.left_col)
                   for l in asm.get("links") or [])

    def _stack_prefilled(self, stack: StackSuggestion) -> bool:
        asm = self._fixed_assembly
        if not asm or asm.get("kind") != "stack":
            return True
        return set(asm.get("paths") or []) == set(stack.paths)

    # ------------------------------------------------------------ modelling
    def _available(self) -> tuple[list[str], list[str], list[str]]:
        involved = self._involved_paths()
        measures, cats, times = [], [], []
        for p in self.profiles:
            if p.path not in involved:
                continue
            measures += [m for m in p.measures if m not in measures]
            cats += [c for c in p.category_columns if c not in cats]
            if p.time_column and p.time_column not in times:
                times.append(p.time_column)
        return measures, cats, times

    def _involved_paths(self) -> set[str]:
        asm = self._assembly()
        if asm["kind"] == "stack":
            return set(asm["paths"])
        if asm["kind"] == "join":
            return {asm["primary"]} | {l["right"] for l in asm["links"]}
        return {asm["path"]}

    def _assembly(self) -> dict:
        if self._fixed_assembly:
            return self._fixed_assembly
        checked_stacks = [self.stacks[i] for i, cb in self._stack_checks.items() if cb.isChecked()]
        if checked_stacks:
            s = checked_stacks[0]
            return {"kind": "stack", "paths": list(s.paths), "label": "source"}
        checked = [self.links[i] for i, row in enumerate(self._link_rows) if row["check"].isChecked()]
        if checked:
            by_path = {p.path: p for p in self.profiles}
            counts: dict[str, int] = {}
            for l in checked:
                counts[l.left] = counts.get(l.left, 0) + 1
                counts[l.right] = counts.get(l.right, 0) + 1
            primary = max(counts, key=lambda p: (counts[p], by_path[p].rows if p in by_path else 0))
            links = []
            for l in checked:
                if l.left == primary:
                    links.append({"right": l.right, "left_on": l.left_col, "right_on": l.right_col})
                elif l.right == primary:
                    links.append({"right": l.left, "left_on": l.right_col, "right_on": l.left_col})
            if links:
                return {"kind": "join", "primary": primary, "links": links}
        if self.profiles:
            biggest = max(self.profiles, key=lambda p: p.rows)
            return {"kind": "single", "path": biggest.path}
        return {"kind": "single", "path": ""}

    def _config(self) -> dict:
        cfg: dict = {"intent": self.intent, "assembly": self._assembly()}
        inp = getattr(self, "_detail_inputs", {})
        for key, widget in inp.items():
            if isinstance(widget, QComboBox):
                if key == "every":
                    cfg[key] = widget.currentData()
                else:
                    cfg[key] = widget.currentText()
        title = self.title_edit.text().strip() if hasattr(self, "title_edit") else ""
        cfg["title"] = title or self._default_title(cfg)
        return cfg

    @staticmethod
    def _default_title(cfg: dict) -> str:
        if cfg.get("intent") == "describe":
            return "Describe the columns"
        if cfg.get("intent") == "over_time":
            return f"{cfg.get('measure', 'Values')} over time"
        if cfg.get("measure") and cfg.get("group"):
            return f"Total {cfg['measure']} by {cfg['group']}"
        return "Totals"

    def _make_plan(self):
        if not self.profiles:
            return None, "No data yet."
        try:
            return plan(self.profiles, self._config()), None
        except ValueError as e:
            return None, str(e)

    def _can_build(self) -> bool:
        p, err = self._make_plan()
        return p is not None

    def _refresh_review(self) -> None:
        p, err = self._make_plan()
        if p is None:
            self.review_sentence.setText("")
            self.review_steps.setText("")
            self.review_warning.setText(err or "Cannot build this yet.")
            return
        self.review_warning.setText("")
        self.review_sentence.setText("“" + p.title + "”")
        steps = [s.type for s in p.steps]
        self.review_steps.setText("I will: " + p.sentence() + f"\n({len(steps)} steps)")
        if self.answer_id:
            self.review_warning.setText("Changing answers rebuilds this answer's steps; any edits you made to them by hand will be replaced. Shared steps are kept.")

    def _build(self) -> None:
        p, err = self._make_plan()
        if p is None:
            self.review_warning.setText(err or "Cannot build this yet.")
            return
        self.requestBuild.emit(p, self.answer_id)
