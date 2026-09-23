"""Centre page for a report: title block, ordered layout of connected items and text, build + open."""
from __future__ import annotations



from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QListWidget, QListWidgetItem,
                               QPushButton, QFormLayout, QInputDialog, QAbstractItemView, QSplitter, QCheckBox)

from .document import Document
from .theme import T
from .common import page_header, status_dot
from .icons import icon, node_icon_name
from .external import open_external


class ReportView(QWidget):
    runRequested = Signal(str)

    def __init__(self, doc: Document, parent=None) -> None:
        super().__init__(parent)
        self.doc = doc
        self.nid: str | None = None
        self._suppress = False
        self._timer = QTimer(self); self._timer.setSingleShot(True); self._timer.setInterval(400); self._timer.timeout.connect(self._commit_text)
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(0)
        head, h = page_header()
        self.heading = QLabel("Report"); self.heading.setObjectName("heading")
        self.status = QLabel(""); self.status.setWordWrap(True); self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.open_btn = QPushButton("Open"); self.open_btn.setIcon(icon("arrow-square-out", T.text, 14)); self.open_btn.clicked.connect(self._open); self.open_btn.hide()
        self.pdf_btn = QPushButton("Open PDF"); self.pdf_btn.clicked.connect(self._open_pdf); self.pdf_btn.hide()
        self.build_btn = QPushButton("Build report"); self.build_btn.setObjectName("primary"); self.build_btn.setIcon(icon("play", "#ffffff", 14))
        self.build_btn.clicked.connect(lambda: self.runRequested.emit(self.nid))
        h.addWidget(self.heading); h.addWidget(self.status, 1); h.addWidget(self.open_btn); h.addWidget(self.pdf_btn); h.addWidget(self.build_btn)
        lay.addWidget(head)
        split = QSplitter(Qt.Horizontal); lay.addWidget(split, 1)
        left = QWidget(); form = QFormLayout(left); form.setContentsMargins(14, 12, 14, 12); form.setSpacing(8)
        self.title = QLineEdit(); self.title.setPlaceholderText("e.g. Monthly summary, June 2024")
        self.company = QLineEdit(); self.company.setPlaceholderText("shown as a letterhead")
        self.author = QLineEdit()
        self.notes = QPlainTextEdit(); self.notes.setPlaceholderText("A few lines shown at the top: what was measured, where, and why."); self.notes.setMaximumHeight(120)
        self.path = QLineEdit(); self.path.setPlaceholderText("report.html (saved next to the project)")
        self.pdf = QCheckBox("Also save a PDF")
        form.addRow("Title", self.title); form.addRow("Company", self.company); form.addRow("Prepared by", self.author)
        form.addRow("Introduction", self.notes); form.addRow("Save as", self.path); form.addRow("", self.pdf)
        for w in (self.title, self.company, self.author, self.path):
            w.editingFinished.connect(self._commit_text)
        self.notes.textChanged.connect(lambda: self._timer.start())
        self.pdf.toggled.connect(lambda on: self._set({"pdf": on}))
        split.addWidget(left)
        right = QWidget(); r = QVBoxLayout(right); r.setContentsMargins(14, 12, 14, 12); r.setSpacing(6)
        lab = QLabel("Layout"); lab.setObjectName("section")
        hint = QLabel("Charts and tables connected to this report appear in this order. Add text between them, and drag to reorder."); hint.setObjectName("muted"); hint.setWordWrap(True)
        self.blocks = QListWidget(); self.blocks.setDragDropMode(QAbstractItemView.InternalMove); self.blocks.setAlternatingRowColors(True)
        self.blocks.model().rowsMoved.connect(lambda *_: self._commit_blocks())
        self.blocks.itemDoubleClicked.connect(self._edit_block)
        btns = QHBoxLayout()
        add_h = QPushButton("Add heading"); add_h.clicked.connect(lambda: self._add_block("heading"))
        add_t = QPushButton("Add text"); add_t.clicked.connect(lambda: self._add_block("text"))
        rm = QPushButton("Remove"); rm.clicked.connect(self._remove_block)
        btns.addWidget(add_h); btns.addWidget(add_t); btns.addWidget(rm); btns.addStretch()
        r.addWidget(lab); r.addWidget(hint); r.addWidget(self.blocks, 1); r.addLayout(btns)
        tip = QLabel("To add a chart or table: open it and press <b>Add to report</b>, or connect it to this step in the Map."); tip.setObjectName("muted"); tip.setWordWrap(True)
        r.addWidget(tip)
        split.addWidget(right); split.setSizes([420, 520])
        doc.nodeChanged.connect(lambda nid: self.refill() if nid == self.nid and not self._timer.isActive() else None)
        doc.edgeAdded.connect(lambda e: self.refill() if e.target == self.nid else None)
        doc.edgeRemoved.connect(lambda e: self.refill() if e.target == self.nid else None)
        doc.nodeState.connect(lambda nid, st: self._refresh_status() if nid == self.nid else None)
        doc.statesChanged.connect(self._refresh_status)
        doc.runFinished.connect(lambda *_: self._refresh_status())
        doc.runStarted.connect(self._refresh_status)
        doc.reloaded.connect(lambda: self.set_node(None))

    def set_node(self, nid: str | None) -> None:
        if self._timer.isActive():
            self._commit_text()
        self.nid = nid if nid in self.doc.pipeline.nodes else None
        self.refill()

    def _params(self) -> dict:
        return self.doc.pipeline.nodes[self.nid].params if self.nid else {}

    def _set(self, changes: dict) -> None:
        if self.nid and not self._suppress:
            self.doc.set_params(self.nid, changes)

    def _items(self) -> list[str]:
        return list(self.doc.pipeline.inputs_of(self.nid).get("items") or []) if self.nid else []

    def refill(self) -> None:
        if self.nid is None:
            return
        p = self._params()
        self._suppress = True
        self.heading.setText(self.doc.pipeline.nodes[self.nid].title)
        self.title.setText(p.get("title") or ""); self.company.setText(p.get("company") or ""); self.author.setText(p.get("author") or "")
        if self.notes.toPlainText() != (p.get("notes") or ""):
            self.notes.setPlainText(p.get("notes") or "")
        self.path.setText(p.get("path") or ""); self.pdf.setChecked(bool(p.get("pdf", True)))
        items = self._items()
        blocks = list(p.get("blocks") or [])
        seen = {b.get("index") for b in blocks if b.get("type") not in ("text", "heading")}
        blocks = [b for b in blocks if b.get("type") in ("text", "heading") or (isinstance(b.get("index"), int) and 0 <= b["index"] < len(items))]
        blocks += [{"type": "item", "index": i} for i in range(len(items)) if i not in seen]
        self.blocks.clear()
        for b in blocks:
            if b.get("type") == "heading":
                it = QListWidgetItem(icon("text-h", T.muted, 16), b.get("text") or "(heading)")
            elif b.get("type") == "text":
                txt = (b.get("text") or "").strip().splitlines()
                it = QListWidgetItem(icon("text-aa", T.muted, 16), (txt[0][:80] if txt else "(text)"))
            else:
                n = self.doc.pipeline.nodes.get(items[b["index"]])
                it = QListWidgetItem(icon(node_icon_name(n.type) if n else "table", T.muted, 16), (n.title if n else "?") + (" (chart)" if n and n.type == "chart" else " (table)"))
            it.setData(Qt.UserRole, dict(b))
            self.blocks.addItem(it)
        self._suppress = False
        self._refresh_status()

    def _refresh_status(self) -> None:
        if self.nid is None or self.nid not in self.doc.pipeline.nodes:
            return
        st = self.doc.state(self.nid)
        rep = st.report or {}
        if st.status == "done" and rep.get("path"):
            txt = f"saved to {rep['path']}"
        elif st.status == "failed":
            txt = f"<span style='color:{T.danger}'>{st.error}</span>"
        elif st.status == "running":
            txt = "building…"
        else:
            n = len(self._items())
            txt = f"{n} item{'s' if n != 1 else ''} · not built yet" if n else "nothing connected yet"
        self.status.setText(status_dot(st.status, txt))
        self.open_btn.setVisible(st.status == "done" and bool(rep.get("path")))
        self.pdf_btn.setVisible(st.status == "done" and bool(rep.get("pdf")))
        self.build_btn.setEnabled(not self.doc.running and bool(self._items()))

    def _commit_text(self) -> None:
        if self.nid is None or self._suppress:
            return
        p = self._params()
        changes = {}
        for key, w in (("title", self.title), ("company", self.company), ("author", self.author), ("path", self.path)):
            if w.text() != (p.get(key) or ""):
                changes[key] = w.text()
        if self.notes.toPlainText() != (p.get("notes") or ""):
            changes["notes"] = self.notes.toPlainText()
        if changes:
            self._set(changes)
            if "title" in changes and changes["title"].strip():
                self.doc.rename(self.nid, changes["title"].strip())

    def _blocks_from_list(self) -> list[dict]:
        return [self.blocks.item(i).data(Qt.UserRole) for i in range(self.blocks.count())]

    def _commit_blocks(self) -> None:
        self._set({"blocks": self._blocks_from_list()})

    def _add_block(self, kind: str) -> None:
        text, ok = QInputDialog.getMultiLineText(self, "Heading" if kind == "heading" else "Text", "Text:") if kind == "text" else QInputDialog.getText(self, "Heading", "Heading:")
        if not ok or not text.strip():
            return
        blocks = self._blocks_from_list()
        row = self.blocks.currentRow()
        blocks.insert(row + 1 if row >= 0 else len(blocks), {"type": kind, "text": text.strip()})
        self._set({"blocks": blocks}); self.refill()

    def _edit_block(self, it: QListWidgetItem) -> None:
        b = it.data(Qt.UserRole)
        if b.get("type") == "heading":
            text, ok = QInputDialog.getText(self, "Heading", "Heading:", text=b.get("text") or "")
        elif b.get("type") == "text":
            text, ok = QInputDialog.getMultiLineText(self, "Text", "Text:", b.get("text") or "")
        else:
            return
        if ok:
            blocks = self._blocks_from_list(); blocks[self.blocks.row(it)] = {"type": b["type"], "text": text.strip()}
            self._set({"blocks": blocks}); self.refill()

    def _remove_block(self) -> None:
        row = self.blocks.currentRow()
        if row < 0:
            return
        b = self.blocks.item(row).data(Qt.UserRole)
        if b.get("type") in ("text", "heading"):
            blocks = self._blocks_from_list(); del blocks[row]
            self._set({"blocks": blocks}); self.refill()
        else:
            items = self._items()
            src = items[b["index"]]
            for e in list(self.doc.pipeline.edges):
                if e.target == self.nid and e.source == src:
                    self.doc.disconnect(e)

    def _open(self) -> None:
        p = (self.doc.state(self.nid).report or {}).get("path") if self.nid else None
        if p:
            open_external(p)

    def _open_pdf(self) -> None:
        p = (self.doc.state(self.nid).report or {}).get("pdf") if self.nid else None
        if p:
            open_external(p)
