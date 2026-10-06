"""Small dialogs and overlays used by the main window."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (QFrame, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QDialog, QListWidget, QListWidgetItem,
                               QDialogButtonBox, QLineEdit, QPlainTextEdit, QFormLayout, QTreeWidget, QTreeWidgetItem,
                               QFileDialog, QScrollArea)

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
        self.setAttribute(Qt.WA_DeleteOnClose)     # freed when closed, so reopening never accumulates dialogs
        self.setWindowTitle("Earlier versions"); self.resize(520, 380)
        lay = QVBoxLayout(self)
        lab = QLabel("DANCR keeps a copy of the project every time it is saved. Pick one to go back to it (your current version is kept too)."); lab.setWordWrap(True)
        self.list = QListWidget()
        versions = doc.versions()
        for p in versions:
            m = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})", p.name)
            when = f"{m.group(1)}  {m.group(2)}:{m.group(3)}:{m.group(4)}" if m else p.stem
            stamp = when + ("   ·   autosave" if p.name.endswith(".auto.json") else "")
            it = QListWidgetItem(icon("clock-counter-clockwise", T.muted, 16), stamp); it.setData(Qt.UserRole, str(p)); self.list.addItem(it)
        if versions:                                  # step counts read on a worker: the files may be on a slow drive
            self._counts = Serial(self, waits_for_run=False)
            self._counts.submit(lambda: [_step_count(p) for p in versions], self._show_counts)
        if not self.list.count():
            self.list.addItem("No earlier versions yet. They appear after you save.")
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


class RelationsDialog(QDialog):
    """How the tables fit together, as DANCR worked it out from the data. Each relation can be built into the
    matching step (a join, a stack, a time alignment) on the canvas — only when you say so."""

    def __init__(self, parent, relations, on_build=None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_DeleteOnClose)
        from ..core.profile import relation_phrase
        self.setWindowTitle("How these tables relate"); self.resize(560, 440)
        self._on_build = on_build
        lay = QVBoxLayout(self)
        intro = QLabel("DANCR worked these out from the data. Build one to add the matching step to the canvas.")
        intro.setWordWrap(True); lay.addWidget(intro)
        body = QWidget(); bl = QVBoxLayout(body); bl.setContentsMargins(0, 4, 0, 4); bl.setSpacing(6)
        for r in relations:
            row = QFrame(); row.setObjectName("panel")
            row.setStyleSheet(f"QFrame#panel {{ background: {T.panel}; border: 1px solid {T.border}; border-radius: 6px; }}")
            h = QHBoxLayout(row); h.setContentsMargins(10, 6, 8, 6)
            lab = QLabel(relation_phrase(r)); lab.setWordWrap(True)
            btn = QPushButton("Build")
            btn.clicked.connect(lambda _=False, rel=r: self._build(rel))
            h.addWidget(lab, 1); h.addWidget(btn)
            bl.addWidget(row)
        if not relations:
            bl.addWidget(QLabel("No relations were found between the tables."))
        bl.addStretch()
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QFrame.NoFrame); scroll.setWidget(body)
        lay.addWidget(scroll, 1)
        bb = QDialogButtonBox(QDialogButtonBox.Close); bb.rejected.connect(self.reject); bb.accepted.connect(self.accept)
        lay.addWidget(bb)

    def _build(self, relation) -> None:
        if self._on_build:
            self._on_build(relation)
        self.accept()


class CatalogDialog(QDialog):
    """Every DANCR project under a folder, with each project's datasets below it. Double-click a dataset or a
    project to open the project. The scan runs on a worker so a big folder never freezes the window."""

    openProject = Signal(str)

    def __init__(self, parent, root: Path | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_DeleteOnClose)     # also cancels the scan worker via the Serial owner
        self.setWindowTitle("Project catalog"); self.resize(820, 560)
        self.root = root
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        self.label = QLabel("Choose a folder of DANCR projects."); self.label.setObjectName("muted")
        self.filter = QLineEdit(); self.filter.setPlaceholderText("Filter…"); self.filter.textChanged.connect(self._apply_filter)
        choose = QPushButton("Choose folder…"); choose.clicked.connect(self._choose)
        top.addWidget(self.label, 1); top.addWidget(self.filter); top.addWidget(choose)
        lay.addLayout(top)
        self.tree = QTreeWidget(); self.tree.setHeaderLabels(["Project / dataset", "What it is"])
        self.tree.setColumnWidth(0, 320); self.tree.itemDoubleClicked.connect(self._open)
        lay.addWidget(self.tree, 1)
        bb = QDialogButtonBox(QDialogButtonBox.Close); bb.rejected.connect(self.reject); bb.accepted.connect(self.accept)
        lay.addWidget(bb)
        if root:
            QTimer.singleShot(0, lambda: self.load(root))

    def _choose(self) -> None:
        start = str(self.root or Path.home())
        d = QFileDialog.getExistingDirectory(self, "Choose a folder of projects", start)
        if d:
            self.load(Path(d))

    def load(self, root) -> None:
        self.root = Path(root)
        self.label.setText(f"Scanning {self.root}…")
        from ..headless import build_catalog
        if getattr(self, "_serial", None) is not None:
            self._serial.cancel()                  # a previous scan must not keep a thread busy after a new one
        self._serial = Serial(self, waits_for_run=False)
        self._serial.submit(lambda: build_catalog(self.root, recursive=True), self._show,
                            lambda m: self.label.setText(f"Could not scan: {m}"))

    def _show(self, cat: dict) -> None:
        self.tree.clear()
        self.label.setText(f"{cat['count']} project(s) under {cat['root']}")
        for proj in cat["projects"]:
            top = QTreeWidgetItem([proj["name"], proj["file"]])
            top.setData(0, Qt.UserRole, proj["file"])
            for d in proj.get("datasets", []):
                child = QTreeWidgetItem([f"[{d['id']}] {d['title']}", (d.get("text") or "")[:140]])
                child.setToolTip(1, d.get("text") or "")
                top.addChild(child)
            self.tree.addTopLevelItem(top)
        for f, why in (cat.get("skipped") or {}).items():
            self.tree.addTopLevelItem(QTreeWidgetItem(["!", f"{f}: {why}"]))
        self.tree.expandToDepth(0)

    def _apply_filter(self, text: str) -> None:
        text = (text or "").lower()
        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            match = text in top.text(0).lower() or text in top.text(1).lower()
            for j in range(top.childCount()):
                c = top.child(j)
                cm = text in c.text(0).lower() or text in c.text(1).lower()
                c.setHidden(bool(text) and not cm and not match)
                match = match or cm
            top.setHidden(bool(text) and not match)

    def _open(self, item: QTreeWidgetItem, _col: int) -> None:
        path = item.data(0, Qt.UserRole)
        if path is None and item.parent() is not None:
            path = item.parent().data(0, Qt.UserRole)
        if path:
            self.openProject.emit(str(path))


class DatasetDialog(QDialog):
    """Who made the data, under what license, how to cite it — the header of a FAIR record. Saved with the project
    (in ``meta["dataset"]``) and used by every metadata export (schema.org, Frictionless, run manifest, RO-Crate)."""

    FIELDS = [
        ("title", "Title", "What the dataset is called"),
        ("creator", "Creator", "A person or team. A name, or name <email> for more detail"),
        ("contact", "Contact", "Who to ask about it"),
        ("publisher", "Publisher / institution", ""),
        ("license", "License", "An SPDX id (CC-BY-4.0) or a URL"),
        ("version", "Version", ""),
        ("identifier", "Identifier", "A DOI, accession or local id"),
        ("citation", "How to cite it", ""),
        ("language", "Language", "e.g. en"),
        ("keywords", "Keywords", "Comma-separated"),
    ]

    def __init__(self, parent, meta: dict) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle("Dataset details"); self.resize(560, 460)
        lay = QVBoxLayout(self)
        intro = QLabel("These travel with the project and appear in every metadata export. They are not sent anywhere by themselves.")
        intro.setWordWrap(True); lay.addWidget(intro)
        form = QFormLayout(); self.edits: dict[str, QLineEdit] = {}
        for key, label, tip in self.FIELDS:
            e = QLineEdit(); e.setPlaceholderText(tip)
            val = meta.get(key, "")
            e.setText(", ".join(str(x) for x in val) if isinstance(val, list) else str(val or ""))
            self.edits[key] = e; form.addRow(label, e)
        self.description = QPlainTextEdit(); self.description.setPlainText(str(meta.get("description") or ""))
        self.description.setPlaceholderText("A short paragraph about what the data is and what it is for")
        form.addRow("Description", self.description)
        lay.addLayout(form)
        bb = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def fields(self) -> dict:
        out: dict = {}
        for key, _, _ in self.FIELDS:
            text = self.edits[key].text().strip()
            if not text:
                continue
            if key == "keywords":
                out[key] = [s.strip() for s in text.replace(";", ",").split(",") if s.strip()]
            elif key == "creator" and "<" in text and text.endswith(">"):
                name, email = text[:-1].split("<", 1)
                out[key] = {"name": name.strip(), "email": email.strip()}
            else:
                out[key] = text
        desc = self.description.toPlainText().strip()
        if desc:
            out["description"] = desc
        return out
