"""The Sources tray: everything you have brought in, side by side, however differently formatted.

A thin filmstrip of chips at the bottom of the canvas — one per source table (a file, a folder, a URL, a
database, typed data) with how many rows it holds and whether it is ready. This is the answer to "a pile of
disparate spreadsheets": at a glance you see all of them, and you can click one to work on it. Collapsible, and
hidden entirely until there is a source."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QFrame, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QToolButton, QScrollArea,
                               QSizePolicy)

from ..core import registry
from .common import listen, status_dot
from .theme import T, category_color
from .icons import icon, node_icon_name


class SourceChip(QFrame):
    clicked = Signal(str)             # node id

    def __init__(self, doc, nid: str, parent=None) -> None:
        super().__init__(parent)
        self.doc, self.nid = doc, nid
        self.setObjectName("sourceChip")
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet(
            f"QFrame#sourceChip {{ background: {T.panel}; border: 1px solid {T.border}; border-radius: 6px; }}"
            f"QFrame#sourceChip:hover {{ border-color: {T.accent}; }}")
        self.setFixedHeight(46)
        node = doc.pipeline.nodes[nid]
        nt = registry.get(node.type)
        h = QHBoxLayout(self); h.setContentsMargins(9, 5, 11, 5); h.setSpacing(7)
        ic = QLabel(); ic.setPixmap(icon(node_icon_name(nt.key), category_color(nt.category).name(), 16).pixmap(16, 16))
        ic.setFixedWidth(18)
        col = QVBoxLayout(); col.setContentsMargins(0, 0, 0, 0); col.setSpacing(0)
        title = QLabel(); title.setStyleSheet("font-weight: 600;")
        title.setMaximumWidth(170); self._elide(title, node.title)
        self.sub = QLabel(); self.sub.setObjectName("faint")
        col.addWidget(title); col.addWidget(self.sub)
        h.addWidget(ic); h.addLayout(col)
        self.refresh()

    def _elide(self, label: QLabel, text: str) -> None:
        fm = label.fontMetrics()
        label.setText(fm.elidedText(text, Qt.ElideRight, 168))

    def refresh(self) -> None:
        node = self.doc.pipeline.nodes.get(self.nid)
        if node is None:
            return
        st = self.doc.cached_state(self.nid)          # never doc.state(): it may read the source file on the GUI thread
        status = st.status if st is not None else "idle"
        if st is not None and st.status == "done" and st.rows is not None:
            self.sub.setText(f"{st.rows:,} rows" + (f" × {len(st.columns)}" if st.columns else ""))
            self.sub.setStyleSheet(f"color: {T.muted};")
        else:
            state = {"idle": "not read yet", "stale": "changed", "running": "reading…", "failed": "failed"}.get(status, status)
            self.sub.setText(state)
            self.sub.setStyleSheet(f"color: {T.danger if status == 'failed' else T.faint};")

    def mouseReleaseEvent(self, e) -> None:  # noqa: N802
        if e.button() == Qt.LeftButton and self.rect().contains(e.position().toPoint()):
            self.clicked.emit(self.nid)
        super().mouseReleaseEvent(e)


class SourcesTray(QFrame):
    addRequested = Signal()
    selected = Signal(str)
    relateRequested = Signal()

    def __init__(self, doc, parent=None) -> None:
        super().__init__(parent)
        self.doc = doc
        self.setObjectName("sources")
        self.setStyleSheet(f"QFrame#sources {{ background: {T.bg}; border-top: 1px solid {T.border}; }}")
        v = QVBoxLayout(self); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(0)
        head = QFrame()
        h = QHBoxLayout(head); h.setContentsMargins(10, 2, 8, 2); h.setSpacing(6)
        self.heading = QLabel("Sources"); self.heading.setObjectName("section")
        self.count = QLabel(""); self.count.setObjectName("faint")
        self.relate_btn = QToolButton(); self.relate_btn.setObjectName("quiet"); self.relate_btn.setIcon(icon("arrows-merge", T.muted, 14))
        self.relate_btn.setText("Relate"); self.relate_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.relate_btn.setToolTip("How these tables relate — see the links, stacks and alignments, and build one")
        self.relate_btn.clicked.connect(self.relateRequested.emit); self.relate_btn.setEnabled(False)
        self.add_btn = QToolButton(); self.add_btn.setObjectName("quiet"); self.add_btn.setIcon(icon("plus", T.muted, 14))
        self.add_btn.setText("Add"); self.add_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.add_btn.setToolTip("Bring in a file, folder, URL or database"); self.add_btn.clicked.connect(self.addRequested.emit)
        self.toggle = QToolButton(); self.toggle.setObjectName("quiet"); self.toggle.setCheckable(True); self.toggle.setChecked(True)
        self.toggle.setIcon(icon("list-bullets", T.muted, 14)); self.toggle.setToolTip("Show or hide the sources")
        self._auto = True                 # auto-collapse until the person opens it by hand: the strip matters at 2+
        self._syncing = False
        h.addWidget(self.heading); h.addWidget(self.count); h.addStretch(1)
        h.addWidget(self.relate_btn); h.addWidget(self.add_btn); h.addWidget(self.toggle)
        v.addWidget(head)
        self.scroll = QScrollArea(); self.scroll.setFrameShape(QFrame.NoFrame); self.scroll.setWidgetResizable(True)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff); self.scroll.setFixedHeight(58)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.row = QWidget(); self.row_lay = QHBoxLayout(self.row)
        self.row_lay.setContentsMargins(10, 4, 10, 6); self.row_lay.setSpacing(8); self.row_lay.addStretch(1)
        self.scroll.setWidget(self.row)
        v.addWidget(self.scroll)
        self.toggle.toggled.connect(self._on_toggle)
        self._chips: dict[str, SourceChip] = {}
        for sig in (doc.nodeAdded, doc.nodeRemoved, doc.reloaded):
            listen(self, sig, lambda *_: self.refresh())
        for sig in (doc.statesChanged, doc.nodeState):
            listen(self, sig, lambda *_: self._refresh_chips())
        self.refresh()

    def source_ids(self) -> list[str]:
        return [nid for nid, n in self.doc.pipeline.nodes.items() if registry.get(n.type).kind == "source"]

    def _on_toggle(self, on: bool) -> None:
        self.scroll.setVisible(on)
        if not self._syncing:
            self._auto = False            # the person opened or closed it: stop auto-setting it from the count

    def refresh(self) -> None:
        ids = set(self.source_ids())
        for nid in list(self._chips):
            if nid not in ids:
                chip = self._chips.pop(nid)
                chip.setParent(None); chip.deleteLater()
        for nid in self.source_ids():
            if nid not in self._chips:
                chip = SourceChip(self.doc, nid)
                chip.clicked.connect(self.selected.emit)
                self._chips[nid] = chip
                self.row_lay.insertWidget(self.row_lay.count() - 1, chip)
        self._refresh_chips()
        n = len(self._chips)
        self.count.setText(f"· {n}" if n else "")
        self.relate_btn.setEnabled(n >= 2)
        if self._auto:
            # collapsed until there are two or more sources to compare; the header (Add, Relate) stays visible
            self._syncing = True
            self.toggle.setChecked(n >= 2)
            self._syncing = False
        self.scroll.setVisible(self.toggle.isChecked())
        self.setVisible(n > 0)

    def _refresh_chips(self) -> None:
        for chip in self._chips.values():
            chip.refresh()
