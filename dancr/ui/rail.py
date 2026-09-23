"""Left rail: the project as a vertical tree.

Two modes, toggled in the header:

- **Flow** (default): the pipeline as a spanning tree. Every step appears once, under the step that
  feeds it (its first input), indented by depth with connector lines. A step with more than one input
  (combine, predict, report, workbook, stack) shows a ``N in`` chip and lists the other inputs as
  muted ``↳ port: step`` rows, so joins are visible without opening the map.
- **Type**: the older flat grouping by Tables / Charts / Reports.

Inputs stay pinned at the bottom. Clicking a step selects it; the window focuses the map on it.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal, QSize, QSettings
from PySide6.QtGui import QColor, QFont, QPen
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTreeWidget, QTreeWidgetItem,
                               QLabel, QStyledItemDelegate, QStyle, QToolButton)

from ..core import registry
from .document import Document
from .theme import T, category_color, STATUS_COLORS
from .icons import icon, node_icon_name

SECTIONS = [("tables", "Tables"), ("charts", "Charts"), ("reports", "Reports")]
VIEW_TYPES = {"chart"}
REPORT_TYPES = {"report", "workbook", "export"}

KIND_ROLE = Qt.UserRole          # "node" | "inputs" | "hint" | "ref" | None (a section header)
ID_ROLE = Qt.UserRole + 1
STATUS_ROLE = Qt.UserRole + 2
INPUTS_ROLE = Qt.UserRole + 3    # total inputs when more than one (drives the "N in" chip)


class RailDelegate(QStyledItemDelegate):
    """Icon, title, an "N in" chip for joins and a status dot; small-caps section headers."""

    def sizeHint(self, option, index) -> QSize:
        return QSize(option.rect.width(), 26 if not index.data(KIND_ROLE) else 22)

    def paint(self, painter, option, index) -> None:
        painter.save()
        r = option.rect
        kind = index.data(KIND_ROLE)
        if kind == "hint":
            painter.setPen(QColor(T.faint)); painter.setFont(option.font)
            painter.drawText(r.adjusted(12, 0, -8, 0), Qt.AlignLeft | Qt.AlignVCenter, str(index.data(Qt.DisplayRole)))
            painter.restore(); return
        if not kind:                                    # a section header
            painter.setPen(QColor(T.muted))
            f = QFont(option.font); f.setBold(True); f.setPointSizeF(f.pointSizeF() - 1.5); painter.setFont(f)
            painter.drawText(r.adjusted(10, 0, -4, -2), Qt.AlignLeft | Qt.AlignBottom, str(index.data(Qt.DisplayRole)).upper())
            painter.restore(); return
        if option.state & QStyle.State_Selected:
            painter.fillRect(r, QColor(T.hover))
            painter.fillRect(r.adjusted(0, 0, -r.width() + 3, 0), QColor(T.accent))
        elif option.state & QStyle.State_MouseOver:
            painter.fillRect(r, QColor(T.hover))
        if kind == "ref":                               # an extra input of a joining step
            painter.setFont(option.font); painter.setPen(QColor(T.faint))
            text = "↳ " + str(index.data(Qt.DisplayRole))
            painter.drawText(r.adjusted(30, 0, -8, 0), Qt.AlignLeft | Qt.AlignVCenter,
                             painter.fontMetrics().elidedText(text, Qt.ElideRight, r.width() - 40))
            painter.restore(); return
        ic = index.data(Qt.DecorationRole)
        if ic is not None:
            ic.paint(painter, r.adjusted(8, 3, -r.width() + 24, -3))
        total = index.data(INPUTS_ROLE) or 0
        right = 46 if total > 1 else 26
        painter.setFont(option.font); painter.setPen(QColor(T.text))
        text_rect = r.adjusted(30, 0, -right, 0)
        painter.drawText(text_rect, Qt.AlignLeft | Qt.AlignVCenter,
                         painter.fontMetrics().elidedText(str(index.data(Qt.DisplayRole)), Qt.ElideRight, text_rect.width()))
        if total > 1:
            f = QFont(option.font); f.setPointSizeF(f.pointSizeF() - 1.5); painter.setFont(f)
            painter.setPen(QColor(T.muted))
            painter.drawText(r.adjusted(0, 0, -24, 0), Qt.AlignRight | Qt.AlignVCenter, f"{total} in")
        status = index.data(STATUS_ROLE)
        if status:
            painter.setPen(Qt.NoPen); painter.setBrush(QColor(STATUS_COLORS.get(status, T.faint)))
            painter.drawEllipse(r.right() - 14, r.center().y() - 3, 6, 6)
        painter.restore()


class RailTree(QTreeWidget):
    """A tree view that draws the usual connector lines (├─ └─) so depth is easy to follow."""

    def drawBranches(self, painter, rect, index) -> None:
        model = self.model()
        ancestors = []
        p = index
        while p.isValid():
            ancestors.append(p); p = p.parent()
        ancestors.reverse()
        depth = len(ancestors) - 1
        if depth >= 1:
            ind = self.indentation()
            top, bottom = rect.top(), rect.bottom()
            mid = rect.top() + rect.height() // 2
            painter.save()
            painter.setPen(QPen(QColor(T.border_soft), 1))
            for level in range(depth):
                x = level * ind + ind // 2
                if level == depth - 1:
                    last = index.row() == model.rowCount(index.parent()) - 1
                    painter.drawLine(x, top, x, mid if last else bottom)
                    painter.drawLine(x, mid, rect.right(), mid)
                else:
                    anc = ancestors[level]
                    if anc.row() != model.rowCount(anc.parent()) - 1:
                        painter.drawLine(x, top, x, bottom)
            painter.restore()
        super().drawBranches(painter, rect, index)


class Rail(QWidget):
    selected = Signal(str, object)     # kind ("node" | "inputs" | "none"), id
    deleteRequested = Signal()

    MODE_SETTING = "rail_mode"

    def __init__(self, doc: Document, parent=None) -> None:
        super().__init__(parent)
        self.doc = doc
        self.setMinimumWidth(190); self.setMaximumWidth(340)
        self.mode = str(QSettings().value(self.MODE_SETTING, "flow") or "flow")
        if self.mode not in ("flow", "type"):
            self.mode = "flow"
        self._items: dict[str, QTreeWidgetItem] = {}
        self._inputs_item: QTreeWidgetItem | None = None
        self._collapsed: set[str] = set()
        self._suppress = False
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(0)
        head = QWidget(); h = QHBoxLayout(head); h.setContentsMargins(12, 8, 6, 6)
        self.title = QLabel("Project"); self.title.setObjectName("section")
        self.mode_btn = QToolButton(); self.mode_btn.setObjectName("quiet")
        self.mode_btn.setToolTip("Show the project as a connection tree, or grouped by kind")
        self.mode_btn.clicked.connect(self._toggle_mode)
        h.addWidget(self.title); h.addStretch(); h.addWidget(self.mode_btn)
        lay.addWidget(head)
        self.tree = RailTree()
        self.tree.setHeaderHidden(True); self.tree.setColumnCount(1)
        self.tree.setItemDelegate(RailDelegate(self.tree))
        self.tree.setIndentation(16); self.tree.setRootIsDecorated(True)
        self.tree.setUniformRowHeights(False); self.tree.setMouseTracking(True)
        self.tree.setSelectionMode(QTreeWidget.SingleSelection)
        self.tree.setEditTriggers(QTreeWidget.NoEditTriggers)
        self.tree.setExpandsOnDoubleClick(False)
        self.tree.setStyleSheet(f"QTreeWidget {{ background: {T.bg}; border: none; border-right: 1px solid {T.border}; }}"
                                f"QTreeWidget::item {{ padding: 0; }}")
        lay.addWidget(self.tree, 1)
        self.tree.itemSelectionChanged.connect(self._on_select)
        self.tree.itemClicked.connect(lambda *_: self._on_select())   # re-clicking the current step re-focuses it
        self.tree.itemExpanded.connect(lambda it: self._on_expanded(it, True))
        self.tree.itemCollapsed.connect(lambda it: self._on_expanded(it, False))
        for sig in (doc.nodeAdded, doc.nodeRemoved, doc.nodeChanged, doc.reloaded, doc.inputsChanged,
                    doc.edgeAdded, doc.edgeRemoved):
            sig.connect(lambda *_: self.refill())
        doc.statesChanged.connect(self.refill_status)
        doc.nodeState.connect(lambda *_: self.refill_status())
        self._update_mode_btn()
        self.refill()

    # ------------------------------------------------------------ building
    def _section_items(self) -> dict[str, list]:
        p = self.doc.pipeline
        tables, charts, reports = [], [], []
        for n in p.nodes.values():
            if n.type in VIEW_TYPES:
                charts.append(n)
            elif n.type in REPORT_TYPES:
                reports.append(n)
            else:
                tables.append(n)
        return {"tables": tables, "charts": charts, "reports": reports}

    def _header(self, text: str) -> QTreeWidgetItem:
        it = QTreeWidgetItem(self.tree)
        it.setText(0, text); it.setFlags(Qt.NoItemFlags)
        return it

    def _hint(self, text: str) -> QTreeWidgetItem:
        it = QTreeWidgetItem(self.tree)
        it.setText(0, text); it.setData(0, KIND_ROLE, "hint"); it.setFlags(Qt.NoItemFlags)
        return it

    def _node_item(self, parent_item: QTreeWidgetItem | None, nid: str, total_inputs: int) -> QTreeWidgetItem:
        node = self.doc.pipeline.nodes[nid]
        nt = registry.get(node.type)
        st = self.doc.state(nid)
        it = QTreeWidgetItem(parent_item) if parent_item is not None else QTreeWidgetItem(self.tree)
        it.setText(0, node.title)
        it.setIcon(0, icon(node_icon_name(node.type), category_color(nt.category).name(), 16))
        it.setData(0, KIND_ROLE, "node"); it.setData(0, ID_ROLE, nid)
        it.setData(0, STATUS_ROLE, st.status); it.setData(0, INPUTS_ROLE, total_inputs)
        it.setToolTip(0, f"{nt.label}: {nt.summarize(node.params)}")
        it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        self._items[nid] = it
        return it

    def _ref_item(self, parent_item: QTreeWidgetItem, edge) -> None:
        target = self.doc.pipeline.nodes.get(edge.target)
        nt = registry.get(target.type) if target else None
        label = next((i.label for i in (nt.inputs if nt else []) if i.name == edge.port), edge.port)
        src = self.doc.pipeline.nodes.get(edge.source)
        title = src.title if src else edge.source
        it = QTreeWidgetItem(parent_item)
        it.setText(0, f"{label}: {title}")
        it.setData(0, KIND_ROLE, "ref"); it.setData(0, ID_ROLE, edge.source)
        it.setToolTip(0, f"Input “{label}” comes from {title}")
        it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)

    def _build_flow(self) -> None:
        p = self.doc.pipeline
        if not p.nodes:
            self._hint("no steps yet — open a data file")
            return
        order = {nid: i for i, nid in enumerate(p.topological_order())}
        incoming: dict[str, list] = {nid: [] for nid in p.nodes}
        for e in p.edges:
            if e.target in incoming:
                incoming[e.target].append(e)
        primary: dict[str, str] = {}
        extra: dict[str, list] = {}
        for nid, edges in incoming.items():
            if edges:
                primary[nid] = edges[0].source
                extra[nid] = edges[1:]
        children: dict[str, list[str]] = {}
        for nid, par in primary.items():
            children.setdefault(par, []).append(nid)
        for lst in children.values():
            lst.sort(key=lambda n: order.get(n, 0))
        roots = sorted((nid for nid in p.nodes if nid not in primary), key=lambda n: order.get(n, 0))
        sources = [n for n in roots if registry.get(p.nodes[n].type).kind == "source"]
        others = [n for n in roots if registry.get(p.nodes[n].type).kind != "source"]
        for nid in sources:
            self._add_subtree(None, nid, children, extra)
        if others:
            self._header("Not connected")
            for nid in others:
                self._add_subtree(None, nid, children, extra)

    def _add_subtree(self, parent_item: QTreeWidgetItem | None, nid: str, children: dict, extra: dict) -> None:
        total = 1 + len(extra.get(nid, []))
        item = self._node_item(parent_item, nid, total if total > 1 else 0)
        for e in extra.get(nid, []):
            self._ref_item(item, e)
        for child in children.get(nid, []):
            self._add_subtree(item, child, children, extra)
        item.setExpanded(nid not in self._collapsed)

    def _build_type(self) -> None:
        secs = self._section_items()
        hints = {"tables": "none yet — open a data file", "charts": "none yet — right-click a column", "reports": "none yet"}
        for key, label in SECTIONS:
            self._header(label)
            for n in secs[key]:
                self._node_item(None, n.id, 0)
            if not secs[key]:
                self._hint(hints[key])

    def _build_inputs(self) -> None:
        self._header("Inputs")
        it = QTreeWidgetItem(self.tree)
        it.setText(0, f"{len(self.doc.pipeline.inputs)} value{'s' if len(self.doc.pipeline.inputs) != 1 else ''}")
        it.setIcon(0, icon("gear", T.muted, 16))
        it.setData(0, KIND_ROLE, "inputs"); it.setData(0, ID_ROLE, "inputs")
        it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        self._inputs_item = it

    def refill(self) -> None:
        cur = self.current()
        self._suppress = True
        self.tree.clear()
        self._items.clear(); self._inputs_item = None
        if self.mode == "type":
            self._build_type()
        else:
            self._build_flow()
        self._build_inputs()
        self._suppress = False
        if cur:
            self.select(*cur, emit=False)

    def refill_status(self) -> None:
        for nid, it in self._items.items():
            it.setData(0, STATUS_ROLE, self.doc.state(nid).status)
        self.tree.viewport().update()

    # ------------------------------------------------------------ selection
    def current(self) -> tuple[str, object] | None:
        it = self.tree.currentItem()
        if it is None:
            return None
        kind = it.data(0, KIND_ROLE)
        ident = it.data(0, ID_ROLE)
        if kind == "ref":
            return ("node", ident)
        if kind in ("node", "inputs"):
            return (kind, ident)
        return None

    def select(self, kind: str, ident: object, emit: bool = True) -> None:
        self._suppress = not emit
        item = self._inputs_item if kind == "inputs" else self._items.get(str(ident))
        if item is not None:
            p = item.parent()
            while p is not None:
                p.setExpanded(True)
                if p.data(0, KIND_ROLE) == "node":
                    self._collapsed.discard(p.data(0, ID_ROLE))
                p = p.parent()
            self.tree.setCurrentItem(item)
            self.tree.scrollToItem(item)
        else:
            self.tree.clearSelection()
        self._suppress = False

    def _on_select(self) -> None:
        if self._suppress:
            return
        cur = self.current()
        self.selected.emit(*(cur if cur else ("none", None)))

    def _on_expanded(self, it: QTreeWidgetItem, expanded: bool) -> None:
        if self._suppress or it.data(0, KIND_ROLE) != "node":
            return
        nid = it.data(0, ID_ROLE)
        if expanded:
            self._collapsed.discard(nid)
        else:
            self._collapsed.add(nid)

    def keyPressEvent(self, e) -> None:
        if e.key() in (Qt.Key_Delete, Qt.Key_Backspace) and self.current() and self.current()[0] == "node":
            self.deleteRequested.emit(); return
        super().keyPressEvent(e)

    # ------------------------------------------------------------ mode
    def _toggle_mode(self) -> None:
        self.mode = "type" if self.mode == "flow" else "flow"
        QSettings().setValue(self.MODE_SETTING, self.mode)
        self._update_mode_btn()
        self.refill()

    def _update_mode_btn(self) -> None:
        self.mode_btn.setText("Flow" if self.mode == "flow" else "Type")
        self.mode_btn.setToolTip("Showing connections. Click to group by kind." if self.mode == "flow"
                                 else "Grouped by kind. Click to show connections.")
