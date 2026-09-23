"""The node canvas."""
from __future__ import annotations

import math
from pathlib import Path

from PySide6.QtCore import Qt, QPointF, QRectF, Signal, QPoint, QSize
from PySide6.QtGui import QColor, QPen, QBrush, QPainter, QPainterPath, QFont, QFontMetrics, QPolygonF, QPainterPathStroker
from PySide6.QtWidgets import (QGraphicsView, QGraphicsScene, QGraphicsObject, QGraphicsItem, QGraphicsPathItem, QMenu,
                               QWidget, QGraphicsSceneMouseEvent, QInputDialog, QToolButton, QHBoxLayout, QLabel)

from ..core import registry, PipelineError
from ..core.model import Edge, Node, Note
from ..core.nodes.load import CSV_EXT, EXCEL_EXT, PARQUET_EXT
from .document import Document
from .theme import T, category_color
from .icons import paint as paint_icon, icon, node_icon_name

NODE_W, NODE_H = 236, 78
PORT_R = 6
PORT_HIT = 18
NODE_MIME = "application/x-dancr-node-type"
DATA_EXT = CSV_EXT | EXCEL_EXT | PARQUET_EXT


def _elide(text: str, font: QFont, width: int) -> str:
    return QFontMetrics(font).elidedText(text, Qt.ElideRight, int(width))


def _font(size: float, bold: bool = False) -> QFont:
    f = QFont(T.ui_font); f.setPointSizeF(size); f.setBold(bold)
    return f


class PortItem(QGraphicsObject):
    def __init__(self, node_item: "NodeItem", name: str, label: str, is_output: bool, index: int, count: int, optional: bool = False) -> None:
        super().__init__(node_item)
        self.node_item = node_item
        self.name, self.label, self.is_output = name, label, is_output
        self.index, self.count, self.optional = index, count, optional
        self.setAcceptHoverEvents(True)
        self.setZValue(2)
        self.hot = False
        self.needs = False        # required input with nothing connected
        self.setCursor(Qt.CrossCursor)
        self.setToolTip("Drag from here to another step's input" if is_output else f"Input: {label}. Drag a step's output here.")
        x = NODE_W if is_output else 0
        y = NODE_H / 2 if count <= 1 else NODE_H * (index + 1) / (count + 1)
        self.setPos(x, y)

    def boundingRect(self) -> QRectF:
        return QRectF(-PORT_HIT - 40, -PORT_HIT, 2 * PORT_HIT + 40, 2 * PORT_HIT)

    def shape(self) -> QPainterPath:
        p = QPainterPath()
        p.addEllipse(QRectF(-PORT_HIT, -PORT_HIT, 2 * PORT_HIT, 2 * PORT_HIT))
        return p

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.Antialiasing)
        r = PORT_R + (2 if self.hot else 0)
        if self.needs and not self.is_output:
            painter.setPen(QPen(QColor(T.warn), 2))
            painter.setBrush(QBrush(QColor(T.node)))
        else:
            painter.setPen(QPen(QColor(T.node), 2))
            painter.setBrush(QBrush(QColor(T.select if self.hot else (T.accent if self.is_output else T.faint))))
        painter.drawEllipse(QPointF(0, 0), r, r)
        if self.count > 1 and not self.is_output:
            painter.setPen(QPen(QColor(T.muted)))
            painter.setFont(_font(7.5))
            painter.drawText(QRectF(-PORT_HIT - 40, -8, PORT_HIT + 40 - PORT_R - 5, 16), Qt.AlignRight | Qt.AlignVCenter,
                             {0: "1st", 1: "2nd", 2: "3rd"}.get(self.index, f"{self.index + 1}th"))

    def hoverEnterEvent(self, e) -> None:
        self.hot = True; self.update()

    def hoverLeaveEvent(self, e) -> None:
        self.hot = False; self.update()

    def mousePressEvent(self, e: QGraphicsSceneMouseEvent) -> None:
        if e.button() == Qt.LeftButton:
            self.scene().begin_connection(self); e.accept()
        else:
            super().mousePressEvent(e)

    def mouseMoveEvent(self, e: QGraphicsSceneMouseEvent) -> None:
        self.scene().drag_connection(e.scenePos()); e.accept()

    def mouseReleaseEvent(self, e: QGraphicsSceneMouseEvent) -> None:
        self.scene().end_connection(e.scenePos()); e.accept()


class PlusButton(QGraphicsObject):
    """The + that appears next to the selected step's output: adds a connected step."""

    def __init__(self, node_item: "NodeItem") -> None:
        super().__init__(node_item)
        self.node_item = node_item
        self.setPos(NODE_W + 22, NODE_H / 2)
        self.setZValue(3)
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("Add a step after this one")
        self.hot = False
        self.hide()

    def boundingRect(self) -> QRectF:
        return QRectF(-11, -11, 22, 22)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor(T.accent), 1.5))
        painter.setBrush(QBrush(QColor(T.accent if self.hot else T.node)))
        painter.drawEllipse(QPointF(0, 0), 10, 10)
        painter.setPen(QPen(QColor("#ffffff" if self.hot else T.accent), 1.8))
        painter.drawLine(QPointF(-4.5, 0), QPointF(4.5, 0)); painter.drawLine(QPointF(0, -4.5), QPointF(0, 4.5))

    def hoverEnterEvent(self, e) -> None:
        self.hot = True; self.update()

    def hoverLeaveEvent(self, e) -> None:
        self.hot = False; self.update()

    def mousePressEvent(self, e) -> None:
        e.accept()

    def mouseReleaseEvent(self, e) -> None:
        self.scene().addAfterRequested.emit(self.node_item.node_id, e.screenPos())
        e.accept()


class NodeItem(QGraphicsObject):
    def __init__(self, scene: "CanvasScene", node: Node) -> None:
        super().__init__()
        self.canvas = scene
        self.node_id = node.id
        self.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemSendsGeometryChanges)
        self.setAcceptHoverEvents(True)
        self.setZValue(1)
        self.setPos(node.x, node.y)
        nt = registry.get(node.type)
        self.inputs = [PortItem(self, i.name, i.label, False, k, len(nt.inputs), i.optional) for k, i in enumerate(nt.inputs)]
        self.output = PortItem(self, "out", "Output", True, 0, 1)
        self.plus = PlusButton(self)
        self._hover = False
        self.status = "idle"
        self.rows: int | None = None
        self.error: str | None = None
        self.problem: str | None = None

    @property
    def node(self) -> Node:
        return self.canvas.doc.pipeline.nodes[self.node_id]

    def port(self, name: str) -> PortItem:
        for p in self.inputs:
            if p.name == name:
                return p
        return self.inputs[0]

    def boundingRect(self) -> QRectF:
        return QRectF(-PORT_HIT - 40, -6, NODE_W + 2 * PORT_HIT + 40 + 14, NODE_H + 30)

    def shape(self) -> QPainterPath:
        p = QPainterPath()
        p.addRoundedRect(QRectF(0, 0, NODE_W, NODE_H), 6, 6)
        return p

    def set_state(self, status: str, rows: int | None, error: str | None) -> None:
        self.status, self.rows, self.error = status, rows, error
        self.setToolTip(error or "")
        self.update()

    def set_problem(self, problem: str | None) -> None:
        self.problem = problem
        self.update()

    def paint(self, painter: QPainter, option, widget=None) -> None:
        if self.node_id not in self.canvas.doc.pipeline.nodes:
            return
        node = self.node
        nt = registry.get(node.type)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = QRectF(0, 0, NODE_W, NODE_H)
        failed = self.status == "failed"
        if self.isSelected():
            border, width = QColor(T.select), 2
        elif failed:
            border, width = QColor(T.danger), 1.5
        elif self._hover:
            border, width = QColor(T.faint), 1
        else:
            border, width = QColor(T.node_border), 1
        # every card carries a soft tint of its category colour
        cat = category_color(nt.category)
        clip = QPainterPath(); clip.addRoundedRect(rect, 6, 6)
        painter.save()
        painter.setClipPath(clip); painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(T.node)); painter.drawRect(rect)
        tint = QColor(cat); tint.setAlpha(48 if T.dark else 30)
        painter.setBrush(tint); painter.drawRect(rect)
        painter.restore()
        painter.setPen(QPen(border, width)); painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(rect, 6, 6)
        # icon
        paint_icon(painter, node_icon_name(node.type), cat.name(), QRectF(12, 12, 20, 20))
        # title
        f = _font(10, True)
        painter.setFont(f); painter.setPen(QColor(T.text))
        painter.drawText(QRectF(40, 9, NODE_W - 50, 20), Qt.AlignLeft | Qt.AlignVCenter, _elide(node.title, f, NODE_W - 52))
        # summary
        f2 = _font(8.5)
        painter.setFont(f2); painter.setPen(QColor(T.muted))
        sub = nt.summarize(node.params) or nt.label
        painter.drawText(QRectF(40, 29, NODE_W - 50, 16), Qt.AlignLeft | Qt.AlignVCenter, _elide(sub, f2, NODE_W - 52))
        # status line
        if failed:
            color, txt = T.danger, (self.error or "failed")
        elif self.problem:
            color, txt = T.warn, self.problem
        elif self.status == "done" and self.rows is not None:
            color, txt = T.ok, f"{self.rows:,} rows"
        elif self.status == "running":
            color, txt = T.accent, "running…"
        elif self.status == "stale":
            color, txt = T.warn, "changed — run again"
        else:
            color, txt = T.faint, "not run yet"
        painter.setPen(Qt.NoPen); painter.setBrush(QColor(color))
        painter.drawEllipse(QPointF(46, NODE_H - 16), 3.5, 3.5)
        painter.setPen(QColor(color if (failed or self.problem) else T.muted))
        painter.drawText(QRectF(54, NODE_H - 25, NODE_W - 62, 18), Qt.AlignLeft | Qt.AlignVCenter, _elide(txt, f2, NODE_W - 66))

    # --- interaction
    def hoverEnterEvent(self, e) -> None:
        self._hover = True; self.update()

    def hoverLeaveEvent(self, e) -> None:
        self._hover = False; self.update()

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged:
            self.canvas.update_edges_of(self.node_id)
        elif change == QGraphicsItem.ItemSelectedHasChanged:
            self.plus.setVisible(bool(value))
        return super().itemChange(change, value)

    def mouseReleaseEvent(self, e: QGraphicsSceneMouseEvent) -> None:
        super().mouseReleaseEvent(e)
        self.canvas.commit_moves()

    def mouseDoubleClickEvent(self, e: QGraphicsSceneMouseEvent) -> None:
        self.canvas.nodeActivated.emit(self.node_id); e.accept()

    def contextMenuEvent(self, e) -> None:
        if not self.isSelected():
            self.scene().clearSelection(); self.setSelected(True)
        self.canvas.node_menu(self.node_id, e.screenPos()); e.accept()


class EdgeItem(QGraphicsPathItem):
    def __init__(self, canvas: "CanvasScene", edge: Edge, src: PortItem, dst: PortItem) -> None:
        super().__init__()
        self.canvas, self.edge, self.src, self.dst = canvas, edge, src, dst
        self.setZValue(0)
        self.setFlag(QGraphicsItem.ItemIsSelectable)
        self.setAcceptHoverEvents(True)
        self._hover = False
        self.update_path()

    def update_path(self) -> None:
        self.setPath(bezier(self.src.scenePos(), self.dst.scenePos()))

    def shape(self) -> QPainterPath:
        s = QPainterPathStroker(); s.setWidth(14)
        return s.createStroke(self.path())

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.Antialiasing)
        active = self._hover or self.isSelected() or self.src.node_item.isSelected() or self.dst.node_item.isSelected()
        color = QColor(T.edge_hover if active else T.edge)
        painter.setPen(QPen(color, 2.2 if active else 1.6, Qt.SolidLine, Qt.RoundCap))
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(self.path())
        end = self.path().pointAtPercent(1.0); prev = self.path().pointAtPercent(0.97)
        ang = math.atan2(end.y() - prev.y(), end.x() - prev.x())
        size = 7
        tri = QPolygonF([end, QPointF(end.x() - size * math.cos(ang - 0.45), end.y() - size * math.sin(ang - 0.45)),
                         QPointF(end.x() - size * math.cos(ang + 0.45), end.y() - size * math.sin(ang + 0.45))])
        painter.setBrush(color); painter.setPen(Qt.NoPen)
        painter.drawPolygon(tri)

    def hoverEnterEvent(self, e) -> None:
        self._hover = True; self.update()

    def hoverLeaveEvent(self, e) -> None:
        self._hover = False; self.update()

    def contextMenuEvent(self, e) -> None:
        m = QMenu()
        act = m.addAction("Disconnect")
        if m.exec(e.screenPos()) == act:
            self.canvas.doc.disconnect(self.edge)


def dot_grid(rect: QRectF, scale: float) -> tuple[list[QPointF], list[QPointF]]:
    """Minor and major dot positions for the map background at this zoom.

    The spacing is picked so dots stay roughly 12-34 px apart on screen whatever the zoom, and every
    fifth dot in both axes is a major dot, so there is always a coarse frame of reference."""
    if scale <= 0.01:
        return [], []
    major_step = 26.0
    while major_step * scale < 60:
        major_step *= 2.0
    while major_step * scale > 170:
        major_step /= 2.0
    minor_step = major_step / 5.0
    x0 = math.floor(rect.left() / minor_step) * minor_step
    y0 = math.floor(rect.top() / minor_step) * minor_step
    cols = int((rect.right() - x0) / minor_step) + 2
    rows = int((rect.bottom() - y0) / minor_step) + 2
    if cols * rows > 80_000:                 # far zoomed out: a plain background beats a stall
        return [], []
    minor: list[QPointF] = []
    major: list[QPointF] = []
    for iy in range(rows):
        y = y0 + iy * minor_step
        for ix in range(cols):
            (major if (ix % 5 == 0 and iy % 5 == 0) else minor).append(QPointF(x0 + ix * minor_step, y))
    return minor, major


def bezier(a: QPointF, b: QPointF) -> QPainterPath:
    p = QPainterPath(a)
    dx = max(40.0, abs(b.x() - a.x()) * 0.5)
    p.cubicTo(QPointF(a.x() + dx, a.y()), QPointF(b.x() - dx, b.y()), b)
    return p


class NoteItem(QGraphicsObject):
    def __init__(self, canvas: "CanvasScene", note: Note) -> None:
        super().__init__()
        self.canvas, self.note_id = canvas, note.id
        self.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemSendsGeometryChanges)
        self.setZValue(-1)
        self.setPos(note.x, note.y)
        self.w, self.h = note.width, note.height

    @property
    def note(self) -> Note | None:
        return next((n for n in self.canvas.doc.pipeline.notes if n.id == self.note_id), None)

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self.w, self.h)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        note = self.note
        if note is None:
            return
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor(T.select) if self.isSelected() else QColor(T.border), 1))
        painter.setBrush(QColor(T.note))
        painter.drawRoundedRect(self.boundingRect(), 4, 4)
        painter.setPen(QColor(T.note_text)); painter.setFont(_font(9))
        painter.drawText(self.boundingRect().adjusted(8, 6, -8, -6), Qt.TextWordWrap | Qt.AlignTop, note.text)

    def edit(self) -> None:
        note = self.note
        if note is None:
            return
        text, ok = QInputDialog.getMultiLineText(None, "Note", "Text:", note.text)
        if ok:
            self.canvas.doc.edit_note(self.note_id, text=text)

    def mouseDoubleClickEvent(self, e) -> None:
        self.edit()

    def mouseReleaseEvent(self, e) -> None:
        super().mouseReleaseEvent(e)
        n = self.note
        if n and (n.x, n.y) != (self.x(), self.y()):
            self.canvas.doc.edit_note(self.note_id, x=self.x(), y=self.y())

    def contextMenuEvent(self, e) -> None:
        m = QMenu()
        edit = m.addAction("Edit note"); delete = m.addAction("Delete note")
        r = m.exec(e.screenPos())
        if r == edit:
            self.edit()
        elif r == delete:
            self.canvas.doc.remove_note(self.note_id)


class CanvasScene(QGraphicsScene):
    nodeActivated = Signal(str)
    selectionChangedTo = Signal(object)
    status = Signal(str)
    addAfterRequested = Signal(str, QPoint)     # node id, screen pos

    def __init__(self, doc: Document, parent=None) -> None:
        super().__init__(parent)
        self.doc = doc
        self.nodes: dict[str, NodeItem] = {}
        self.edges: dict[tuple, EdgeItem] = {}
        self.notes: dict[str, NoteItem] = {}
        self._temp: QGraphicsPathItem | None = None
        self._drag_from: PortItem | None = None
        self._reroute_edge: Edge | None = None     # an existing edge being dragged to a new input
        self._reroute_item: EdgeItem | None = None
        self.setSceneRect(-6000, -6000, 12000, 12000)
        self.selectionChanged.connect(self._on_selection)
        doc.nodeAdded.connect(self._add_node)
        doc.nodeRemoved.connect(self._remove_node)
        doc.nodeChanged.connect(self._node_changed)
        doc.nodeMoved.connect(self._node_moved)
        doc.edgeAdded.connect(self._add_edge)
        doc.edgeRemoved.connect(self._remove_edge)
        doc.noteAdded.connect(self._add_note)
        doc.noteRemoved.connect(self._remove_note)
        doc.noteChanged.connect(lambda nid: self.notes[nid].update() if nid in self.notes else None)
        doc.reloaded.connect(self.rebuild)
        doc.statesChanged.connect(self.refresh_states)
        doc.nodeState.connect(self._node_state)
        self.rebuild()

    # ---- sync with document
    def rebuild(self) -> None:
        self.blockSignals(True)
        self.clear()
        self.nodes.clear(); self.edges.clear(); self.notes.clear()
        for n in self.doc.pipeline.nodes.values():
            self._add_node(n.id)
        for e in self.doc.pipeline.edges:
            self._add_edge(e)
        for n in self.doc.pipeline.notes:
            self._add_note(n.id)
        self.blockSignals(False)
        self.refresh_states()
        self.selectionChangedTo.emit(None)

    def _add_node(self, nid: str) -> None:
        if nid in self.nodes or nid not in self.doc.pipeline.nodes:
            return
        item = NodeItem(self, self.doc.pipeline.nodes[nid])
        self.nodes[nid] = item
        self.addItem(item)
        self._apply_state(nid)

    def _remove_node(self, nid: str) -> None:
        item = self.nodes.pop(nid, None)
        if item:
            self.removeItem(item)

    def _node_changed(self, nid: str) -> None:
        if nid in self.nodes:
            self.nodes[nid].update()
            self.refresh_problems([nid])

    def _node_moved(self, nid: str) -> None:
        item = self.nodes.get(nid); n = self.doc.pipeline.nodes.get(nid)
        if item and n and (item.x(), item.y()) != (n.x, n.y):
            item.setPos(n.x, n.y)

    def _add_edge(self, e: Edge) -> None:
        src = self.nodes.get(e.source); dst = self.nodes.get(e.target)
        if not src or not dst or e.key() in self.edges:
            return
        item = EdgeItem(self, e, src.output, dst.port(e.port))
        self.edges[e.key()] = item
        self.addItem(item)
        self.refresh_problems([e.target])

    def _remove_edge(self, e: Edge) -> None:
        item = self.edges.pop(e.key(), None)
        if item:
            self.removeItem(item)
        self.refresh_problems([e.target])

    def _add_note(self, nid: str) -> None:
        note = next((n for n in self.doc.pipeline.notes if n.id == nid), None)
        if note is None or nid in self.notes:
            return
        item = NoteItem(self, note)
        self.notes[nid] = item
        self.addItem(item)

    def _remove_note(self, nid: str) -> None:
        item = self.notes.pop(nid, None)
        if item:
            self.removeItem(item)

    def _apply_state(self, nid: str) -> None:
        st = self.doc.state(nid)
        self.nodes[nid].set_state(st.status, st.rows, st.error)

    def refresh_states(self) -> None:
        for nid in list(self.nodes):
            if nid in self.doc.pipeline.nodes:
                self._apply_state(nid)
        self.refresh_problems()

    def refresh_problems(self, only: list[str] | None = None) -> None:
        """Mark unconnected required inputs and missing settings on the node itself."""
        p = self.doc.pipeline
        for nid in (only or list(self.nodes)):
            item = self.nodes.get(nid)
            if not item or nid not in p.nodes:
                continue
            node = p.nodes[nid]
            nt = registry.get(node.type)
            ins = p.inputs_of(nid)
            missing = [spec for spec in nt.inputs if not spec.optional and not ins.get(spec.name)]
            for port in item.inputs:
                port.needs = any(s.name == port.name for s in missing)
                port.update()
            probs = nt.param_problems(node.params)
            if missing:
                item.set_problem("connect " + (", ".join(s.label.lower() for s in missing) if len(nt.inputs) > 1 else "an input"))
            elif probs:
                item.set_problem(probs[0])
            else:
                item.set_problem(None)

    def _node_state(self, nid: str, st) -> None:
        if nid in self.nodes:
            self.nodes[nid].set_state(st.status, st.rows, st.error)

    def update_edges_of(self, nid: str) -> None:
        for key, e in self.edges.items():
            if key[0] == nid or key[1] == nid:
                e.update_path()

    def commit_moves(self) -> None:
        moves = {}
        for nid, item in self.nodes.items():
            n = self.doc.pipeline.nodes.get(nid)
            if n and (n.x, n.y) != (item.x(), item.y()):
                moves[nid] = ((n.x, n.y), (item.x(), item.y()))
        if moves:
            self.doc.move_nodes(moves)

    def _on_selection(self) -> None:
        sel = [i for i in self.selectedItems() if isinstance(i, NodeItem)]
        for e in self.edges.values():
            e.update()
        self.selectionChangedTo.emit(sel[-1].node_id if sel else None)

    def selected_node_ids(self) -> list[str]:
        return [i.node_id for i in self.selectedItems() if isinstance(i, NodeItem)]

    def select_node(self, nid: str | None) -> None:
        self.clearSelection()
        if nid and nid in self.nodes:
            self.nodes[nid].setSelected(True)

    # ---- connection dragging
    def begin_connection(self, port: PortItem) -> None:
        """Start dragging a connection. Grabbing an input that already has one picks that edge up
        (its input end is dragged, the source stays put); otherwise a fresh connection is started."""
        self._reroute_edge = None
        self._reroute_item = None
        if not port.is_output:
            edge = next((e for e in self.doc.pipeline.edges
                         if e.target == port.node_item.node_id and e.port == port.name), None)
            if edge is not None and edge.source in self.nodes:
                self._reroute_edge = edge
                self._reroute_item = self.edges.get(edge.key())
                if self._reroute_item is not None:
                    self._reroute_item.setVisible(False)        # hidden while it is dragged
                port = self.nodes[edge.source].output
        self._drag_from = port
        self._temp = QGraphicsPathItem()
        self._temp.setPen(QPen(QColor(T.accent), 2, Qt.DashLine))
        self._temp.setZValue(5)
        self.addItem(self._temp)
        self.drag_connection(port.scenePos())

    def _target_port_at(self, pos: QPointF) -> PortItem | None:
        if self._drag_from is None:
            return None
        want_output = not self._drag_from.is_output
        for it in self.items(pos):
            if isinstance(it, PortItem) and it.is_output == want_output and it.node_item is not self._drag_from.node_item:
                return it
            if isinstance(it, NodeItem) and it is not self._drag_from.node_item:
                if want_output:
                    return it.output
                for p in it.inputs:
                    spec = next(i for i in registry.get(it.node.type).inputs if i.name == p.name)
                    if spec.multiple or not self.doc.pipeline.inputs_of(it.node_id).get(p.name):
                        return p
                return it.inputs[0] if it.inputs else None
        return None

    def drag_connection(self, pos: QPointF) -> None:
        if not self._temp or not self._drag_from:
            return
        a = self._drag_from.scenePos()
        target = self._target_port_at(pos)
        for it in self.items():
            if isinstance(it, PortItem) and it.hot and it is not target and it is not self._drag_from:
                it.hot = False; it.update()
        if target:
            target.hot = True; target.update()
            pos = target.scenePos()
        self._temp.setPath(bezier(a, pos) if self._drag_from.is_output else bezier(pos, a))

    def end_connection(self, pos: QPointF) -> None:
        target = self._target_port_at(pos)
        if self._temp:
            self.removeItem(self._temp)
        self._temp = None
        src = self._drag_from
        self._drag_from = None
        for it in self.items():
            if isinstance(it, PortItem) and it.hot:
                it.hot = False; it.update()
        edge, item = self._reroute_edge, self._reroute_item
        self._reroute_edge = None
        self._reroute_item = None
        if edge is not None:
            if target is None:                              # dropped in empty space: sever the connection
                self.doc.disconnect(edge)
                return
            if (target.node_item.node_id, target.name) == (edge.target, edge.port):
                if item is not None:                        # dropped back where it was: no change
                    item.setVisible(True)
                return
            self.doc.undo.beginMacro("Move connection")     # detach here, attach there, as one undo step
            self.doc.disconnect(edge)
            try:
                self.doc.connect(edge.source, target.node_item.node_id, target.name)
            except PipelineError as e:
                self.status.emit(str(e))
            self.doc.undo.endMacro()
            return
        if not src or not target:
            return
        out, inp = (src, target) if src.is_output else (target, src)
        try:
            self.doc.connect(out.node_item.node_id, inp.node_item.node_id, inp.name)
        except PipelineError as e:
            self.status.emit(str(e))

    # ---- menus
    def node_menu(self, nid: str, screen_pos) -> None:
        m = QMenu()
        run_here = m.addAction(icon("play", T.text), "Run up to here")
        show = m.addAction(icon("table", T.text), "Show output")
        add_after = m.addAction(icon("plus", T.text), "Add a step after this…")
        m.addSeparator()
        rename = m.addAction(icon("note-pencil", T.text), "Rename…")
        dup = m.addAction(icon("copy", T.text), "Duplicate")
        m.addSeparator()
        disc = m.addAction("Disconnect all")
        delete = m.addAction(icon("trash", T.text), "Delete")
        run_here.setEnabled(not self.doc.running)
        r = m.exec(screen_pos)
        if r == run_here:
            self.doc.run(targets=[nid])
        elif r == show:
            self.nodeActivated.emit(nid)
        elif r == add_after:
            self.addAfterRequested.emit(nid, screen_pos)
        elif r == rename:
            text, ok = QInputDialog.getText(None, "Rename", "Name:", text=self.doc.pipeline.nodes[nid].title)
            if ok and text.strip():
                self.doc.rename(nid, text.strip())
        elif r == dup:
            self.doc.duplicate_nodes(self.selected_node_ids() or [nid])
        elif r == disc:
            for e in [e for e in self.doc.pipeline.edges if e.source == nid or e.target == nid]:
                self.doc.disconnect(e)
        elif r == delete:
            self.doc.remove_nodes(self.selected_node_ids() or [nid])

    def delete_selection(self) -> None:
        ids = self.selected_node_ids()
        edges = [i.edge for i in self.selectedItems() if isinstance(i, EdgeItem)]
        notes = [i.note_id for i in self.selectedItems() if isinstance(i, NoteItem)]
        if ids or edges or notes:
            self.doc.undo.beginMacro("Delete")
            for e in edges:
                if e.source not in ids and e.target not in ids:
                    self.doc.disconnect(e)
            if ids:
                self.doc.remove_nodes(ids)
            for n in notes:
                self.doc.remove_note(n)
            self.doc.undo.endMacro()

    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:
        painter.fillRect(rect, QColor(T.canvas))
        scale = painter.worldTransform().m11()
        minor, major = dot_grid(rect, scale)
        if minor:
            painter.setPen(QPen(QColor(T.grid), 2.2 / scale, Qt.SolidLine, Qt.RoundCap))
            painter.drawPoints(minor)
        if major:
            painter.setPen(QPen(QColor(T.dot_major), 3.4 / scale, Qt.SolidLine, Qt.RoundCap))
            painter.drawPoints(major)


class CanvasView(QGraphicsView):
    fileDropped = Signal(str, QPointF)
    nodeTypeDropped = Signal(str, QPointF)
    addStepRequested = Signal(QPoint, object)   # screen pos, scene pos (or None) for the picker
    addNoteRequested = Signal(QPointF)
    deleteRequested = Signal()

    def __init__(self, scene: CanvasScene, parent=None) -> None:
        super().__init__(scene, parent)
        self.canvas = scene
        self.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setAcceptDrops(True)
        # No view background brush: it would paint over CanvasScene.drawBackground, which fills the
        # canvas colour and draws the dot grid.
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFrameShape(QGraphicsView.NoFrame)
        self._panning = False
        self._pan_start = QPointF()
        self._pan_moved = False
        self._build_overlay()
        scene.changed.connect(lambda _: self._update_empty())
        self._update_empty()

    # ---- overlay: zoom buttons + empty state
    def _build_overlay(self) -> None:
        self.zoom_box = QWidget(self)
        h = QHBoxLayout(self.zoom_box); h.setContentsMargins(0, 0, 0, 0); h.setSpacing(2)
        self.zoom_box.setStyleSheet(f"QToolButton {{ background: {T.panel}; border: 1px solid {T.border}; border-radius: 4px; padding: 3px; }} QToolButton:hover {{ background: {T.hover}; }}")
        for name, tip, fn in (("magnifying-glass-minus", "Zoom out", lambda: self.zoom_by(1 / 1.2)),
                              ("magnifying-glass-plus", "Zoom in", lambda: self.zoom_by(1.2)),
                              ("arrows-out", "Fit everything in view", self.fit_all)):
            b = QToolButton(); b.setIcon(icon(name, T.text)); b.setIconSize(QSize(16, 16)); b.setToolTip(tip); b.clicked.connect(fn)
            h.addWidget(b)
        self.empty = QLabel(self)
        self.empty.setAlignment(Qt.AlignCenter)
        self.empty.setStyleSheet(f"QLabel {{ color: {T.muted}; border: 1.5px dashed {T.border}; border-radius: 10px; padding: 24px 32px; background: transparent; }}")
        self.empty.setText("<b style='font-size:12pt'>Drop a CSV or Excel file here</b><br><br>"
                           "or press <b>+ Add step</b> in the toolbar.<br>"
                           "<span style='color:%s'>Drag to move around · scroll to zoom</span>" % T.faint)
        self.empty.adjustSize()

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self.zoom_box.adjustSize()
        self.zoom_box.move(self.width() - self.zoom_box.width() - 12, self.height() - self.zoom_box.height() - 12)
        self.empty.adjustSize()
        self.empty.move((self.width() - self.empty.width()) // 2, (self.height() - self.empty.height()) // 2)

    def _update_empty(self) -> None:
        self.empty.setVisible(not self.canvas.nodes and not self.canvas.notes)

    # ---- zoom / pan
    def zoom_by(self, f: float) -> None:
        cur = self.transform().m11()
        if 0.25 <= cur * f <= 2.5:
            self.scale(f, f)

    def wheelEvent(self, e) -> None:
        """Wheel and two-finger touchpad scrolling zoom the map; pan by dragging it (no modifier needed).
        The zoom is continuous, so a touchpad's many small deltas feel smooth and a mouse notch is a step."""
        angle = e.angleDelta().y()
        delta = angle if angle else e.pixelDelta().y()
        if not delta:
            e.accept(); return
        factor = 2 ** max(-0.8, min(0.8, delta / 500.0))
        self.zoom_by(factor)
        e.accept()

    def fit_all(self) -> None:
        items = [i for i in self.canvas.items() if isinstance(i, (NodeItem, NoteItem))]
        if not items:
            self.resetTransform(); self.centerOn(0, 0); return
        r = items[0].sceneBoundingRect()
        for i in items[1:]:
            r = r.united(i.sceneBoundingRect())
        self.fitInView(r.adjusted(-80, -80, 80, 80), Qt.KeepAspectRatio)
        scale = self.transform().m11()
        if scale > 1.0:
            self.resetTransform(); self.centerOn(r.center())
        elif scale < 0.6:
            self.resetTransform(); self.scale(0.6, 0.6)
            self.centerOn(r.left() + self.viewport().width() / 2 / 0.6 - 80, r.center().y())

    def reveal(self, nid: str) -> None:
        """Make sure a node is visible; pan the view if it is not."""
        it = self.canvas.nodes.get(nid)
        if it:
            self.ensureVisible(it, 60, 60)

    def focus_node(self, nid: str) -> None:
        """Centre the map on a node (used when a step is picked in the side rail)."""
        it = self.canvas.nodes.get(nid)
        if it is not None:
            self.ensureVisible(it, 80, 80)
            self.centerOn(it)               # centre last, so ensureVisible cannot leave it off-centre

    def keyPressEvent(self, e) -> None:
        if e.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self.deleteRequested.emit(); return
        if e.key() in (Qt.Key_Plus, Qt.Key_Equal) and not e.modifiers():
            self.addStepRequested.emit(self.mapToGlobal(self.viewport().rect().center()), None); return
        super().keyPressEvent(e)

    def mousePressEvent(self, e) -> None:
        item = self.itemAt(e.position().toPoint())
        on_empty = item is None
        if e.button() == Qt.MiddleButton or (e.button() == Qt.LeftButton and on_empty and not (e.modifiers() & (Qt.ShiftModifier | Qt.ControlModifier))):
            self._panning = True; self._pan_moved = False; self._pan_start = e.position()
            self.setCursor(Qt.ClosedHandCursor); e.accept(); return
        if e.button() == Qt.LeftButton and on_empty:
            self.setDragMode(QGraphicsView.RubberBandDrag)
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e) -> None:
        if self._panning:
            d = e.position() - self._pan_start
            self._pan_start = e.position()
            self._pan_moved = True
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(d.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(d.y()))
            e.accept(); return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e) -> None:
        if self._panning:
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
            if not self._pan_moved and e.button() == Qt.LeftButton:
                self.canvas.clearSelection()
            e.accept(); return
        super().mouseReleaseEvent(e)
        self.setDragMode(QGraphicsView.NoDrag)

    # drops
    def dragEnterEvent(self, e) -> None:
        md = e.mimeData()
        if md.hasFormat(NODE_MIME) or (md.hasUrls() and any(Path(u.toLocalFile()).suffix.lower() in DATA_EXT for u in md.urls())):
            e.acceptProposedAction()
        else:
            e.ignore()

    def dragMoveEvent(self, e) -> None:
        e.acceptProposedAction()

    def dropEvent(self, e) -> None:
        pos = self.mapToScene(e.position().toPoint())
        md = e.mimeData()
        if md.hasFormat(NODE_MIME):
            self.nodeTypeDropped.emit(bytes(md.data(NODE_MIME)).decode(), pos)
            e.acceptProposedAction(); return
        for u in md.urls():
            f = u.toLocalFile()
            if Path(f).suffix.lower() in DATA_EXT:
                self.fileDropped.emit(f, pos)
                pos = QPointF(pos.x(), pos.y() + NODE_H + 40)
        e.acceptProposedAction()

    def contextMenuEvent(self, e) -> None:
        item = self.itemAt(e.pos())
        if item is not None and not isinstance(item, PortItem):
            super().contextMenuEvent(e); return
        pos = self.mapToScene(e.pos())
        m = QMenu(self)
        add = m.addAction(icon("plus", T.text), "Add a step here…")
        add.triggered.connect(lambda: self.addStepRequested.emit(e.globalPos(), pos))
        note = m.addAction(icon("note-pencil", T.text), "Add a note here")
        note.triggered.connect(lambda: self.addNoteRequested.emit(pos))
        m.addSeparator()
        fit = m.addAction(icon("arrows-out", T.text), "Fit everything in view"); fit.triggered.connect(self.fit_all)
        run = m.addAction(icon("play", T.text), "Run everything"); run.triggered.connect(lambda: self.canvas.doc.run())
        run.setEnabled(not self.canvas.doc.running)
        m.exec(e.globalPos())
