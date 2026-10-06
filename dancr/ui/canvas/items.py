"""The graphics items drawn on the canvas (nodes, ports, edges, notes, answers, ghosts)."""
from __future__ import annotations

import math
from pathlib import Path

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, QPointF, QRectF, Signal, QPoint, QSize
from PySide6.QtGui import QColor, QPen, QBrush, QPainter, QPainterPath, QFont, QFontMetrics, QPolygonF, QPainterPathStroker
from PySide6.QtWidgets import (QGraphicsView, QGraphicsScene, QGraphicsObject, QGraphicsItem, QGraphicsPathItem, QMenu,
                               QWidget, QGraphicsSceneMouseEvent, QInputDialog, QToolButton, QHBoxLayout, QLabel)

from ...core import registry, PipelineError
from ...core.model import Edge, Node, Note, Answer
from ...core.nodes.load import CSV_EXT, EXCEL_EXT, PARQUET_EXT
from ..document import Document
from ..common import listen
from ..theme import T, category_color
from ..icons import paint as paint_icon, icon, node_icon_name

if TYPE_CHECKING:                       # the items are created by the scene; only annotations name it
    from . import CanvasScene


NODE_W, NODE_H = 236, 98
ANSWER_W, ANSWER_H = 230, 70
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
    def __init__(self, node_item: "NodeItem", name: str, label: str, is_output: bool, index: int, count: int) -> None:
        super().__init__(node_item)
        self.node_item = node_item
        self.name, self.label, self.is_output = name, label, is_output
        self.index, self.count = index, count
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
        self.inputs = [PortItem(self, i.name, i.label, False, k, len(nt.inputs)) for k, i in enumerate(nt.inputs)]
        self.output = PortItem(self, "out", "Output", True, 0, 1)
        self.plus = PlusButton(self)
        self._hover = False
        self.status = "idle"
        self.rows: int | None = None
        self.error: str | None = None
        self.problem: str | None = None
        self.finding: str = ""

    @property
    def node(self) -> Node:
        return self.canvas.doc.pipeline.nodes[self.node_id]

    @property
    def is_ai(self) -> bool:
        ids = self.canvas.doc.pipeline.meta.get("ai_steps") or []
        return self.node_id in ids

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

    def set_state(self, status: str, rows: int | None, error: str | None, finding: str = "") -> None:
        self.status, self.rows, self.error, self.finding = status, rows, error, finding or ""
        self.setToolTip(error or self.finding or "")
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
        # a small "AI" tag: this step was built by the Assistant, and can be removed as a set
        if self.is_ai:
            tag = QRectF(NODE_W - 32, 8, 24, 14)
            fill = QColor(T.accent); fill.setAlpha(38 if T.dark else 26)
            painter.setPen(QPen(QColor(T.accent), 1)); painter.setBrush(fill)
            painter.drawRoundedRect(tag, 7, 7)
            painter.setPen(QColor(T.accent)); painter.setFont(_font(7.5, True))
            painter.drawText(tag, Qt.AlignCenter, "AI")
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
        # the step's finding: the plain sentence it produced, when there is one and nothing is wrong
        if self.finding and self.status == "done" and not self.problem:
            f3 = _font(8.0)
            painter.setFont(f3); painter.setPen(QColor(T.accent))
            painter.drawText(QRectF(12, 47, NODE_W - 24, 16), Qt.AlignLeft | Qt.AlignVCenter,
                             "→ " + _elide(self.finding, f3, NODE_W - 40))
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
            color, txt = T.warn, "changed, run again"
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


class AnswerItem(QGraphicsObject):
    """A guided answer: an unconnected card that points at the step whose output answers a question.

    It is not part of the dataflow (no ports, no wires). Click it to see its result; drag to move;
    right-click to change the question or delete it."""

    def __init__(self, canvas: "CanvasScene", answer: Answer) -> None:
        super().__init__()
        self.canvas, self.answer_id = canvas, answer.id
        self.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemSendsGeometryChanges)
        self.setZValue(1.5)
        self.setPos(answer.x, answer.y)
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.PointingHandCursor)
        self._hover = False
        self._press = QPointF()

    @property
    def answer(self) -> Answer | None:
        return self.canvas.doc.pipeline.answer(self.answer_id)

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, ANSWER_W, ANSWER_H)

    def shape(self) -> QPainterPath:
        p = QPainterPath(); p.addRoundedRect(self.boundingRect(), 8, 8)
        return p

    def paint(self, painter: QPainter, option, widget=None) -> None:
        a = self.answer
        if a is None:
            return
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.boundingRect()
        accent = QColor(T.accent)
        painter.setPen(QPen(accent if self.isSelected() else (QColor(T.accent) if self._hover else QColor(T.border)), 2 if self.isSelected() else 1))
        tint = QColor(accent); tint.setAlpha(34 if T.dark else 22)
        painter.setBrush(QColor(T.node)); painter.drawRoundedRect(rect, 8, 8)
        painter.setBrush(tint); painter.drawRoundedRect(rect, 8, 8)
        paint_icon(painter, "sparkle", accent.name(), QRectF(12, 13, 20, 20))
        f = _font(10.5, True); painter.setFont(f); painter.setPen(QColor(T.text))
        painter.drawText(QRectF(40, 11, ANSWER_W - 50, 20), Qt.AlignLeft | Qt.AlignVCenter, _elide(a.title, f, ANSWER_W - 52))
        f2 = _font(8.5); painter.setFont(f2); painter.setPen(QColor(T.muted))
        painter.drawText(QRectF(40, 32, ANSWER_W - 50, 15), Qt.AlignLeft | Qt.AlignVCenter, "Answer. Click to see or change it")
        p = self.canvas.doc.pipeline
        st = self.canvas.doc.state(a.terminal) if a.terminal in p.nodes else None
        status = st.status if st else "idle"
        if status == "failed":
            color, txt = T.danger, "failed"
        elif status == "running":
            color, txt = T.accent, "running…"
        elif status == "done" and st is not None and st.rows is not None:
            color, txt = T.ok, f"{st.rows:,} rows"
        elif status == "stale":
            color, txt = T.warn, "changed, run again"
        else:
            color, txt = T.faint, "not run yet"
        painter.setPen(Qt.NoPen); painter.setBrush(QColor(color))
        painter.drawEllipse(QPointF(46, ANSWER_H - 15), 3.5, 3.5)
        painter.setPen(QColor(color if status in ("failed", "stale") else T.muted))
        painter.drawText(QRectF(54, ANSWER_H - 24, ANSWER_W - 62, 18), Qt.AlignLeft | Qt.AlignVCenter, txt)

    def hoverEnterEvent(self, e) -> None:
        self._hover = True; self.update()

    def hoverLeaveEvent(self, e) -> None:
        self._hover = False; self.update()

    def mousePressEvent(self, e: QGraphicsSceneMouseEvent) -> None:
        self._press = e.scenePos()
        super().mousePressEvent(e)

    def mouseReleaseEvent(self, e: QGraphicsSceneMouseEvent) -> None:
        moved = (e.scenePos() - self._press).manhattanLength() > 4
        super().mouseReleaseEvent(e)
        a = self.answer
        if a and (a.x, a.y) != (self.x(), self.y()):
            self.canvas.doc.edit_answer(self.answer_id, x=self.x(), y=self.y())
        if not moved and e.button() == Qt.LeftButton:
            self.canvas.answerActivated.emit(self.answer_id)

    def contextMenuEvent(self, e) -> None:
        m = QMenu(); m.setAttribute(Qt.WA_DeleteOnClose)
        show = m.addAction(icon("table", T.text), "Show the result")
        change = m.addAction(icon("sliders", T.text), "Change the question…")
        m.addSeparator()
        delete = m.addAction(icon("trash", T.text), "Delete this answer…")
        r = m.exec(e.screenPos())
        if r == show:
            self.canvas.answerActivated.emit(self.answer_id)
        elif r == change:
            self.canvas.answerChangeRequested.emit(self.answer_id)
        elif r == delete:
            self.canvas.answerDeleteRequested.emit(self.answer_id)


class GhostItem(QGraphicsObject):
    """A proposed step: dashed, not part of the project, waiting to be approved. Nothing here touches the dataflow."""

    def __init__(self, title: str, label: str) -> None:
        super().__init__()
        self.title, self.label = title, label
        self.setZValue(0.5)
        self.setAcceptedMouseButtons(Qt.NoButton)

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, NODE_W, NODE_H)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.Antialiasing)
        rect = QRectF(0, 0, NODE_W, NODE_H)
        painter.setPen(QPen(QColor(T.accent), 1.6, Qt.DashLine)); painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(rect, 6, 6)
        f = _font(10, True)
        painter.setPen(QColor(T.text)); painter.setFont(f)
        painter.drawText(rect.adjusted(12, 12, -12, -12), Qt.AlignLeft | Qt.AlignTop, _elide(self.title, f, NODE_W - 24))
        painter.setPen(QColor(T.faint)); painter.setFont(_font(8.5))
        painter.drawText(QRectF(12, NODE_H - 28, NODE_W - 24, 18), Qt.AlignLeft | Qt.AlignVCenter, self.label)

