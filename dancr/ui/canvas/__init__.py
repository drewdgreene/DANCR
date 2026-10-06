"""The node canvas: a QGraphicsScene of items and the QGraphicsView that shows it."""
from __future__ import annotations

import math
from pathlib import Path

from PySide6.QtCore import Qt, QPointF, QRectF, Signal, QPoint, QSize
from PySide6.QtGui import QColor, QPen, QBrush, QPainter, QPainterPath, QFont, QFontMetrics, QPolygonF, QPainterPathStroker
from PySide6.QtWidgets import (QGraphicsView, QGraphicsScene, QGraphicsObject, QGraphicsItem, QGraphicsPathItem, QMenu,
                               QWidget, QGraphicsSceneMouseEvent, QInputDialog, QToolButton, QHBoxLayout, QLabel,
                               QDialog, QPlainTextEdit, QVBoxLayout, QDialogButtonBox)

from ...core import registry, PipelineError
from ...core.model import Edge, Node, Note, Answer
from ...core.executor import Executor
from ...core.lineage import build_lineage, proof_card
from ...core.nodes.load import CSV_EXT, EXCEL_EXT, PARQUET_EXT
from ..document import Document
from ..common import listen
from ..theme import T, category_color
from ..icons import paint as paint_icon, icon, node_icon_name

from .items import (  # noqa: E402
    ANSWER_H, ANSWER_W, DATA_EXT, EdgeItem, GhostItem, NODE_H, NODE_MIME, NODE_W, NodeItem, NoteItem,
    PORT_HIT, PORT_R, PlusButton, PortItem, AnswerItem, bezier, dot_grid,
)

def _show_text(parent, title: str, text: str) -> None:
    """A read-only monospace panel for the lineage and proof cards (long, plain text a person can copy)."""
    dlg = QDialog(parent)
    dlg.setWindowTitle(title)
    dlg.resize(760, 560)
    lay = QVBoxLayout(dlg)
    box = QPlainTextEdit(); box.setReadOnly(True); box.setPlainText(text)
    box.setStyleSheet("font-family: monospace;")
    lay.addWidget(box)
    bb = QDialogButtonBox(QDialogButtonBox.Close); bb.rejected.connect(dlg.reject); bb.accepted.connect(dlg.accept)
    lay.addWidget(bb)
    dlg.exec()


class CanvasScene(QGraphicsScene):
    nodeActivated = Signal(str)
    answerActivated = Signal(str)
    answerChangeRequested = Signal(str)
    answerDeleteRequested = Signal(str)
    selectionChangedTo = Signal(object)
    status = Signal(str)
    addAfterRequested = Signal(str, QPoint)     # node id, screen pos
    runRequested = Signal(object)               # step ids to run up to, or None for everything (the window checks first)

    def __init__(self, doc: Document, parent=None) -> None:
        super().__init__(parent)
        self.doc = doc
        self.nodes: dict[str, NodeItem] = {}
        self.edges: dict[tuple, EdgeItem] = {}
        self.notes: dict[str, NoteItem] = {}
        self.answers: dict[str, "AnswerItem"] = {}
        self._temp: QGraphicsPathItem | None = None
        self._drag_from: PortItem | None = None
        self._reroute_edge: Edge | None = None     # an existing edge being dragged to a new input
        self._reroute_item: EdgeItem | None = None
        self.setSceneRect(-6000, -6000, 12000, 12000)
        self.selectionChanged.connect(self._on_selection)
        listen(self, doc.nodeAdded, self._add_node)
        listen(self, doc.nodeRemoved, self._remove_node)
        listen(self, doc.nodeChanged, self._node_changed)
        listen(self, doc.nodeMoved, self._node_moved)
        listen(self, doc.edgeAdded, self._add_edge)
        listen(self, doc.edgeRemoved, self._remove_edge)
        listen(self, doc.noteAdded, self._add_note)
        listen(self, doc.noteRemoved, self._remove_note)
        listen(self, doc.noteChanged, lambda nid: self.notes[nid].update() if nid in self.notes else None)
        listen(self, doc.answerAdded, self._add_answer)
        listen(self, doc.answerRemoved, self._remove_answer)
        listen(self, doc.answerChanged, lambda aid: self.answers[aid].update() if aid in self.answers else None)
        listen(self, doc.reloaded, self.rebuild)
        listen(self, doc.statesChanged, self.refresh_states)
        listen(self, doc.nodeState, self._node_state)
        self.rebuild()

    # ---- sync with document
    def rebuild(self) -> None:
        self.blockSignals(True)
        self.clear()
        self.nodes.clear(); self.edges.clear(); self.notes.clear(); self.answers.clear()
        self._ghost_items = []
        for n in self.doc.pipeline.nodes.values():
            self._add_node(n.id)
        for e in self.doc.pipeline.edges:
            self._add_edge(e)
        for n in self.doc.pipeline.notes:
            self._add_note(n.id)
        for a in self.doc.pipeline.answers:
            self._add_answer(a.id)
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

    def _add_answer(self, aid: str) -> None:
        answer = self.doc.pipeline.answer(aid)
        if answer is None or aid in self.answers:
            return
        item = AnswerItem(self, answer)
        self.answers[aid] = item
        self.addItem(item)

    def _remove_answer(self, aid: str) -> None:
        item = self.answers.pop(aid, None)
        if item:
            self.removeItem(item)

    def selected_answer_ids(self) -> list[str]:
        return [i.answer_id for i in self.selectedItems() if isinstance(i, AnswerItem)]

    def select_answer(self, aid: str | None) -> None:
        self.clearSelection()
        if aid and aid in self.answers:
            self.answers[aid].setSelected(True)

    def highlight_branch(self, nids: set[str]) -> None:
        """Dim everything not in the branch, so one answer's pipeline stands out in a busy project."""
        for nid, item in self.nodes.items():
            item.setOpacity(1.0 if nid in nids else 0.28)
        for (src, dst, _port), edge in self.edges.items():
            edge.setOpacity(1.0 if (src in nids and dst in nids) else 0.2)

    def clear_highlight(self) -> None:
        for item in self.nodes.values():
            item.setOpacity(1.0)
        for edge in self.edges.values():
            edge.setOpacity(1.0)

    def show_ghost(self, steps: list[dict]) -> None:
        """Draw a proposed chain of steps as dashed, non-committed cards to the right of the graph. The person
        approves or dismisses it; nothing here is in the pipeline."""
        self.clear_ghost()
        if not steps:
            return
        items: list[QGraphicsItem] = []
        r = self.itemsBoundingRect()
        x0 = r.right() + 90 if self.nodes else 80.0
        y0 = r.top() if self.nodes else 120.0
        prev: GhostItem | None = None
        for i, s in enumerate(steps):
            x = x0 + i * (NODE_W + 70)
            item = GhostItem(str(s.get("title") or s.get("type") or "step"), str(s.get("type") or ""))
            item.setPos(x, y0)
            self.addItem(item); items.append(item)
            after = s.get("after")
            if after and after in self.nodes:
                a = QGraphicsPathItem(bezier(self.nodes[after].output.scenePos(), QPointF(x, y0 + NODE_H / 2)))
                a.setPen(QPen(QColor(T.accent), 1.6, Qt.DashLine)); a.setZValue(0.4)
                self.addItem(a); items.append(a)
            if prev is not None:
                e = QGraphicsPathItem(bezier(prev.scenePos() + QPointF(NODE_W, NODE_H / 2), QPointF(x, y0 + NODE_H / 2)))
                e.setPen(QPen(QColor(T.accent), 1.6, Qt.DashLine)); e.setZValue(0.4)
                self.addItem(e); items.append(e)
            prev = item
        self._ghost_items = items

    def clear_ghost(self) -> None:
        for item in getattr(self, "_ghost_items", []):
            self.removeItem(item)
        self._ghost_items = []

    def _apply_state(self, nid: str) -> None:
        st = self.doc.state(nid)
        finding = ((st.report or {}).get("finding") or {}).get("statement") or ""
        self.nodes[nid].set_state(st.status, st.rows, st.error, finding)

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
            elif nid in self.doc.held:
                item.set_problem("changed elsewhere. Run to save its file")
            else:
                item.set_problem(None)

    def _node_state(self, nid: str, st) -> None:
        if nid in self.nodes:
            finding = ((st.report or {}).get("finding") or {}).get("statement") or ""
            self.nodes[nid].set_state(st.status, st.rows, st.error, finding)

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
            try:
                self.doc.move_edge(edge, target.node_item.node_id, target.name)
            except PipelineError as e:                      # not allowed: the connection stays where it was
                if item is not None:
                    item.setVisible(True)
                self.status.emit(str(e))
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
        m.addSeparator()
        lineage = m.addAction("Lineage…")
        proof = m.addAction("Proof && verify…")
        run_here.setEnabled(not self.doc.running)
        r = m.exec(screen_pos)
        if r == run_here:
            self.runRequested.emit([nid])
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
            with self.doc.macro("Disconnect all"):
                for e in [e for e in self.doc.pipeline.edges if e.source == nid or e.target == nid]:
                    self.doc.disconnect(e)
        elif r == delete:
            self.doc.remove_nodes(self.selected_node_ids() or [nid])
        elif r == lineage:
            self._show_lineage(nid)
        elif r == proof:
            self._show_proof(nid)

    def _dialog_parent(self):
        views = self.views()
        return views[0] if views else None

    def _show_lineage(self, nid: str) -> None:
        p = self.doc.pipeline
        parent = self._dialog_parent()
        try:
            res = build_lineage(p, nid, direction="both", executor=Executor(p))
        except Exception as e:  # noqa: BLE001 - show the reason, never a traceback in a dialog
            _show_text(parent, "Lineage", str(e))
            return
        lines = [res["sentence"], ""]
        lines.append("Upstream (what it is built from):")
        up = (res.get("up") or {}).get("nodes") or []
        lines += [f"  {n}  {p.nodes[n].title}" for n in up] or ["  (nothing — a source)"]
        down = res.get("down") or {}
        lines.append("")
        lines.append("Downstream (what depends on it):")
        nodes = down.get("nodes") or []
        lines += [f"  {n}  {p.nodes[n].title}" for n in nodes] or ["  (nothing)"]
        for x in down.get("answers", []):
            lines.append(f"  answer: {x['title']} ({x['id']})")
        for x in down.get("artifacts", []):
            lines.append(f"  file: {x['file']}")
        for x in down.get("agent_turns", []):
            lines.append(f"  agent turn {x.get('turn')}: {x.get('question', '')}")
        _show_text(parent, f"Lineage — {p.nodes[nid].title}", "\n".join(lines))

    def _show_proof(self, nid: str) -> None:
        p = self.doc.pipeline
        parent = self._dialog_parent()
        try:
            card = proof_card(p, Executor(p), nid)
        except Exception as e:  # noqa: BLE001
            _show_text(parent, "Proof", str(e))
            return
        rows = card.get("rows")
        lines = [f"{card['title']}  [{card['node']}]",
                 f"result: {rows:,} row(s)" if isinstance(rows, int) else "result: not computed"]
        if card.get("output_hash"):
            lines.append(f"content hash: {card['output_hash']}")
        if card.get("finding"):
            lines.append(f"finding: {card['finding']}")
        lines.append("")
        lines.append("Steps:")
        for s in card["steps"]:
            lines.append(f"  {s['id']}  ({s['type']})  {s.get('plan_hash')}")
        if card["sources"]:
            lines.append("")
            lines.append("Sources:")
            for s in card["sources"]:
                lines.append(f"  {s.get('path')}  [{s.get('sample')}]")
        if card["assumptions"]:
            lines.append("")
            lines.append("Assumptions:")
            lines += [f"  {a}" for a in card["assumptions"]]
        lines.append("")
        lines.append(f"Verify: {card['verify']}")
        _show_text(parent, f"Proof — {card['title']}", "\n".join(lines))

    def delete_selection(self) -> None:
        answer_ids = self.selected_answer_ids()
        if answer_ids:
            for aid in answer_ids:
                self.answerDeleteRequested.emit(aid)
            return
        ids = self.selected_node_ids()
        edges = [i.edge for i in self.selectedItems() if isinstance(i, EdgeItem)]
        notes = [i.note_id for i in self.selectedItems() if isinstance(i, NoteItem)]
        if ids or edges or notes:
            with self.doc.macro("Delete"):
                for e in edges:
                    if e.source not in ids and e.target not in ids:
                        self.doc.disconnect(e)
                if ids:
                    self.doc.remove_nodes(ids)
                for n in notes:
                    self.doc.remove_note(n)

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
        listen(self, scene.changed, lambda _: self._update_empty())
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
        self.empty.setText("<b style='font-size:13pt'>Drop your spreadsheets here</b><br>"
                           "<span style='color:%s'>CSV, Excel, Parquet, JSON, a folder, a database or a URL</span><br><br>"
                           "or use <b>Auto</b> to ask a question in plain words.<br>"
                           "<span style='color:%s'>Drag to move · scroll to zoom</span>" % (T.muted, T.faint))
        self.empty.adjustSize()

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self.zoom_box.adjustSize()
        self.zoom_box.move(self.width() - self.zoom_box.width() - 12, self.height() - self.zoom_box.height() - 12)
        self.empty.adjustSize()
        self.empty.move((self.width() - self.empty.width()) // 2, (self.height() - self.empty.height()) // 2)

    def _update_empty(self) -> None:
        self.empty.setVisible(not self.canvas.nodes and not self.canvas.notes and not self.canvas.answers)

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
        items = [i for i in self.canvas.items() if isinstance(i, (NodeItem, NoteItem, AnswerItem))]
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
        m = QMenu(self); m.setAttribute(Qt.WA_DeleteOnClose)
        add = m.addAction(icon("plus", T.text), "Add a step here…")
        add.triggered.connect(lambda: self.addStepRequested.emit(e.globalPos(), pos))
        note = m.addAction(icon("note-pencil", T.text), "Add a note here")
        note.triggered.connect(lambda: self.addNoteRequested.emit(pos))
        m.addSeparator()
        fit = m.addAction(icon("arrows-out", T.text), "Fit everything in view"); fit.triggered.connect(self.fit_all)
        run = m.addAction(icon("play", T.text), "Run everything"); run.triggered.connect(lambda: self.canvas.runRequested.emit(None))
        run.setEnabled(not self.canvas.doc.running)
        m.exec(e.globalPos())
