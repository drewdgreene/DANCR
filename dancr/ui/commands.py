"""Undo commands for Document edits. Each one mutates ``doc.pipeline`` and emits the matching signals."""
from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

from PySide6.QtGui import QUndoCommand

from ..core import PipelineError
from ..core.model import Node, Edge, Note, Input

if TYPE_CHECKING:
    from .document import Document


class AddNode(QUndoCommand):
    def __init__(self, doc: Document, node: Node, connect_from: str | None, port: str | None) -> None:
        super().__init__(f"Add {node.title}")
        self.doc, self.node, self.connect_from, self.port = doc, node, connect_from, port
        self.edge: Edge | None = None

    def redo(self) -> None:
        p = self.doc.pipeline
        p.nodes[self.node.id] = self.node
        self.doc.nodeAdded.emit(self.node.id)
        if self.connect_from and self.connect_from in p.nodes:
            try:
                self.edge = p.connect(self.connect_from, self.node.id, self.port)
                self.doc.edgeAdded.emit(self.edge)
            except PipelineError:
                self.edge = None
        self.doc.refresh_states()

    def undo(self) -> None:
        p = self.doc.pipeline
        if self.node.id in p.nodes:
            for e in p.remove_node(self.node.id):
                self.doc.edgeRemoved.emit(e)
            self.doc.nodeRemoved.emit(self.node.id)
        self.doc.refresh_states()


class RemoveNodes(QUndoCommand):
    def __init__(self, doc: Document, ids: list[str]) -> None:
        super().__init__("Delete")
        self.doc = doc
        self.nodes = [doc.pipeline.nodes[i] for i in ids]
        self.edges: list[Edge] = []

    def redo(self) -> None:
        p = self.doc.pipeline
        self.edges = []
        for n in self.nodes:
            if n.id not in p.nodes:
                continue
            for e in p.remove_node(n.id):
                self.edges.append(e)
                self.doc.edgeRemoved.emit(e)
            self.doc.nodeRemoved.emit(n.id)
        self.doc.refresh_states()

    def undo(self) -> None:
        p = self.doc.pipeline
        for n in self.nodes:
            p.nodes[n.id] = n
            self.doc.nodeAdded.emit(n.id)
        for e in self.edges:
            if e.key() not in {x.key() for x in p.edges} and e.source in p.nodes and e.target in p.nodes:
                p.edges.append(e)
                self.doc.edgeAdded.emit(e)
        self.doc.refresh_states()


class SetParams(QUndoCommand):
    def __init__(self, doc: Document, nid: str, old: dict, new: dict, keys: set[str]) -> None:
        super().__init__(f"Edit {doc.pipeline.nodes[nid].title}")
        self.doc, self.nid, self.old, self.new, self.keys = doc, nid, old, new, keys
        self.at = time.time()

    def id(self) -> int:
        return 1001

    def mergeWith(self, other: QUndoCommand) -> bool:
        if isinstance(other, SetParams) and other.nid == self.nid and other.keys == self.keys and other.at - self.at < 1.5:
            self.new = other.new
            self.at = other.at
            return True
        return False

    def _apply(self, params: dict) -> None:
        if self.nid in self.doc.pipeline.nodes:
            self.doc.pipeline.nodes[self.nid].params = params
            self.doc.nodeChanged.emit(self.nid)
        self.doc.refresh_states()

    def redo(self) -> None:
        self._apply(self.new)

    def undo(self) -> None:
        self._apply(self.old)


class Rename(QUndoCommand):
    def __init__(self, doc: Document, nid: str, old: str, new: str) -> None:
        super().__init__("Rename")
        self.doc, self.nid, self.old, self.new = doc, nid, old, new

    def _apply(self, title: str) -> None:
        if self.nid in self.doc.pipeline.nodes:
            self.doc.pipeline.nodes[self.nid].title = title
            self.doc.nodeChanged.emit(self.nid)

    def redo(self) -> None:
        self._apply(self.new)

    def undo(self) -> None:
        self._apply(self.old)


class Move(QUndoCommand):
    def __init__(self, doc: Document, moves: dict) -> None:
        super().__init__("Move")
        self.doc, self.moves = doc, moves

    def _apply(self, idx: int) -> None:
        for nid, (a, b) in self.moves.items():
            n = self.doc.pipeline.nodes.get(nid)
            if n:
                n.x, n.y = (a, b)[idx]
                self.doc.nodeMoved.emit(nid)

    def redo(self) -> None:
        self._apply(1)

    def undo(self) -> None:
        self._apply(0)


class Connect(QUndoCommand):
    def __init__(self, doc: Document, edge: Edge) -> None:
        super().__init__("Connect")
        self.doc, self.edge = doc, edge

    def redo(self) -> None:
        try:
            e = self.doc.pipeline.connect(self.edge.source, self.edge.target, self.edge.port)
        except PipelineError:
            return
        self.edge = e
        self.doc.edgeAdded.emit(e)
        self.doc.refresh_states()

    def undo(self) -> None:
        try:
            self.doc.pipeline.disconnect(self.edge.source, self.edge.target, self.edge.port)
        except PipelineError:
            return
        self.doc.edgeRemoved.emit(self.edge)
        self.doc.refresh_states()


class Disconnect(Connect):
    def __init__(self, doc: Document, edge: Edge) -> None:
        super().__init__(doc, edge)
        self.setText("Disconnect")

    def redo(self) -> None:
        Connect.undo(self)

    def undo(self) -> None:
        Connect.redo(self)


class AddNote(QUndoCommand):
    def __init__(self, doc: Document, note: Note) -> None:
        super().__init__("Add note")
        self.doc, self.note = doc, note

    def redo(self) -> None:
        self.doc.pipeline.notes.append(self.note)
        self.doc.noteAdded.emit(self.note.id)

    def undo(self) -> None:
        if self.note in self.doc.pipeline.notes:
            self.doc.pipeline.notes.remove(self.note)
            self.doc.noteRemoved.emit(self.note.id)


class RemoveNote(QUndoCommand):
    def __init__(self, doc: Document, nid: str) -> None:
        super().__init__("Delete note")
        self.doc = doc
        self.note = next(n for n in doc.pipeline.notes if n.id == nid)

    def redo(self) -> None:
        if self.note in self.doc.pipeline.notes:
            self.doc.pipeline.notes.remove(self.note)
            self.doc.noteRemoved.emit(self.note.id)

    def undo(self) -> None:
        self.doc.pipeline.notes.append(self.note)
        self.doc.noteAdded.emit(self.note.id)


class EditNote(QUndoCommand):
    def __init__(self, doc: Document, nid: str, old: dict, new: dict) -> None:
        super().__init__("Edit note")
        self.doc, self.nid, self.old, self.new = doc, nid, old, new

    def _apply(self, d: dict) -> None:
        note = next((n for n in self.doc.pipeline.notes if n.id == self.nid), None)
        if note is None:
            return
        for k, v in d.items():
            setattr(note, k, v)
        self.doc.noteChanged.emit(self.nid)

    def redo(self) -> None:
        self._apply(self.new)

    def undo(self) -> None:
        self._apply(self.old)


class SetInputs(QUndoCommand):
    def __init__(self, doc: Document, before: list, after: list, text: str) -> None:
        super().__init__(text)
        self.doc, self.before, self.after = doc, before, after

    def _apply(self, items: list) -> None:
        self.doc.pipeline.inputs = [Input(i["name"], i.get("value"), i.get("unit", ""), i.get("note", "")) for i in items]
        self.doc.inputsChanged.emit()
        self.doc.refresh_states()

    def redo(self) -> None:
        self._apply(self.after)

    def undo(self) -> None:
        self._apply(self.before)


class SetColumns(QUndoCommand):
    def __init__(self, doc: Document, before: dict, after: dict, text: str) -> None:
        super().__init__(text)
        self.doc, self.before, self.after = doc, before, after

    def _apply(self, cols: dict) -> None:
        self.doc.pipeline.columns = json.loads(json.dumps(cols))
        self.doc.columnsChanged.emit()

    def redo(self) -> None:
        self._apply(self.after)

    def undo(self) -> None:
        self._apply(self.before)
