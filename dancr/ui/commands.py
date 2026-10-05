"""Undo commands for Document edits. Each one mutates ``doc.pipeline`` and emits the matching signals."""
from __future__ import annotations

import copy
import json
import time
from typing import TYPE_CHECKING

from PySide6.QtGui import QUndoCommand

from ..core import PipelineError
from ..core.model import Node, Edge, Note, Input, Answer

if TYPE_CHECKING:
    from .document import Document

# Commands that hold settings implement ``rebase(fn)``: after Save As into another folder, fn(type, params)
# rewrites path settings so undo and redo still point at the same files. Nodes that are in the project
# right now were already rewritten by the save and are left alone.


class AddNode(QUndoCommand):
    def __init__(self, doc: Document, node: Node, connect_from: str | None, port: str | None) -> None:
        super().__init__(f"Add {node.title}")
        self.doc, self.node, self.connect_from, self.port = doc, node, connect_from, port
        self.edge: Edge | None = None

    def rebase(self, fn) -> None:
        if self.doc.pipeline.nodes.get(self.node.id) is not self.node:
            self.node.params = fn(self.node.type, self.node.params)

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
        # per removed node, its connections with their places in the edge list: the order of a step's
        # inputs (stack, workbook, report) is the order of the list, so undo puts each one back in place
        self.removed: list[list[tuple[int, Edge]]] = []

    def rebase(self, fn) -> None:
        for n in self.nodes:
            if self.doc.pipeline.nodes.get(n.id) is not n:
                n.params = fn(n.type, n.params)

    def redo(self) -> None:
        p = self.doc.pipeline
        self.removed = []
        for n in self.nodes:
            if n.id not in p.nodes:
                continue
            placed = [(i, e) for i, e in enumerate(p.edges) if e.source == n.id or e.target == n.id]
            for e in p.remove_node(n.id):
                self.doc.edgeRemoved.emit(e)
            self.removed.append(placed)
            self.doc.nodeRemoved.emit(n.id)
        self.doc.refresh_states()

    def undo(self) -> None:
        p = self.doc.pipeline
        for n in self.nodes:
            p.nodes[n.id] = n
            self.doc.nodeAdded.emit(n.id)
        for placed in reversed(self.removed):
            for i, e in placed:
                if e.key() not in {x.key() for x in p.edges} and e.source in p.nodes and e.target in p.nodes:
                    p.edges.insert(min(i, len(p.edges)), e)
                    self.doc.edgeAdded.emit(e)
        self.doc.refresh_states()


class SetParams(QUndoCommand):
    def __init__(self, doc: Document, nid: str, old: dict, new: dict, keys: set[str]) -> None:
        super().__init__(f"Edit {doc.pipeline.nodes[nid].title}")
        self.doc, self.nid, self.old, self.new, self.keys = doc, nid, old, new, keys
        self.type = doc.pipeline.nodes[nid].type
        self.at = time.time()

    def rebase(self, fn) -> None:
        self.old, self.new = fn(self.type, self.old), fn(self.type, self.new)

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
            self.doc.refresh_states()          # titles shape what file-writing steps write

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
        self.index: int | None = None       # where the edge sat when it was detached, so it comes back in place

    def _attach(self) -> None:
        p = self.doc.pipeline
        try:
            e = p.connect(self.edge.source, self.edge.target, self.edge.port)
        except PipelineError:
            return
        if self.index is not None and p.edges and p.edges[-1] is e:
            p.edges.insert(min(self.index, len(p.edges) - 1), p.edges.pop())
        self.edge = e
        self.doc.edgeAdded.emit(e)
        self.doc.refresh_states()

    def _detach(self) -> None:
        p = self.doc.pipeline
        idx = next((i for i, e in enumerate(p.edges) if e.key() == self.edge.key()), None)
        if idx is None:
            return
        self.index = idx
        p.disconnect(self.edge.source, self.edge.target, self.edge.port)
        self.doc.edgeRemoved.emit(self.edge)
        self.doc.refresh_states()

    def redo(self) -> None:
        self._attach()

    def undo(self) -> None:
        self._detach()


class Disconnect(Connect):
    def __init__(self, doc: Document, edge: Edge) -> None:
        super().__init__(doc, edge)
        self.setText("Disconnect")

    def redo(self) -> None:
        self._detach()

    def undo(self) -> None:
        self._attach()


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
        self.doc.refresh_states()          # labels shape what file-writing steps write

    def redo(self) -> None:
        self._apply(self.after)

    def undo(self) -> None:
        self._apply(self.before)


class AddAnswer(QUndoCommand):
    def __init__(self, doc: Document, answer: Answer) -> None:
        super().__init__(f"Add answer {answer.title}")
        self.doc, self.answer = doc, answer

    def redo(self) -> None:
        if not any(a.id == self.answer.id for a in self.doc.pipeline.answers):
            self.doc.pipeline.answers.append(self.answer)
        self.doc.answerAdded.emit(self.answer.id)
        self.doc.refresh_states()

    def undo(self) -> None:
        if any(a.id == self.answer.id for a in self.doc.pipeline.answers):
            self.doc.pipeline.answers.remove(self.answer)
            self.doc.answerRemoved.emit(self.answer.id)
        self.doc.refresh_states()


class RemoveAnswer(QUndoCommand):
    def __init__(self, doc: Document, answer: Answer) -> None:
        super().__init__(f"Delete answer {answer.title}")
        self.doc, self.answer = doc, answer

    def redo(self) -> None:
        if any(a.id == self.answer.id for a in self.doc.pipeline.answers):
            self.doc.pipeline.answers.remove(self.answer)
            self.doc.answerRemoved.emit(self.answer.id)

    def undo(self) -> None:
        if not any(a.id == self.answer.id for a in self.doc.pipeline.answers):
            self.doc.pipeline.answers.append(self.answer)
            self.doc.answerAdded.emit(self.answer.id)


class EditAnswer(QUndoCommand):
    def __init__(self, doc: Document, answer_id: str, before: dict, after: dict, text: str = "Change answer") -> None:
        super().__init__(text)
        self.doc, self.answer_id, self.before, self.after = doc, answer_id, before, after

    def _apply(self, fields: dict) -> None:
        answer = self.doc.pipeline.answer(self.answer_id)
        if answer is None:
            return
        for k, v in fields.items():
            setattr(answer, k, v)
        self.doc.answerChanged.emit(self.answer_id)

    def redo(self) -> None:
        self._apply(self.after)

    def undo(self) -> None:
        self._apply(self.before)


class SetThread(QUndoCommand):
    """The Assistant's conversation changed. It lives in ``meta``, so it is not part of the dataflow; keeping
    it as a command makes the change undoable and marks the project as changed so it is saved with the file."""

    def __init__(self, doc: Document, before: dict | None, after: dict | None, text: str = "Assistant") -> None:
        super().__init__(text)
        self.doc, self.before, self.after = doc, before, after

    def _apply(self, value: dict | None) -> None:
        meta = self.doc.pipeline.meta
        if value is None:
            meta.pop("assistant", None)
        else:
            meta["assistant"] = value
        self.doc.message.emit("Assistant conversation saved with the project")

    def redo(self) -> None:
        self._apply(self.after)

    def undo(self) -> None:
        self._apply(self.before)


class SetDatasetMeta(QUndoCommand):
    """The project's dataset-level metadata (creator, license, description…) changed. It lives in ``meta``, so it
    is not part of the dataflow, but keeping it as a command makes the change undoable and saves it with the file."""

    def __init__(self, doc: "Document", before: dict | None, after: dict | None, text: str = "Dataset details") -> None:
        super().__init__(text)
        self.doc, self.before, self.after = doc, before, after

    def _apply(self, value: dict | None) -> None:
        meta = self.doc.pipeline.meta
        if value:
            meta["dataset"] = copy.deepcopy(value)
        else:
            meta.pop("dataset", None)
        self.doc.message.emit("Dataset details saved with the project")

    def redo(self) -> None:
        self._apply(self.after)

    def undo(self) -> None:
        self._apply(self.before)


class ReplacePipeline(QUndoCommand):
    """Replace the whole canvas (nodes, edges, answers, inputs, columns, meta) as one undo step. Used by the
    Assistant's "replace the canvas with this": the file and the cache stay, so shared steps keep their results."""

    def __init__(self, doc: Document, before: dict, after: dict, text: str = "Replace the canvas") -> None:
        super().__init__(text)
        self.doc, self.before, self.after = doc, before, after

    def _apply(self, data: dict) -> None:
        from ..core.model import Pipeline
        p = Pipeline.from_dict(json.loads(json.dumps(data)), self.doc.pipeline.path)
        cur = self.doc.pipeline
        cur.name = p.name
        cur.nodes, cur.edges, cur.notes = p.nodes, p.edges, p.notes
        cur.answers, cur.inputs, cur.columns, cur.meta = p.answers, p.inputs, p.columns, p.meta
        self.doc._states_cache = {}
        try:
            self.doc._rewatch()
        except Exception:  # noqa: BLE001 - a watcher hiccup must not undo the replacement
            pass
        self.doc.reloaded.emit()
        self.doc.refresh_states()

    def redo(self) -> None:
        self._apply(self.after)

    def undo(self) -> None:
        self._apply(self.before)
