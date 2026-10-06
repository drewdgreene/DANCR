"""The undoable edits an Answer plan makes, expressed as Document commands."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..core import registry
from ..core.model import Edge

if TYPE_CHECKING:
    from .document import Document


class _DocEdits:
    """Plan edits as undoable commands (inside the caller's undo macro)."""

    def __init__(self, doc: "Document") -> None:
        self.doc = doc
        self.pipe = doc.pipeline

    def create(self, step, ins: dict[str, list[str]], x: float, y: float) -> str:
        nid = self.doc.add_node(step.type, x, y, params=step.params, title=step.title)
        for port, srcs in ins.items():
            for s in srcs:
                self.doc.connect(s, nid, port)
        return nid

    def set_params(self, nid: str, params: dict[str, Any]) -> None:
        node = self.doc.pipeline.nodes[nid]
        full = registry.get(node.type).normalize_params(params)
        self.doc.set_params(nid, {k: v for k, v in full.items() if node.params.get(k) != v})

    def set_title(self, nid: str, title: str) -> None:
        self.doc.rename(nid, title)

    def connect(self, source: str, target: str, port: str) -> None:
        self.doc.connect(source, target, port)

    def disconnect(self, source: str, target: str, port: str) -> None:
        self.doc.disconnect(Edge(source, target, port))

    def remove(self, ids: list[str]) -> None:
        self.doc.remove_nodes(ids)
