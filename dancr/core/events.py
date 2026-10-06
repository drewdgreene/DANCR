"""The repository event log: a typed, append-only record of what changed and what it affects.

Events are the "living" half of the context layer (roadmap A2): a source file
changed, a dataset was invalidated, a project was recomputed, the graph was
updated, an index went stale, a gateway action was taken. The log is one JSON
object per line under ``<root>/.dancr/events/events.jsonl`` with a monotonic
``seq``; it is append-only and never rewritten.

This module is pure (standard library + :mod:`dancr.core.repo`). Reading, ripple
invalidation and the repository watcher live in :mod:`dancr.headless._events`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from .repo import append_jsonl, read_jsonl

EVENT_VERSION = 1
KINDS = ("source_changed", "dataset_invalidated", "project_recomputed", "index_stale",
         "graph_updated", "answer_stale", "gateway_action")


def make_event(kind: str, **fields: Any) -> dict[str, Any]:
    """One event: its kind plus whatever fields describe it. Unknown kinds are refused so the vocabulary stays
    closed and downstream consumers can rely on it."""
    if kind not in KINDS:
        raise ValueError(f"Unknown event kind {kind!r}. Known: {', '.join(KINDS)}")
    return {"kind": "dancr.event", "version": EVENT_VERSION, "type": kind,
            **{k: fields[k] for k in sorted(fields)}}


class EventLog:
    """An append-only log at one path. ``append`` assigns the next sequence number; the caller must hold the
    repository lock so two writers never pick the same one."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path).expanduser()
        self._last: int | None = None

    def last_seq(self) -> int:
        if self._last is None:
            self._last = max((int(r.get("seq", 0)) for r in read_jsonl(self.path)), default=0)
        return self._last

    def append(self, kind: str, **fields: Any) -> dict[str, Any]:
        seq = self.last_seq() + 1
        record = {"seq": seq, **make_event(kind, **fields)}
        append_jsonl(self.path, record)
        self._last = seq
        return record

    def read(self, *, since: int = 0, type: str | None = None, limit: int | None = None) -> list[dict[str, Any]]:
        """Events with ``seq > since``, optionally of one type, in order. ``limit`` keeps the most recent ones."""
        out = [r for r in read_jsonl(self.path) if int(r.get("seq", 0)) > since and (type is None or r.get("type") == type)]
        if limit is not None and limit >= 0 and len(out) > limit:
            out = out[-limit:]
        return out
