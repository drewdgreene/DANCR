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

import json
import os
from pathlib import Path
from typing import Any

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
        # A tiny sidecar with the last sequence number, so a fresh EventLog (each append_event call) does not
        # rescan the whole log to find the next number: N appends would otherwise read the log N times (O(N²)).
        self._seq_path = self.path.with_name(self.path.name + ".seq")
        self._last: int | None = None

    def _last_seq_in_log(self) -> int:
        """The ``seq`` of the last complete line in the log, read from the tail only (cheap on a long log)."""
        try:
            with open(self.path, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - 65536))            # events are small; the last line is well within this
                tail = f.read().splitlines()
        except OSError:
            return 0
        for line in reversed(tail):                     # a torn last line is skipped for the one before it
            if not line.strip():
                continue
            try:
                return int(json.loads(line).get("seq", 0))
            except (ValueError, TypeError):
                continue
        return 0

    def _read_last_seq(self) -> int:
        sidecar = -1
        try:
            n = int(self._seq_path.read_text().strip())
            if n >= 0:
                sidecar = n
        except (OSError, ValueError):
            pass
        # Reconcile the sidecar with the log: a crash between the append and the sidecar write leaves the sidecar
        # stale, and trusting it alone would reissue a sequence number. The log is authoritative.
        return max(sidecar, self._last_seq_in_log()) if sidecar >= 0 else self._last_seq_in_log()

    def last_seq(self) -> int:
        if self._last is None:
            self._last = self._read_last_seq()
        return self._last

    def append(self, kind: str, **fields: Any) -> dict[str, Any]:
        seq = self.last_seq() + 1
        record = {"seq": seq, **make_event(kind, **fields)}
        append_jsonl(self.path, record)
        self._last = seq
        try:                                  # the caller holds the repository lock, so a plain write is safe
            self._seq_path.write_text(str(seq))
        except OSError:
            pass
        return record

    def read(self, *, since: int = 0, type: str | None = None, limit: int | None = None) -> list[dict[str, Any]]:
        """Events with ``seq > since``, optionally of one type, in order. ``limit`` keeps the most recent ones.
        A damaged line's sequence is skipped rather than raising."""
        out: list[dict[str, Any]] = []
        for r in read_jsonl(self.path):
            try:
                seq = int(r.get("seq", 0))
            except (ValueError, TypeError):
                continue
            if seq > since and (type is None or r.get("type") == type):
                out.append(r)
        if limit is not None and limit >= 0 and len(out) > limit:
            out = out[-limit:] if limit > 0 else []
        return out
