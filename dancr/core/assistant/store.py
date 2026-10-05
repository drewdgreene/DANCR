"""The conversation, saved with the project.

The thread lives in the project's ``meta`` under one key, so the project file format does not change and an
older DANCR simply ignores it (exactly as it ignores the words a project has learned). Each turn keeps just
enough to be trusted and replayed: the question, the proposal, the assumptions, and which step or answer it
built. It never holds raw model prompts or the project's data.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

from ..dtypes import json_safe

META_KEY = "assistant"
FORMAT = 1
MAX_TURNS = 40
MAX_TEXT = 4000


@dataclass
class Turn:
    role: str                       # user | assistant
    text: str = ""
    kind: str = ""                  # answer | steps | text | error | paused | running
    proposal: dict[str, Any] | None = None
    finding: str = ""               # the engine's own sentence, once the proposal has run
    node: str | None = None         # the terminal step the proposal built
    answer: str | None = None       # the Answer it made, if any
    assumptions: list[str] = field(default_factory=list)
    next_questions: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    unverified: list[str] = field(default_factory=list)   # figures in the reply no tool result backed
    allowed: list[str] = field(default_factory=list)      # figures this turn's tools/profile did back (for later turns)
    usage: dict[str, int] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Turn":
        known = {k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class Thread:
    turns: list[Turn] = field(default_factory=list)
    chips: list[str] = field(default_factory=list)      # attached step ids
    connections: list[dict[str, Any]] = field(default_factory=list)   # the last connection map the engine reported
    model: str = ""
    allow_samples: bool = False
    updated: float = 0.0

    def add(self, turn: Turn) -> Turn:
        self.turns.append(turn)
        if len(self.turns) > MAX_TURNS:
            self.turns = self.turns[-MAX_TURNS:]
        self.updated = time.time()
        return turn

    def to_dict(self) -> dict[str, Any]:
        return {"v": FORMAT, "turns": [t.to_dict() for t in self.turns][-MAX_TURNS:],
                "chips": [str(c) for c in self.chips][:20], "connections": json_safe(self.connections[:40]),
                "model": self.model, "allow_samples": bool(self.allow_samples), "updated": float(self.updated)}

    @classmethod
    def from_dict(cls, d: Any) -> "Thread":
        if not isinstance(d, dict):
            return cls()
        turns: list[Turn] = []
        for t in (d.get("turns") or [])[-(MAX_TURNS + 20):]:
            if isinstance(t, dict):
                try:
                    turn = Turn.from_dict(t)
                    turn.text = str(turn.text or "")[:MAX_TEXT]
                    turns.append(turn)
                except (TypeError, ValueError):
                    continue
        conns = [c for c in (d.get("connections") or []) if isinstance(c, dict)][:40]
        return cls(turns=turns, chips=[str(c) for c in (d.get("chips") or []) if c][:20], connections=conns,
                   model=str(d.get("model") or ""), allow_samples=bool(d.get("allow_samples")),
                   updated=float(d.get("updated") or 0.0))


def load_thread(pipeline) -> Thread:
    """The project's saved conversation, or an empty one."""
    meta = getattr(pipeline, "meta", None) or {}
    return Thread.from_dict(meta.get(META_KEY))


def save_thread(pipeline, thread: Thread) -> None:
    """Write the conversation into the project's meta (in memory; the caller saves the file)."""
    if not hasattr(pipeline, "meta") or pipeline.meta is None:
        return
    pipeline.meta[META_KEY] = thread.to_dict()


def clear_thread(pipeline) -> None:
    if getattr(pipeline, "meta", None) is not None:
        pipeline.meta.pop(META_KEY, None)
