"""What a project remembers about how its person words questions.

The grammar and the lexicon let a person ask in the project's own vocabulary; memory lets them ask in *their*
vocabulary. Two small things, kept in the project file's ``meta`` so they travel with it and never change the
project format:

- **aliases**: a word the person keeps using for something ("turnover" for the sales column), learned from a
  spelling repair the first time and reused instantly after;
- **recent**: the last few questions asked, so they can be offered back as one click.

This is not AI and not a model: it is a small dictionary that grows from use. It changes the questions asked
of a project, never the data or the numbers.
"""
from __future__ import annotations

from typing import Any

KEY = "ask"
MAX_RECENT = 12


def _store(pipeline: Any) -> dict[str, Any]:
    meta = getattr(pipeline, "meta", None)
    if not isinstance(meta, dict):
        return {}
    store = meta.get(KEY)
    if not isinstance(store, dict):
        store = {}
        meta[KEY] = store
    return store


def aliases(pipeline: Any) -> dict[str, str]:
    """The words this project has learned to read, as {typed word (lower-case): the project's word}."""
    a = _store(pipeline).get("aliases") or {}
    if not isinstance(a, dict):          # a hand-edited or damaged project: read it as nothing learned
        return {}
    return {str(k).strip().lower(): str(v).strip() for k, v in a.items()
            if str(k).strip() and str(v).strip()}


def remember_alias(pipeline: Any, typed: Any, means: Any) -> None:
    """Learn that a word the person types means one of the project's own words. Blank or identical pairs are ignored."""
    k = str(typed or "").strip().lower()
    v = str(means or "").strip()
    if not k or not v or k == v:
        return
    store = _store(pipeline)
    if not isinstance(store.get("aliases"), dict):       # a hand-edited/damaged project: start the map afresh
        store["aliases"] = {}
    store["aliases"][k] = v


def remember_corrections(pipeline: Any, corrected: list[dict[str, Any]] | None) -> None:
    """Learn every spelling repair a question needed, so the same word is understood next time."""
    for c in corrected or []:
        if isinstance(c, dict):
            remember_alias(pipeline, c.get("from"), c.get("to"))


def recent(pipeline: Any, limit: int = MAX_RECENT) -> list[str]:
    """The questions asked of this project, newest first."""
    r = _store(pipeline).get("recent") or []
    return [str(x) for x in r[:limit]] if isinstance(r, list) else []


def remember_question(pipeline: Any, text: Any, limit: int = MAX_RECENT) -> None:
    """Keep a question in the recent list, newest first, without duplicates."""
    t = " ".join(str(text or "").split())
    if not t:
        return
    store = _store(pipeline)
    r = store.setdefault("recent", [])
    if not isinstance(r, list):
        r = store["recent"] = []
    if t in r:
        r.remove(t)
    r.insert(0, t)
    del r[limit:]
