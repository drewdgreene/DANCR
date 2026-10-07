"""Structural cross-project questions (roadmap C2).

A deterministic, model-free grammar over the entity graph: *what relates to X*,
*what joins A and B*, *which keys link projects*, *where does X come from*, *what
feeds X*, *find X*. Every answer cites the edges (and their evidence) that support
it. There is **no cross-project execution**: a structural answer is a fact about
the graph, and the caller may build a join into one project as a follow-up.

Pure: operates on a :class:`dancr.core.graph.Graph`, no Qt, no execution.
"""
from __future__ import annotations

from typing import Any

from .graph import Graph

INTENTS = ("path", "shared_keys", "upstream", "downstream", "neighbors", "find")

_PATH = ("path between", "path from", "join path", "how do", "how does", "connect", "joins", "join ")
_SHARED = ("shared key", "shared keys", "common key", "common keys", "which keys", "keys that link", "keys link")
_UP = ("come from", "comes from", "upstream", "lineage", "where does", "where do", "source of", "produced by", "provenance")
_DOWN = ("what feeds", "feeds ", "downstream", "depends on", "what uses", "what uses", "impact of", "what breaks")
_NEIGH = ("relates to", "related to", "what relates to", "relate", "relates", "related", "neighbour", "neighbor",
          "neighbours", "neighbors", "connected to", "links to", "what is near")


def _datasets(graph: Graph, allow_restricted: bool) -> list[Any]:
    return graph.all_datasets(allow_restricted=allow_restricted)


def resolve(graph: Graph, needle: str, *, allow_restricted: bool = False) -> tuple[Any | None, list[Any]]:
    """The dataset a name refers to: an exact id, then an exact title or node, then a unique substring of the id,
    title, node or a column name. Returns ``(dataset, candidates)``; candidates are non-empty when ambiguous."""
    datasets = _datasets(graph, allow_restricted)
    n = str(needle or "").strip().lower()
    if not n:
        return None, []
    for d in datasets:
        if d.id.lower() == n:
            return d, []
    exact = [d for d in datasets if d.title.lower() == n or d.node.lower() == n]
    if len(exact) == 1:
        return exact[0], []
    sub = [d for d in datasets if n in d.id.lower() or n in d.title.lower() or n in d.node.lower()
           or any(n in c.name.lower() for c in d.columns)]
    if len(sub) == 1:
        return sub[0], []
    if len(n) >= 3:                              # the name appears *inside* the question's phrase
        inside = [d for d in datasets if d.id.lower() in n or d.title.lower() in n or d.node.lower() in n]
        if len(inside) == 1:
            return inside[0], []
        sub = sub or inside
    return None, sub[:10]


def _split_two(text: str) -> tuple[str, str]:
    for sep in (" and ", " to ", " vs ", " versus ", "→", "->"):
        if sep in text:
            a, b = text.split(sep, 1)
            return a.strip(" ?."), b.strip(" ?.")
    return "", ""


def _strip(text: str, prefixes: tuple[str, ...]) -> str:
    low = text.lower()
    for p in prefixes:
        i = low.find(p)
        if i >= 0:
            rest = text[i + len(p):].strip(" ?.")
            return rest
    return text.strip(" ?.")


def _strip_best(graph: Graph, text: str, prefixes: tuple[str, ...], *, allow_restricted: bool) -> str:
    """The name a question refers to: the shortest phrase left after removing an intent keyword (or the whole
    question) that resolves to a dataset; else the longest non-empty one."""
    low = text.lower()
    candidates = [text[i + len(p):].strip(" ?.") for p in prefixes if (i := low.find(p)) >= 0]
    candidates.append(text.strip(" ?."))

    def interesting(c: str) -> bool:
        d, cand = resolve(graph, c, allow_restricted=allow_restricted)
        return d is not None or bool(cand)          # a dataset, or a name with candidates to disambiguate

    resolving = [c for c in candidates if c and interesting(c)]
    if resolving:
        return min(resolving, key=len)
    nonempty = [c for c in candidates if c]
    return max(nonempty, key=len) if nonempty else ""


def classify(question: str) -> str:
    q = (question or "").lower()
    if any(k in q for k in _SHARED):
        return "shared_keys"
    if any(k in q for k in _PATH):
        return "path"
    if any(k in q for k in _UP):
        return "upstream"
    if any(k in q for k in _DOWN):
        return "downstream"
    if any(k in q for k in _NEIGH):
        return "neighbors"
    return "find"


def ask(graph: Graph, question: str, *, allow_restricted: bool = False) -> dict[str, Any]:
    """Answer a structural question against the graph, or explain why it cannot (with candidates when a name is
    ambiguous). Deterministic: the same graph and question always give the same answer."""
    intent = classify(question)
    out: dict[str, Any] = {"kind": "dancr.crossask", "version": 1, "question": question, "intent": intent}

    if intent == "shared_keys":
        keys = graph.shared_keys(allow_restricted=allow_restricted)
        out.update({"ok": True, "answer": f"{len(keys)} key(s) link datasets across different projects",
                    "keys": keys, "evidence": keys})
        return out

    if intent == "path":
        a_needle, b_needle = _split_two(_strip(question, _PATH))
        da, ca = resolve(graph, a_needle, allow_restricted=allow_restricted)
        db, cb = resolve(graph, b_needle, allow_restricted=allow_restricted)
        if da is None or db is None:
            return _ambiguous(out, (a_needle, ca), (b_needle, cb))
        p = graph.path(da.id, db.id, allow_restricted=allow_restricted)
        edge_dicts = [graph.edges[e].to_dict() for e in p["edges"] if e in graph.edges]
        out.update({"ok": p["found"], "datasets": p["datasets"], "edges": edge_dicts, "evidence": edge_dicts, "hops": p["hops"],
                    "answer": (f"{da.id} and {db.id} are joined by {p['hops']} relation(s): "
                               + " → ".join(p["datasets"])) if p["found"] else f"No relation joins {da.id} and {db.id}"})
        return out

    if intent in ("neighbors", "upstream", "downstream"):
        needle = _strip_best(graph, question, _UP if intent == "upstream" else (_DOWN if intent == "downstream" else _NEIGH),
                             allow_restricted=allow_restricted)
        d, candidates = resolve(graph, needle, allow_restricted=allow_restricted)
        if d is None:
            return _ambiguous(out, (needle, candidates))
        if intent == "neighbors":
            edges = graph.edges_of(d.id, allow_restricted=allow_restricted)
            out["answer"] = f"{d.title} relates to {len(edges)} dataset(s)"
        else:
            edges = _directed(graph, d.id, up=(intent == "upstream"), allow_restricted=allow_restricted)
            out["answer"] = f"{d.title} {'comes from' if intent == 'upstream' else 'feeds'} {len(edges)} dataset(s)"
        out.update({"ok": True, "dataset": d.to_dict(), "edges": [e.to_dict() for e in edges],
                    "evidence": [e.to_dict() for e in edges]})
        return out

    # find
    needle = _strip(question, ("find ", "which datasets", "what datasets", "show me", "list "))
    res = graph.query(text=needle or None, allow_restricted=allow_restricted)
    out.update({"ok": True, "datasets": res["datasets"], "edges": res["edges"],
                "answer": f"{res['count']} dataset(s) match {needle!r}", "evidence": res["edges"]})
    return out


def _directed(graph: Graph, dataset_id: str, *, up: bool, allow_restricted: bool) -> list[Any]:
    """The edges that lead into (upstream) or out of (downstream) a dataset, one hop, deterministic. A link is
    treated as a dependency in both directions, so a shared key shows on either side."""
    hidden = graph._hidden(allow_restricted)      # noqa: SLF001 - the graph owns its sensitivity rule
    out = []
    for e in sorted(graph.edges.values(), key=lambda x: x.id):
        if e.left in hidden or e.right in hidden:
            continue
        if e.kind == "link" and dataset_id in e.endpoints:
            out.append(e)                         # a shared key relates both ways
        elif up and e.right == dataset_id or not up and e.left == dataset_id:
            out.append(e)
    return out


def _ambiguous(out: dict[str, Any], *pairs: tuple[str, list[Any]]) -> dict[str, Any]:
    names = [p[0] for p in pairs if p[0]]
    candidates = [d.id for p in pairs for d in p[1]]
    out.update({"ok": False, "answer": f"Could not find a dataset named {', '.join(repr(n) for n in names)}",
                "candidates": candidates, "evidence": []})
    return out


def suggest(graph: Graph, *, allow_restricted: bool = False) -> list[dict[str, Any]]:
    """The structural questions the graph can answer, best first, grounded in what is actually there."""
    out: list[dict[str, Any]] = []
    if graph.shared_keys(allow_restricted=allow_restricted):
        out.append({"intent": "shared_keys", "question": "which keys link datasets across projects?"})
    links = graph.all_edges(allow_restricted=allow_restricted, kind="link")
    if links:
        e = links[0]
        out.append({"intent": "neighbors", "question": f"what relates to {e.left}?"})
        out.append({"intent": "path", "question": f"what joins {e.left} and {e.right}?"})
    visible = graph.all_datasets(allow_restricted=allow_restricted)
    if visible:
        out.append({"intent": "upstream", "question": f"where does {visible[0].id} come from?"})
        out.append({"intent": "find", "question": "find orders"})
    return out
