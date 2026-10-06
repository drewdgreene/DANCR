"""Lineage: what produced a step (up), and what consumed it (down).

The step graph gives the dataflow lineage directly. The *impact* view adds the things a graph alone does not
show: the Answers whose question resolves to a step, the files the project wrote from it, and the Assistant
turns that built or discussed it. Together they answer "where did this number come from?" and "what would
break if this source changed?".

:func:`proof_card` is the one-node summary a report, an export or a reviewer can read: the chain of steps,
their plan hashes, the source files with their content samples, the result's content hash, and the exact
command that re-checks it. Pure core, no Qt.
"""
from __future__ import annotations

from typing import Any

from .executor import Executor
from .fair.manifest import run_manifest
from .registry import registry
from .verify import output_hash


def _require(pipe, node_id: str) -> str:
    if node_id not in pipe.nodes:
        raise ValueError(f"No step called {node_id!r}. Steps: {list(pipe.nodes)}")
    return node_id


def _assistant_turns(pipe, node_id: str) -> list[dict[str, Any]]:
    """The Assistant turns that built a step depending on this one (read from the thread saved in the project),
    so a source shows the turns whose work consumed it, not only a turn that named it directly."""
    try:
        from .assistant import load_thread
        thread = load_thread(pipe)
    except Exception:  # noqa: BLE001 - a project without a thread simply has no turns
        return []
    out: list[dict[str, Any]] = []
    question = ""
    for i, t in enumerate(thread.turns):
        if getattr(t, "role", "") == "user":
            question = str(getattr(t, "text", "") or "")
        tnode = getattr(t, "node", None)
        if not tnode or tnode not in pipe.nodes:
            continue
        if tnode == node_id or node_id in pipe.upstream_closure(tnode):
            out.append({"turn": i, "kind": getattr(t, "kind", ""), "node": tnode,
                        "question": question[:300], "finding": getattr(t, "finding", "")})
    return out


def _answer_nodes(a) -> set[str]:
    """Every step an Answer touches: the step it built, its terminal step, and any step named anywhere in its
    spec (the tables a question started from, so a change to a source shows the answers that depend on it)."""
    ids: set[str] = set()
    if a.terminal:
        ids.add(a.terminal)
    ids.update(a.nodes)

    def walk(o: Any) -> None:
        if isinstance(o, str):
            ids.add(o)
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)

    walk(a.spec)
    return ids


def build_lineage(pipe, node_id: str, *, direction: str = "both", executor: Executor | None = None) -> dict[str, Any]:
    """The lineage of one step: its upstream chain and/or everything downstream that depends on it."""
    _require(pipe, node_id)
    if direction not in ("up", "down", "both"):
        raise ValueError("direction must be 'up', 'down' or 'both'")
    ex = executor if executor is not None else Executor(pipe)

    up = pipe.upstream_closure(node_id)
    down = pipe.downstream_closure(node_id)

    out: dict[str, Any] = {"kind": "dancr.lineage", "version": 1, "root": node_id,
                           "title": pipe.nodes[node_id].title, "type": pipe.nodes[node_id].type}
    if direction in ("up", "both"):
        out["up"] = {
            "nodes": sorted(up),
            "edges": [[e.source, e.target] for e in pipe.edges
                      if e.target in (up | {node_id}) and e.source in up],
        }
    if direction in ("down", "both"):
        artifacts: list[dict[str, Any]] = []
        for nid in [node_id, *sorted(down)]:
            st = ex.state(nid)
            for f in sorted((st.files or {}), key=str):
                artifacts.append({"node": nid, "type": pipe.nodes[nid].type, "file": str(f)})
        answers = [{"id": a.id, "title": a.title, "terminal": a.terminal}
                   for a in pipe.answers if node_id in _answer_nodes(a)]
        out["down"] = {
            "nodes": sorted(down),
            "edges": [[e.source, e.target] for e in pipe.edges
                      if e.source in ({node_id} | down) and e.target in down],
            "answers": answers,
            "artifacts": artifacts,
            "agent_turns": _assistant_turns(pipe, node_id),
        }

    up_n = len(up)
    down_n = len(down)
    d = out.get("down") or {}
    pieces = [f"{out['title']} [{node_id}]"]
    if direction in ("up", "both"):
        pieces.append(f"is produced from {up_n} step(s)" if up_n else "is a source")
    if direction in ("down", "both"):
        bits = [f"{down_n} later step(s)"]
        if d.get("answers"):
            bits.append(f"{len(d['answers'])} answer(s)")
        if d.get("artifacts"):
            bits.append(f"{len(d['artifacts'])} file(s)")
        if d.get("agent_turns"):
            bits.append(f"{len(d['agent_turns'])} agent turn(s)")
        pieces.append("feeds " + ", ".join(bits))
    out["sentence"] = " ".join(pieces) + "." if len(pieces) == 2 else ", and ".join(pieces) + "."
    return out


def proof_card(pipe, executor: Executor | None = None, node_id: str | None = None, *,
               any_node: str | None = None) -> dict[str, Any]:
    """A one-node, human-readable proof: the chain of steps with their plan hashes, the source files with
    their content samples, the result's content hash and finding, and the command that re-checks it."""
    nid = _require(pipe, node_id or any_node or "")
    ex = executor if executor is not None else Executor(pipe)
    man = run_manifest(pipe, ex)
    chain = pipe.upstream_closure(nid) | {nid}

    steps = [{"id": n["id"], "type": n["type"], "title": n["title"], "plan_hash": n.get("hash"), "rows": n.get("rows")}
             for n in man["nodes"] if n["id"] in chain]
    sources: list[dict[str, Any]] = []
    for s in man["sources"]:
        if s["node"] in chain:
            sources += s.get("files") or []

    st = ex.state(nid)
    nt = registry.get(pipe.nodes[nid].type)
    assumptions: list[str] = []
    for a in pipe.answers:
        if nid in _answer_nodes(a):
            for x in a.assumptions or []:
                text = x.get("text") if isinstance(x, dict) else str(x)
                if text:
                    assumptions.append(str(text))

    project = pipe.path.name if pipe.path else "project.json"
    return {
        "kind": "dancr.proof",
        "version": 1,
        "node": nid,
        "title": pipe.nodes[nid].title,
        "type": pipe.nodes[nid].type,
        "steps": steps,
        "sources": sources,
        "materialize": bool(nt.materialize),
        "output_hash": output_hash(ex, nid) if nt.materialize else None,
        "rows": st.rows,
        "columns": list(st.columns or []),
        "finding": ((st.report or {}).get("finding") or {}).get("statement", ""),
        "assumptions": assumptions,
        "verify": f"dancr verify {project} --manifest <attestation.json>",
    }
