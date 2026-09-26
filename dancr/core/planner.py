"""Plans: the steps an answer needs, and how they land on the map.

A :class:`Plan` names steps and wires them; ``apply_plan`` maps each step to a node of the project, reusing
one that already computes the same thing (so two answers over the same files share their links) and
creating the rest, or updating an answer's own steps in place. A step of type ``"@"`` is an existing node,
named by ``params["node"]``.
No execution happens here, and nothing touches Qt. The plans themselves come from ``recipes.py``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PlanStep:
    key: str
    type: str
    title: str
    params: dict[str, Any]
    inputs: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class Plan:
    steps: list[PlanStep]
    terminal: str
    title: str
    config: dict[str, Any] = field(default_factory=dict)      # the spec this plan answers
    view: str = "chart"
    assumptions: list[dict[str, Any]] = field(default_factory=list)
    why: str = ""
    chips: list[dict[str, Any]] = field(default_factory=list)
    set_aside: list[str] = field(default_factory=list)     # after building: titles of hand-edited steps left as they were

    @property
    def new_steps(self) -> list[PlanStep]:
        return [s for s in self.steps if s.type != "@"]

    def sentence(self) -> str:
        """A short plain-English description of what will be built."""
        return " → ".join(s.title for s in self.new_steps if s.title)

    def to_dict(self) -> dict[str, Any]:
        return {"title": self.title, "view": self.view, "spec": self.config, "why": self.why,
                "assumptions": self.assumptions, "chips": self.chips, "terminal": self.terminal,
                "steps": [{"key": s.key, "type": s.type, "title": s.title, "params": s.params, "inputs": s.inputs}
                          for s in self.steps]}


# ------------------------------------------------------------------- reuse
def signature(node_type: str, params: dict[str, Any], inputs: dict[str, list[str]], directory: Path | None = None) -> str:
    """A structural fingerprint of a step, so the same computation is shared instead of duplicated.

    ``inputs`` keeps its order (a stack's table order matters), and the ids are the resolved node ids
    of the inputs, so identical steps that read the same upstream nodes share one node. Params are
    normalised first (defaults filled in), so a minimal plan matches a node that already exists, and
    file settings are compared as the files they name (``data.csv`` and ``/proj/data.csv`` are one file)."""
    from .registry import registry, resolve_path
    try:
        nt = registry.get(node_type)
        norm = nt.normalize_params(params, strict=False)
        if directory is not None:
            for p in nt.params:
                if p.kind == "path" and isinstance(norm.get(p.name), str) and norm[p.name].strip():
                    norm[p.name] = str(resolve_path(directory, norm[p.name]).resolve())
    except Exception:  # noqa: BLE001 - an unknown type still gets a stable fingerprint
        norm = params
    return json.dumps({"t": node_type, "p": norm, "i": inputs}, sort_keys=True, default=str)


def signatures_of(pipe) -> dict[str, str]:
    """Map every existing node's signature to its id, for reuse when applying a plan."""
    out: dict[str, str] = {}
    for nid, node in pipe.nodes.items():
        ins = {port: list(srcs) for port, srcs in pipe.inputs_of(nid).items()}
        out.setdefault(signature(node.type, node.params, ins, pipe.directory), nid)
    return out


def place_near(pipe, inputs: list[str], taken: list[tuple[float, float]] | None = None) -> tuple[float, float]:
    """Where a new step goes: one column right of its inputs, level with the first, never on another step."""
    ups = [pipe.nodes[n] for n in inputs if n in pipe.nodes]
    if ups:
        x, y = max(n.x for n in ups) + 290.0, ups[0].y
    elif pipe.nodes:
        x, y = 60.0, max(n.y for n in pipe.nodes.values()) + 160.0
    else:
        x, y = 60.0, 140.0
    spots = [(n.x, n.y) for n in pipe.nodes.values()] + list(taken or [])
    while any(abs(sx - x) < 230 and abs(sy - y) < 130 for sx, sy in spots):
        y += 155.0
    return x, y


class Edits:
    """What applying a plan may do to a project. ``PipelineEdits`` changes a Pipeline directly; the window
    passes its own, which makes each change an undoable command."""

    def __init__(self, pipe) -> None:
        self.pipe = pipe

    def create(self, step: PlanStep, ins: dict[str, list[str]], x: float, y: float) -> str:
        raise NotImplementedError

    def set_params(self, nid: str, params: dict[str, Any]) -> None:
        raise NotImplementedError

    def set_title(self, nid: str, title: str) -> None:
        raise NotImplementedError

    def connect(self, source: str, target: str, port: str) -> None:
        raise NotImplementedError

    def disconnect(self, source: str, target: str, port: str) -> None:
        raise NotImplementedError

    def remove(self, ids: list[str]) -> None:
        raise NotImplementedError


class PipelineEdits(Edits):
    def create(self, step, ins, x, y):
        node = self.pipe.add_node(step.type, title=step.title, params=step.params, x=x, y=y)
        for port, srcs in ins.items():
            for s in srcs:
                self.pipe.connect(s, node.id, port)
        return node.id

    def set_params(self, nid, params):
        from .registry import registry
        node = self.pipe.nodes[nid]
        node.params = registry.get(node.type).normalize_params(params)

    def set_title(self, nid, title):
        self.pipe.rename_node(nid, title)

    def connect(self, source, target, port):
        self.pipe.connect(source, target, port)

    def disconnect(self, source, target, port):
        self.pipe.disconnect(source, target, port)

    def remove(self, ids):
        for nid in ids:
            if nid in self.pipe.nodes:
                self.pipe.remove_node(nid)


def _norm_params(node_type: str, params: dict[str, Any]) -> str:
    from .registry import registry
    try:
        params = registry.get(node_type).normalize_params(params, strict=False)
    except Exception:  # noqa: BLE001
        pass
    return json.dumps(params, sort_keys=True, default=str)


def protected_nodes(pipe, answer_id: str | None) -> set[str]:
    """Steps another answer depends on: changing or removing them would change that answer."""
    out: set[str] = set()
    for a in pipe.answers:
        if a.id == answer_id:
            continue
        if a.terminal in pipe.nodes:
            out |= pipe.upstream_closure(a.terminal) | {a.terminal}
        out |= {n for n in a.nodes if n in pipe.nodes}
    return out


@dataclass
class Applied:
    resolved: dict[str, str]                 # plan key -> node id
    record: dict[str, dict]                  # the steps this answer owns: plan key -> {"node", "made", "title", "inputs", "hand"?}
    left: list[str] = field(default_factory=list)   # steps edited by hand that the change no longer fits: kept, set aside


def apply_plan(pipe, plan: Plan, edits: Edits, previous: dict[str, dict] | None = None,
               protected: set[str] | None = None) -> Applied:
    """Make the project hold ``plan``. With ``previous`` (the steps an answer built last time), steps are
    updated in place rather than rebuilt:

    - a step this answer built and nobody changed since gets the new settings and inputs;
    - a step the person edited by hand keeps its settings while the change leaves that step's plan as it was;
      when the change needs that step to be different, the person's step is left as it is (set aside, with its
      edits) and the answer makes a new one;
    - steps the person put in front of an answer's step stay there, and when the answer's step is to read from
      somewhere new, their steps are moved onto the new input, so they stay in the answer's path;
    - a step another answer depends on (``protected``), or any step the answer did not make itself (found by
      its settings), is used as it is and never changed or removed by the answer;
    - steps no longer needed are removed, unless the person edited them or something outside the answer reads
      from them."""
    previous = previous or {}
    protected = protected or set()
    resolved: dict[str, str] = {}
    record: dict[str, dict] = {}
    left: list[str] = []
    sigs = signatures_of(pipe)
    kept_prev: set[str] = set()
    owned = {p.get("node") for p in previous.values()}
    for step in plan.steps:
        if step.type == "@":
            if step.params["node"] not in pipe.nodes:
                raise KeyError(f"The step {step.params['node']!r} is not in the project any more")
            resolved[step.key] = step.params["node"]
            continue
        ins = {port: [resolved[k] for k in keys] for port, keys in step.inputs.items()}
        prev = previous.get(step.key)
        nid = prev.get("node") if prev else None
        if nid in pipe.nodes and pipe.nodes[nid].type == step.type and nid not in protected and nid not in kept_prev:
            node = pipe.nodes[nid]
            now = _norm_params(node.type, node.params)
            made = _norm_params(node.type, prev.get("made") or {})
            wanted = _norm_params(step.type, step.params)
            made_ins = _wiring(prev.get("inputs", pipe.inputs_of(nid)))
            hand = now != made
            rewired = _wiring(pipe.inputs_of(nid)) != made_ins     # the person put a step in front of it
            moved = (rewired and not (hand and wanted != made) and _wiring(ins) != made_ins
                     and _move_inserted(pipe, edits, nid, made_ins, _wiring(ins), owned | protected))
            if (hand and wanted != made) or (rewired and _wiring(ins) != made_ins and not moved):
                left.append(nid)                        # the person's edited step stays theirs; the answer moves on
            else:
                if not hand and now != wanted:
                    edits.set_params(nid, dict(step.params))
                titled_by_hand = node.title != prev.get("title")
                if not titled_by_hand and node.title != step.title and step.title:
                    edits.set_title(nid, step.title)
                if not rewired:
                    _rewire(pipe, edits, nid, ins)
                kept_prev.add(nid)
                resolved[step.key] = nid
                record[step.key] = {"node": nid, "made": dict(prev.get("made") or {}) if hand else dict(step.params),
                                    "title": pipe.nodes[nid].title if titled_by_hand else step.title, "inputs": ins}
                if hand or rewired:
                    record[step.key]["hand"] = True     # the person's own settings (or wiring) were kept
                continue
        sig = signature(step.type, step.params, ins, pipe.directory)
        existing = sigs.get(sig)
        if existing and existing in pipe.nodes and existing not in left:
            resolved[step.key] = existing               # used, not owned: whoever made it keeps it
            continue
        x, y = place_near(pipe, [n for srcs in ins.values() for n in srcs])
        new = edits.create(step, ins, x, y)
        sigs[sig] = new
        resolved[step.key] = new
        record[step.key] = {"node": new, "made": dict(step.params), "title": step.title, "inputs": ins}
    # what the answer built before and needs no more; a step the person edited is theirs, and stays
    now_used = set(resolved.values())
    stale = {p["node"] for p in previous.values() if p.get("node") in pipe.nodes} - now_used - protected - set(left)
    for p in previous.values():
        if p.get("node") in stale and _edited(pipe.nodes[p["node"]], p):
            stale.discard(p["node"]); left.append(p["node"])
    removable = set(stale)
    changed = True
    while changed:                     # keep a step something outside the answer still reads from
        changed = False
        for nid in sorted(removable):
            if any(out not in removable for out in pipe.outputs_of(nid)):
                removable.discard(nid); changed = True
    if removable:
        edits.remove(sorted(removable))
    return Applied(resolved, record, left)


def _wiring(ins: dict[str, list[str]]) -> dict[str, list[str]]:
    return {port: list(srcs) for port, srcs in ins.items() if srcs}


def _move_inserted(pipe, edits: Edits, nid: str, made: dict[str, list[str]], want: dict[str, list[str]],
                   fixed: set[str]) -> bool:
    """The person put steps of their own in front of an answer's step, and the answer now wants that step to read
    from somewhere else: their steps stay in the path, now reading from the new input (a filter they added between
    the file and the totals still filters the totals). False, changing nothing, when their steps are not a simple
    chain back to what the answer's step read before, or belong to the answer or to another answer (``fixed``)."""
    current = _wiring(pipe.inputs_of(nid))
    if set(current) != set(made) or set(want) != set(made):
        return False
    moves = []
    for port, was in made.items():
        have, new = current[port], want[port]
        if have == was:
            if new != was:
                moves += [("port", nid, port, was, new)]
            continue
        if len(have) != 1 or len(was) != 1 or len(new) != 1:
            return False
        node, seen = have[0], set()
        while True:
            if node in fixed or node in seen or node not in pipe.nodes:
                return False
            seen.add(node)
            ins = _wiring(pipe.inputs_of(node))
            if len(ins) != 1 or len(next(iter(ins.values()))) != 1:
                return False
            (p, (src,)), = ins.items()
            if src == was[0]:
                moves.append(("port", node, p, was, new))
                break
            node = src
    for _, target, port, was, new in moves:
        for src in was:
            edits.disconnect(src, target, port)
        for src in new:
            edits.connect(src, target, port)
    return True


def _edited(node, rec: dict) -> bool:
    """Whether the person changed a step since the answer made it: its settings, or its title."""
    return (bool(rec.get("hand")) or _norm_params(node.type, node.params) != _norm_params(node.type, rec.get("made") or {})
            or node.title != rec.get("title"))


def _rewire(pipe, edits: Edits, nid: str, ins: dict[str, list[str]]) -> None:
    """Make a step's inputs exactly ``ins`` (order kept on a multi-input port)."""
    current = pipe.inputs_of(nid)
    for port in set(current) | set(ins):
        have, want = list(current.get(port) or []), list(ins.get(port) or [])
        if have == want:
            continue
        for src in have:
            edits.disconnect(src, nid, port)
        for src in want:
            edits.connect(src, nid, port)


def instantiate(pipe, plan: Plan) -> dict[str, str]:
    """Apply a plan straight to a pipeline (no undo, nothing replaced): see apply_plan."""
    return apply_plan(pipe, plan, PipelineEdits(pipe)).resolved
