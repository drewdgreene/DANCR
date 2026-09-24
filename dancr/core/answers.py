"""Answers as a whole: understand the tables, plan a spec, put the steps in the project, keep the record.

The window, the command line and the MCP server all build answers through here (the window passes edits
that make each change undoable), so an answer is the same whichever of them made it.
"""
from __future__ import annotations

import copy
from typing import Any

from .model import Answer
from .planner import Plan, Edits, PipelineEdits, apply_plan, protected_nodes
from .recipes import RULES_VERSION, plan as make_plan, apply_choice, PlanError
from .understand import DataModel, understand, deepen, default_tables

CARD_OFFSET = (300.0, 40.0)             # where an answer card sits relative to the step it points at


def model_for(pipe, executor, deep: bool = True, nodes: list[str] | None = None) -> DataModel:
    """The data model of a project's tables (every source step by default), read in full unless ``deep`` is False."""
    m = understand(pipe, executor, nodes)
    return deepen(pipe, executor, m) if deep else m


def tables_for(pipe, spec: dict[str, Any]) -> list[str]:
    """The tables a spec's model needs: the project's sources, plus the table asked about if it is a step."""
    nodes = default_tables(pipe)
    t = spec.get("table")
    if t and t in pipe.nodes and t not in nodes:
        nodes.append(t)
    return nodes


def card_position(pipe, terminal: str) -> tuple[float, float]:
    n = pipe.nodes[terminal]
    return n.x + CARD_OFFSET[0], n.y + CARD_OFFSET[1]


def answer_fields(pipe, plan: Plan, resolved: dict[str, str], record: dict[str, dict]) -> dict[str, Any]:
    terminal = resolved[plan.terminal]
    x, y = card_position(pipe, terminal)
    return {"title": plan.title, "terminal": terminal, "view": plan.view, "spec": copy.deepcopy(plan.config),
            "steps": record, "assumptions": copy.deepcopy(plan.assumptions), "rules": RULES_VERSION, "x": x, "y": y}


def build(pipe, model: DataModel, spec: dict[str, Any], edits: Edits | None = None,
          answer_id: str | None = None) -> tuple[Answer, Plan]:
    """Answer ``spec``: add (or, for ``answer_id``, update) its steps and its Answer. Changes ``pipe`` directly
    unless ``edits`` says otherwise. Raises PlanError when the question cannot be answered."""
    plan = make_plan(model, spec)
    edits = edits or PipelineEdits(pipe)
    existing = pipe.answer(answer_id) if answer_id else None
    previous = existing.steps if existing is not None else None
    resolved, record = apply_plan(pipe, plan, edits, previous, protected_nodes(pipe, answer_id))
    fields = answer_fields(pipe, plan, resolved, record)
    if existing is None:
        a = pipe.add_answer(fields["title"], fields["terminal"], fields["x"], fields["y"], fields["view"], fields["spec"],
                            steps=fields["steps"], assumptions=fields["assumptions"], rules=fields["rules"])
    else:
        for k, v in fields.items():
            setattr(existing, k, v)
        a = existing
    return a, plan


def change(pipe, model: DataModel, answer_id: str, key: str, value: Any, edits: Edits | None = None) -> tuple[Answer, Plan]:
    """Apply one chip or assumption choice to an answer and rebuild it in place."""
    a = pipe.answer(answer_id)
    if a is None:
        raise PlanError(f"No answer called {answer_id!r}")
    return build(pipe, model, apply_choice(a.spec, key, value), edits, answer_id)


def remove(pipe, answer_id: str, remove_steps: bool, edits: Edits | None = None) -> list[str]:
    """Delete an answer, and with ``remove_steps`` the steps only it uses. Returns the steps removed."""
    a = pipe.answer(answer_id)
    if a is None:
        return []
    gone = exclusive_steps(pipe, answer_id) if remove_steps else []
    if gone:
        (edits or PipelineEdits(pipe)).remove(gone)
    pipe.remove_answer(answer_id)
    return gone


def exclusive_steps(pipe, answer_id: str) -> list[str]:
    """Steps this answer built that no other answer depends on and nothing else reads from."""
    a = pipe.answer(answer_id)
    if a is None:
        return []
    mine = {n for n in a.nodes if n in pipe.nodes} - protected_nodes(pipe, answer_id)
    keep = set(mine)
    changed = True
    while changed:
        changed = False
        for nid in sorted(keep):
            if any(out not in keep for out in pipe.outputs_of(nid)):
                keep.discard(nid); changed = True
    return sorted(keep)


def kept_by_hand(pipe, a: Answer) -> list[str]:
    """Titles of this answer's steps whose settings were edited by hand, and so were kept as they are."""
    return [pipe.nodes[s["node"]].title for s in a.steps.values() if s.get("hand") and s.get("node") in pipe.nodes]


def describe(pipe, a: Answer) -> dict[str, Any]:
    """How the command line and MCP report an answer."""
    return {**a.to_dict(), "built": a.terminal in pipe.nodes, "kept_by_hand": kept_by_hand(pipe, a)}
