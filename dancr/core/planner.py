"""Turn a wizard's answers into a tidy pipeline of steps.

Deterministic: the same answers on the same data always produce the same graph. The planner only
names steps and wires them; the document resolves each step to an existing node (so shared loads and
links are reused) or creates a new one. No execution happens here, and nothing touches Qt.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .profile import TableProfile

# Supported intents (the wizard in dancr/ui/wizard.py owns their labels, blurbs and icons):
#   total      group rows by a category and sum/aggregate a measure, then a bar chart
#   over_time  bucket by a time column and average a measure, then a line chart
#   describe   one row per column (count, missing, average, range) as a table


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
    config: dict[str, Any] = field(default_factory=dict)
    view: str = "chart"

    def sentence(self) -> str:
        """A short plain-English description of what will be built, for the review step."""
        parts = []
        for s in self.steps:
            if s.type == "load_file":
                parts.append(f"load {Path(str(s.params.get('path', ''))).stem}")
            elif s.type == "combine":
                on = ", ".join(s.params.get("on") or [])
                parts.append(f"link on {on}" if on else "link")
            elif s.type == "stack":
                parts.append("stack the files")
            elif s.type == "group_summary":
                by = ", ".join(s.params.get("by") or [])
                parts.append(f"total by {by}")
            elif s.type == "time_buckets":
                parts.append(f"average every {s.params.get('every')}")
            elif s.type == "calculate":
                parts.append("add a column")
            elif s.type == "chart":
                parts.append("chart")
        return " → ".join(parts)


def _load_step(path: str, title: str | None = None) -> PlanStep:
    return PlanStep(key=f"load:{path}", type="load_file", title=title or Path(str(path)).stem,
                    params={"path": str(path)})


def plan(profiles: list[TableProfile], config: dict[str, Any]) -> Plan:
    """Build a Plan from profiled tables and a wizard config. Raises ValueError on an impossible request."""
    assembly = config.get("assembly") or {"kind": "single", "path": profiles[0].path}
    intent = config.get("intent") or "total"
    title = (config.get("title") or "").strip() or _default_title(intent, config)
    steps: list[PlanStep] = []

    # --- data layer: load the files, then stack or link them into one prepared table
    if assembly["kind"] == "stack":
        paths = list(assembly["paths"])
        for p in paths:
            steps.append(_load_step(p))
        stack_key = "stack"
        steps.append(PlanStep(stack_key, "stack", "Stack the files",
                              {"label_column": assembly.get("label") or "source"},
                              {"tables": [f"load:{p}" for p in paths]}))
        current = stack_key
    elif assembly["kind"] == "join":
        primary = assembly["primary"]
        steps.append(_load_step(primary))
        current = f"load:{primary}"
        for i, link in enumerate(assembly.get("links") or []):
            right = link["right"]
            if f"load:{right}" not in {s.key for s in steps}:
                steps.append(_load_step(right))
            key = f"link:{i}"
            steps.append(PlanStep(
                key, "combine", f"Link {Path(str(right)).stem}",
                {"method": "match", "on": [link["left_on"]], "right_on": [link["right_on"]], "how": "left"},
                {"left": [current], "right": [f"load:{right}"]}))
            current = key
    else:
        path = assembly["path"]
        steps.append(_load_step(path))
        current = f"load:{path}"

    # --- answer branch
    measure = config.get("measure")
    if intent == "describe":
        steps.append(PlanStep("describe", "summarize", "Describe the columns", {}, {"in": [current]}))
        return Plan(steps=steps, terminal="describe", title=title, config=config, view="table")
    if intent == "over_time":
        time_col = config.get("time_column")
        every = config.get("every") or "1d"
        if not time_col:
            raise ValueError("This answer needs a date or time column to average over.")
        buckets = "buckets"
        steps.append(PlanStep(buckets, "time_buckets", f"Average every {every}",
                              {"every": every, "columns": [measure] if measure else [],
                               "default_stats": ["mean"], "time_column": time_col},
                              {"in": [current]}))
        steps.append(PlanStep("chart", "chart", title,
                              {"kind": "line", "x": time_col,
                               "series": [{"column": measure}] if measure else [], "title": title},
                              {"in": [buckets]}))
        terminal = "chart"
    else:                                    # total
        group = config.get("group")
        if not group:
            raise ValueError("This answer needs a category column to group by.")
        steps.append(PlanStep("total", "group_summary", f"Total by {group}",
                              {"by": [group], "columns": [measure] if measure else [], "default_stats": ["sum"]},
                              {"in": [current]}))
        steps.append(PlanStep("chart", "chart", title,
                              {"kind": "bar", "category": group, "value": measure or "", "stat": "sum", "title": title},
                              {"in": ["total"]}))
        terminal = "chart"

    return Plan(steps=steps, terminal=terminal, title=title, config=config, view="chart")


def _default_title(intent: str, config: dict[str, Any]) -> str:
    if intent == "describe":
        return "Describe the columns"
    if intent == "over_time":
        m = config.get("measure")
        return f"{m} over time" if m else "Change over time"
    m, g = config.get("measure"), config.get("group")
    if m and g:
        return f"Total {m} by {g}"
    return "Totals"


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


def plan_layout(plan: Plan) -> dict[str, tuple[float, float]]:
    """A tidy left-to-right layout: sources in the first column, then each step one column to the
    right of its inputs, stacked when several share a column."""
    cols: dict[str, int] = {}
    for step in plan.steps:
        c = 0
        for keys in step.inputs.values():
            for k in keys:
                c = max(c, cols.get(k, 0) + 1)
        cols[step.key] = c
    used: dict[int, int] = {}
    pos: dict[str, tuple[float, float]] = {}
    for step in plan.steps:
        c = cols[step.key]
        r = used.get(c, 0); used[c] = r + 1
        pos[step.key] = (60.0 + c * 290.0, 140.0 + r * 155.0)
    return pos


CreateFn = Callable[[PlanStep, dict[str, list[str]], float, float], str]


def resolve_plan(pipe, plan: Plan, create: CreateFn) -> dict[str, str]:
    """Map every step of a plan to a node of ``pipe``, reusing what is already there and calling
    ``create(step, inputs, x, y)`` for the rest (the window makes that undoable). Returns {plan key: node id}.

    Reused: a loader of the same file (however its path is written), and any step with the same type,
    settings and upstream nodes, so a second answer that reads the same files branches off the first
    answer's data layer instead of duplicating it."""
    sigs = signatures_of(pipe)
    resolved: dict[str, str] = {}
    pos = plan_layout(plan)
    for step in plan.steps:
        ins = {port: [resolved[k] for k in keys] for port, keys in step.inputs.items()}
        if step.type == "load_file":
            existing = find_load(pipe, step.params.get("path"))
            if existing:
                resolved[step.key] = existing
                continue
        sig = signature(step.type, step.params, ins, pipe.directory)
        existing = sigs.get(sig)
        if existing and existing in pipe.nodes:
            resolved[step.key] = existing
            continue
        x, y = pos.get(step.key, (60.0, 200.0))
        resolved[step.key] = sigs[sig] = create(step, ins, x, y)
    return resolved


def instantiate(pipe, plan: Plan) -> dict[str, str]:
    """Apply a plan straight to a pipeline (no undo): see resolve_plan."""
    def create(step: PlanStep, ins: dict[str, list[str]], x: float, y: float) -> str:
        node = pipe.add_node(step.type, title=step.title, params=step.params, x=x, y=y)
        for port, srcs in ins.items():
            for s in srcs:
                pipe.connect(s, node.id, port)
        return node.id
    return resolve_plan(pipe, plan, create)


def find_load(pipe, path: Any) -> str | None:
    """An existing load step for the same file, so a guided build over files already on the map reads
    them once instead of adding a second loader."""
    if not path:
        return None
    from .registry import resolve_path
    target = resolve_path(pipe.directory, str(path)).resolve()
    for nid, node in pipe.nodes.items():
        if node.type == "load_file" and node.params.get("path") and resolve_path(pipe.directory, str(node.params["path"])).resolve() == target:
            return nid
    return None
