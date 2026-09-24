"""What the command line and the MCP server share: finding steps, adding them, reporting them, charting and
reading results. Both front ends call these, so a step is reported the same way wherever it is asked for."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from .core import Pipeline, registry
from .core.dtypes import json_safe
from .core.executor import Executor, NodeState
from .core.model import Node


def require_node(p: Pipeline, node_id: str) -> str:
    """Every command or tool that names a step checks it here, so the message is always the same."""
    if node_id not in p.nodes:
        raise ValueError(f"No step called {node_id!r}. Steps: {list(p.nodes)}")
    return node_id


def resolve_type(key_or_label: str) -> str:
    """A step type by key (`keep_rows`) or by the label people see (`Filter rows`)."""
    if registry.has(key_or_label):
        return key_or_label
    match = [t for t in registry.all() if t.label.lower() == key_or_label.lower()]
    if not match:
        raise ValueError(f"Unknown step type {key_or_label!r}. Known: {sorted(t.key for t in registry.all())}")
    return match[0].key


def place(p: Pipeline, after: str | None) -> tuple[float, float]:
    """Where a new step goes on the map: right of the step it follows, else below the last one; never on top of another."""
    x, y = 0.0, 0.0
    if after and after in p.nodes:
        x, y = p.nodes[after].x + 280, p.nodes[after].y
    elif p.nodes:
        last = list(p.nodes.values())[-1]
        x, y = last.x, last.y + 120
    while any(abs(n.x - x) < 200 and abs(n.y - y) < 80 for n in p.nodes.values()):
        y += 120
    return x, y


def add_step(p: Pipeline, type_key: str, params: dict[str, Any] | None = None, title: str | None = None,
             node_id: str | None = None, after: str | None = None, port: str | None = None,
             also_after: list[str] | None = None) -> Node:
    """Add a step, connected after `after` (into `port`) and from each of `also_after`. Nothing is changed if any
    step it names does not exist."""
    type_key = resolve_type(type_key)
    for nid in [after, *(also_after or [])]:
        if nid:
            require_node(p, nid)
    x, y = place(p, after)
    node = p.add_node(type_key, title=title, params=params or {}, x=x, y=y, id=node_id)
    try:
        if after:
            p.connect(after, node.id, port)
        for extra in also_after or []:
            p.connect(extra, node.id)
    except Exception:
        p.remove_node(node.id)
        raise
    return node


def node_record(p: Pipeline, st: NodeState) -> dict[str, Any]:
    """How a step is reported by both front ends: node_id, title, type, status, rows, columns ({name, dtype}),
    error, messages, report, elapsed, from_cache, and the rest of its state."""
    n = p.nodes[st.node_id]
    return json_safe({**st.to_dict(), "title": n.title, "type": n.type})


def run_record(p: Pipeline, ex: Executor, res: dict[str, NodeState], elapsed: float) -> dict[str, Any]:
    """How a run is reported by both front ends; `nodes` is keyed by step id."""
    failed = [nid for nid, s in res.items() if s.status == "failed"]
    return {"ok": not failed, "failed": failed, "nodes": {nid: node_record(p, s) for nid, s in res.items()},
            "problems": p.problems(), "elapsed": elapsed, "cache_dir": str(ex.cache_dir)}


def result_frame(p: Pipeline, ex: Executor, node_id: str, run: bool) -> pl.LazyFrame:
    """A step's output, running it first when `run` is set and it is not computed yet."""
    require_node(p, node_id)
    st = ex.state(node_id)
    if st.status != "done":
        if not run:
            raise ValueError(f"{node_id} has not been run yet (status: {st.status}). Run the pipeline first, or ask to run it")
        res = ex.run(targets=[node_id])
        if res[node_id].status != "done":
            raise ValueError(f"{node_id} failed: {res[node_id].error}")
    return ex.frame(node_id)


def select_columns(lf: pl.LazyFrame, columns: list[str] | None) -> pl.LazyFrame:
    """Only these columns, with a plain message for one that does not exist."""
    if not columns:
        return lf
    have = lf.collect_schema().names()
    missing = [c for c in columns if c not in have]
    if missing:
        raise ValueError(f"There is no column called {missing[0]!r}. Columns: {have[:20]}")
    return lf.select(columns)


def chart_params(node: Node, kind: str | None = None, x: str | None = None, y: list[str] | None = None,
                 column: str | None = None, title: str | None = None) -> dict[str, Any]:
    """A chart step's settings, or an ad-hoc chart of any step; each argument given overrides."""
    params = dict(node.params) if node.type == "chart" else {}
    if kind:
        params["kind"] = kind
    if x:
        params["x"] = x
    if y:
        params["series"] = [{"column": c} for c in y]
    if column:
        params["column"] = column
    if title:
        params["title"] = title
    params.setdefault("kind", "line")
    return params


def build_template(out: Path, template: str, data_file: str | Path | None) -> tuple[Pipeline, Path]:
    """A starter project at `out` on `data_file` (relative to the project's folder), or on a generated sample
    file when none is given. The template name and the data file are checked before anything is written."""
    from .core.samples import build_template as _build, write_sample, check_template
    check_template(template)
    out = out.expanduser().resolve()
    if data_file:
        data = Path(data_file).expanduser()
        data = (data if data.is_absolute() else out.parent / data).resolve()
        if not data.is_file():
            raise ValueError(f"No data file at {data}")
    else:
        data = write_sample(out.parent)
    pipe = Pipeline(out.stem)
    pipe.path = out
    _build(template, pipe, data)
    pipe.save(out)
    return pipe, data


def output_paths_outside(p: Pipeline, node_id: str, folder: Path) -> list[str]:
    """The files a step would write outside `folder` (its path settings, for steps that save files)."""
    node = p.nodes[node_id]
    nt = registry.get(node.type)
    if nt.kind != "sink":
        return []
    out = []
    for prm in nt.params:
        v = node.params.get(prm.name)
        if prm.kind == "path" and isinstance(v, str) and v.strip():
            target = Path(v).expanduser()
            target = (target if target.is_absolute() else p.directory / target).resolve()
            if not target.is_relative_to(folder):
                out.append(str(target))
    return out
