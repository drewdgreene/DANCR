"""MCP server exposing DANCR to AI coding agents (Claude Code, OpenCode, Cursor...).

Run with:  dancr mcp
Register in an agent, e.g. Claude Code:  claude mcp add dancr -- dancr mcp

Every tool works on a pipeline file path. The GUI watches that file, so a person
can have it open and see the pipeline change and results appear while the agent works.
"""
from __future__ import annotations


import json


from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer, Image
from mcp.server.mcpserver.exceptions import ToolError

import polars as pl

from . import __version__
from .core import Pipeline, PipelineError, registry
from .core.executor import Executor
from .core.dtypes import json_safe

mcp = MCPServer("dancr", version=__version__, instructions=(
    "DANCR builds and runs data pipelines (node graphs) over large CSV/Excel/Parquet files. "
    "Workflow: create_pipeline -> add_node (load_file first, with path) -> add more nodes with after= to chain them -> "
    "run_pipeline -> inspect with get_schema / get_sample / get_stats / render_chart. "
    "Call list_node_types once to learn node types and their settings. Paths inside a pipeline are relative to the pipeline file. "
    "Use open_in_gui so the person can watch; the GUI reloads the file whenever it changes. "
    "Files written by render_chart(out_png) and export_node must be inside the pipeline file's folder."))


def friendly(fn):
    """Turn validation errors into ToolErrors so the agent sees the actual message."""
    import functools

    @functools.wraps(fn)
    def wrapper(*a, **k):
        try:
            return fn(*a, **k)
        except (ValueError, PipelineError, OSError, pl.exceptions.PolarsError) as e:
            raise ToolError(str(e)) from e
        except (KeyError, TypeError, AttributeError) as e:
            raise ToolError(str(e).strip("'\"") or type(e).__name__) from e
    return wrapper


def _load(path: str) -> Pipeline:
    p = Path(path).expanduser()
    if not p.exists():
        raise ValueError(f"No pipeline at {p}. Call create_pipeline first.")
    return Pipeline.load(p)


def _inside_project(p: Pipeline, out: str) -> Path:
    """Resolve a write target and refuse anything outside the folder of the pipeline file (relative paths are taken from there)."""
    folder = p.path.resolve().parent
    target = Path(out).expanduser()
    target = (target if target.is_absolute() else folder / target).resolve()
    if not target.is_relative_to(folder):
        raise ToolError(f"Can only write inside the project folder {folder}, not {target}")
    return target


def _state(p: Pipeline, ex: Executor, nid: str) -> dict[str, Any]:
    if nid not in p.nodes:
        raise ValueError(f"No node {nid!r}. Nodes: {list(p.nodes)}")
    st = ex.state(nid)
    n = p.nodes[nid]
    return json_safe({"id": nid, "title": n.title, "type": n.type, "status": st.status, "rows": st.rows,
                      "columns": [c["name"] for c in st.columns], "error": st.error, "messages": st.messages,
                      "report": st.report, "elapsed_s": st.elapsed})


@mcp.tool()
@friendly
def list_node_types(type_key: str | None = None) -> str:
    """List every node type with its settings (params), inputs and description. Pass type_key for one type in full."""
    types = registry.all() if not type_key else [registry.get(type_key)]
    if type_key:
        return json.dumps(types[0].to_json(), indent=1)
    brief = [{"key": t.key, "label": t.label, "category": t.category, "description": t.description,
              "inputs": [i.name for i in t.inputs], "params": [p.name for p in t.params]} for t in types]
    return json.dumps(brief, indent=1)


@mcp.tool()
@friendly
def formula_reference() -> str:
    """Syntax and functions of the formula language used by the calculate node and keep_rows.formula."""
    from .core.expr import function_docs
    lines = ["Columns: bare name (Pressure), [with spaces], or `backticks`. Operators: + - * / ^ %  = != < > <= >=  and or not  & (join text).",
             "IF(test, a, b). One-argument SUM/AVERAGE/MIN/MAX/MEDIAN/STDEV aggregate the whole column (broadcast); with several arguments they work row-wise.",
             "Functions:"]
    lines += [f"  {n}: {d}" for n, d in function_docs()]
    return "\n".join(lines)


@mcp.tool()
@friendly
def create_pipeline(path: str, name: str | None = None, overwrite: bool = False) -> str:
    """Create an empty pipeline file (JSON). Use an absolute path ending in .json, ideally next to the data files."""
    p = Path(path).expanduser()
    if p.exists() and not overwrite:
        return json.dumps({"ok": True, "path": str(p), "note": "already exists; loaded as-is"})
    pipe = Pipeline(name or p.stem)
    pipe.save(p)
    return json.dumps({"ok": True, "path": str(p)})


@mcp.tool()
@friendly
def describe_pipeline(path: str) -> str:
    """Nodes, connections, settings, statuses and configuration problems of a pipeline."""
    p = _load(path)
    ex = Executor(p)
    return json.dumps({
        "name": p.name, "path": str(p.path),
        "nodes": [{**n.to_dict(), "inputs": p.inputs_of(n.id), "state": _state(p, ex, n.id)} for n in p.nodes.values()],
        "edges": [e.to_dict() for e in p.edges],
        "inputs": [{"name": i.name, "value": i.value, "unit": i.unit, "note": i.note} for i in p.inputs],
        "columns": p.columns,
        "problems": p.problems(),
    }, indent=1, default=str)


@mcp.tool()
@friendly
def add_node(path: str, type_key: str, params: dict[str, Any] | None = None, title: str | None = None,
             node_id: str | None = None, after: str | None = None, port: str | None = None,
             also_after: list[str] | None = None) -> str:
    """Add a node. `after` connects an existing node's output to the new node's input (port: for combine use 'left'/'right';
    for stack all inputs go to 'tables'). `also_after` connects extra upstream nodes (e.g. the second table of a combine)."""
    p = _load(path)
    if not registry.has(type_key):
        match = [t for t in registry.all() if t.label.lower() == type_key.lower()]
        if not match:
            raise ValueError(f"Unknown node type {type_key!r}. Known: {[t.key for t in registry.all()]}")
        type_key = match[0].key
    x, y = 0.0, 0.0
    if after and after in p.nodes:
        x, y = p.nodes[after].x + 280, p.nodes[after].y
    elif p.nodes:
        last = list(p.nodes.values())[-1]
        x, y = last.x, last.y + 120
    while any(abs(n.x - x) < 200 and abs(n.y - y) < 80 for n in p.nodes.values()):
        y += 120
    node = p.add_node(type_key, title=title, params=params or {}, x=x, y=y, id=node_id)
    if after:
        p.connect(after, node.id, port)
    for extra in also_after or []:
        p.connect(extra, node.id)
    p.save()
    return json.dumps({"ok": True, "node": node.to_dict(), "inputs": p.inputs_of(node.id), "problems": [x for x in p.problems() if x.startswith(node.title + ":")]})


@mcp.tool()
@friendly
def set_input(path: str, name: str, value: Any = None, unit: str = "", note: str = "") -> str:
    """Create or change a named input: a number or text usable by name in formulas (e.g. `[value] / [maximum allowed]`),
    filter values and limit lines. Changing it recomputes only the steps that use it."""
    p = _load(path)
    i = p.set_input(name, value, unit, note)
    p.save()
    return json.dumps({"ok": True, "input": {"name": i.name, "value": i.value, "unit": i.unit, "note": i.note}, "all": [x.name for x in p.inputs]})


@mcp.tool()
@friendly
def remove_input(path: str, name: str) -> str:
    """Remove a named input."""
    p = _load(path)
    p.remove_input(name); p.save()
    return json.dumps({"ok": True, "inputs": [x.name for x in p.inputs]})


@mcp.tool()
@friendly
def set_column_label(path: str, column: str, label: str | None = None, unit: str | None = None) -> str:
    """Give a column a display name and/or unit used on charts, reports and in the table header (the data is unchanged)."""
    p = _load(path)
    p.set_column_meta(column, label, unit); p.save()
    return json.dumps({"ok": True, "columns": p.columns})


@mcp.tool()
@friendly
def build_template(path: str, template: str, data_file: str | None = None) -> str:
    """Create a starter project from a template: compare | limits | fit | report. Uses a generated sample
    file (two values over time) unless data_file is given. The pipeline file must not exist yet."""
    from .core.samples import build_template as _bt, write_sample, TEMPLATES
    out = Path(path).expanduser()
    if out.exists():
        raise ValueError(f"{out} already exists")
    data = Path(data_file).expanduser().resolve() if data_file else write_sample(out.parent)
    pipe = Pipeline(out.stem); pipe.path = out.resolve()
    _bt(template, pipe, data)
    pipe.save(out)
    return json.dumps({"ok": True, "path": str(out), "data": str(data), "nodes": list(pipe.nodes), "templates": TEMPLATES})


@mcp.tool()
@friendly
def set_params(path: str, node_id: str, params: dict[str, Any]) -> str:
    """Change settings on a node. Only the keys given are changed. Values are validated."""
    p = _load(path)
    p.set_params(node_id, **params)
    p.save()
    return json.dumps({"ok": True, "node": p.nodes[node_id].to_dict()})


@mcp.tool()
@friendly
def connect_nodes(path: str, source: str, target: str, port: str | None = None) -> str:
    """Connect source's output to target's input. port is only needed for nodes with several inputs (combine: left/right)."""
    p = _load(path)
    e = p.connect(source, target, port)
    p.save()
    return json.dumps({"ok": True, "edge": e.to_dict()})


@mcp.tool()
@friendly
def disconnect_nodes(path: str, source: str, target: str, port: str | None = None) -> str:
    """Remove a connection (every connection between the two nodes, or only the one into `port`)."""
    p = _load(path)
    p.disconnect(source, target, port)
    p.save()
    return json.dumps({"ok": True})


@mcp.tool()
@friendly
def remove_node(path: str, node_id: str) -> str:
    """Delete a node and its connections."""
    p = _load(path)
    p.remove_node(node_id)
    p.save()
    return json.dumps({"ok": True})


@mcp.tool()
@friendly
def rename_node(path: str, node_id: str, title: str) -> str:
    """Give a node a human-friendly title (shown on the canvas)."""
    p = _load(path)
    p.rename_node(node_id, title)
    p.save()
    return json.dumps({"ok": True})


@mcp.tool()
@friendly
def run_pipeline(path: str, node_ids: list[str] | None = None, force: bool = False) -> str:
    """Execute the pipeline (or only the given nodes and what they depend on). Unchanged nodes are served from cache.
    Returns per-node status, row counts, messages, reports and errors."""
    p = _load(path)
    ex = Executor(p)
    res = ex.run(targets=node_ids, force=force)
    out = {nid: _state(p, ex, nid) for nid in res}
    failed = [nid for nid, s in out.items() if s["status"] == "failed"]
    return json.dumps({"ok": not failed, "failed": failed, "nodes": out, "problems": p.problems()}, indent=1, default=str)


@mcp.tool()
@friendly
def node_status(path: str, node_id: str) -> str:
    """Status, row count, columns, messages and report (e.g. fit coefficients, gap statistics) of one node."""
    p = _load(path)
    return json.dumps(_state(p, Executor(p), node_id), indent=1, default=str)


def _frame(path: str, node_id: str, run: bool):
    p = _load(path)
    ex = Executor(p)
    if node_id not in p.nodes:
        raise ValueError(f"No node {node_id!r}. Nodes: {list(p.nodes)}")
    st = ex.state(node_id)
    if st.status != "done":
        if not run:
            raise ValueError(f"{node_id} has not been run (status {st.status}). Call run_pipeline or pass run=true.")
        res = ex.run(targets=[node_id])
        if res[node_id].status != "done":
            raise ValueError(f"{node_id} failed: {res[node_id].error}")
    return p, ex, ex.frame(node_id)


@mcp.tool()
@friendly
def get_schema(path: str, node_id: str) -> str:
    """Column names and types of a node's output (works before running, as long as upstream files exist)."""
    p = _load(path)
    ex = Executor(p)
    st = ex.state(node_id)
    if st.status == "done":
        return json.dumps({"rows": st.rows, "columns": st.columns})
    sch = ex.schema(node_id)
    if sch is None:
        raise ValueError("Cannot determine the schema yet; check the node's inputs and settings")
    return json.dumps({"rows": None, "columns": [{"name": k, "dtype": str(v)} for k, v in sch.items()]})


@mcp.tool()
@friendly
def get_sample(path: str, node_id: str, rows: int = 20, offset: int = 0, columns: list[str] | None = None, run: bool = True) -> str:
    """Rows from a node's output as JSON records (at most 500 rows; pass columns to narrow wide tables)."""
    _, _, lf = _frame(path, node_id, run)
    rows = max(1, min(int(rows), 500))
    offset = max(0, int(offset))
    if columns:
        lf = lf.select(columns)
    df = lf.slice(offset, rows).collect(engine="streaming")
    return df.write_json()


@mcp.tool()
@friendly
def get_stats(path: str, node_id: str, columns: list[str] | None = None, run: bool = True) -> str:
    """Summary statistics (count, missing, mean, std, min, quartiles, max) for the columns of a node's output."""
    from .views.stats import column_summary
    _, _, lf = _frame(path, node_id, run)
    return column_summary(lf, columns).write_json()


@mcp.tool()
@friendly
def render_chart(path: str, node_id: str, out_png: str | None = None, kind: str = "line", x: str | None = None,
                 y: list[str] | None = None, column: str | None = None, title: str | None = None,
                 width: int = 1200, height: int = 600, run: bool = True) -> Image:
    """Render a chart of a node's output to PNG and return the image. If node is a chart node its settings are used;
    otherwise give kind (line|scatter|histogram|bar), x and y. Big data is downsampled per pixel.
    out_png (optional) must be inside the pipeline file's folder; otherwise the PNG goes to the cache."""
    from .views.render import render_chart as _render
    p, ex, lf = _frame(path, node_id, run)
    node = p.nodes[node_id]
    params = dict(node.params) if node.type == "chart" else {}
    if node.type != "chart" or kind != "line":
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
    out = _inside_project(p, out_png) if out_png else (ex.cache_dir / "charts" / f"{node_id}.png")
    _render(lf, params, out, width=width, height=height)
    return Image(path=str(out))


@mcp.tool()
@friendly
def export_node(path: str, node_id: str, out_path: str, run: bool = True) -> str:
    """Write a node's full output to a .csv, .parquet or .xlsx file inside the pipeline file's folder."""
    p, _, lf = _frame(path, node_id, run)
    from .core.nodes.outputs import write_table
    out = _inside_project(p, out_path)
    write_table(lf, out)
    return json.dumps({"ok": True, "path": str(out)})


@mcp.tool()
@friendly
def open_in_gui(path: str) -> str:
    """Open the pipeline in the DANCR desktop app so the person can watch and explore. Safe to call repeatedly."""
    from .cli import launch_gui
    launch_gui(path)
    return json.dumps({"ok": True, "note": "DANCR is opening. It reloads the file automatically when you change it."})


@mcp.tool()
@friendly
def inspect_file(file_path: str, rows: int = 5) -> str:
    """Peek at a data file before building a pipeline: detected columns, types and the first rows."""
    from .core.registry import Ctx
    from .core.nodes.load import scan_file
    fp = Path(file_path).expanduser()
    ctx = Ctx(fp.parent, "inspect", "inspect", preview=True)
    lf, messages = scan_file(ctx, {"path": str(fp), "has_header": True, "parse_dates": True})
    schema = lf.collect_schema()
    head = lf.head(rows).collect(engine="streaming")
    return json.dumps({"columns": [{"name": k, "dtype": str(v)} for k, v in schema.items()], "messages": messages,
                       "head": json.loads(head.write_json())}, default=str)


def main() -> None:
    import logging
    from .logsetup import configure
    configure(stderr_level=logging.WARNING)      # stdout carries the protocol; the log file and stderr get the rest
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
