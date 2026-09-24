"""MCP server exposing DANCR to AI coding agents (Claude Code, OpenCode, Cursor...).

Run with:  dancr mcp [--root DIR]
Register in an agent, e.g. Claude Code:  claude mcp add dancr -- dancr mcp

Every tool works on a pipeline file path. The GUI watches that file, so a person
can have it open and see the pipeline change and results appear while the agent works.

Where the server may write: pipeline files only inside its root folder; every other file (exports, reports,
workbooks, chart images) only inside the folder of the pipeline file, checked when a step is added or changed
and again when it runs. Tools may be called in parallel: edits to one pipeline file are made one at a time.
"""
from __future__ import annotations

import functools
import json
import logging
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import polars as pl
from mcp.server.mcpserver import MCPServer, Image
from mcp.server.mcpserver.exceptions import ToolError

from . import __version__
from . import headless as hl
from .core import Pipeline, PipelineError, registry
from .core.executor import Executor

log = logging.getLogger("dancr.mcp")

mcp = MCPServer("dancr", version=__version__, instructions=(
    "DANCR builds and runs data pipelines (node graphs) over large CSV/Excel/Parquet files. "
    "Workflow: create_pipeline -> add_node (load_file first, with path) -> add more nodes with after= to chain them -> "
    "run_pipeline -> inspect with get_schema / get_sample / get_stats / render_chart. "
    "Call list_node_types once to learn node types and their settings. Paths inside a pipeline are relative to the pipeline file. "
    "Use open_in_gui so the person can watch; the GUI reloads the file whenever it changes. "
    "Pipeline files must be inside the server's root folder, and every file a pipeline or tool writes "
    "(export, workbook and report steps, render_chart out_png, export_node) must be inside the pipeline file's folder. "
    "Reading data files is not restricted; relative data paths are taken from the root folder."))

ROOT = Path.cwd().resolve()        # where pipelines may be created; `dancr mcp --root DIR` sets it

_locks: dict[Path, threading.Lock] = {}
_locks_guard = threading.Lock()


def friendly(fn):
    """Turn expected failures into ToolErrors carrying their plain message. Anything else is a DANCR bug:
    it is logged with its traceback and the agent is told so, rather than shown a bare 'KeyError: x'."""
    @functools.wraps(fn)
    def wrapper(*a, **k):
        try:
            return fn(*a, **k)
        except ToolError:
            raise
        except (ValueError, PipelineError, OSError, pl.exceptions.PolarsError) as e:
            raise ToolError(str(e)) from e
        except Exception as e:  # noqa: BLE001
            log.exception("MCP tool %s failed", fn.__name__)
            raise ToolError(f"DANCR hit an internal error in {fn.__name__} ({type(e).__name__}: {e}). "
                            "The details are in the log (dancr log).") from e
    return wrapper


def _in_root(path: str | Path) -> Path:
    """A pipeline file path (relative to the root), refused outside the server's root folder."""
    p = Path(path).expanduser()
    p = (p if p.is_absolute() else ROOT / p).resolve()
    if not p.is_relative_to(ROOT):
        raise ToolError(f"Pipelines can only be created inside {ROOT} (the folder the DANCR MCP server was started in), not {p}")
    return p


def _from_root(path: str | Path) -> Path:
    """A file to read, relative to the root folder (reading is not confined)."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else ROOT / p


def _load(path: str) -> Pipeline:
    p = _in_root(path)
    if not p.exists():
        raise ValueError(f"No pipeline at {p}. Call create_pipeline first.")
    return Pipeline.load(p)


@contextmanager
def _editing(path: str) -> Iterator[Pipeline]:
    """Load, change and save one pipeline file with no other tool call editing it at the same time (agents call
    tools in parallel; without this, one call's save would drop another's change)."""
    target = _in_root(path)
    with _locks_guard:
        lock = _locks.setdefault(target, threading.Lock())
    with lock:
        p = _load(path)
        yield p
        p.save()


def _folder(p: Pipeline) -> Path:
    return p.path.resolve().parent


def _executor(p: Pipeline) -> Executor:
    """Steps run from MCP may only write inside the pipeline file's folder."""
    return Executor(p, output_root=_folder(p))


def _check_outputs(p: Pipeline, node_id: str) -> None:
    """Refuse a step that would save a file outside the pipeline's folder, when it is added or changed: the
    window runs steps too (open_in_gui), and it must never be handed one that writes elsewhere."""
    outside = hl.output_paths_outside(p, node_id, _folder(p))
    if outside:
        raise ToolError(f"Can only save inside the project folder {_folder(p)}, not {outside[0]}")


def _inside_project(p: Pipeline, out: str) -> Path:
    """Resolve a write target and refuse anything outside the folder of the pipeline file (relative paths are taken from there)."""
    folder = _folder(p)
    target = Path(out).expanduser()
    target = (target if target.is_absolute() else folder / target).resolve()
    if not target.is_relative_to(folder):
        raise ToolError(f"Can only write inside the project folder {folder}, not {target}")
    return target


def _dump(data: Any) -> str:
    return json.dumps(data, indent=1, default=str)


def _record(p: Pipeline, ex: Executor, nid: str) -> dict[str, Any]:
    return hl.node_record(p, ex.state(hl.require_node(p, nid)))


# ----------------------------------------------------------------- reference
@mcp.tool()
@friendly
def list_node_types(type_key: str | None = None) -> str:
    """List every node type with its settings (params), inputs and description. Pass type_key for one type in full."""
    if type_key:
        return _dump(registry.get(hl.resolve_type(type_key)).to_json())
    return _dump([{"key": t.key, "label": t.label, "category": t.category, "description": t.description,
                   "inputs": [i.name for i in t.inputs], "params": [p.name for p in t.params]} for t in registry.all()])


@mcp.tool()
@friendly
def formula_reference() -> str:
    """Syntax and functions of the formula language used by the calculate node and keep_rows.formula."""
    from .core.expr import function_docs
    lines = ["Columns: bare name (Pressure), [with spaces], or `backticks`. Operators: + - * / ^ %  = != < > <= >=  and or not  & (join text).",
             "Text in quotes; a quote inside text is doubled (\"say \"\"hi\"\"\"). A text column compared with a number is compared as numbers.",
             "IF(test, a, b). One-argument SUM/AVERAGE/MIN/MAX/MEDIAN/STDEV aggregate the whole column (broadcast); with several arguments they work row-wise.",
             "Functions:"]
    lines += [f"  {n}: {d}" for n, d in function_docs()]
    return "\n".join(lines)


# ----------------------------------------------------------------- building
@mcp.tool()
@friendly
def create_pipeline(path: str, name: str | None = None, overwrite: bool = False) -> str:
    """Create an empty pipeline file (JSON) inside the server's root folder. Use a path ending in .json,
    ideally next to the data files."""
    p = _in_root(path)
    if p.suffix.lower() != ".json":
        raise ToolError(f"A pipeline file name must end in .json, not {p.name}")
    if p.exists() and not overwrite:
        return _dump({"ok": True, "path": str(p), "note": "already exists; loaded as-is"})
    Pipeline(name or p.stem).save(p)
    return _dump({"ok": True, "path": str(p)})


@mcp.tool()
@friendly
def build_template(path: str, template: str, data_file: str | None = None) -> str:
    """Create a starter project from a template: compare | limits | fit | report. Uses a generated sample
    file (two values over time) unless data_file is given (relative paths are taken from the pipeline's folder).
    The pipeline file must not exist yet."""
    from .core.samples import TEMPLATES
    out = _in_root(path)
    if out.suffix.lower() != ".json":
        raise ToolError(f"A pipeline file name must end in .json, not {out.name}")
    if out.exists():
        raise ValueError(f"{out} already exists")
    pipe, data = hl.build_template(out, template, data_file)
    return _dump({"ok": True, "path": str(out), "data": str(data), "nodes": list(pipe.nodes), "templates": TEMPLATES})


@mcp.tool()
@friendly
def describe_pipeline(path: str) -> str:
    """Nodes, connections, settings, statuses and configuration problems of a pipeline."""
    p = _load(path)
    ex = Executor(p)
    return _dump({
        "name": p.name, "path": str(p.path),
        "nodes": [{**n.to_dict(), "inputs": p.inputs_of(n.id), "state": _record(p, ex, n.id)} for n in p.nodes.values()],
        "edges": [e.to_dict() for e in p.edges],
        "inputs": [{"name": i.name, "value": i.value, "unit": i.unit, "note": i.note} for i in p.inputs],
        "columns": p.columns,
        "answers": [a.to_dict() for a in p.answers],
        "problems": p.problems(),
    })


@mcp.tool()
@friendly
def add_node(path: str, type_key: str, params: dict[str, Any] | None = None, title: str | None = None,
             node_id: str | None = None, after: str | None = None, port: str | None = None,
             also_after: list[str] | None = None) -> str:
    """Add a node. `after` connects an existing node's output to the new node's input (port: for combine use 'left'/'right';
    for stack all inputs go to 'tables'). `also_after` connects extra upstream nodes (e.g. the second table of a combine)."""
    with _editing(path) as p:
        node = hl.add_step(p, type_key, params, title, node_id, after, port, also_after)
        _check_outputs(p, node.id)
    return _dump({"ok": True, "node": node.to_dict(), "inputs": p.inputs_of(node.id),
                  "problems": [x for x in p.problems() if x.startswith(node.title + ":")]})


@mcp.tool()
@friendly
def set_params(path: str, node_id: str, params: dict[str, Any]) -> str:
    """Change settings on a node. Only the keys given are changed. Values are validated."""
    with _editing(path) as p:
        p.set_params(hl.require_node(p, node_id), **params)
        _check_outputs(p, node_id)
    return _dump({"ok": True, "node": p.nodes[node_id].to_dict()})


@mcp.tool()
@friendly
def connect_nodes(path: str, source: str, target: str, port: str | None = None) -> str:
    """Connect source's output to target's input. port is only needed for nodes with several inputs (combine: left/right)."""
    with _editing(path) as p:
        e = p.connect(hl.require_node(p, source), hl.require_node(p, target), port)
    return _dump({"ok": True, "edge": e.to_dict()})


@mcp.tool()
@friendly
def disconnect_nodes(path: str, source: str, target: str, port: str | None = None) -> str:
    """Remove a connection (every connection between the two nodes, or only the one into `port`)."""
    with _editing(path) as p:
        p.disconnect(hl.require_node(p, source), hl.require_node(p, target), port)
    return _dump({"ok": True})


@mcp.tool()
@friendly
def remove_node(path: str, node_id: str) -> str:
    """Delete a node and its connections."""
    with _editing(path) as p:
        p.remove_node(hl.require_node(p, node_id))
    return _dump({"ok": True})


@mcp.tool()
@friendly
def rename_node(path: str, node_id: str, title: str) -> str:
    """Give a node a human-friendly title (shown on the canvas)."""
    with _editing(path) as p:
        p.rename_node(hl.require_node(p, node_id), title)
    return _dump({"ok": True})


@mcp.tool()
@friendly
def set_input(path: str, name: str, value: Any = None, unit: str = "", note: str = "") -> str:
    """Create or change a named input: a number or text usable by name in formulas (e.g. `[value] / [maximum allowed]`),
    filter values and limit lines. Changing it recomputes only the steps that use it."""
    with _editing(path) as p:
        i = p.set_input(name, value, unit, note)
    return _dump({"ok": True, "input": {"name": i.name, "value": i.value, "unit": i.unit, "note": i.note}, "all": [x.name for x in p.inputs]})


@mcp.tool()
@friendly
def remove_input(path: str, name: str) -> str:
    """Remove a named input."""
    with _editing(path) as p:
        if not any(i.name.lower() == name.lower() for i in p.inputs):
            raise ValueError(f"No input called {name!r}. Inputs: {[i.name for i in p.inputs]}")
        p.remove_input(name)
    return _dump({"ok": True, "inputs": [x.name for x in p.inputs]})


@mcp.tool()
@friendly
def set_column_label(path: str, column: str, label: str | None = None, unit: str | None = None) -> str:
    """Give a column a display name and/or unit used on charts, reports and in the table header (the data is unchanged)."""
    with _editing(path) as p:
        p.set_column_meta(column, label, unit)
    return _dump({"ok": True, "columns": p.columns})


# ----------------------------------------------------------------- running and reading
@mcp.tool()
@friendly
def run_pipeline(path: str, node_ids: list[str] | None = None, force: bool = False) -> str:
    """Execute the pipeline (or only the given nodes and what they depend on). Unchanged nodes are served from cache.
    Returns per-node status, row counts, messages, reports and errors."""
    p = _load(path)
    for nid in node_ids or []:
        hl.require_node(p, nid)
    ex = _executor(p)
    t0 = time.perf_counter()
    res = ex.run(targets=node_ids, force=force)
    return _dump(hl.run_record(p, ex, res, time.perf_counter() - t0))


@mcp.tool()
@friendly
def node_status(path: str, node_id: str) -> str:
    """Status, row count, columns, messages and report (e.g. fit coefficients, gap statistics) of one node."""
    p = _load(path)
    return _dump(_record(p, Executor(p), node_id))


def _frame(path: str, node_id: str, run: bool) -> tuple[Pipeline, pl.LazyFrame]:
    p = _load(path)
    return p, hl.result_frame(p, _executor(p), node_id, run)


@mcp.tool()
@friendly
def get_schema(path: str, node_id: str) -> str:
    """Column names and types of a node's output (works before running, as long as upstream files exist)."""
    p = _load(path)
    hl.require_node(p, node_id)
    ex = Executor(p)
    st = ex.state(node_id)
    if st.status == "done":
        return _dump({"node_id": node_id, "rows": st.rows, "columns": st.columns})
    sch = ex.schema(node_id)
    if sch is None:
        raise ValueError("Cannot determine the schema yet; check the node's inputs and settings")
    return _dump({"node_id": node_id, "rows": None, "columns": [{"name": k, "dtype": str(v)} for k, v in sch.items()]})


@mcp.tool()
@friendly
def get_sample(path: str, node_id: str, rows: int = 20, offset: int = 0, columns: list[str] | None = None, run: bool = True) -> str:
    """Rows from a node's output as JSON records (1 to 500 rows; pass columns to narrow wide tables)."""
    if not 1 <= int(rows) <= 500 or int(offset) < 0:
        raise ValueError("rows must be between 1 and 500, and offset at least 0")
    _, lf = _frame(path, node_id, run)
    return hl.select_columns(lf, columns).slice(int(offset), int(rows)).collect(engine="streaming").write_json()


@mcp.tool()
@friendly
def get_stats(path: str, node_id: str, columns: list[str] | None = None, run: bool = True) -> str:
    """Summary statistics (count, missing, mean, std, min, quartiles, max) for the columns of a node's output."""
    from .views.stats import column_summary
    _, lf = _frame(path, node_id, run)
    return column_summary(lf, columns).write_json()


@mcp.tool()
@friendly
def render_chart(path: str, node_id: str, out_png: str | None = None, kind: str | None = None, x: str | None = None,
                 y: list[str] | None = None, column: str | None = None, title: str | None = None,
                 width: int = 1200, height: int = 600, run: bool = True) -> Image:
    """Render a chart of a node's output to PNG and return the image. If node is a chart node its settings are used
    and any of kind/x/y/column/title given here override them; otherwise give kind (line|scatter|histogram|bar,
    default line), x and y. Big data is downsampled per pixel.
    out_png (optional) must be inside the pipeline file's folder; otherwise the PNG goes to the cache."""
    from .views.render import render_chart as _render
    p, lf = _frame(path, node_id, run)
    params = hl.chart_params(p.nodes[node_id], kind, x, y, column, title)
    out = _inside_project(p, out_png) if out_png else (Executor(p).cache_dir / ".charts" / f"{node_id}.png")
    _render(lf, params, out, width=width, height=height, columns=p.columns, inputs=p.input_values())
    return Image(path=str(out))


@mcp.tool()
@friendly
def export_node(path: str, node_id: str, out_path: str, run: bool = True) -> str:
    """Write a node's full output to a .csv, .tsv, .parquet or .xlsx file inside the pipeline file's folder."""
    from .core.nodes.outputs import write_table
    p, lf = _frame(path, node_id, run)
    out = _inside_project(p, out_path)
    write_table(lf, out)
    return _dump({"ok": True, "path": str(out)})


@mcp.tool()
@friendly
def open_in_gui(path: str) -> str:
    """Open the pipeline in the DANCR desktop app so the person can watch and explore. Safe to call repeatedly."""
    from .cli import launch_gui
    p = _load(path)
    for nid in p.nodes:                           # the window runs steps with no confinement of its own
        _check_outputs(p, nid)
    launch_gui(str(p.path))
    return _dump({"ok": True, "note": "DANCR is opening. It reloads the file automatically when you change it."})


@mcp.tool()
@friendly
def inspect_file(file_path: str, rows: int = 5) -> str:
    """Peek at a data file before building a pipeline: detected columns, types and the first rows.
    A relative path is taken from the server's root folder."""
    from .core.registry import Ctx
    from .core.nodes.load import scan_file
    fp = _from_root(file_path)
    ctx = Ctx(fp.parent, "inspect", "inspect", preview=True)
    lf, messages = scan_file(ctx, {"path": str(fp), "has_header": True, "parse_dates": True})
    schema = lf.collect_schema()
    head = lf.head(max(1, min(int(rows), 100))).collect(engine="streaming")
    return json.dumps({"columns": [{"name": k, "dtype": str(v)} for k, v in schema.items()], "messages": messages,
                       "head": json.loads(head.write_json())}, default=str)


def main(root: str | None = None) -> None:
    from .logsetup import configure
    global ROOT
    if root:
        ROOT = Path(root).expanduser().resolve()
        if not ROOT.is_dir():
            raise SystemExit(f"--root {ROOT} is not a folder")
    configure(stderr_level=logging.WARNING)      # stdout carries the protocol; the log file and stderr get the rest
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
