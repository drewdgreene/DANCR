"""dancr command line: build, run and inspect pipelines without the GUI.

Every command prints human-readable text; add --json for machine-readable
output. AI agents should use --json.
"""
from __future__ import annotations

import argparse
import json

import subprocess
import sys

import time
from pathlib import Path
from typing import Any

import polars as pl

from . import __version__
from .core import Pipeline, PipelineError, registry
from .core.executor import Executor, NodeState, ExecutionCancelled


class CliError(Exception):
    pass


# ----------------------------------------------------------------- helpers
def _load(path: str) -> Pipeline:
    p = Path(path)
    if not p.exists():
        raise CliError(f"No such pipeline file: {p}. Create one with: dancr new {p}")
    return Pipeline.load(p)


def _check_node(p: Pipeline, node_id: str) -> str:
    """Every command that names a step validates it here, before anything else can produce a vaguer error."""
    if node_id not in p.nodes:
        raise CliError(f"No step called {node_id!r}. Steps: {list(p.nodes)}")
    return node_id


def _parse_kv(items: list[str]) -> dict[str, Any]:
    """key=value pairs. Values that look like JSON (objects, lists, numbers, true/false/null) are decoded."""
    import re
    out: dict[str, Any] = {}
    for it in items or []:
        if "=" not in it:
            raise CliError(f"Expected key=value, got {it!r}")
        k, v = it.split("=", 1)
        vs = v.strip()
        if vs and (vs[0] in "{[" or vs in ("true", "false", "null") or re.fullmatch(r"-?\d+(\.\d+)?([eE][-+]?\d+)?", vs)):
            try:
                out[k] = json.loads(vs)
                continue
            except json.JSONDecodeError:
                pass
        out[k] = v
    return out


def _params_from_args(a: argparse.Namespace) -> dict[str, Any]:
    params = _parse_kv(getattr(a, "set", []) or [])
    if getattr(a, "params", None):
        try:
            params.update(json.loads(a.params))
        except json.JSONDecodeError as e:
            raise CliError(f"--params is not valid JSON: {e}") from e
    return params


def _state_dict(p: Pipeline, st: NodeState) -> dict[str, Any]:
    n = p.nodes[st.node_id]
    d = st.to_dict()
    d["title"] = n.title
    d["type"] = n.type
    return d


def _print(a: argparse.Namespace, data: Any, text: str | None = None) -> None:
    from .core.dtypes import json_safe
    if a.json:
        print(json.dumps(json_safe(data), indent=2, default=str))
    elif text is not None:
        print(text)
    else:
        print(json.dumps(data, indent=2, default=str))


def _fmt_state(st: NodeState, title: str) -> str:
    mark = {"done": "✓", "failed": "✗", "running": "…", "stale": "~", "idle": "·"}.get(st.status, "?")
    parts = [f"{mark} {title} [{st.node_id}]"]
    if st.status == "done":
        parts.append(f"{st.rows:,} rows × {len(st.columns)} cols")
        if st.elapsed is not None:
            parts.append("cached" if st.from_cache else f"{st.elapsed:.2f}s")
    elif st.status == "failed":
        parts.append(f"ERROR: {st.error}")
    else:
        parts.append(st.status)
    return "  ".join(parts)


# ----------------------------------------------------------------- commands
def cmd_nodes(a: argparse.Namespace) -> None:
    types = registry.all()
    if a.type:
        types = [t for t in types if t.key == a.type or t.label.lower() == a.type.lower()]
        if not types:
            raise CliError(f"No node type {a.type!r}")
    if a.json:
        _print(a, [t.to_json() for t in types])
        return
    for t in types:
        print(f"{t.key:20s} {t.label:28s} [{t.category}]  {t.description}")
        if a.type or a.verbose:
            from .core.examples import EXAMPLES
            if EXAMPLES.get(t.key):
                print(f"    e.g. {EXAMPLES[t.key]}")
            for inp in t.inputs:
                print(f"    input  {inp.name}: {inp.label}{' (many)' if inp.multiple else ''}")
            for p in t.params:
                extra = ""
                if p.choices:
                    extra = "  one of: " + ", ".join(f"{v}" for v, _ in p.choices)
                if p.visible_when:
                    extra += f"  (when {p.visible_when})"
                print(f"    {p.name:18s} {p.kind:12s} default={p.default!r}  {p.label}{extra}")
                if p.help:
                    print(f"    {'':18s} {'':12s} {p.help}")


def cmd_formulas(a: argparse.Namespace) -> None:
    from .core.expr import function_docs
    docs = function_docs()
    if a.json:
        _print(a, [{"name": n, "doc": d} for n, d in docs])
        return
    print("Formula syntax: column names bare or in [brackets]; + - * / ^ %; = != < > <= >=; and or not; & joins text.")
    for n, d in docs:
        print(f"  {n:18s} {d}")


def cmd_new(a: argparse.Namespace) -> None:
    p = Path(a.pipeline)
    if p.exists() and not a.force:
        raise CliError(f"{p} already exists (use --force to overwrite)")
    pipe = Pipeline(a.name or p.stem)
    pipe.save(p)
    _print(a, {"path": str(p), "name": pipe.name}, f"Created {p}")


def cmd_add(a: argparse.Namespace) -> None:
    p = _load(a.pipeline)
    if not registry.has(a.type):
        # try label match
        match = [t for t in registry.all() if t.label.lower() == a.type.lower()]
        if not match:
            raise CliError(f"Unknown node type {a.type!r}. Run: dancr nodes")
        a.type = match[0].key
    params = _params_from_args(a)
    for nid in [a.after, *(a.also_after or [])]:
        if nid:
            _check_node(p, nid)
    x, y = 0.0, 0.0
    if a.after:
        src = p.nodes[a.after]
        x, y = src.x + 260, src.y
    else:
        if p.nodes:
            last = list(p.nodes.values())[-1]
            x, y = last.x, last.y + 140
    node = p.add_node(a.type, title=a.title, params=params, x=x, y=y, id=a.id)
    if a.after:
        p.connect(a.after, node.id, a.port)
    for extra in a.also_after or []:
        p.connect(extra, node.id)
    p.save()
    _print(a, node.to_dict(), f"Added {node.title} as {node.id}" + (f", connected from {a.after}" if a.after else ""))


def cmd_set(a: argparse.Namespace) -> None:
    p = _load(a.pipeline)
    _check_node(p, a.node)
    params = _params_from_args(a)
    if not params:
        raise CliError("Nothing to set. Use key=value or --params '{...}'")
    p.set_params(a.node, **params)
    p.save()
    _print(a, p.nodes[a.node].to_dict(), f"Updated {a.node}: {', '.join(params)}")


def cmd_rename(a: argparse.Namespace) -> None:
    p = _load(a.pipeline)
    _check_node(p, a.node)
    p.rename_node(a.node, a.title)
    p.save()
    _print(a, p.nodes[a.node].to_dict(), f"Renamed {a.node} to {a.title!r}")


def cmd_connect(a: argparse.Namespace) -> None:
    p = _load(a.pipeline)
    _check_node(p, a.source); _check_node(p, a.target)
    e = p.connect(a.source, a.target, a.port)
    p.save()
    _print(a, e.to_dict(), f"Connected {e.source} → {e.target} ({e.port})")


def cmd_disconnect(a: argparse.Namespace) -> None:
    p = _load(a.pipeline)
    _check_node(p, a.source); _check_node(p, a.target)
    p.disconnect(a.source, a.target, a.port)
    p.save()
    _print(a, {"ok": True}, f"Disconnected {a.source} → {a.target}")


def cmd_remove(a: argparse.Namespace) -> None:
    p = _load(a.pipeline)
    _check_node(p, a.node)
    removed = p.remove_node(a.node)
    p.save()
    _print(a, {"removed": a.node, "edges_removed": [e.to_dict() for e in removed]}, f"Removed {a.node}")


def cmd_show(a: argparse.Namespace) -> None:
    p = _load(a.pipeline)
    ex = Executor(p)
    states = ex.states()
    data = {
        "path": str(p.path), "name": p.name,
        "nodes": [{**n.to_dict(), "state": _state_dict(p, states[n.id])} for n in p.nodes.values()],
        "edges": [e.to_dict() for e in p.edges],
        "problems": p.problems(),
    }
    if a.json:
        _print(a, data)
        return
    print(f"{p.name}  ({p.path})")
    for n in p.nodes.values():
        nt = registry.get(n.type)
        ins = p.inputs_of(n.id)
        src = ", ".join(f"{port}←{'+'.join(s)}" for port, s in ins.items()) if ins else "(source)"
        print(f"  {_fmt_state(states[n.id], n.title)}")
        print(f"      {n.type}: {nt.summarize(n.params)}    inputs: {src}")
    if data["problems"]:
        print("Problems:")
        for pr in data["problems"]:
            print(f"  ! {pr}")


def cmd_run(a: argparse.Namespace) -> None:
    p = _load(a.pipeline)
    for nid in a.nodes or []:
        _check_node(p, nid)
    ex = Executor(p)
    problems = p.problems()
    if problems and not a.json:
        for pr in problems:
            print(f"! {pr}", file=sys.stderr)
    events: list[dict[str, Any]] = []

    def on_event(e: dict[str, Any]) -> None:
        if e["type"] in ("node_finished", "node_failed", "node_cached"):
            st: NodeState = e["state"]
            if a.json:
                events.append({"event": e["type"], **_state_dict(p, st)})
            else:
                print(_fmt_state(st, p.nodes[st.node_id].title), flush=True)
        elif e["type"] == "node_started" and not a.json and sys.stdout.isatty():
            print(f"… {p.nodes[e['node']].title}", end="\r", flush=True)

    t0 = time.perf_counter()
    try:
        res = ex.run(targets=a.nodes or None, on_event=on_event, force=a.force)
    except ExecutionCancelled:
        raise CliError("Cancelled")
    failed = [s for s in res.values() if s.status == "failed"]
    if a.json:
        _print(a, {"ok": not failed, "elapsed": time.perf_counter() - t0, "nodes": events,
                   "problems": problems, "cache_dir": str(ex.cache_dir)})
    else:
        print(f"{'Done' if not failed else f'{len(failed)} node(s) failed'} in {time.perf_counter() - t0:.1f}s. Cache: {ex.cache_dir}")
    if failed:
        sys.exit(1)


def cmd_status(a: argparse.Namespace) -> None:
    p = _load(a.pipeline)
    ex = Executor(p)
    nodes = [_check_node(p, a.node)] if a.node else list(p.nodes)
    states = {n: ex.state(n) for n in nodes}
    if a.json:
        _print(a, [_state_dict(p, s) for s in states.values()])
        return
    for n, s in states.items():
        print(_fmt_state(s, p.nodes[n].title))
        for m in s.messages:
            print(f"      {m}")
        if s.report:
            for k, v in s.report.items():
                print(f"      {k}: {v}")


def _frame(a: argparse.Namespace, p: Pipeline, ex: Executor, node: str) -> pl.LazyFrame:
    _check_node(p, node)
    st = ex.state(node)
    if st.status != "done":
        if getattr(a, "run", False):
            res = ex.run(targets=[node])
            if res[node].status != "done":
                raise CliError(f"{node} failed: {res[node].error}")
        else:
            raise CliError(f"{node} has not been run yet (status: {st.status}). Run: dancr run {p.path} {node}   or add --run")
    return ex.frame(node)


def cmd_schema(a: argparse.Namespace) -> None:
    p = _load(a.pipeline)
    _check_node(p, a.node)
    ex = Executor(p)
    st = ex.state(a.node)
    if st.status == "done":
        cols = st.columns
        rows = st.rows
    else:
        sch = ex.schema(a.node)
        if sch is None:
            raise CliError(f"Cannot work out the columns of {a.node} yet (upstream missing or misconfigured)")
        cols = [{"name": n, "dtype": str(d)} for n, d in sch.items()]
        rows = None
    _print(a, {"node": a.node, "rows": rows, "columns": cols},
           "\n".join(f"{c['name']:30s} {c['dtype']}" for c in cols) + (f"\n{rows:,} rows" if rows is not None else ""))


def cmd_sample(a: argparse.Namespace) -> None:
    p = _load(a.pipeline)
    ex = Executor(p)
    lf = _frame(a, p, ex, a.node)
    df = lf.slice(a.offset, a.rows).collect(engine="streaming")
    if a.json:
        print(df.write_json())
    elif a.csv:
        print(df.write_csv(), end="")
    else:
        with pl.Config(tbl_rows=a.rows, tbl_cols=-1, tbl_width_chars=200, fmt_str_lengths=40):
            print(df)


def cmd_stats(a: argparse.Namespace) -> None:
    from .views.stats import column_summary
    p = _load(a.pipeline)
    ex = Executor(p)
    lf = _frame(a, p, ex, a.node)
    df = column_summary(lf, a.columns.split(",") if a.columns else None)
    if a.json:
        print(df.write_json())
    else:
        with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=220):
            print(df)


def cmd_chart(a: argparse.Namespace) -> None:
    from .views.render import render_chart
    p = _load(a.pipeline)
    ex = Executor(p)
    lf = _frame(a, p, ex, a.node)
    node = p.nodes[a.node]
    params = dict(node.params) if node.type == "chart" else {}
    if a.kind:
        params["kind"] = a.kind
    if a.x:
        params["x"] = a.x
    if a.y:
        params["series"] = [{"column": c} for c in a.y.split(",")]
    if a.column:
        params["column"] = a.column
    if a.title:
        params["title"] = a.title
    params.setdefault("kind", "line")
    out = render_chart(lf, params, a.out, width=a.width, height=a.height)
    _print(a, {"path": str(out), "params": params}, f"Wrote {out}")


def cmd_export(a: argparse.Namespace) -> None:
    p = _load(a.pipeline)
    ex = Executor(p)
    lf = _frame(a, p, ex, a.node)
    from .core.nodes.outputs import write_table
    out = Path(a.out)
    write_table(lf, out)
    _print(a, {"path": str(out)}, f"Wrote {out}")


def cmd_clear_cache(a: argparse.Namespace) -> None:
    p = _load(a.pipeline)
    ex = Executor(p)
    size = ex.cache_size()
    ex.clear_cache()
    _print(a, {"cleared_bytes": size}, f"Cleared {size / 1e6:.1f} MB")


def launch_gui(pipeline: str | None, wait: bool = False) -> None:
    """Start the desktop app in its own process; its output goes to gui-stdio.log next to the log file."""
    from .logsetup import log_path, rotate_if_large
    target = [str(Path(pipeline).expanduser().resolve())] if pipeline else []
    if getattr(sys, "frozen", False):                  # packaged app: the windowed DANCR binary sits next to dancr-cli
        gui = Path(sys.executable).with_name("DANCR.exe" if sys.platform == "win32" else "DANCR")
        args = [str(gui), "gui", *target]
    else:
        args = [sys.executable, "-m", "dancr.ui.app", *target]
    if wait:
        subprocess.call(args); return
    lp = log_path()
    try:
        lp.parent.mkdir(parents=True, exist_ok=True)
        stdio = lp.parent / "gui-stdio.log"
        rotate_if_large(stdio)
        out = open(stdio, "a")
    except OSError:
        out = subprocess.DEVNULL
    subprocess.Popen(args, start_new_session=True, stdout=out, stderr=out)


def cmd_gui(a: argparse.Namespace) -> None:
    """Run the window in this process (what the packaged app's double-click does)."""
    from .ui.app import main as gui_main
    sys.exit(gui_main([sys.argv[0], *([a.pipeline] if a.pipeline else [])]))


def cmd_open(a: argparse.Namespace) -> None:
    launch_gui(a.pipeline, a.wait)
    if not a.wait:
        _print(a, {"launched": True}, "Opened DANCR" + (f" with {a.pipeline}" if a.pipeline else ""))


def cmd_log(a: argparse.Namespace) -> None:
    from .logsetup import log_path
    lp = log_path()
    if a.json:
        _print(a, {"path": str(lp)})
        return
    print(lp)
    if lp.exists():
        print("".join(lp.read_text(errors="replace").splitlines(True)[-a.lines:]), end="")


def cmd_mcp(a: argparse.Namespace) -> None:
    from .mcp_server import main as mcp_main
    mcp_main()


def cmd_synth(a: argparse.Namespace) -> None:
    from .synth import write_dataset
    if a.hours <= 0 or a.rate <= 0:
        raise CliError("--hours and --rate must be greater than 0")
    t = write_dataset(Path(a.out_dir), a.hours, a.rate, a.seed, a.format)
    _print(a, t, f"Wrote probe_A and probe_B ({t['rows_a']:,} / {t['rows_b']:,} rows) to {a.out_dir}")


def cmd_template(a: argparse.Namespace) -> None:
    from .core.samples import build_template, write_sample, TEMPLATES
    if a.list or not a.pipeline:
        _print(a, TEMPLATES, "\n".join(f"{t['key']:10} {t['title']} — {t['blurb']}" for t in TEMPLATES)); return
    out = Path(a.pipeline)
    if out.exists() and not a.force:
        raise CliError(f"{out} already exists (use --force to overwrite)")
    data = Path(a.data) if a.data else write_sample(out.parent)
    pipe = Pipeline(out.stem); pipe.path = out.resolve()
    build_template(a.key, pipe, data.resolve())
    pipe.save(out)
    _print(a, {"path": str(out), "data": str(data), "nodes": list(pipe.nodes)}, f"Built the {a.key!r} template in {out} on {data}")


def cmd_inputs(a: argparse.Namespace) -> None:
    p = _load(a.pipeline)
    if a.remove:
        p.remove_input(a.remove); p.save()
        _print(a, {"removed": a.remove}, f"Removed input {a.remove!r}"); return
    if a.name:
        p.set_input(a.name, a.value, a.unit, a.note); p.save()
    rows = [{"name": i.name, "value": i.value, "unit": i.unit, "note": i.note} for i in p.inputs]
    _print(a, rows, "\n".join(f"{r['name']:24} {r['value']!s:>12} {r['unit']:8} {r['note']}" for r in rows) or "no inputs")


def cmd_columns(a: argparse.Namespace) -> None:
    p = _load(a.pipeline)
    if a.name:
        p.set_column_meta(a.name, a.label, a.unit); p.save()
    _print(a, p.columns, "\n".join(f"{k:24} label={v.get('label', '')!r} unit={v.get('unit', '')!r}" for k, v in p.columns.items()) or "no column names or units set")


# ----------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    kv = ("KEY=VALUE settings: values that look like JSON (numbers, true/false/null, [lists], {objects}) are parsed as JSON, "
          "so sheet=1 is the number 1; quote it as sheet='\"1\"' for the text 1. Anything else is text.")
    ap = argparse.ArgumentParser(prog="dancr", description="DANCR: visual data pipelines for big tables. Headless CLI.", epilog=kv)
    ap.add_argument("--version", action="version", version=f"dancr {__version__}")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("nodes", help="list node types and their settings"); s.add_argument("type", nargs="?"); s.add_argument("-v", "--verbose", action="store_true"); s.set_defaults(fn=cmd_nodes)
    s = sub.add_parser("formulas", help="list formula functions"); s.set_defaults(fn=cmd_formulas)
    s = sub.add_parser("new", help="create an empty pipeline file"); s.add_argument("pipeline"); s.add_argument("--name"); s.add_argument("--force", action="store_true"); s.set_defaults(fn=cmd_new)
    s = sub.add_parser("add", help="add a node", epilog=kv); s.add_argument("pipeline"); s.add_argument("type"); s.add_argument("--id"); s.add_argument("--title")
    s.add_argument("--after", help="connect from this node"); s.add_argument("--port", help="input port on the new node (for combine: left/right)")
    s.add_argument("--also-after", action="append", help="additional upstream nodes (e.g. second input of combine)")
    s.add_argument("--set", action="append", metavar="KEY=VALUE", help="a setting (see below)"); s.add_argument("--params", help="JSON object of settings"); s.set_defaults(fn=cmd_add)
    s = sub.add_parser("set", help="change a node's settings", epilog=kv); s.add_argument("pipeline"); s.add_argument("node"); s.add_argument("set", nargs="*", metavar="KEY=VALUE", help="settings to change (see below)"); s.add_argument("--params", help="JSON object of settings"); s.set_defaults(fn=cmd_set)
    s = sub.add_parser("rename", help="rename a node"); s.add_argument("pipeline"); s.add_argument("node"); s.add_argument("title"); s.set_defaults(fn=cmd_rename)
    s = sub.add_parser("connect", help="connect two nodes"); s.add_argument("pipeline"); s.add_argument("source"); s.add_argument("target"); s.add_argument("--port"); s.set_defaults(fn=cmd_connect)
    s = sub.add_parser("disconnect", help="remove a connection"); s.add_argument("pipeline"); s.add_argument("source"); s.add_argument("target"); s.add_argument("--port"); s.set_defaults(fn=cmd_disconnect)
    s = sub.add_parser("remove", help="delete a node"); s.add_argument("pipeline"); s.add_argument("node"); s.set_defaults(fn=cmd_remove)
    s = sub.add_parser("show", help="print the pipeline and node statuses"); s.add_argument("pipeline"); s.set_defaults(fn=cmd_show)
    s = sub.add_parser("run", help="execute the pipeline (or just some nodes and what they need)"); s.add_argument("pipeline"); s.add_argument("nodes", nargs="*"); s.add_argument("--force", action="store_true", help="ignore the cache"); s.set_defaults(fn=cmd_run)
    s = sub.add_parser("status", help="node status, messages and reports"); s.add_argument("pipeline"); s.add_argument("node", nargs="?"); s.set_defaults(fn=cmd_status)
    s = sub.add_parser("schema", help="columns of a node's output"); s.add_argument("pipeline"); s.add_argument("node"); s.set_defaults(fn=cmd_schema)
    s = sub.add_parser("sample", help="print rows of a node's output"); s.add_argument("pipeline"); s.add_argument("node"); s.add_argument("--rows", type=int, default=20); s.add_argument("--offset", type=int, default=0); s.add_argument("--csv", action="store_true"); s.add_argument("--run", action="store_true", help="run first if needed"); s.set_defaults(fn=cmd_sample)
    s = sub.add_parser("stats", help="summary statistics of a node's output"); s.add_argument("pipeline"); s.add_argument("node"); s.add_argument("--columns"); s.add_argument("--run", action="store_true"); s.set_defaults(fn=cmd_stats)
    s = sub.add_parser("chart", help="render a chart of a node's output to PNG"); s.add_argument("pipeline"); s.add_argument("node"); s.add_argument("--out", required=True)
    s.add_argument("--kind", choices=["line", "scatter", "histogram", "bar"]); s.add_argument("--x"); s.add_argument("--y", help="comma-separated columns"); s.add_argument("--column"); s.add_argument("--title")
    s.add_argument("--width", type=int, default=1400); s.add_argument("--height", type=int, default=700); s.add_argument("--run", action="store_true"); s.set_defaults(fn=cmd_chart)
    s = sub.add_parser("export", help="write a node's output to csv/parquet/xlsx"); s.add_argument("pipeline"); s.add_argument("node"); s.add_argument("out"); s.add_argument("--run", action="store_true"); s.set_defaults(fn=cmd_export)
    s = sub.add_parser("clear-cache", help="delete cached outputs"); s.add_argument("pipeline"); s.set_defaults(fn=cmd_clear_cache)
    s = sub.add_parser("gui", help="run the window in this process"); s.add_argument("pipeline", nargs="?"); s.set_defaults(fn=cmd_gui)
    s = sub.add_parser("open", help="open the GUI (optionally on a pipeline)"); s.add_argument("pipeline", nargs="?"); s.add_argument("--wait", action="store_true"); s.set_defaults(fn=cmd_open)
    s = sub.add_parser("mcp", help="start the MCP server (stdio) for AI agents"); s.set_defaults(fn=cmd_mcp)
    s = sub.add_parser("log", help="print the log file path and its last lines"); s.add_argument("--lines", type=int, default=40); s.set_defaults(fn=cmd_log)
    s = sub.add_parser("template", help="build a starter project (on sample data unless --data is given)"); s.add_argument("key", nargs="?"); s.add_argument("pipeline", nargs="?"); s.add_argument("--data"); s.add_argument("--list", action="store_true"); s.add_argument("--force", action="store_true"); s.set_defaults(fn=cmd_template)
    s = sub.add_parser("inputs", help="list, set or remove named inputs (values usable in formulas, filters and limits)"); s.add_argument("pipeline"); s.add_argument("name", nargs="?"); s.add_argument("value", nargs="?"); s.add_argument("--unit", default=""); s.add_argument("--note", default=""); s.add_argument("--remove"); s.set_defaults(fn=cmd_inputs)
    s = sub.add_parser("columns", help="list or set display names and units for columns"); s.add_argument("pipeline"); s.add_argument("name", nargs="?"); s.add_argument("--label"); s.add_argument("--unit"); s.set_defaults(fn=cmd_columns)
    s = sub.add_parser("synth", help="generate a synthetic two-probe test dataset"); s.add_argument("out_dir"); s.add_argument("--hours", type=float, default=1.0); s.add_argument("--rate", type=float, default=20.0); s.add_argument("--seed", type=int, default=1); s.add_argument("--format", choices=["csv", "parquet"], default="csv"); s.set_defaults(fn=cmd_synth)
    return ap


def main(argv: list[str] | None = None) -> None:
    ap = build_parser()
    a = ap.parse_args(argv)
    if a.cmd != "gui":                                  # the window configures its own logging
        import logging
        from .logsetup import configure
        configure(stderr_level=logging.WARNING)         # file as usual, warnings to stderr, never stdout
    try:
        a.fn(a)
    except (CliError, PipelineError, ValueError, OSError, pl.exceptions.PolarsError) as e:
        if a.json:
            print(json.dumps({"error": str(e)}))
        else:
            print(f"error: {e}", file=sys.stderr)
        sys.exit(2)
    except (KeyError, TypeError, AttributeError) as e:
        msg = str(e).strip("'\"") or type(e).__name__
        if a.json:
            print(json.dumps({"error": msg}))
        else:
            print(f"error: {msg}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
