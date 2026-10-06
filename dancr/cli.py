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
from . import headless as hl
from .core import Pipeline, PipelineError, registry
from .core.executor import Executor, NodeState


class CliError(Exception):
    pass


# Exit codes: 0 ok, 1 a step failed (dancr run), 2 a usage or input error with a message, 3 an internal error
# (a DANCR bug; the traceback is in the log).
EXIT_FAILED, EXIT_ERROR, EXIT_BUG = 1, 2, 3


class _Parser(argparse.ArgumentParser):
    """Argument errors in the same shape as every other error: JSON with --json (anywhere on the line)."""

    def error(self, message: str) -> None:  # type: ignore[override]
        if "--json" in sys.argv[1:]:
            print(json.dumps({"error": f"{self.prog}: {message}"}))
            sys.exit(EXIT_ERROR)
        super().error(message)


# ----------------------------------------------------------------- helpers
def _load(path: str) -> Pipeline:
    p = Path(path)
    if not p.exists():
        raise CliError(f"There's no project file at {p}. To create one, run dancr new {p}")
    return Pipeline.load(p)


def _with_files(path: str, files: list[str] | None) -> Pipeline:
    """The project with a load step for each file named, added under the lock (only when there are files).
    The slow reading that follows is done without the lock, so nobody waits on it."""
    if files:
        with _editing(path) as p:
            hl.add_files(p, _abs(files))
        return p
    return _load(path)


def _editing(path: str):
    """Load, change and save a project file holding its lock, so the window, an agent or another command never
    saves over the change (and it never saves over theirs). Saved only when something changed."""
    p = Path(path)
    if not p.exists():
        raise CliError(f"There's no project file at {p}. To create one, run dancr new {p}")
    return hl.editing(p)


def _check_node(p: Pipeline, node_id: str) -> str:
    """Every command that names a step validates it here, before anything else can produce a vaguer error."""
    return hl.require_node(p, node_id)


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
            given = json.loads(a.params)
        except json.JSONDecodeError as e:
            raise CliError(f"--params is not valid JSON: {e}") from e
        if not isinstance(given, dict):
            raise CliError('--params must be a JSON object of settings, like \'{"path": "data.csv"}\'')
        params.update(given)
    return params


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
            raise CliError(f"No step type {a.type!r}")
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
    with hl.project_lock(p):
        if p.exists() and not a.force:
            raise CliError(f"{p} already exists. Use --force to overwrite it")
        pipe = Pipeline(a.name or p.stem)
        pipe.save(p)
    _print(a, {"path": str(p), "name": pipe.name}, f"Created {p}")


def cmd_add(a: argparse.Namespace) -> None:
    with _editing(a.pipeline) as p:
        node = hl.add_step(p, a.type, _params_from_args(a), a.title, a.id, a.after, a.port, a.also_after)
    _print(a, hl.node_public(p, node.id), f"Added {node.title} as {node.id}" + (f", connected from {a.after}" if a.after else ""))


def cmd_set(a: argparse.Namespace) -> None:
    params = _params_from_args(a)
    with _editing(a.pipeline) as p:
        _check_node(p, a.node)
        if not params:
            raise CliError("Nothing to set. Use key=value or --params '{...}'")
        p.set_params(a.node, **params)
    _print(a, hl.node_public(p, a.node), f"Updated {a.node}: {', '.join(params)}")


def cmd_rename(a: argparse.Namespace) -> None:
    with _editing(a.pipeline) as p:
        _check_node(p, a.node)
        p.rename_node(a.node, a.title)
    _print(a, hl.node_public(p, a.node), f"Renamed {a.node} to {a.title!r}")


def cmd_connect(a: argparse.Namespace) -> None:
    with _editing(a.pipeline) as p:
        _check_node(p, a.source); _check_node(p, a.target)
        e = p.connect(a.source, a.target, a.port)
    _print(a, e.to_dict(), f"Connected {e.source} → {e.target} ({e.port})")


def cmd_disconnect(a: argparse.Namespace) -> None:
    with _editing(a.pipeline) as p:
        _check_node(p, a.source); _check_node(p, a.target)
        p.disconnect(a.source, a.target, a.port)
    _print(a, {"ok": True}, f"Disconnected {a.source} → {a.target}")


def cmd_remove(a: argparse.Namespace) -> None:
    with _editing(a.pipeline) as p:
        _check_node(p, a.node)
        removed = p.remove_node(a.node)
    _print(a, {"removed": a.node, "edges_removed": [e.to_dict() for e in removed]}, f"Removed {a.node}")


def cmd_show(a: argparse.Namespace) -> None:
    from .core.secrets import redact_params
    p = _load(a.pipeline)
    ex = Executor(p)
    states = ex.states()
    data = {
        "path": str(p.path), "name": p.name,
        "nodes": [{**n.to_dict(), "params": redact_params(registry.get(n.type), n.params),
                   "state": hl.node_record(p, states[n.id])} for n in p.nodes.values()],
        "edges": [e.to_dict() for e in p.edges],
        "answers": [a.to_dict() for a in p.answers],
        "problems": p.problems(),
    }
    if a.json:
        _print(a, data)
        return
    print(f"{p.name}  ({p.path})")
    for ans in p.answers:
        st = states.get(ans.terminal)
        mark = {"done": "✓", "failed": "✗"}.get(st.status if st else "", "·")
        print(f"  {mark} ★ {ans.title} [{ans.id}] → {ans.terminal}")
    for n in p.nodes.values():
        nt = registry.get(n.type)
        ins = p.inputs_of(n.id)
        src = ", ".join(f"{port}←{'+'.join(s)}" for port, s in ins.items()) if ins else "(source)"
        print(f"  {_fmt_state(states[n.id], n.title)}")
        print(f"      {n.type}: {nt.summarize(redact_params(nt, n.params))}    inputs: {src}")
    if data["problems"]:
        print("Problems:")
        for pr in data["problems"]:
            print(f"  ! {pr}")


def _abs(files: list[str] | None) -> list[str]:
    """Files named on the command line are relative to where the command runs, not to the project."""
    return [str(Path(f).expanduser().resolve()) for f in files or []]


def cmd_understand(a: argparse.Namespace) -> None:
    p = _with_files(a.pipeline, a.file)
    m = hl.data_model(p, deep=not a.quick)
    data = m.to_dict()
    if a.json:
        _print(a, data); return
    for t in data["tables"]:
        rows = f"{t['rows']:,} rows" if t["rows"] is not None else f"{t['sampled']:,}+ rows"
        print(f"{t['title']} [{t['node']}]  {t['shape']}, {rows}" + (f", time {t['time']} {t['start']} → {t['end']}" if t["time"] else ""))
        for c in t["columns"]:
            extra = f" ({c['unit']})" if c["unit"] else ""
            print(f"    {c['name']}{extra}: {c['role']}" + (f", {len(c['values'])} values" if c["values"] else ""))
    for r in data["relations"]:
        print(f"  {r['kind']}: {r['why']}")
    for nid, why in data["skipped"].items():
        print(f"  could not read {nid}: {why}")


def cmd_context(a: argparse.Namespace) -> None:
    """A knowledge-base export of the project's datasets: schema, relations, statistics, an optional sample
    and a prose doc card per table. JSON by default (--jsonl for one line per dataset), or --output to a file."""
    from .core.dtypes import json_safe
    p = _with_files(a.pipeline, a.file)
    ctx = hl.build_context(p, nodes=[a.node] if a.node else None, deep=not a.quick,
                           stats=not a.no_stats, samples=a.samples, sample_rows=a.sample_rows)
    if a.changed:
        ctx = hl.context_changes(ctx, a.changed)
    if a.jsonl:
        text = hl.context_jsonl(ctx)
    elif a.json:
        text = json.dumps(json_safe(ctx), indent=2, default=str)
    else:
        text = hl.context_text(ctx)
    if a.output:
        target = hl.write_text_atomic(a.output, text)
        print(f"Wrote {target} ({len(ctx['tables'])} dataset(s), context version {ctx['version']})")
        return
    print(text)


def cmd_dataset(a: argparse.Namespace) -> None:
    """Show or set the project's dataset-level metadata (creator, license, description…)."""
    if a.set or a.remove:
        with _editing(a.pipeline) as p:
            if a.remove:
                p.set_dataset_meta(**{a.remove: None})
            if a.set:
                p.set_dataset_meta(**_parse_kv(a.set))
            meta = p.dataset_meta()
    else:
        p = _load(a.pipeline)
        meta = p.dataset_meta()
    _print(a, meta, "\n".join(f"{k}: {v}" for k, v in meta.items()) or "no dataset metadata")


def cmd_fair(a: argparse.Namespace) -> None:
    """Write a FAIR descriptor for the project's datasets."""
    p = _load(a.pipeline)
    ex = Executor(p)
    out = Path(a.out) if a.out else None
    doc = hl.export_fair(p, ex, fmt=a.format, samples=a.samples, out=out)
    if out is not None:
        _print(a, {"path": str(out), "format": a.format}, f"Wrote {out}")
    else:
        _print(a, doc)


def cmd_trace(a: argparse.Namespace) -> None:
    """The Assistant conversation saved with the project, as an audit log."""
    p = _load(a.pipeline)
    tr = hl.assistant_trace(p)
    if a.json:
        _print(a, tr)
        return
    if not tr["turns"]:
        print("No Assistant conversation is saved in this project.")
        return
    print(f"{tr['count']} turn(s)" + (f" · model {tr['model']}" if tr["model"] else ""))
    for i, t in enumerate(tr["turns"]):
        print(f"[{i}] {t.get('role', '?')}: " + " ".join(str(t.get("text") or "").split())[:160])
        if t.get("node"):
            print(f"     built: {t['node']}" + (f" · answer {t['answer']}" if t.get("answer") else ""))
        if t.get("finding"):
            print(f"     finding: {t['finding']}")
        if t.get("unverified"):
            print(f"     UNVERIFIED FIGURES: {t['unverified']}")
        if t.get("flags"):
            print(f"     flags: {t['flags']}")


def cmd_eval(a: argparse.Namespace) -> None:
    """Score a set of questions against the project: the deterministic engine, or the Assistant with --model."""
    p = _load(a.pipeline)
    cases = hl.load_eval_set(a.set)
    res = hl.run_eval(p, cases, model=a.model)
    if a.json:
        _print(a, res)
    else:
        s = res["summary"]
        print(f"{s['passed']}/{s['total']} passed" + (" (Assistant)" if res["model"] else " (engine)"))
        for c in res["cases"]:
            print(f"  {'✓' if c['ok'] else '✗'} {c['id']}: {c['question']}")
            if not c["ok"]:
                print(f"      {c.get('reason', '')}")
    if not res["ok"]:
        sys.exit(EXIT_FAILED)


def cmd_search(a: argparse.Namespace) -> None:
    """Search the project's own text index (built by a 'Build search index' step)."""
    p = _load(a.pipeline)
    res = hl.search_knowledge(p, a.query, node=a.node, k=a.k, min_score=a.min_score,
                              allow_restricted=a.allow_restricted, executor=Executor(p))
    if a.json:
        _print(a, res)
        return
    extra = f" ({res['withheld']} restricted withheld)" if res.get("withheld") else ""
    if not res["hits"]:
        print(f"No passages matched {a.query!r}{extra}")
        return
    print(f"{res['count']} passage(s) for {a.query!r}{extra}:")
    for h in res["hits"]:
        src = h.get("source") or h.get("doc") or h.get("chunk_id")
        text = " ".join(str(h.get("text", "")).split())
        print(f"  [{h['rank']}] {h['score']:.3f}  {src}")
        print(f"      {text[:200]}{'…' if len(text) > 200 else ''}")


def cmd_watch(a: argparse.Namespace) -> None:
    """Watch a project and its data files, and rerun when anything changes."""
    if a.batch and (not a.files or not a.out_dir):
        raise CliError("--batch needs --files and --out-dir")

    def on_event(e: dict[str, Any]) -> None:
        if a.json:
            return
        t = e.get("type")
        if t == "watch_started":
            print(f"Watching {e['path']} every {a.interval:g}s (Ctrl+C to stop)")
        elif t == "watch_changed":
            print("changed: " + ", ".join(Path(x).name for x in e["files"]), flush=True)
        elif t == "watch_ran":
            rec = e["record"]
            print(f"ran: {'ok' if rec.get('ok') else 'failed'}", flush=True)
        elif t == "watch_error":
            print(f"error: {e['error']}", file=sys.stderr)

    try:
        rec = hl.watch(a.pipeline, node=a.node, batch=a.batch, files=a.files, out_dir=a.out_dir,
                       loader=a.loader, interval=a.interval, once=a.once, on_event=on_event)
    except KeyboardInterrupt:
        rec = {"ok": True, "stopped": True}
    if a.json:
        _print(a, rec)
    else:
        print("Stopped." if rec.get("stopped") else "Done.")


def cmd_catalog(a: argparse.Namespace) -> None:
    """A catalog of every DANCR project in a folder: each project's datasets, for an index or an overview."""
    from .core.dtypes import json_safe
    cat = hl.build_catalog(a.root, pattern=a.pattern, recursive=not a.no_recursive,
                           stats=not a.no_stats, samples=a.samples, fair_format=a.fair, jobs=a.jobs)
    if a.changed:
        cat = hl.catalog_changes(cat, a.changed)
    if a.jsonl:
        text = hl.catalog_jsonl(cat, by_project=a.by_project)
    elif a.json:
        text = json.dumps(json_safe(cat), indent=2, default=str)
    else:
        lines = [f"{cat['count']} project(s) under {cat['root']}"]
        for proj in cat["projects"]:
            lines.append(f"  {proj['name']}  ({proj['file']})")
            for d in proj.get("datasets", []):
                lines.append(f"      [{d['id']}] {d['title']}" + (f"  ({d['rows']:,} rows)" if d.get("rows") else ""))
        for f, why in cat.get("skipped", {}).items():
            lines.append(f"  ! {f}: {why}")
        text = "\n".join(lines)
    if a.output:
        target = hl.write_text_atomic(a.output, text)
        print(f"Wrote {target} ({cat['count']} project(s))")
        return
    print(text)


def cmd_package(a: argparse.Namespace) -> None:
    """Write a self-contained RO-Crate (FAIR descriptors, the project and its run manifest)."""
    p = _load(a.pipeline)
    rec = hl.package_rocrate(p, Executor(p), out=a.out, copy=a.copy, zip=not a.dir, overwrite=a.force)
    _print(a, rec, f"Wrote a RO-Crate to {rec['path']} ({rec['format']}, {rec['count']} data/result file(s))")


def cmd_verify(a: argparse.Namespace) -> None:
    """Record an attestation of the project, or verify the project still reproduces one."""
    p = _load(a.pipeline)
    ex = Executor(p)
    if a.manifest:
        ref = Path(a.manifest).expanduser()
        res = hl.verify_pipeline(p, ref, mode="rerun" if a.rerun else "stored", strict_sources=a.strict_sources,
                                 hash_outputs=not a.no_hash)
        if a.json:
            _print(a, res)
        else:
            lines = [f"{res['verdict'].upper()}: {res['summary']}"]
            for c in res["checks"]:
                if c.get("match") is False:
                    lines.append(f"  x {c['scope']} {c.get('id') or c['name']}: {c.get('expected')} -> {c.get('actual')}")
            for m in res["mismatches"]:
                if m.get("scope") == "finding":
                    lines.append(f"  x finding {m['id']}: {m.get('expected')!r} -> {m.get('actual')!r}")
            for n in res["notices"]:
                lines.append(f"  ! {n.get('message') or n.get('name')}")
            for i in res.get("incomplete", []):
                lines.append(f"  ... {i}")
            _print(a, res, "\n".join(lines))
        if not res["ok"]:
            sys.exit(EXIT_FAILED)
        return
    att = hl.build_attestation(p, ex, hash_outputs=not a.no_hash)
    if a.record:
        target = hl.write_text_atomic(a.record, hl.dump_attestation(att))
        _print(a, {"ok": True, "path": str(target), "attestation_hash": att["attestation_hash"],
                   "nodes": len(att["nodes"]), "answers": len(att.get("answers", []))},
               f"Wrote attestation {target} ({att['attestation_hash']}, {len(att['nodes'])} step(s))")
        return
    if a.json:
        _print(a, att)
    else:
        print(f"Attestation {att['attestation_hash']}: {len(att['nodes'])} step(s), "
              f"{len(att.get('answers', []))} answer(s), {len(att.get('sources', []))} source(s).")
        print("Record it with --record FILE, or check a project against one with --manifest FILE.")


def cmd_lineage(a: argparse.Namespace) -> None:
    """What produced a step, and what depends on it."""
    p = _load(a.pipeline)
    direction = "up" if a.up else ("down" if a.down else "both")
    res = hl.build_lineage(p, a.node, direction=direction, executor=Executor(p))
    if a.json:
        _print(a, res)
        return
    lines = [res["sentence"]]
    if res.get("up"):
        lines.append("  up:   " + (", ".join(res["up"]["nodes"]) or "(nothing)"))
    if res.get("down"):
        lines.append("  down: " + (", ".join(res["down"]["nodes"]) or "(nothing)"))
        for x in res["down"].get("answers", []):
            lines.append(f"    answer: {x['id']} ({x['title']})")
        for x in res["down"].get("artifacts", []):
            lines.append(f"    file: {x['file']}")
        for x in res["down"].get("agent_turns", []):
            lines.append(f"    agent turn {x['turn']}: {x.get('question', '')}")
    _print(a, res, "\n".join(lines))


def cmd_proof(a: argparse.Namespace) -> None:
    """A proof card for a step: the chain, the hashes, the sources and the command that re-checks it."""
    p = _load(a.pipeline)
    card = hl.proof_card(p, Executor(p), a.node)
    if a.json:
        _print(a, card)
        return
    lines = [f"{card['title']} [{card['node']}]"]
    rows = card.get("rows")
    lines.append(f"result: {rows:,} row(s)" if isinstance(rows, int) else "result: not computed")
    if card.get("output_hash"):
        lines.append(f"content hash: {card['output_hash']}")
    if card.get("finding"):
        lines.append(f"finding: {card['finding']}")
    lines.append("steps:")
    for s in card["steps"]:
        lines.append(f"  {s['id']} ({s['type']})  {s.get('plan_hash')}")
    for s in card["sources"]:
        lines.append(f"  source: {s.get('path')}  [{s.get('sample')}]")
    for x in card["assumptions"]:
        lines.append(f"  assumed: {x}")
    lines.append(f"verify: {card['verify']}")
    _print(a, card, "\n".join(lines))


def cmd_suggest(a: argparse.Namespace) -> None:
    p = _with_files(a.pipeline, a.file)
    sugs = hl.suggestions(p, a.focus)
    if a.build is not None:
        if not 0 <= a.build < len(sugs):
            raise CliError(f"There are {len(sugs)} suggestions (numbered from 0)")
        out = hl.editing_deferred(a.pipeline, lambda pp: hl.build_answer(pp, sugs[a.build]["spec"]))
    if a.build is not None:
        _print(a, out, f"Built “{out['title']}” as {out['id']}, with its result in step {out['terminal']}. Use dancr run to compute it.")
        return
    _print(a, sugs, "\n".join(f"{s['index']}. {s['title']}  — {s['why']}" for s in sugs) or "Nothing to suggest yet. Add a data file first.")


def cmd_ask(a: argparse.Namespace) -> None:
    def work(pp):                                   # the files are added even when the question is not understood
        hl.add_files(pp, _abs(a.file))
        return hl.ask_question(pp, a.question, build=not a.dry_run)

    out = hl.editing_deferred(a.pipeline, work)     # reading the data runs with the lock released
    q = out["question"]
    if not q["ok"]:
        if a.json:
            print(json.dumps({"error": q["message"], "unknown": q["unknown"], "hints": q["hints"]}))
            sys.exit(EXIT_ERROR)
        raise CliError(q["message"])
    if a.dry_run:
        _print(a, out, f"Would build “{q['title']}”: " + ", ".join(c["text"] for c in q["chips"]))
        return
    ans = out["answer"]
    lines = [f"Built “{ans['title']}” as {ans['id']}, with its result in step {ans['terminal']}. Use dancr run to compute it."]
    lines += [f"  assumed: {x['text']}" for x in ans["assumptions"]]
    lines += [f"  note: “{x['text']}” could also mean " + ", ".join(c["label"] for c in x["choices"]) for x in q["ambiguous"]]
    _print(a, out, "\n".join(lines))


def cmd_connections(a: argparse.Namespace) -> None:
    out = hl.editing_deferred(a.pipeline, lambda pp: hl.connection_map(pp, recompute=a.recompute))
    lines = []
    for c in out.get("connections", []):
        bits = [c.get("kind", "")]
        if c.get("left_on"):
            bits.append(f"{c['left_on']} = {c.get('right_on')}")
        if c.get("match_pct") is not None:
            bits.append(f"{c['match_pct']}% match")
        if c.get("cardinality"):
            bits.append(c["cardinality"])
        lines.append(" ↔ ".join(c.get("tables", [])) + "  (" + ", ".join(str(b) for b in bits if b) + ")")
    _print(a, out, "\n".join(lines) or "No relations found between the tables")


def cmd_assistant(a: argparse.Namespace) -> None:
    def work(pp):
        hl.add_files(pp, _abs(a.file))
        return hl.assistant_turn(pp, a.question, allow_samples=a.samples, focus=a.focus, build=a.build)

    out = hl.editing_deferred(a.pipeline, work)     # a model turn holds no project lock while it thinks
    if out.get("error"):
        if a.json:
            print(json.dumps(out))
            sys.exit(EXIT_ERROR)
        raise CliError(out["error"])
    text = out.get("text") or ""
    if out.get("answer"):
        text += f"\nBuilt “{out['answer']['title']}” as {out['answer']['id']} (step {out.get('terminal')}, {out.get('status')})."
        if out.get("finding"):
            text += f"\nFinding: {out['finding']}"
    elif out.get("terminal"):
        text += f"\nBuilt and ran step {out['terminal']} ({out.get('status')})."
    _print(a, out, text)


def cmd_answer(a: argparse.Namespace) -> None:
    p = _load(a.pipeline)
    if not a.answer:
        _print(a, [x.to_dict() for x in p.answers], "\n".join(f"{x.id}: {x.title} → {x.terminal}" for x in p.answers) or "No answers yet")
        return
    ans = p.answer(a.answer)
    if ans is None:
        raise CliError(f"No answer called {a.answer!r}. Answers: {[x.id for x in p.answers]}")
    if a.remove:
        from .core import answers
        with _editing(a.pipeline) as p:
            gone = answers.remove(p, a.answer, remove_steps=a.steps)
        _print(a, {"removed": a.answer, "steps_removed": gone}, f"Deleted {a.answer}" + (f" and {len(gone)} steps" if gone else ""))
        return
    if a.set is not None and not a.set:
        raise CliError("Say what to change with --set KEY=VALUE, for example stat=mean or every=1d")
    if a.set or a.choose is not None:
        with _editing(a.pipeline) as p:
            if a.choose is not None:
                out = hl.change_answer(p, a.answer, assumption=a.choose[0], choice=a.choose[1] if len(a.choose) > 1 else 0)
            else:
                kv = _parse_kv(a.set)
                out = None
                for k, v in kv.items():
                    out = hl.change_answer(p, a.answer, k, None if v in ("", None) else v)
        _print(a, out, f"Changed {a.answer}. It's now “{out['title']}”")
        return
    data = ans.to_dict()
    text = [f"{ans.title} [{ans.id}] → {ans.terminal}", f"  question: {json.dumps(ans.spec)}"]
    text += [f"  assumption {i}: {x['text']}" + "".join(f"\n      choice {j}: {c['label']}" for j, c in enumerate(x.get("choices") or []))
             for i, x in enumerate(ans.assumptions)]
    _print(a, data, "\n".join(text))


def cmd_run(a: argparse.Namespace) -> None:
    p = _load(a.pipeline)
    for nid in a.nodes or []:
        _check_node(p, nid)
    ex = Executor(p)
    problems = p.problems()
    if problems and not a.json:
        for pr in problems:
            print(f"! {pr}", file=sys.stderr)
    def on_event(e: dict[str, Any]) -> None:
        if a.json:
            return
        if e["type"] in ("node_finished", "node_failed", "node_cached"):
            st: NodeState = e["state"]
            print(_fmt_state(st, p.nodes[st.node_id].title), flush=True)
        elif e["type"] == "node_started" and sys.stdout.isatty():
            print(f"… {p.nodes[e['node']].title}", end="\r", flush=True)

    t0 = time.perf_counter()
    res = ex.run(targets=a.nodes or None, on_event=on_event, force=a.force)
    failed = [s for s in res.values() if s.status == "failed"]
    if a.json:
        _print(a, hl.run_record(p, ex, res, time.perf_counter() - t0))
    else:
        print(f"{'Done' if not failed else f'{len(failed)} step(s) failed'} in {time.perf_counter() - t0:.1f}s. Cache: {ex.cache_dir}")
        rec = hl.run_record(p, ex, res, 0.0)
        if rec.get("headline"):
            print(f"→ {rec['headline']}")
    if failed:
        sys.exit(EXIT_FAILED)


def cmd_batch(a: argparse.Namespace) -> None:
    """Run the project's target step over many files, writing one output each and a combined table."""
    p = _load(a.pipeline)
    if a.node:
        _check_node(p, a.node)

    def on_event(e: dict[str, Any]) -> None:
        if not a.json and e.get("type") == "batch_file":
            print(f"  {e.get('status', ''):7} {e.get('file', '')}", flush=True)

    rec = hl.run_batch(p, a.files, target=a.node, out_dir=a.out_dir, loader=a.loader,
                       ext=a.format, jobs=a.jobs, force=a.force, combined=not a.no_combined,
                       manifest=a.manifest, on_event=on_event)
    if a.json:
        _print(a, rec)
    else:
        print(f"{rec['count']} file(s) → {rec['out_dir']}  ({'all done' if rec['ok'] else 'some failed'})")
        if rec.get("combined"):
            print(f"combined: {rec['combined']}")
        if rec.get("manifest"):
            print(f"manifest: {rec['manifest']}")
    if not rec["ok"]:
        sys.exit(EXIT_FAILED)


def cmd_status(a: argparse.Namespace) -> None:
    p = _load(a.pipeline)
    ex = Executor(p)
    nodes = [_check_node(p, a.node)] if a.node else list(p.nodes)
    memo: dict[str, str] = {}                   # each step's hash once, not again for every step below it
    states = {n: ex.state(n, memo) for n in nodes}
    if a.json:                                  # one step: its record (as MCP node_status); all: keyed by id
        _print(a, hl.node_record(p, states[a.node]) if a.node else {n: hl.node_record(p, s) for n, s in states.items()})
        return
    for n, s in states.items():
        print(_fmt_state(s, p.nodes[n].title))
        for m in s.messages:
            print(f"      {m}")
        fnd = (s.report or {}).get("finding", {}).get("statement")
        if fnd:
            print(f"→ {fnd}")
        if s.report:
            for k, v in s.report.items():
                if k == "finding":
                    continue
                print(f"      {k}: {v}")


def _frame(a: argparse.Namespace, p: Pipeline, ex: Executor, node: str):
    """A step's output, held while the block reads it (``headless.result_frame``)."""
    _check_node(p, node)
    if ex.state(node).status != "done" and not getattr(a, "run", False):
        raise CliError(f"{node} hasn't been run yet (it's {ex.state(node).status}). Run dancr run {p.path} {node}, or add --run")
    return hl.result_frame(p, ex, node, run=True)


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
            raise CliError(f"Can't work out the columns of {a.node} yet. A step before it is missing or not set up.")
        cols = [{"name": n, "dtype": str(d)} for n, d in sch.items()]
        rows = None
    _print(a, {"node": a.node, "rows": rows, "columns": cols},
           "\n".join(f"{c['name']:30s} {c['dtype']}" for c in cols) + (f"\n{rows:,} rows" if rows is not None else ""))


def cmd_sample(a: argparse.Namespace) -> None:
    if a.rows < 1 or a.offset < 0:
        raise CliError("--rows must be at least 1 and --offset at least 0")
    p = _load(a.pipeline)
    ex = Executor(p)
    with _frame(a, p, ex, a.node) as lf:
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
    with _frame(a, p, ex, a.node) as lf:
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
    _check_node(p, a.node)
    params = hl.chart_params(p.nodes[a.node], a.kind, a.x, a.y.split(",") if a.y else None, a.column, a.title)
    with _frame(a, p, ex, a.node) as lf:
        out = render_chart(lf, params, a.out, width=a.width, height=a.height, columns=p.columns, inputs=p.input_values())
    _print(a, {"path": str(out), "params": params}, f"Wrote {out}")


def cmd_map(a: argparse.Namespace) -> None:
    from .views.render import render_map
    p = _load(a.pipeline)
    ex = Executor(p)
    _check_node(p, a.node)
    node = p.nodes[a.node]
    params = dict(node.params) if node.type == "map" else {}
    if a.lat:
        params["lat"] = a.lat
    if a.lon:
        params["lon"] = a.lon
    if a.color:
        params["color_by"] = a.color
    if a.size_by:
        params["size_by"] = a.size_by
    if a.label:
        params["label"] = a.label
    if a.cell_size:
        params["cell_size"] = a.cell_size
    if a.title is not None:
        params["title"] = a.title
    if a.no_basemap:
        params["basemap"] = False
    with _frame(a, p, ex, a.node) as lf:
        out = render_map(lf, params, a.out, width=a.width, height=a.height, columns=p.columns, inputs=p.input_values())
    _print(a, {"path": str(out), "params": params}, f"Wrote {out}")


def cmd_export(a: argparse.Namespace) -> None:
    p = _load(a.pipeline)
    ex = Executor(p)
    from .core.nodes.outputs import write_table
    out = Path(a.out)
    with _frame(a, p, ex, a.node) as lf:
        write_table(lf, out)
    _print(a, {"path": str(out)}, f"Wrote {out}")


def cmd_clear_cache(a: argparse.Namespace) -> None:
    p = _load(a.pipeline)
    ex = Executor(p)
    before = ex.cache_size()
    ex.clear_cache()
    size = before - ex.cache_size()         # results another DANCR program is using right now are kept
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
        out = None
    try:
        # the window gets no stdin: under `dancr mcp` ours is the protocol pipe
        subprocess.Popen(args, start_new_session=True, stdin=subprocess.DEVNULL,
                         stdout=out or subprocess.DEVNULL, stderr=out or subprocess.DEVNULL)
    finally:
        if out is not None:
            out.close()                             # the window has its own copy of the handle


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


# every capability ships in the one install; doctor says which libraries this build actually carries
_DOCTOR_PACKAGES = ("shapely", "pyproj", "pyogrio", "sqlalchemy", "psycopg", "xarray", "netCDF4", "h5py", "httpx")


def cmd_doctor(a: argparse.Namespace) -> None:
    import importlib
    import importlib.metadata as md
    found: dict[str, str] = {}
    missing: list[str] = []
    for name in _DOCTOR_PACKAGES:
        try:
            importlib.import_module(name)
            try:
                found[name] = md.version(name if name != "netCDF4" else "netCDF4")
            except Exception:  # noqa: BLE001
                found[name] = "?"
        except Exception:  # noqa: BLE001
            missing.append(name)
    out = {"ok": not missing, "present": found, "missing": missing}
    try:
        from .core.nodes.document import mineru_tool, mineru_version, mineru_home
        tool = mineru_tool({})
        out["documents"] = {"present": bool(tool), "command": tool or "", "version": mineru_version(tool) or "",
                            "models": mineru_home() or ""}
    except Exception:  # noqa: BLE001 - a diagnostic must never fail
        out["documents"] = {"present": False}
    if missing:
        _print(a, out, "This build is missing: " + ", ".join(missing) + ". Reinstall DANCR (or run 'uv sync').")
    else:
        doc = out.get("documents") or {}
        extra = (f" Documents: MinerU {doc.get('version') or 'present'}" if doc.get("present")
                 else " Documents: MinerU not found (install it to read PDF/Office files)")
        _print(a, out, "All capabilities present: " + ", ".join(f"{k} {v}" for k, v in found.items()) + "." + extra)


# the formats DANCR reads, and what to do with the ones it does not (mirrors dancr/help/formats.md)
_FORMATS = {
    "reads": [
        "CSV, TSV, TXT, DAT, TAB, LOG (separator and types detected)",
        "Excel .xlsx, .xlsm, .xls, .xlsb, .ods (every sheet; several tables on one sheet)",
        "Parquet .parquet, .pq",
        "GeoJSON .geojson (or a .json that is GeoJSON); GeoPackage .gpkg; shapefile .shp",
        "a folder or glob of the above (Load folder, one table with a file-name column)",
        "a table at a URL (Load from a URL); a database query or table (SQLite built in, servers included)",
        "NetCDF .nc (Load NetCDF: one variable); HDF5 .h5/.hdf5 (Load HDF5: one dataset)",
        "a small table you type (Type in a table)",
        "FASTA/FASTQ .fa/.fasta/.fna/.fq/.fastq (also .gz) (Load sequences: one row per sequence)",
        "VCF .vcf (also .vcf.gz) (Load variants: one row per variant, or genotype rows with dosage)",
        "GFF3/GTF/BED .gff3/.gff/.gtf/.bed (Load features: one row per feature, attributes lifted)",
        "GenBank .gb/.gbk (Load GenBank features: one row per feature)",
        "PLINK .map + .ped (Load markers: one row per marker, a genotype column per sample)",
        "PDF/Office/EPUB/HTML documents (Load document: one row per block, or the tables found, via MinerU)",
    ],
    "not_tables": {
        "JSON Lines .jsonl / plain JSON": "spread the objects into rows and columns, save CSV/Parquet (a GeoJSON file loads as a table)",
        "images / zip / XML / HTML / Markdown / YAML": "export the data part to CSV or Excel first (or, for a scanned page or a document, use Load document via MinerU)",
        "HDF5 .h5 or NetCDF .nc opened with Load file": "use Load HDF5 / Load NetCDF and name the dataset or variable",
    },
}


def cmd_formats(a: argparse.Namespace) -> None:
    text = ("DANCR reads tables:\n  - " + "\n  - ".join(_FORMATS["reads"])
            + "\n\nNot tables (and what to do):\n"
            + "\n".join(f"  - {k}: {v}" for k, v in _FORMATS["not_tables"].items())
            + "\n\nTip: 'dancr doctor' confirms every reader is present in this build.")
    _print(a, _FORMATS, text)


def cmd_inspect(a: argparse.Namespace) -> None:
    from . import headless as hl
    out = hl.inspect_file(a.file, a.rows)
    if a.json:
        _print(a, out)
        return
    lines = [f"{c['name']}  ({c['dtype']})" for c in out["columns"]]
    lines += [str(m) for m in out.get("messages", [])]
    _print(a, out, "\n".join(lines + ["", json.dumps(out["head"], indent=2, default=str)]))


def cmd_mcp(a: argparse.Namespace) -> None:
    from .mcp_server import main as mcp_main
    mcp_main(a.root)


def cmd_synth(a: argparse.Namespace) -> None:
    from .synth import write_dataset
    if a.hours <= 0 or a.rate <= 0:
        raise CliError("--hours and --rate must be greater than 0")
    t = write_dataset(Path(a.out_dir), a.hours, a.rate, a.seed, a.format)
    _print(a, t, f"Wrote probe_A and probe_B ({t['rows_a']:,} / {t['rows_b']:,} rows) to {a.out_dir}")


def cmd_template(a: argparse.Namespace) -> None:
    from .core.samples import TEMPLATES
    if a.list or not a.key:
        _print(a, TEMPLATES, "\n".join(f"{t['key']:10} {t['title']} — {t['blurb']}" for t in TEMPLATES)); return
    if not a.pipeline:
        raise CliError(f"Name the project file to create, for example dancr template {a.key} my_project.json")
    out = Path(a.pipeline)
    from .core.samples import check_template
    check_template(a.key)                           # before anything is written, the lock included
    with hl.project_lock(out):
        if out.exists() and not a.force:
            raise CliError(f"{out} already exists. Use --force to overwrite it")
        pipe, data = hl.build_template(out, a.key, Path(a.data).resolve() if a.data else None)   # --data: relative to here
    _print(a, {"path": str(out), "data": str(data), "nodes": list(pipe.nodes)}, f"Built the {a.key!r} template in {out}, using the data in {data}")


def cmd_inputs(a: argparse.Namespace) -> None:
    if a.remove or a.name:
        with _editing(a.pipeline) as p:
            if a.remove:
                if not any(i.name.lower() == a.remove.lower() for i in p.inputs):
                    raise CliError(f"No input called {a.remove!r}. Inputs: {[i.name for i in p.inputs]}")
                p.remove_input(a.remove)
            else:
                exists = any(i.name.lower() == a.name.lower() for i in p.inputs)
                if a.value is None and not exists:
                    raise CliError(f"Give {a.name!r} a value, for example dancr inputs {a.pipeline} {a.name!r} 12.5")
                p.set_input(a.name, a.value, a.unit or None, a.note or None)
    else:
        p = _load(a.pipeline)                       # only listing them: no lock, so a read-only folder works
    if a.remove:
        _print(a, {"removed": a.remove}, f"Removed input {a.remove!r}"); return
    rows = [{"name": i.name, "value": i.value, "unit": i.unit, "note": i.note} for i in p.inputs]
    _print(a, rows, "\n".join(f"{r['name']:24} {r['value']!s:>12} {r['unit']:8} {r['note']}" for r in rows) or "no inputs")


def cmd_columns(a: argparse.Namespace) -> None:
    if a.name:
        with _editing(a.pipeline) as p:
            p.set_column_meta(a.name, a.label, a.unit)
    else:
        p = _load(a.pipeline)
    _print(a, p.columns, "\n".join(f"{k:24} label={v.get('label', '')!r} unit={v.get('unit', '')!r}" for k, v in p.columns.items()) or "no column names or units set")


# ----------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    kv = ("In KEY=VALUE settings, values that look like JSON (numbers, true/false/null, [lists], {objects}) are read as JSON, "
          "so sheet=1 is the number 1. For the text 1, quote it as sheet='\"1\"'. Anything else is text.")
    ap = _Parser(prog="dancr", description="Build, run and inspect DANCR projects without opening the window.", epilog=kv)
    ap.add_argument("--version", action="version", version=f"dancr {__version__}")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    sub = ap.add_subparsers(dest="cmd", required=True, parser_class=_Parser)

    s = sub.add_parser("nodes", help="list step types and their settings"); s.add_argument("type", nargs="?"); s.add_argument("-v", "--verbose", action="store_true"); s.set_defaults(fn=cmd_nodes)
    s = sub.add_parser("formulas", help="list formula functions"); s.set_defaults(fn=cmd_formulas)
    s = sub.add_parser("new", help="create an empty project file"); s.add_argument("pipeline"); s.add_argument("--name"); s.add_argument("--force", action="store_true"); s.set_defaults(fn=cmd_new)
    s = sub.add_parser("add", help="add a step", epilog=kv); s.add_argument("pipeline"); s.add_argument("type"); s.add_argument("--id"); s.add_argument("--title")
    s.add_argument("--after", help="connect from this step"); s.add_argument("--port", help="which input of the new step to connect (left or right for combine)")
    s.add_argument("--also-after", action="append", help="more steps to connect from (for example the second input of combine)")
    s.add_argument("--set", action="append", metavar="KEY=VALUE", help="a setting (see below)"); s.add_argument("--params", help="JSON object of settings"); s.set_defaults(fn=cmd_add)
    s = sub.add_parser("set", help="change a step's settings", epilog=kv); s.add_argument("pipeline"); s.add_argument("node"); s.add_argument("set", nargs="*", metavar="KEY=VALUE", help="settings to change (see below)"); s.add_argument("--params", help="JSON object of settings"); s.set_defaults(fn=cmd_set)
    s = sub.add_parser("rename", help="rename a step"); s.add_argument("pipeline"); s.add_argument("node"); s.add_argument("title"); s.set_defaults(fn=cmd_rename)
    s = sub.add_parser("connect", help="connect two steps"); s.add_argument("pipeline"); s.add_argument("source"); s.add_argument("target"); s.add_argument("--port"); s.set_defaults(fn=cmd_connect)
    s = sub.add_parser("disconnect", help="remove a connection"); s.add_argument("pipeline"); s.add_argument("source"); s.add_argument("target"); s.add_argument("--port"); s.set_defaults(fn=cmd_disconnect)
    s = sub.add_parser("remove", help="delete a step"); s.add_argument("pipeline"); s.add_argument("node"); s.set_defaults(fn=cmd_remove)
    s = sub.add_parser("show", help="print the project and the status of each step"); s.add_argument("pipeline"); s.set_defaults(fn=cmd_show)
    s = sub.add_parser("understand", help="describe the project's tables: column roles, table shapes, how tables relate"); s.add_argument("pipeline"); s.add_argument("--file", action="append", help="add a data file first (repeatable)"); s.add_argument("--quick", action="store_true", help="from a sample only, without reading every row"); s.set_defaults(fn=cmd_understand)
    s = sub.add_parser("context", aliases=["profile"], help="knowledge-base export: schema, stats, samples and a doc card per dataset")
    s.add_argument("pipeline"); s.add_argument("--file", action="append", help="add a data file first (repeatable)"); s.add_argument("--node", help="only this step's output")
    s.add_argument("--samples", action="store_true", help="include example rows (up to 100)"); s.add_argument("--sample-rows", type=int, default=10)
    s.add_argument("--no-stats", action="store_true", help="skip per-column statistics"); s.add_argument("--quick", action="store_true", help="from a sample only, without reading every row")
    s.add_argument("--jsonl", action="store_true", help="one JSON object per dataset, for a search index"); s.add_argument("--output", help="write to this file instead of printing")
    s.add_argument("--changed", metavar="OLD_CONTEXT", help="only datasets whose content changed since this earlier context file (JSON or JSONL)")
    s.set_defaults(fn=cmd_context)
    s = sub.add_parser("search", help="search the project's text index (built by a 'Build search index' step)")
    s.add_argument("pipeline"); s.add_argument("query"); s.add_argument("--node", help="the index step (needed only if there are several)")
    s.add_argument("--k", type=int, default=5, help="how many passages"); s.add_argument("--min-score", type=float, default=0.0, dest="min_score")
    s.add_argument("--allow-restricted", action="store_true", dest="allow_restricted", help="include confidential/restricted passages (withheld by default)")
    s.set_defaults(fn=cmd_search)
    s = sub.add_parser("eval", help="score questions against the project (the engine, or the Assistant with --model)")
    s.add_argument("pipeline"); s.add_argument("--set", required=True, help="an evaluation set file (JSON {'cases': [...]}, a JSON list, or JSON Lines)")
    s.add_argument("--model", action="store_true", help="score the Assistant (needs a model key, or DANCR_ASSISTANT_FAKE=1)")
    s.set_defaults(fn=cmd_eval)
    s = sub.add_parser("trace", help="the Assistant conversation saved with the project, as an audit log")
    s.add_argument("pipeline"); s.set_defaults(fn=cmd_trace)
    s = sub.add_parser("dataset", help="show or set the project's dataset metadata (creator, license, description…)")
    s.add_argument("pipeline"); s.add_argument("--set", action="append", metavar="KEY=VALUE", help="a field, e.g. license=CC-BY-4.0 or keywords=a,b")
    s.add_argument("--remove", metavar="FIELD", help="remove a field"); s.set_defaults(fn=cmd_dataset)
    s = sub.add_parser("fair", help="export a FAIR descriptor (schema.org, frictionless, manifest or rocrate)")
    s.add_argument("pipeline"); s.add_argument("--format", default="schema.org", choices=["schema.org", "frictionless", "manifest", "rocrate"])
    s.add_argument("--out", help="write to this file instead of printing"); s.add_argument("--samples", action="store_true", help="include example rows")
    s.set_defaults(fn=cmd_fair)
    s = sub.add_parser("package", help="write a self-contained RO-Crate (FAIR descriptors + pipeline + manifest)")
    s.add_argument("pipeline"); s.add_argument("--out", required=True, help="a .zip file, or a folder with --dir")
    s.add_argument("--copy", default="metadata", choices=list(hl.ROCRATE_COPY), help="include the data files, the result files, both, or neither")
    s.add_argument("--dir", action="store_true", help="write a folder instead of a .zip"); s.add_argument("--force", action="store_true", help="replace an existing crate")
    s.set_defaults(fn=cmd_package)
    s = sub.add_parser("verify", help="record an attestation of the project, or check that it still reproduces one")
    s.add_argument("pipeline"); s.add_argument("--record", metavar="FILE", help="write an attestation to this file")
    s.add_argument("--manifest", metavar="FILE", help="verify against this attestation or run manifest")
    s.add_argument("--rerun", action="store_true", help="recompute every step instead of using the cache")
    s.add_argument("--strict-sources", action="store_true", dest="strict_sources", help="treat a changed source file as a mismatch")
    s.add_argument("--no-hash", action="store_true", dest="no_hash", help="skip the output content hashes (a faster, weaker record/check)")
    s.set_defaults(fn=cmd_verify)
    s = sub.add_parser("lineage", help="what produced a step, and what depends on it")
    s.add_argument("pipeline"); s.add_argument("node")
    g = s.add_mutually_exclusive_group(); g.add_argument("--up", action="store_true"); g.add_argument("--down", action="store_true")
    s.set_defaults(fn=cmd_lineage)
    s = sub.add_parser("proof", help="a proof card for a step: chain, hashes, sources and how to re-check it")
    s.add_argument("pipeline"); s.add_argument("node"); s.set_defaults(fn=cmd_proof)
    s = sub.add_parser("watch", help="watch a project and its data files, and rerun when anything changes")
    s.add_argument("pipeline"); s.add_argument("--node", help="only this step and what it needs"); s.add_argument("--interval", type=float, default=2.0, help="seconds between checks")
    s.add_argument("--once", action="store_true", help="run once and exit instead of watching")
    s.add_argument("--batch", action="store_true", help="run a batch over --files instead of the whole project")
    s.add_argument("--files", action="append", help="with --batch, a file/folder/glob (repeatable)"); s.add_argument("--out-dir", dest="out_dir", help="with --batch, where results go"); s.add_argument("--loader", help="with --batch, the source step to swap")
    s.set_defaults(fn=cmd_watch)
    s = sub.add_parser("catalog", help="a catalog of every DANCR project in a folder, for an index or an overview")
    s.add_argument("root", help="the folder to search"); s.add_argument("--pattern", default="*.json"); s.add_argument("--no-recursive", action="store_true", dest="no_recursive")
    s.add_argument("--samples", action="store_true"); s.add_argument("--no-stats", action="store_true", dest="no_stats"); s.add_argument("--fair", help="include a FAIR descriptor per project (schema.org, frictionless, manifest, rocrate)")
    s.add_argument("--jsonl", action="store_true", help="one JSON object per dataset"); s.add_argument("--by-project", action="store_true", dest="by_project", help="with --jsonl, one line per project")
    s.add_argument("--changed", metavar="OLD", help="only what changed since an earlier catalog (JSON or JSONL)"); s.add_argument("--output", help="write to this file"); s.add_argument("--jobs", type=int, default=1)
    s.set_defaults(fn=cmd_catalog)
    s = sub.add_parser("suggest", help="answers DANCR can give for the project's tables, best first"); s.add_argument("pipeline"); s.add_argument("--file", action="append", help="add a data file first (repeatable)"); s.add_argument("--focus", help="only answers about this step's output"); s.add_argument("--build", type=int, metavar="N", help="build suggestion N"); s.set_defaults(fn=cmd_suggest)
    s = sub.add_parser("ask", help="answer a question typed in plain words (for example \"total sales by region\")"); s.add_argument("pipeline"); s.add_argument("question"); s.add_argument("--file", action="append", help="add a data file first (repeatable)"); s.add_argument("--dry-run", action="store_true", help="show how the question is read without building it"); s.set_defaults(fn=cmd_ask)
    s = sub.add_parser("connections", help="how the project's tables relate (links, stacks, alignments); saved with the project")
    s.add_argument("pipeline"); s.add_argument("--recompute", action="store_true", help="work it out again instead of using the saved map")
    s.set_defaults(fn=cmd_connections)
    s = sub.add_parser("assistant", help="ask the Assistant (an AI front end that builds real steps; needs a model key)")
    s.add_argument("pipeline"); s.add_argument("question"); s.add_argument("--file", action="append", help="add a data file first (repeatable)")
    s.add_argument("--build", action="store_true", help="build and run what it proposes")
    s.add_argument("--samples", action="store_true", help="allow sample rows to be sent to the model")
    s.add_argument("--focus", help="focus on this step's output")
    s.set_defaults(fn=cmd_assistant)
    s = sub.add_parser("answer", help="list answers, show one, change it or delete it"); s.add_argument("pipeline"); s.add_argument("answer", nargs="?"); s.add_argument("--set", nargs="*", metavar="KEY=VALUE", help="change a chip, for example stat=mean every=1d"); s.add_argument("--choose", type=int, nargs="+", metavar="N", help="take alternative M (default 0) of assumption N"); s.add_argument("--remove", action="store_true"); s.add_argument("--steps", action="store_true", help="with --remove, also delete the steps that only this answer uses"); s.set_defaults(fn=cmd_answer)
    s = sub.add_parser("run", help="run the project, or only some steps and what they need"); s.add_argument("pipeline"); s.add_argument("nodes", nargs="*"); s.add_argument("--force", action="store_true", help="ignore the cache"); s.set_defaults(fn=cmd_run)
    s = sub.add_parser("batch", help="run one step over many files, writing an output each and a combined table")
    s.add_argument("pipeline"); s.add_argument("--files", action="append", required=True, metavar="PATH", help="a file, folder or glob, relative to the project file's folder (repeatable)")
    s.add_argument("--out-dir", required=True, dest="out_dir", help="where results go (inside the project folder)"); s.add_argument("--node", help="the step to write out (default: the last step)")
    s.add_argument("--loader", help="the step whose file setting to swap (default: the only source step)"); s.add_argument("--format", default="csv", choices=list(hl.BATCH_EXT))
    s.add_argument("--jobs", type=int, default=1, help="files to run at once (default 1)"); s.add_argument("--force", action="store_true", help="ignore the cache")
    s.add_argument("--no-combined", action="store_true", dest="no_combined", help="skip the combined table"); s.add_argument("--manifest", help="write a JSON manifest here")
    s.set_defaults(fn=cmd_batch)
    s = sub.add_parser("status", help="step status, messages and reports"); s.add_argument("pipeline"); s.add_argument("node", nargs="?"); s.set_defaults(fn=cmd_status)
    s = sub.add_parser("schema", help="columns of a step's output"); s.add_argument("pipeline"); s.add_argument("node"); s.set_defaults(fn=cmd_schema)
    s = sub.add_parser("sample", help="print rows of a step's output"); s.add_argument("pipeline"); s.add_argument("node"); s.add_argument("--rows", type=int, default=20); s.add_argument("--offset", type=int, default=0); s.add_argument("--csv", action="store_true"); s.add_argument("--run", action="store_true", help="run first if needed"); s.set_defaults(fn=cmd_sample)
    s = sub.add_parser("stats", help="summary statistics of a step's output"); s.add_argument("pipeline"); s.add_argument("node"); s.add_argument("--columns"); s.add_argument("--run", action="store_true"); s.set_defaults(fn=cmd_stats)
    s = sub.add_parser("chart", help="draw a chart of a step's output to a PNG file"); s.add_argument("pipeline"); s.add_argument("node"); s.add_argument("--out", required=True)
    s.add_argument("--kind", choices=["line", "scatter", "histogram", "bar"]); s.add_argument("--x"); s.add_argument("--y", help="comma-separated columns"); s.add_argument("--column"); s.add_argument("--title")
    s.add_argument("--width", type=int, default=1400); s.add_argument("--height", type=int, default=700); s.add_argument("--run", action="store_true"); s.set_defaults(fn=cmd_chart)
    s = sub.add_parser("map", help="draw a map of a step's output to a PNG file"); s.add_argument("pipeline"); s.add_argument("node"); s.add_argument("--out", required=True)
    s.add_argument("--lat"); s.add_argument("--lon"); s.add_argument("--color"); s.add_argument("--size", dest="size_by"); s.add_argument("--label")
    s.add_argument("--cell", dest="cell_size", help="draw grid squares of this size (e.g. 0.1 or 5km)")
    s.add_argument("--title"); s.add_argument("--no-basemap", action="store_true", dest="no_basemap")
    s.add_argument("--width", type=int, default=1200); s.add_argument("--height", type=int, default=800); s.add_argument("--run", action="store_true"); s.set_defaults(fn=cmd_map)
    s = sub.add_parser("export", help="write a step's output to csv/parquet/xlsx/geojson"); s.add_argument("pipeline"); s.add_argument("node"); s.add_argument("out"); s.add_argument("--run", action="store_true"); s.set_defaults(fn=cmd_export)
    s = sub.add_parser("clear-cache", help="delete cached outputs"); s.add_argument("pipeline"); s.set_defaults(fn=cmd_clear_cache)
    s = sub.add_parser("gui", help="run the window in this process"); s.add_argument("pipeline", nargs="?"); s.set_defaults(fn=cmd_gui)
    s = sub.add_parser("open", help="open the window, optionally on a project"); s.add_argument("pipeline", nargs="?"); s.add_argument("--wait", action="store_true"); s.set_defaults(fn=cmd_open)
    s = sub.add_parser("mcp", help="start the MCP server (stdio) for AI agents"); s.add_argument("--root", help="the folder agents may create projects in (default: the current folder)"); s.set_defaults(fn=cmd_mcp)
    s = sub.add_parser("log", help="print the log file path and its last lines"); s.add_argument("--lines", type=int, default=40); s.set_defaults(fn=cmd_log)
    s = sub.add_parser("doctor", help="check that every capability (geo, database, NetCDF/HDF5, the model client) is present"); s.set_defaults(fn=cmd_doctor)
    s = sub.add_parser("formats", help="list the file formats DANCR reads, and what to do with the ones it can't"); s.set_defaults(fn=cmd_formats)
    s = sub.add_parser("inspect", help="peek at any data file: columns, types and the first rows"); s.add_argument("file"); s.add_argument("--rows", type=int, default=5); s.set_defaults(fn=cmd_inspect)
    s = sub.add_parser("template", help="build a starter project (on sample data unless --data is given)"); s.add_argument("key", nargs="?"); s.add_argument("pipeline", nargs="?"); s.add_argument("--data"); s.add_argument("--list", action="store_true"); s.add_argument("--force", action="store_true"); s.set_defaults(fn=cmd_template)
    s = sub.add_parser("inputs", help="list, set or remove named inputs (values usable in formulas, filters and limits)"); s.add_argument("pipeline"); s.add_argument("name", nargs="?"); s.add_argument("value", nargs="?"); s.add_argument("--unit", default=""); s.add_argument("--note", default=""); s.add_argument("--remove"); s.set_defaults(fn=cmd_inputs)
    s = sub.add_parser("columns", help="list or set display names and units for columns"); s.add_argument("pipeline"); s.add_argument("name", nargs="?"); s.add_argument("--label"); s.add_argument("--unit"); s.set_defaults(fn=cmd_columns)
    s = sub.add_parser("synth", help="generate a synthetic two-probe test dataset"); s.add_argument("out_dir"); s.add_argument("--hours", type=float, default=1.0); s.add_argument("--rate", type=float, default=20.0); s.add_argument("--seed", type=int, default=1); s.add_argument("--format", choices=["csv", "parquet"], default="csv"); s.set_defaults(fn=cmd_synth)
    return ap


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["pdf-helper"]:                      # internal: a report's PDF written in its own process
        from .views.pdf import _main
        sys.exit(_main(argv[1:]))
    ap = build_parser()
    if "--json" in argv[1:]:                            # --json is accepted after the command too
        argv = ["--json", *[x for x in argv if x != "--json"]]
    a = ap.parse_args(argv)
    if a.cmd != "gui":                                  # the window configures its own logging
        import logging
        from .logsetup import configure
        configure(stderr_level=logging.WARNING)         # file as usual, warnings to stderr, never stdout
    try:
        a.fn(a)
    except hl.StepFailed as e:
        _fail(a, str(e), EXIT_FAILED)
    except (CliError, PipelineError, ValueError, OSError, pl.exceptions.PolarsError) as e:
        _fail(a, str(e), EXIT_ERROR)
    except Exception as e:  # noqa: BLE001 - anything else is a DANCR bug: say so and keep the traceback
        import logging
        logging.getLogger("dancr.cli").exception("dancr %s failed", a.cmd)
        _fail(a, f"internal error ({type(e).__name__}: {e}). The details are in the log (see dancr log).", EXIT_BUG)


def _fail(a: argparse.Namespace, msg: str, code: int) -> None:
    if a.json:
        print(json.dumps({"error": msg}))
    else:
        print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)


if __name__ == "__main__":
    main()
