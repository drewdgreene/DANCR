"""MCP server exposing DANCR to AI coding agents (Claude Code, OpenCode, Cursor...).

Run with:  dancr mcp [--root DIR]
Register in an agent, e.g. Claude Code:  claude mcp add dancr -- dancr mcp

Every tool works on a pipeline file path. The GUI watches that file, so a person
can have it open and see the pipeline change and results appear while the agent works.

Where the server may write: pipeline files only inside its root folder; every other file (exports, reports,
workbooks, chart images) only inside the folder of the pipeline file and never over a data file the pipeline
reads, checked when a step is added or changed and again before it runs. Edits to one pipeline file are made one
at a time, also against the window and the command line (the file's lock, see headless.project_lock).
"""
from __future__ import annotations

import functools
import json
import os
import logging
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
from .core.model import FORMAT_VERSION
from .core.executor import Executor
from .core.registry import in_dancr_folder

log = logging.getLogger("dancr.mcp")

mcp = MCPServer("dancr", version=__version__, instructions=(
    "DANCR builds and runs data pipelines (node graphs) over large CSV/Excel/Parquet files. "
    "Fastest route to an answer: create_pipeline -> suggest_answers(files=[...]) to see what DANCR can answer on its own, "
    "or ask(question, files=[...]) with a plain question such as 'total sales by region' or 'average pressure per hour'; "
    "both build ordinary steps plus an Answer, deterministically, and say what they assumed. "
    "Hand-built workflow: create_pipeline -> add_node (load_file first, with path) -> add more nodes with after= to chain them -> "
    "run_pipeline -> inspect with get_schema / get_sample / get_stats / render_chart. "
    "Call list_node_types once to learn node types and their settings. Paths inside a pipeline are relative to the pipeline file. "
    "Use open_in_gui so the person can watch; the GUI reloads the file whenever it changes. "
    "Pipeline files must be inside the server's root folder, and every file a pipeline or tool writes "
    "(export, workbook and report steps, render_chart out_png, export_node) must be inside the pipeline file's folder "
    "and must not be a data file the pipeline reads or inside a .dancr folder. "
    "Reading data files is not restricted; relative data paths are taken from the root folder."))

ROOT = Path.cwd().resolve()        # where pipelines may be created; `dancr mcp --root DIR` sets it
ROOT_REFUSED = False               # started in or above the home folder, or at the top of a drive, with no --root: no pipelines


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


def _in_root(path: str | Path, change: bool = True) -> Path:
    """A pipeline file path (relative to the root), refused outside the server's root folder. A server started
    in or above the home folder, or at the top of a drive, reads pipelines there but does not create or change them."""
    if ROOT_REFUSED and change:
        raise ToolError(f"The DANCR MCP server was started in {ROOT}, so it won't create or change projects "
                        "anywhere under it. Start it in the project's folder, or with --root <folder>.")
    p = Path(path).expanduser()
    p = (p if p.is_absolute() else ROOT / p).resolve()
    if not p.is_relative_to(ROOT):
        raise ToolError(f"Projects can only be created inside {ROOT} (the folder the DANCR MCP server was started in), not {p}")
    if change and in_dancr_folder(p, ROOT):
        raise ToolError(f"Won't create or change a project inside DANCR's own .dancr folder: {p}")
    return p


def _from_root(path: str | Path) -> Path:
    """A file to read, relative to the root folder (reading is not confined)."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else ROOT / p


def _load(path: str) -> Pipeline:
    p = _in_root(path, change=False)
    if not p.exists():
        raise ValueError(f"No pipeline at {p}. Call create_pipeline first.")
    return Pipeline.load(p)


@contextmanager
def _editing(path: str) -> Iterator[Pipeline]:
    """Load, change and save one pipeline file holding its lock, so no other tool call (agents call tools in
    parallel), command or window saves over the change."""
    target = _in_root(path)
    if not target.exists():
        raise ValueError(f"No pipeline at {target}. Call create_pipeline first.")
    with hl.editing(target) as p:
        yield p


def _editing_deferred(path: str, compute: Any) -> Any:
    """Run a long operation against the project (a model turn, a build over the whole data) with the file's lock
    released, saving briefly at the end. Keeps an agent's slow call from blocking the window and other tools."""
    target = _in_root(path)
    if not target.exists():
        raise ValueError(f"No pipeline at {target}. Call create_pipeline first.")
    return hl.editing_deferred(target, compute)


def _folder(p: Pipeline) -> Path:
    return p.path.resolve().parent


def _executor(p: Pipeline) -> Executor:
    """Steps run from MCP may only write inside the pipeline file's folder (the executor refuses the rest)."""
    return Executor(p, output_root=_folder(p))


def _unsafe_steps(p: Pipeline, targets: list[str] | None) -> dict[str, str]:
    """Steps among those a run would do that would save outside the pipeline's folder or over a file it reads,
    with why. Checked before running, since the file may have been changed by something other than these tools."""
    folder = _folder(p)
    return {nid: f"{p.nodes[nid].title}: {bad[0]}" for nid in p.topological_order(targets or None)
            if (bad := hl.unsafe_outputs(p, nid, folder))}


def _check_outputs(p: Pipeline, node_id: str) -> None:
    """Refuse a step that would save a file outside the pipeline's folder or over a data file the pipeline
    reads, when it is added or changed: the window runs steps too (open_in_gui), and it must never be handed
    one that writes elsewhere."""
    bad = hl.unsafe_outputs(p, node_id, _folder(p))
    if bad:
        raise ToolError(f"{p.nodes[node_id].title}: {bad[0]}")


def _inside_project(p: Pipeline, out: str) -> Path:
    """Resolve a write target (relative paths are taken from the pipeline file's folder) and refuse anything
    outside that folder or over a data file the pipeline reads."""
    folder = _folder(p)
    target = Path(out).expanduser()
    target = (target if target.is_absolute() else folder / target).resolve()
    why = hl.unsafe_write(p, target, folder)
    if why:
        raise ToolError(why)
    return target


def _dump(data: Any) -> str:
    from .core.dtypes import json_safe
    return json.dumps(json_safe(data), indent=1, default=str, allow_nan=False)


def _record(p: Pipeline, ex: Executor, nid: str, memo: dict[str, str] | None = None) -> dict[str, Any]:
    return hl.node_record(p, ex.state(hl.require_node(p, nid), memo))


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
    ideally next to the data files. overwrite replaces an existing DANCR pipeline (never any other file)."""
    p = _in_root(path)
    if p.suffix.lower() != ".json":
        raise ToolError(f"A project file name must end in .json, not {p.name}")
    with hl.project_lock(p):
        if p.exists():
            if not _is_pipeline(p):
                raise ToolError(f"{p} already exists and is not a DANCR project. Choose another file name.")
            if not overwrite:
                return _dump({"ok": True, "path": str(p), "note": "already exists; loaded as-is"})
        Pipeline(name or p.stem).save(p)
    return _dump({"ok": True, "path": str(p)})


def _is_pipeline(p: Path) -> bool:
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    # the same strict test find_pipelines uses: a file that merely mentions "dancr" is not a project, and must
    # never be clobbered by overwrite=True
    return isinstance(data, dict) and data.get("dancr") == FORMAT_VERSION


@mcp.tool()
@friendly
def build_template(path: str, template: str, data_file: str | None = None) -> str:
    """Create a starter project from a template: compare | limits | fit | report. Uses a generated sample
    file (two values over time) unless data_file is given (relative paths are taken from the pipeline's folder).
    The pipeline file must not exist yet."""
    from .core.samples import TEMPLATES, check_template
    out = _in_root(path)
    if out.suffix.lower() != ".json":
        raise ToolError(f"A project file name must end in .json, not {out.name}")
    check_template(template)                        # before anything is written, the lock included
    with hl.project_lock(out):
        if out.exists():
            raise ValueError(f"{out} already exists")
        pipe, data = hl.build_template(out, template, data_file)
    return _dump({"ok": True, "path": str(out), "data": str(data), "nodes": list(pipe.nodes), "templates": TEMPLATES})


@mcp.tool()
@friendly
def describe_pipeline(path: str) -> str:
    """Nodes, connections, settings, statuses and configuration problems of a pipeline."""
    from .core.secrets import redact_params
    p = _load(path)
    ex = Executor(p)
    memo: dict[str, str] = {}                           # each step's hash once, not again for every step below it
    return _dump({
        "name": p.name, "path": str(p.path),
        "nodes": [{**n.to_dict(), "params": redact_params(registry.get(n.type), n.params),
                   "inputs": p.inputs_of(n.id), "state": _record(p, ex, n.id, memo)} for n in p.nodes.values()],
        "edges": [e.to_dict() for e in p.edges],
        "inputs": [{"name": i.name, "value": i.value, "unit": i.unit, "note": i.note} for i in p.inputs],
        "columns": p.columns,
        "answers": [a.to_dict() for a in p.answers],
        "problems": p.problems(),
    })


def _data_files(files: list[str] | None) -> list[str]:
    return [str(_from_root(f).resolve()) for f in files or []]


def _with_files(path: str, files: list[str] | None) -> Pipeline:
    """The pipeline with a load step for each data file named, added under its lock (only when there are files).
    The slow reading that follows is done without the lock, so the window and other tools never wait on it."""
    if files:
        with _editing(path) as p:
            hl.add_files(p, _data_files(files))
        return p
    return _load(path)


@mcp.tool()
@friendly
def understand_data(path: str, files: list[str] | None = None) -> str:
    """How DANCR reads the project's tables: each column's role (time, id, category, measure...), each table's shape
    (series, lookup, events), and how tables relate (links on a key with how many match, stacks, time alignments).
    `files` adds load steps for data files first (relative to the server's root)."""
    p = _with_files(path, files)
    return _dump(hl.data_model(p).to_dict())


@mcp.tool()
@friendly
def connections(path: str, recompute: bool = False) -> str:
    """How the project's tables relate: links (each with its match percentage and cardinality), stacks and time
    alignments. The map is saved with the project; recompute=true works it out again."""
    out = _editing_deferred(path, lambda pp: hl.connection_map(pp, recompute=recompute))
    return _dump(out)


@mcp.tool()
@friendly
def profile(path: str, files: list[str] | None = None, node: str | None = None, samples: bool = False,
            sample_rows: int = 10, stats: bool = True, deep: bool = True, changed: str | None = None) -> str:
    """A knowledge-base document for the project's datasets: each table's schema (columns with roles, types,
    units and ranges), how tables relate, per-column statistics, and a prose doc card per table. With samples=true,
    up to `sample_rows` example rows (max 100) are included too. One call to index DANCR's metadata for a RAG;
    DANCR still computes exact figures with run_pipeline/get_stats. `files` adds data files first; `node` limits it
    to one step. With stats or samples, a table that has not run yet is computed first. Each document carries a
    `content_hash`; pass `changed` = an earlier context file (JSON or JSONL) to return only what changed since."""
    p = _with_files(path, files)
    ctx = hl.build_context(p, _executor(p), nodes=[node] if node else None, deep=deep,
                           stats=stats, samples=samples, sample_rows=sample_rows)
    if changed:
        cp = _from_root(changed)
        text = cp.read_text(encoding="utf-8") if cp.is_file() else changed
        ctx = hl.context_changes(ctx, text)
    return _dump(ctx)


@mcp.tool()
@friendly
def get_dataset_meta(path: str) -> str:
    """The project's dataset-level metadata (creator, license, description, citation…), the header of its FAIR record."""
    p = _load(path)
    return _dump(p.dataset_meta())


@mcp.tool()
@friendly
def set_dataset_meta(path: str, fields: dict[str, Any]) -> str:
    """Set dataset-level metadata on the project, for a FAIR record: fields is an object such as
    {"license": "CC-BY-4.0", "creator": {"name": "Jane Doe", "email": "jane@example.org"}, "description": "…",
    "keywords": ["maize", "yield"]}. A field set to null removes it. Known fields: title, description, creator,
    contact, publisher, license, keywords, version, created, identifier, citation, language."""
    with _editing(path) as p:
        meta = p.set_dataset_meta(**fields)
    return _dump({"ok": True, "dataset": meta})


@mcp.tool()
@friendly
def export_fair(path: str, format: str = "schema.org", out_path: str | None = None, samples: bool = False) -> str:
    """A FAIR descriptor for the project's datasets: `schema.org` (JSON-LD Dataset), `frictionless` (Data Package),
    `manifest` (what produced the results: engine, library and input versions, per-step hashes) or `rocrate`.
    With out_path (inside the pipeline file's folder) it is written to a file; otherwise the JSON is returned."""
    from .core.fair import FORMATS
    key = format.strip().lower().replace("_", ".")
    if key not in FORMATS and key not in ("provenance", "run", "data-package", "datapackage", "jsonld", "schemaorg"):
        raise ToolError(f"Unknown format {format!r}. Choose one of: {', '.join(FORMATS)}")
    p = _load(path)
    out = _inside_project(p, out_path) if out_path else None
    doc = hl.export_fair(p, _executor(p), fmt=format, samples=samples, out=out)
    return _dump(doc)


@mcp.tool()
@friendly
def catalog(root: str = ".", pattern: str = "*.json", recursive: bool = True, samples: bool = False,
            stats: bool = True, fair: str | None = None, changed: str | None = None, jobs: int = 1) -> str:
    """A catalog of every DANCR project under `root` (relative to the server's root folder): each project's
    datasets — schema, relations, a prose doc card, and each dataset's `content_hash`. `fair` adds a descriptor
    per project (schema.org, frictionless, manifest, rocrate). `changed` limits it to what moved since an
    earlier catalog (JSON or JSONL). Everything is read-only; `stats`/`samples` compute a table if needed."""
    base = _from_root(root)
    cat = hl.build_catalog(base, pattern=pattern, recursive=recursive, stats=stats, samples=samples,
                           fair_format=fair, jobs=int(jobs))
    if changed:
        cp = _from_root(changed)
        text = cp.read_text(encoding="utf-8") if cp.is_file() else changed
        cat = hl.catalog_changes(cat, text)
    return _dump(cat)


@mcp.tool()
@friendly
def package_project(path: str, out_path: str, copy: str = "metadata", zip: bool = True, overwrite: bool = False) -> str:
    """Write a self-contained RO-Crate: the FAIR descriptors (schema.org, Frictionless, run manifest), the
    pipeline file and a knowledge-base context file, as a `.zip` or a folder inside the pipeline's folder.
    `copy` controls extra files: `metadata` (none), `data` (the source files), `results` (files the project
    wrote) or `all`."""
    p = _load(path)
    out = _inside_project(p, out_path)
    rec = hl.package_rocrate(p, _executor(p), out=out, copy=copy, zip=bool(zip), overwrite=overwrite)
    return _dump(rec)


@mcp.tool()
@friendly
def record_attestation(path: str, out_path: str | None = None) -> str:
    """Record an attestation of the project: a self-contained, checkable record of exactly what produced its
    results (engine and library versions, the code fingerprint, each source file's content sample, and every
    step's plan hash, output content hash, row count and reported numbers). With out_path (inside the pipeline
    file's folder) it is written to a file; otherwise the attestation is returned. Re-check it with verify_pipeline."""
    p = _load(path)
    att = hl.build_attestation(p, _executor(p))
    out = _inside_project(p, out_path) if out_path else None
    if out is not None:
        hl.write_text_atomic(out, hl.dump_attestation(att))
    return _dump({"ok": True, "path": str(out) if out else None, "attestation_hash": att["attestation_hash"],
                  "nodes": len(att["nodes"]), "answers": len(att.get("answers", [])), "attestation": att})


@mcp.tool()
@friendly
def verify_pipeline(path: str, reference_path: str, rerun: bool = False, strict_sources: bool = False) -> str:
    """Verify the project still reproduces an attestation (or run manifest): re-run and compare the engine
    fingerprint, source content samples, every step's plan hash and output content hash, and the numbers in each
    report. reference_path is relative to the server's root folder (or absolute). rerun recomputes every step
    instead of using the cache (the strong claim); strict_sources treats a changed source file as a mismatch.
    Returns {verdict, ok, checks, mismatches, notices, summary}."""
    p = _load(path)
    ref = _from_root(reference_path)
    reference: Any = ref.read_text(encoding="utf-8") if ref.is_file() else reference_path
    res = hl.verify_pipeline(p, reference, mode="rerun" if rerun else "stored", strict_sources=strict_sources)
    return _dump(res)


@mcp.tool()
@friendly
def lineage(path: str, node_id: str, direction: str = "both") -> str:
    """What produced a step and/or what depends on it: the upstream chain (direction='up'), everything
    downstream (direction='down': later steps, Answers, files written and Assistant turns), or both."""
    p = _load(path)
    return _dump(hl.build_lineage(p, node_id, direction=direction, executor=_executor(p)))


@mcp.tool()
@friendly
def get_proof(path: str, node_id: str) -> str:
    """A proof card for one step: its chain of steps with plan hashes, the source files with content samples,
    its output content hash and finding, and the command that re-checks it. Show this next to a result."""
    p = _load(path)
    return _dump(hl.proof_card(p, _executor(p), node_id))


@mcp.tool()
@friendly
def search_knowledge(path: str, query: str, node: str | None = None, k: int = 5, min_score: float = 0.0,
                     allow_restricted: bool = False, retriever: str = "lexical") -> str:
    """Search the project's own text index (built by a 'Build search index' step) and return the closest
    passages, each with the source columns it came from and a provenance block (which retrievers and embedder
    were used). Offline and deterministic; `node` names the index step when there is more than one.
    `retriever` is 'lexical' (the offline embedding, default), 'bm25' (keyword) or 'hybrid'. When the index has
    a 'sensitivity' column, confidential/restricted passages are withheld unless allow_restricted is true (the
    withheld count is reported)."""
    p = _load(path)
    return _dump(hl.search_knowledge(p, query, node=node, k=int(k), min_score=float(min_score),
                                     allow_restricted=bool(allow_restricted), retriever=retriever,
                                     executor=_executor(p)))


@mcp.tool()
@friendly
def get_trace(path: str) -> str:
    """The Assistant conversation saved with the project, as an audit log: each turn's question, reply, the step
    or answer it built, its flags and any figure no tool backed. Read-only."""
    return _dump(hl.assistant_trace(_load(path)))


@mcp.tool()
@friendly
def run_eval(path: str, cases: list[dict[str, Any]] | None = None, cases_path: str | None = None,
             model: bool = False) -> str:
    """Score a set of questions against the project. Give `cases` (a list of {"id","question","expect_recipe" |
    "expect_text" | "expect_no_text" | "expect_refuse"}) or `cases_path` (a file inside the server's root folder,
    or JSON/JSONL text). Without `model` it scores the deterministic engine; with `model=true` it scores the
    Assistant (no unverified figures, injection payloads not echoed, refusals honoured). Returns per-case results,
    a summary and `ok`."""
    p = _load(path)
    if cases_path:
        cp = _from_root(cases_path)
        src: Any = cp.read_text(encoding="utf-8") if cp.is_file() else cases_path
    elif cases is not None:
        src = cases
    else:
        raise ToolError("Give cases (a list) or cases_path")
    return _dump(hl.run_eval(p, hl.load_eval_set(src), model=bool(model)))


@mcp.tool()
@friendly
def suggest_answers(path: str, files: list[str] | None = None, focus: str | None = None, build: int | None = None) -> str:
    """Answers DANCR can give on its own for the project's tables, best first ({index, title, recipe, why, spec}).
    `files` adds data files first; `focus` limits them to one step's output; `build` = an index builds that answer
    (its steps and an Answer) and returns it. Run run_pipeline afterwards to compute it."""
    p = _with_files(path, files)
    sugs = hl.suggestions(p, focus)
    if build is None:
        return _dump({"suggestions": sugs})
    if not 0 <= build < len(sugs):
        raise ToolError(f"There are {len(sugs)} suggestions (numbered from 0)")

    def work(pp: Pipeline) -> dict[str, Any]:
        out = hl.build_answer(pp, sugs[build]["spec"])
        _check_outputs(pp, out["terminal"])
        return out

    out = _editing_deferred(path, work)      # building an answer reads the data: do it off the lock
    return _dump({"answer": out})


@mcp.tool()
@friendly
def ask(path: str, question: str, files: list[str] | None = None, dry_run: bool = False) -> str:
    """Answer a question typed in plain words, using the project's own column, table and value names:
    'total qty by region', 'average pressure per hour for MJ03F', 'top 10 customers by sales', 'compare A and B',
    'orders where qty above 2', 'gaps in probe_A'. Builds the steps and an Answer (unless dry_run) and reports how
    the question was read (chips), what was assumed, and any word it did not know with 'did you mean' hints."""
    def work(pp: Pipeline) -> dict[str, Any]:
        hl.add_files(pp, _data_files(files))          # kept even when the question is not understood, as on the CLI
        return hl.ask_question(pp, question, build=not dry_run)

    out = _editing_deferred(path, work)               # reading the data to answer runs off the lock
    if not out["question"]["ok"]:
        q = out["question"]
        raise ToolError(q["message"] + (f" (unknown: {', '.join(q['unknown'])})" if q["unknown"] else ""))
    return _dump(out)


@mcp.tool()
@friendly
def assistant(path: str, question: str, files: list[str] | None = None, build: bool = False,
              allow_samples: bool = False, focus: str | None = None) -> str:
    """Talk to the Assistant: an AI front end to the same engine. It proposes steps and answers; the engine does
    every number, so no figure is ever made up. Needs a model key (DANCR_ASSISTANT_API_KEY, with optional
    DANCR_ASSISTANT_BASE_URL / DANCR_ASSISTANT_MODEL; DANCR_ASSISTANT_FAKE=1 runs a scripted fake with no key).
    `files` adds data files first; `focus` limits it to one step's output. With build=true it applies the proposal
    (its steps and an Answer) and runs it. Returns {kind, text, proposal, flags, usage, tool_calls}."""
    def work(pp: Pipeline) -> dict[str, Any]:
        hl.add_files(pp, _data_files(files))
        return hl.assistant_turn(pp, question, allow_samples=allow_samples, focus=focus, build=build,
                                 output_root=_folder(pp), check_output=_check_outputs)

    # a model turn can run for minutes; the project file is locked only for the read and the final save, so the
    # window can still autosave and other tools still work while the agent is thinking
    out = _editing_deferred(path, work)
    if out.get("error"):
        raise ToolError(out["error"])
    return _dump(out)


@mcp.tool()
@friendly
def apply_edits(path: str, edits: list[dict[str, Any]]) -> str:
    """Apply a batch of project edits and save, as one change: each edit is
    {"op": "rename", "node": id, "title": "…"}, {"op": "set_params", "node": id, "params": {…}},
    {"op": "set_input", "name": "…", "value": …, "unit": "…", "note": "…"}, or
    {"op": "column_label", "column": "…", "label": "…", "unit": "…"}. Returns the summaries applied."""
    from .core.assistant.edits import apply_edits as _apply_edits
    with _editing(path) as p:
        applied = _apply_edits(p, edits)
        # a repointed sink (an edit that changes a step's output path) must obey the same confinement as add/set
        for nid in {str(e.get("node")) for e in edits if isinstance(e, dict) and e.get("node")}:
            if nid in p.nodes:
                _check_outputs(p, nid)
    return _dump({"ok": True, "applied": applied})


@mcp.tool()
@friendly
def change_answer(path: str, answer_id: str, key: str | None = None, value: Any = None,
                  assumption: int | None = None, choice: int = 0) -> str:
    """Change an Answer and rebuild its steps in place (steps edited by hand keep their settings). Either set one chip
    (`key`/`value`, e.g. key='stat' value='mean', key='every' value='1d', key='by' value=['node','column'];
    value null removes it) or take alternative `choice` of assumption number `assumption`."""
    with _editing(path) as p:
        out = hl.change_answer(p, answer_id, key, value, assumption, choice)
        _check_outputs(p, out["terminal"])
    return _dump({"answer": out})


@mcp.tool()
@friendly
def remove_answer(path: str, answer_id: str, remove_steps: bool = False) -> str:
    """Delete an Answer; with remove_steps also the steps only it uses (steps other answers need are kept)."""
    from .core import answers
    with _editing(path) as p:
        if p.answer(answer_id) is None:
            raise ToolError(f"No answer called {answer_id!r}")
        gone = answers.remove(p, answer_id, remove_steps)
    return _dump({"ok": True, "steps_removed": gone})


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
    return _dump({"ok": True, "node": hl.node_public(p, node.id), "inputs": p.inputs_of(node.id),
                  "problems": [x for x in p.problems() if x.startswith(node.title + ":")]})


@mcp.tool()
@friendly
def set_params(path: str, node_id: str, params: dict[str, Any]) -> str:
    """Change settings on a node. Only the keys given are changed. Values are validated."""
    with _editing(path) as p:
        p.set_params(hl.require_node(p, node_id), **params)
        _check_outputs(p, node_id)
    return _dump({"ok": True, "node": hl.node_public(p, node_id)})


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
    from .core.executor import NodeState
    p = _load(path)
    targets = list(node_ids) if node_ids else None      # an empty list is every step, for the check as for the run
    for nid in targets or []:
        hl.require_node(p, nid)
    ex = _executor(p)
    t0 = time.perf_counter()
    unsafe = _unsafe_steps(p, targets)
    # those steps, and what needs them, are not run: they fail with the reason
    wanted = [n for n in (targets or list(p.nodes))
              if n not in unsafe and not set(unsafe) & p.upstream_closure(n)]
    res = ex.run(targets=wanted, force=force) if wanted else {}
    for nid, why in unsafe.items():
        res[nid] = NodeState(nid, status="failed", error=why)
    return _dump(hl.run_record(p, ex, res, time.perf_counter() - t0))


@mcp.tool()
@friendly
def run_batch(path: str, files: list[str], out_dir: str, node_id: str | None = None, loader: str | None = None,
              format: str = "csv", jobs: int = 1, force: bool = False, combined: bool = True,
              manifest: str | None = None) -> str:
    """Run one step of the pipeline over many files (a folder or globs in `files`), writing one output per file
    to `out_dir` plus a combined table with a `source_file` column. The project's cache is shared, so unchanged
    files are served from cache. `node_id` defaults to the last step; `loader` is the source step whose file
    setting is swapped (default: the only source). All files are confined to the pipeline file's folder."""
    p = _load(path)
    out = _inside_project(p, out_dir)
    man = _inside_project(p, manifest) if manifest else None
    rec = hl.run_batch(p, list(files), target=node_id, out_dir=out, loader=loader,
                       ext=format, jobs=int(jobs), force=force, combined=combined, manifest=man)
    return _dump(rec)


@mcp.tool()
@friendly
def node_status(path: str, node_id: str) -> str:
    """Status, row count, columns, messages and report (e.g. fit coefficients, gap statistics) of one node."""
    p = _load(path)
    return _dump(_record(p, Executor(p), node_id))


@contextmanager
def _frame(path: str, node_id: str, run: bool) -> Iterator[tuple[Pipeline, pl.LazyFrame]]:
    """The pipeline and a step's output, held while the block reads it (``headless.result_frame``)."""
    p = _load(path)
    hl.require_node(p, node_id)
    unsafe = _unsafe_steps(p, [node_id])
    if unsafe and run:
        raise ToolError(next(iter(unsafe.values())))
    with hl.result_frame(p, _executor(p), node_id, run) as lf:
        yield p, lf


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
    with _frame(path, node_id, run) as (_, lf):
        return hl.select_columns(lf, columns).slice(int(offset), int(rows)).collect(engine="streaming").write_json()


@mcp.tool()
@friendly
def get_stats(path: str, node_id: str, columns: list[str] | None = None, run: bool = True) -> str:
    """Summary statistics (count, missing, mean, std, min, quartiles, max) for the columns of a node's output."""
    from .views.stats import column_summary
    with _frame(path, node_id, run) as (_, lf):
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
    import tempfile
    with _frame(path, node_id, run) as (p, lf):
        params = hl.chart_params(p.nodes[node_id], kind, x, y, column, title)
        if out_png:
            out = _inside_project(p, out_png)
            _render(lf, params, out, width=width, height=height, columns=p.columns, inputs=p.input_values())
            return Image(data=out.read_bytes(), format="png")
        # a file of its own, read back before returning: calls run side by side, and a shared file could be
        # overwritten by another call (or read half-written) before the image reached the client
        fd, tmp = tempfile.mkstemp(suffix=".png", prefix="dancr-chart-")
        os.close(fd)
        try:
            _render(lf, params, Path(tmp), width=width, height=height, columns=p.columns, inputs=p.input_values())
            return Image(data=Path(tmp).read_bytes(), format="png")
        finally:
            Path(tmp).unlink(missing_ok=True)


@mcp.tool()
@friendly
def render_map(path: str, node_id: str, out_png: str | None = None, lat: str | None = None, lon: str | None = None,
               color_by: str | None = None, size_by: str | None = None, label: str | None = None,
               cell_size: str | None = None, title: str | None = None, basemap: bool = True,
               width: int = 1200, height: int = 800, run: bool = True) -> Image:
    """Render a map of a node's output to PNG and return the image. If the node is a map node its settings are
    used and any argument given here overrides them; otherwise give lat and lon (and optionally color_by, size_by,
    cell_size). Country outlines come from the app's offline basemap. out_png (optional) must be inside the
    pipeline file's folder; otherwise the PNG goes to the cache."""
    from .views.render import render_map as _render
    import tempfile
    with _frame(path, node_id, run) as (p, lf):
        params = dict(p.nodes[node_id].params) if p.nodes[node_id].type == "map" else {}
        for k, v in (("lat", lat), ("lon", lon), ("color_by", color_by), ("size_by", size_by),
                     ("label", label), ("cell_size", cell_size), ("title", title)):
            if v is not None:
                params[k] = v
        if not basemap:
            params["basemap"] = False
        if out_png:
            out = _inside_project(p, out_png)
            _render(lf, params, out, width=width, height=height, columns=p.columns, inputs=p.input_values())
            return Image(data=out.read_bytes(), format="png")
        fd, tmp = tempfile.mkstemp(suffix=".png", prefix="dancr-map-")
        os.close(fd)
        try:
            _render(lf, params, Path(tmp), width=width, height=height, columns=p.columns, inputs=p.input_values())
            return Image(data=Path(tmp).read_bytes(), format="png")
        finally:
            Path(tmp).unlink(missing_ok=True)


@mcp.tool()
@friendly
def export_node(path: str, node_id: str, out_path: str, run: bool = True) -> str:
    """Write a node's full output to a .csv, .tsv, .parquet, .xlsx or .geojson file inside the pipeline file's folder."""
    from .core.nodes.outputs import write_table
    with _frame(path, node_id, run) as (p, lf):
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
    """Peek at a data file before building a pipeline: detected columns, types and the first rows. Reads any
    format DANCR can open, including the bio formats (FASTA/FASTQ, VCF, GFF/GTF/BED, GenBank, PLINK) and
    documents (PDF/Office/EPUB/HTML) through MinerU. A relative path is taken from the server's root folder."""
    return _dump(hl.inspect_file(_from_root(file_path), rows))


@mcp.tool()
@friendly
def read_document(file_path: str, what: str = "blocks", tier: str | None = None, pages: str | None = None,
                  allow_remote: bool = False, rows: int = 200) -> str:
    """Read a PDF/Office/EPUB/HTML document as a table through MinerU: one row per content block
    (doc, page, block, type, text, locator) with what='blocks', or a catalog of the tables found with
    what='tables' (each written to a CSV next to the document). A relative path is taken from the server's root
    folder. Needs MinerU installed, or a folder MinerU already produced, or a configured endpoint."""
    if what == "tables" and ROOT_REFUSED:
        # no --root was given (started in or above home): writing table CSVs would scatter files through the
        # person's home folder, so refuse rather than do that
        raise ValueError("Reading tables writes CSV files, so the server needs a folder to write in: "
                         "start it with `dancr mcp --root DIR`. (what='blocks' writes nothing and still works.)")
    # extracted table CSVs are confined to the server's root folder, never written next to an arbitrary document
    return _dump(hl.read_document(_from_root(file_path), what=what, tier=tier, pages=pages,
                                  allow_remote=allow_remote, rows=rows, output_root=ROOT))


# ----------------------------------------------------------------- cross-project graph
def _repo_root(root: str | None) -> Path:
    """The repository folder a graph tool works on: inside the server's root, never a .dancr folder. `root` is
    optional and defaults to the server's root."""
    if ROOT_REFUSED:
        raise ToolError(f"The DANCR MCP server was started in {ROOT}, so it won't build or change derived state "
                        "anywhere under it. Start it in the repository folder, or with --root <folder>.")
    p = Path(root).expanduser() if root else ROOT
    p = (p if p.is_absolute() else ROOT / p).resolve()
    if not p.is_relative_to(ROOT):
        raise ToolError(f"Only repositories inside {ROOT} are allowed, not {p}")
    if in_dancr_folder(p, ROOT):
        raise ToolError(f"Won't use DANCR's own .dancr folder as a repository: {p}")
    if not p.is_dir():
        raise ToolError(f"{p} is not a folder")
    return p


@mcp.tool()
@friendly
def graph_build(root: str | None = None) -> str:
    """Build or refresh the cross-project entity graph for a repository folder (default: the server's root):
    one vertex per project, dataset, column and source, and one edge per relation the engine found (link,
    stack, align, nearest-place, containment), each carrying its evidence and confidence. Incremental: projects
    whose file and source files have not changed are carried over. Writes <root>/.dancr/graph/graph.db."""
    return _dump(hl.build_graph(_repo_root(root)))


@mcp.tool()
@friendly
def graph_query(root: str | None = None, kind: str | None = None, project: str | None = None,
                text: str | None = None, allow_restricted: bool = False, limit: int | None = None) -> str:
    """Datasets and edges in the repository graph matching a filter. `kind` limits edges by relation; `project`
    limits to one project file; `text` matches a dataset id, title, node, column name or source file name.
    Confidential/restricted datasets (and the edges touching them) are withheld unless allow_restricted."""
    return _dump(hl.graph_query(_repo_root(root), kind=kind, project=project, text=text,
                                allow_restricted=bool(allow_restricted), limit=limit))


@mcp.tool()
@friendly
def graph_neighbors(root: str | None = None, dataset: str = "", allow_restricted: bool = False) -> str:
    """What one dataset relates to: its links, stacks, time alignments, nearest-place and containment edges,
    each with the evidence the engine recorded. `dataset` is a dataset id from graph_query (file#node)."""
    if not dataset:
        raise ToolError("Give a dataset id (from graph_query), e.g. data/shop.json#orders")
    return _dump(hl.graph_neighbors(_repo_root(root), dataset, allow_restricted=bool(allow_restricted)))


@mcp.tool()
@friendly
def graph_path(root: str | None = None, source: str = "", target: str = "", allow_restricted: bool = False) -> str:
    """The shortest chain of relations joining two datasets in the repository graph, with the edges between
    them. Deterministic: ties are broken by id."""
    if not source or not target:
        raise ToolError("Give two dataset ids (from graph_query)")
    return _dump(hl.graph_path(_repo_root(root), source, target, allow_restricted=bool(allow_restricted)))


@mcp.tool()
@friendly
def graph_shared_keys(root: str | None = None, allow_restricted: bool = False) -> str:
    """Keys that link datasets across different projects: the same join key reaching across a repository, with
    its cardinality, match percentage and evidence."""
    return _dump(hl.graph_shared_keys(_repo_root(root), allow_restricted=bool(allow_restricted)))


@mcp.tool()
@friendly
def get_events(root: str | None = None, since: int = 0, type: str | None = None, limit: int | None = None) -> str:
    """The repository event log (roadmap A2): an append-only record of what changed and what it affected —
    source_changed, dataset_invalidated (rippling across projects via the graph), project_recomputed,
    index_stale, graph_updated. Read-only. `since` returns only events after a sequence number; `type` filters
    to one kind; `limit` keeps the most recent."""
    return _dump(hl.read_events(_repo_root(root), since=int(since), type=type, limit=limit))


@mcp.tool()
@friendly
def policy_check(root: str | None = None, principal: str = "anonymous", tool: str = "", category: str | None = None,
                 project: str | None = None) -> str:
    """Whether a repository's governance policy (roadmap B1) would allow a principal to use a tool, without
    doing it: verdict allow/approve/deny with the reason. When no policy is configured every action is allowed,
    so this changes nothing until a repository opts in with <root>/.dancr/gateway/policy.json. Read-only."""
    return _dump(hl.policy_check(_repo_root(root), principal, tool or "", category=category, project=project))


@mcp.tool()
@friendly
def graph_suggest(root: str | None = None, allow_restricted: bool = False) -> str:
    """The structural cross-project questions the graph can answer, best first."""
    return _dump(hl.cross_suggest(_repo_root(root), allow_restricted=bool(allow_restricted)))


@mcp.tool()
@friendly
def graph_ask(root: str | None = None, question: str = "", allow_restricted: bool = False) -> str:
    """Answer a structural cross-project question against the graph, citing edge evidence. No cross-project
    execution: it returns graph facts (a shared key, a join path, where a dataset comes from, what it feeds).
    Example questions: 'what joins data/a.json#orders and data/b.json#customers?', 'which keys link projects?',
    'where does shop.json#sales come from?', 'what relates to shop.json#orders?'."""
    if not question:
        raise ToolError("Give a question")
    return _dump(hl.cross_ask(_repo_root(root), question, allow_restricted=bool(allow_restricted)))


@mcp.tool()
@friendly
def run_scenarios(path: str, spec: dict[str, Any] | None = None, spec_path: str | None = None,
                  target: str | None = None, out_dir: str = "scenarios", ext: str = "csv",
                  jobs: int = 1, force: bool = False) -> str:
    """Run a project across many scenarios (roadmap F2) and return a record with the per-scenario plan/output
    hashes. Give `spec` (a scenario spec: named/sweep/monte_carlo/sensitivity) or `spec_path` (a JSON file inside
    the server's root folder, or JSON text). One output per scenario plus a combined table land in `out_dir`
    inside the pipeline file's folder; the project file itself is never changed."""
    p = _load(path)
    if spec_path:
        sp = _from_root(spec_path)
        src: Any = json.loads(sp.read_text(encoding="utf-8")) if sp.is_file() else json.loads(spec_path)
    elif spec is not None:
        src = spec if isinstance(spec, dict) else json.loads(spec)
    else:
        raise ToolError("Give spec (a scenario spec) or spec_path")
    return _dump(hl.run_scenarios(p, src, target=target, out_dir=out_dir, ext=ext, jobs=int(jobs),
                                  force=bool(force)))


def main(root: str | None = None) -> None:
    from .logsetup import configure
    global ROOT, ROOT_REFUSED
    if root:
        ROOT = Path(root).expanduser().resolve()
        if not ROOT.is_dir():
            raise ValueError(f"--root {ROOT} is not a folder")      # a usage error: exit 2, JSON with --json
        ROOT_REFUSED = False
    else:
        # started from the home folder, a folder holding it (/home, /Users, C:\Users) or the top of a drive (an
        # agent launched from there): every file of the person's would be in reach, so creating or changing
        # pipelines is refused until --root names a folder. Reading and running existing ones still works.
        ROOT_REFUSED = Path.home().resolve().is_relative_to(ROOT) or ROOT.parent == ROOT
    configure(stderr_level=logging.WARNING)      # stdout carries the protocol; the log file and stderr get the rest
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
