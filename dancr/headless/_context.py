"""Knowledge-base context, FAIR descriptors and the project catalog."""
from __future__ import annotations

from ._atomic import write_text_atomic
from ._safety import source_files

import json
import os
from pathlib import Path
from typing import Any

import polars as pl

from ..core import Pipeline, registry
from ..core.registry import in_dancr_folder
from ..core.dtypes import json_safe
from ..core.executor import Executor, CODE_FINGERPRINT
from ..core.model import FORMAT_VERSION
CONTEXT_VERSION = 2          # the context document's own format, independent of the pipeline format
CONTEXT_MAX_SAMPLE_ROWS = 100
CONTEXT_DOC = "dancr.table"


def engine_version() -> str:
    """The DANCR version that built or read something (the context document, a run manifest)."""
    from .. import __version__
    return __version__


def _dataset_meta(p: Pipeline) -> dict[str, Any]:
    """The project's dataset-level metadata (creator, license, description…), kept in ``meta["dataset"]`` so an
    older DANCR opening the project preserves it (unknown top-level keys are dropped, unknown meta keys are not)."""
    meta = p.meta.get("dataset") if isinstance(getattr(p, "meta", None), dict) else None
    return dict(meta) if isinstance(meta, dict) else {}


def build_context(p: Pipeline, executor: Executor | None = None, *, nodes: list[str] | None = None,
                  deep: bool = True, stats: bool = False, samples: bool = False, sample_rows: int = 10,
                  run: bool | None = None, root: Path | str | None = None,
                  allow_restricted: bool = False) -> dict[str, Any]:
    """One document describing the project's datasets for a knowledge base to index: the compact profile
    (schemas, roles, relations) plus, optionally, per-column statistics and capped sample rows, and a prose
    *doc card* per table that a search index can match on. Nothing is changed; only the cache may be written
    (when ``run`` lets a table be computed for real statistics).

    ``stats``/``samples`` need a table's result; a source that has not run yet is computed when ``run`` is left
    unset (defaults on for either), and statistics are skipped, with a note, when ``run=False``. Samples fall
    back to a preview of the first rows for a table that has not been computed. The Assistant's own profile is
    a subset of this document (see :mod:`dancr.core.profile`)."""
    from ..core.answers import model_for
    from ..core.profile import clean, project_profile, table_card
    from ..core.dtypes import json_safe
    ex = executor if executor is not None else Executor(p)
    model = model_for(p, ex, deep=deep, nodes=nodes)
    do_run = bool(stats or samples) if run is None else bool(run)
    node = nodes[0] if nodes and len(nodes) == 1 else None
    held: dict[str, str | None] = {}
    memo: dict[str, str] = {}           # one memo for the whole loop: a shared node is hashed once, not per table
    for nid in model.tables:
        try:
            held[nid] = ex.safe_hash(nid, memo)
        except Exception:  # noqa: BLE001 - a step whose hash cannot be worked out simply holds nothing
            held[nid] = None
    ex.hold(held)                       # before looking for results, so a cache sweep elsewhere keeps them
    try:
        prof = project_profile(p, model, node=node)
        by_node = {t["node"]: t for t in prof["tables"]}
        documents: list[dict[str, Any]] = []
        for nid, table in model.tables.items():
            entry = by_node.get(nid)
            if entry is None:
                continue
            st = ex.state(nid)
            if do_run and st.status != "done":
                st = (ex.run(targets=[nid]) or {}).get(nid, st)
            card = table_card(table, [r for r in model.relations if nid in r.tables])
            entry["summary"] = card
            # The plan hash folds in the code fingerprint, the step's settings, its inputs and its source
            # files' size/time/content: it is exactly what changes when the data behind a document changes,
            # so a search index can tell a stale document from a fresh one (see ``context_changes``).
            entry["content_hash"] = held.get(nid)
            document: dict[str, Any] = {"id": nid, "node": nid, "title": entry["title"], "text": card,
                                        "content_hash": held.get(nid), "rows": entry.get("rows"),
                                        "source": entry.get("file")}
            if stats:
                summary = _table_stats(ex, nid)
                if summary is not None:
                    entry["stats"] = summary
                    document["stats"] = summary
                else:
                    entry["stats_note"] = "not computed; run the step first"
            if samples:
                rows, origin = _table_sample(ex, nid, sample_rows)
                entry["sample"] = rows
                entry["sample_from"] = origin
                document["sample"] = rows
            documents.append(document)
        from datetime import datetime
        out = {
            "kind": "dancr.context",
            "version": CONTEXT_VERSION,
            "engine_version": engine_version(),
            "fingerprint": CODE_FINGERPRINT,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "project": {"name": clean(getattr(p, "name", "") or "Untitled"),
                        "file": p.path.name if p.path else None},
            "dataset": _dataset_meta(p),
            "tables": prof["tables"],
            "relations": prof["relations"],
            "inputs": prof.get("inputs", []),
            "skipped": prof.get("tables_that_could_not_be_read", {}),
            "documents": documents,
        }
        # The cross-project graph block: the edges incident to this project's datasets and their neighbours,
        # inferred from a graph built near the project (or an explicit root). Absent when there is no graph.
        try:
            from ._graph import graph_context_for
            block = graph_context_for(p, root=root, allow_restricted=allow_restricted)
        except Exception:  # noqa: BLE001 - a graph problem must never break the context export
            block = None
        if block:
            out["graph"] = block
            per = block.get("per_dataset") or {}
            for doc in out["documents"]:
                if (slice_ := per.get(doc["node"])):
                    doc["graph"] = slice_
            for table in out["tables"]:
                if (slice_ := per.get(table.get("node"))):
                    table["graph"] = slice_
        return json_safe(out)
    finally:
        ex.release()


def export_fair(p: Pipeline, executor: Executor | None = None, *, fmt: str = "schema.org",
                nodes: list[str] | None = None, deep: bool = True, samples: bool = False,
                sample_rows: int = 10, out: Path | str | None = None, path_root: str = "",
                run: bool | None = None, root: Path | str | None = None,
                allow_restricted: bool = False) -> dict[str, Any]:
    """A FAIR descriptor for the project's datasets: ``schema.org`` (JSON-LD), ``frictionless`` (Data Package),
    ``manifest`` (what produced the results) or ``rocrate`` (a metadata graph). Built from the context document
    and the project's dataset metadata; ``out`` writes it atomically. Nothing is changed but the cache."""
    from ..core import fair
    ex = executor if executor is not None else Executor(p)
    ctx = build_context(p, ex, nodes=nodes, deep=deep, stats=fmt in ("manifest", "rocrate"),
                        samples=samples, sample_rows=sample_rows, run=run, root=root,
                        allow_restricted=allow_restricted)
    meta = p.dataset_meta()
    manifest = fair.run_manifest(p, ex, meta=meta, graph=ctx.get("graph")) if fmt.strip().lower().replace("_", ".") in (
        "manifest", "provenance", "run", "rocrate", "ro-crate") else None
    doc = fair.fair_document(ctx, fmt, pipe=p, executor=ex, manifest=manifest, meta=meta,
                             pipeline_file=p.path.name if p.path else None)
    if out is not None:
        write_text_atomic(out, fair.dump(doc))
    return doc


ROCRATE_COPY = ("metadata", "data", "results", "all")


def package_rocrate(p: Pipeline, executor: Executor | None = None, *, out: Path | str,
                    copy: str = "metadata", zip: bool = False, meta: dict[str, Any] | None = None,
                    manifest: dict[str, Any] | None = None, samples: bool = False,
                    overwrite: bool = False) -> dict[str, Any]:
    """Write a self-contained RO-Crate: the FAIR descriptors, the project file and its run manifest, and — with
    ``copy`` ``data``/``results``/``all`` — the source files and the files the project writes. Written as a
    directory or a ``.zip``. Everything lands inside the project's folder, never over a data file it reads."""
    import shutil
    import zipfile
    from ..core import fair
    ex = executor if executor is not None else Executor(p)
    folder = p.directory.resolve()
    meta = dict(meta if meta is not None else p.dataset_meta())
    mode = (copy or "metadata").lower()
    if mode not in ROCRATE_COPY:
        raise ValueError(f"copy must be one of: {', '.join(ROCRATE_COPY)}")
    ctx = build_context(p, ex, stats=False, samples=samples)
    man = manifest if manifest is not None else fair.run_manifest(p, ex, meta=meta, graph=ctx.get("graph"))

    destination = Path(out).expanduser()
    destination = (destination if destination.is_absolute() else folder / destination).resolve()
    if not destination.is_relative_to(folder):
        raise ValueError(f"A crate must be written inside the project folder {folder}, not {destination}")
    if in_dancr_folder(destination, folder):
        raise ValueError(f"Won't write a crate into DANCR's own .dancr folder: {destination}")

    from ..core.dtypes import json_safe
    from ..core.secrets import redact_params
    from ..core.verify import build_attestation, dump_attestation
    public = p.to_dict()
    for nd in public.get("nodes", []):                  # a crate is shared: never ship a password or token in it
        try:
            nd["params"] = redact_params(registry.get(nd["type"]), nd.get("params") or {})
        except Exception:  # noqa: BLE001 - an unknown node type is left as it is
            pass
    pipeline_text = json.dumps(json_safe(public), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    descriptors: dict[str, str] = {
        "dancr-pipeline.json": pipeline_text,
        "datapackage.json": fair.dump(fair.datapackage(ctx, meta)),
        "dataset.jsonld": fair.dump(fair.dataset_jsonld(ctx, meta)),
        "dancr-manifest.json": fair.dump(man),
        "dancr-attestation.json": dump_attestation(build_attestation(p, ex, states=ex.states())),
        "context.jsonl": context_jsonl(ctx),
    }
    copies: dict[str, Path] = {}                        # arcname -> source file
    if mode in ("data", "all"):
        for f in sorted(source_files(p), key=str):
            copies[_unique_arc("data", f.name, copies)] = f
    if mode in ("results", "all"):
        for _nid, st in ex.states().items():
            for f in sorted(st.files or {}, key=str):
                fp = Path(f)
                if fp.exists():
                    copies[_unique_arc("results", fp.name, copies)] = fp

    files = [{"path": name, "name": _descriptor_name(name), "encodingFormat": "application/json"}
             for name in descriptors]
    for arc, src in copies.items():
        files.append({"path": arc, "name": src.name, "source": src.name})
    graph = fair.ro_crate_graph(ctx, meta, files)

    payload: dict[str, str] = {"ro-crate-metadata.json": fair.dump(graph), **descriptors}
    if zip and destination.suffix.lower() != ".zip":
        destination = destination.with_suffix(destination.suffix + ".zip") if destination.suffix else destination.with_name(destination.name + ".zip")
    if zip:
        if destination.exists() and not overwrite:
            raise ValueError(f"{destination} already exists. Choose another name, or pass overwrite")
        tmp = destination.with_name(f".{destination.name}.{os.getpid()}.tmp.zip")
        try:
            with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
                for name, text in payload.items():
                    z.writestr(name, text)
                for arc, src in copies.items():
                    z.write(src, arc)
            os.replace(tmp, destination)
        finally:
            tmp.unlink(missing_ok=True)
        written = [str(destination)]
        kind = "zip"
    else:
        if destination.exists() and any(destination.iterdir()) and not overwrite:
            raise ValueError(f"{destination} is not empty. Choose another folder, or pass overwrite")
        destination.mkdir(parents=True, exist_ok=True)
        for name, text in payload.items():
            write_text_atomic(destination / name, text)
        for arc, src in copies.items():
            target = destination / arc
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
        written = [str(destination / n) for n in payload] + [str(destination / a) for a in copies]
        kind = "directory"
    return json_safe({"ok": True, "path": str(destination), "format": kind, "copy": mode,
                      "files": written, "datasets": len(ctx.get("documents", [])),
                      "count": len(copies)})


def _unique_arc(folder: str, name: str, taken: dict[str, Any]) -> str:
    arc = f"{folder}/{name}"
    n = 2
    while arc in taken:
        stem, dot, ext = name.rpartition(".")
        arc = f"{folder}/{stem}_{n}.{ext}" if dot else f"{folder}/{name}_{n}"
        n += 1
    return arc


def _descriptor_name(name: str) -> str:
    return {"dancr-pipeline.json": "DANCR pipeline", "datapackage.json": "Frictionless data package",
            "dataset.jsonld": "schema.org dataset", "dancr-manifest.json": "DANCR run manifest",
            "dancr-attestation.json": "DANCR attestation", "context.jsonl": "Knowledge-base context"}.get(name, name)


def _table_stats(ex: Executor, node_id: str) -> list[dict[str, Any]] | None:
    """Per-column summary statistics (count, missing, mean, std, min, quartiles, max) of a computed table, or
    None when it has not been computed (statistics on a preview would be misleading)."""
    st = ex.state(node_id)
    if st.status != "done" or not st.output:
        return None
    from ..core.dtypes import json_safe
    from ..views.stats import column_summary
    return json_safe(column_summary(pl.scan_parquet(st.output)).to_dicts())


def _table_sample(ex: Executor, node_id: str, rows: int) -> tuple[list[dict[str, Any]], str]:
    """Up to ``rows`` rows of a table and where they came from: "cache" (a spread/head of a computed result)
    or "preview" (the first rows of a table not computed yet)."""
    from ..core.dtypes import json_safe
    n = max(1, min(int(rows), CONTEXT_MAX_SAMPLE_ROWS))
    lf, kind = ex.sample_frame(node_id, n)
    return json_safe(lf.head(n).collect(engine="streaming").to_dicts()), ("cache" if kind in ("all", "spread") else "preview")


def context_jsonl(ctx: dict[str, Any]) -> str:
    """A context document as JSON Lines, one line per dataset — the unit a RAG index ingests. Each line carries
    the engine version, when it was generated and the dataset's ``content_hash``, so an index can spot a stale
    document and refresh only what changed (see ``context_changes``)."""
    from ..core.dtypes import json_safe
    lines = []
    for d in ctx.get("documents", []):
        line = {"kind": CONTEXT_DOC, "version": ctx.get("version", CONTEXT_VERSION),
                "project": (ctx.get("project") or {}).get("name"),
                "engine_version": ctx.get("engine_version"),
                "generated_at": ctx.get("generated_at"), **d}
        if ctx.get("changes") is not None:          # a --changed export: keep the marker on every line so a
            line["changes"] = ctx["changes"]        # JSON Lines changed feed is recognisable downstream
        lines.append(json.dumps(json_safe(line), ensure_ascii=False, allow_nan=False, default=str))
    return "\n".join(lines) + ("\n" if lines else "")


def parse_context(text: str) -> dict[str, Any]:
    """A context document, or its JSON Lines form, as a dict. A blank or unreadable text is an empty context."""
    t = (text or "").strip()
    if not t:
        return {}
    try:                                        # the whole thing as one JSON document
        data = json.loads(t)
        if isinstance(data, dict):
            return data
    except ValueError:
        pass                                    # not one JSON object: try it as JSON Lines, one dataset per line
    documents = []
    for line in t.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if isinstance(d, dict):
            documents.append(d)
    return {"documents": documents}


def _as_context(previous: Any) -> dict[str, Any]:
    """A context dict, or a context document given as text or a file path, as a dict."""
    if isinstance(previous, dict):
        return previous
    p = Path(str(previous)).expanduser()
    try:
        if p.is_file():
            return parse_context(p.read_text(encoding="utf-8"))
    except OSError:
        pass
    return parse_context(str(previous))


def context_changes(ctx: dict[str, Any], previous: Any) -> dict[str, Any]:
    """``ctx`` reduced to the datasets whose content changed since ``previous`` (a context document, its JSON
    Lines text, or a file path), best for re-indexing only what moved.

    A dataset is *added* when the previous document did not have it, *changed* when its ``content_hash`` differs,
    and *unchanged* otherwise; the engine's own version is compared too, so a new DANCR that changes how a step
    computes marks every document changed. ``context_changes`` returns a context-shaped dict (so the usual
    printers and writers work on it) with a ``changes`` block summarising added/changed/unchanged/removed."""
    prev = _as_context(previous)
    prev_docs = {str(d.get("id") or d.get("node")): d for d in prev.get("documents", []) if isinstance(d, dict)}
    prev_engine = prev.get("engine_version")
    if not prev_engine and prev.get("documents"):
        first = prev["documents"][0]
        prev_engine = first.get("engine_version") if isinstance(first, dict) else None
    prev_fp = prev.get("fingerprint")
    if not prev:
        engine_changed = False                       # nothing to compare with: everything is "added"
    elif prev_fp is None:
        engine_changed = prev_engine is not None and prev_engine != ctx.get("engine_version")
    else:
        engine_changed = (prev_engine, prev_fp) != (ctx.get("engine_version"), ctx.get("fingerprint"))

    added, changed, unchanged = [], [], []
    for d in ctx.get("documents", []):
        did = str(d.get("id") or d.get("node"))
        old = prev_docs.get(did)
        if old is None:
            added.append(did)
        elif engine_changed or old.get("content_hash") != d.get("content_hash"):
            changed.append(did)
        else:
            unchanged.append(did)
    current = {str(d.get("id") or d.get("node")) for d in ctx.get("documents", [])}
    removed = [k for k in prev_docs if k not in current]
    keep = set(added) | set(changed)
    out = dict(ctx)
    out["documents"] = [d for d in ctx.get("documents", []) if str(d.get("id") or d.get("node")) in keep]
    out["tables"] = [t for t in ctx.get("tables", []) if str(t.get("node")) in keep]
    out["changes"] = {"engine_changed": engine_changed, "added": added, "changed": changed,
                      "unchanged": unchanged, "removed": removed,
                      "graph_changed": (prev.get("graph") or {}).get("content_hash")
                      != (ctx.get("graph") or {}).get("content_hash"),
                      "previous": prev_engine}
    return json_safe(out)


# ---------------------------------------------------------------- a catalog of many projects
def find_pipelines(root: Path | str, pattern: str = "*.json", recursive: bool = True) -> list[Path]:
    """Every DANCR project file under ``root`` (never inside a ``.dancr`` folder), sorted by path."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"{root} is not a folder")
    it = root.rglob(pattern) if recursive else root.glob(pattern)
    out: list[Path] = []
    for p in sorted(it, key=str):
        try:
            if not p.is_file() or p.suffix.lower() != ".json":
                continue
            if any(part.lower() == ".dancr" for part in p.relative_to(root).parts):
                continue
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and data.get("dancr") == FORMAT_VERSION:
            out.append(p)
    return out


def build_catalog(root: Path | str, *, pattern: str = "*.json", recursive: bool = True, stats: bool = False,
                  samples: bool = False, fair_format: str | None = None, jobs: int = 1,
                  allow_restricted: bool = False) -> dict[str, Any]:
    """One document describing every DANCR project under ``root``: each project's datasets (the context export)
    and, optionally, a FAIR descriptor each. A broken project is listed under ``skipped`` with the reason, never
    raised. Statistics and samples need a table's result and compute it (each project's own cache)."""
    from datetime import datetime
    from ..core import fair
    root = Path(root).expanduser().resolve()
    files = find_pipelines(root, pattern, recursive)

    def one(f: Path) -> tuple[dict[str, Any] | None, str | None, str | None]:
        try:
            p = Pipeline.load(f)
        except Exception as e:  # noqa: BLE001 - a broken project is reported, not fatal
            return None, str(f), f"cannot read: {e}"
        try:
            ex = Executor(p)
            ctx = build_context(p, ex, stats=stats, samples=samples, root=root, allow_restricted=allow_restricted)
        except Exception as e:  # noqa: BLE001
            return None, str(f), str(e)
        entry: dict[str, Any] = {"file": str(f), "name": p.name,
                                 "datasets": ctx.get("documents", []),
                                 "relations": ctx.get("relations", []),
                                 "tables": len(ctx.get("tables", [])),
                                 "engine_version": ctx.get("engine_version")}
        if fair_format:
            try:
                entry["fair"] = fair.fair_document(ctx, fair_format, pipe=p, executor=ex)
            except Exception as e:  # noqa: BLE001
                entry["fair_error"] = str(e)
        return entry, None, None

    projects: list[dict[str, Any]] = []
    skipped: dict[str, str] = {}
    if int(jobs or 1) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=int(jobs)) as pool:
            for entry, bad, why in pool.map(one, files):
                if entry is not None:
                    projects.append(entry)
                elif bad is not None and why is not None:
                    skipped[bad] = why
    else:
        for f in files:
            entry, bad, why = one(f)
            if entry is not None:
                projects.append(entry)
            elif bad is not None and why is not None:
                skipped[bad] = why
    projects.sort(key=lambda e: e["file"])
    return json_safe({
        "kind": "dancr.catalog",
        "version": CONTEXT_VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "engine_version": engine_version(),
        "fingerprint": CODE_FINGERPRINT,
        "root": str(root), "pattern": pattern,
        "count": len(projects), "projects": projects, "skipped": skipped,
    })


def catalog_jsonl(catalog: dict[str, Any], by_project: bool = False) -> str:
    """A catalog as JSON Lines. By default one line per dataset (the unit an index ingests); ``by_project``
    gives one line per project with its datasets nested."""
    from ..core.dtypes import json_safe
    lines = []
    for proj in catalog.get("projects", []):
        if by_project:
            line = {"kind": "dancr.project", "version": catalog.get("version"), "file": proj.get("file"),
                    "name": proj.get("name"), "engine_version": catalog.get("engine_version"),
                    "generated_at": catalog.get("generated_at"), "datasets": proj.get("datasets", [])}
            lines.append(json.dumps(json_safe(line), ensure_ascii=False, allow_nan=False, default=str))
            continue
        for d in proj.get("datasets", []):
            line = {"kind": CONTEXT_DOC, "version": catalog.get("version"), "project": proj.get("name"),
                    "file": proj.get("file"), "engine_version": catalog.get("engine_version"),
                    "generated_at": catalog.get("generated_at"), **d}
            lines.append(json.dumps(json_safe(line), ensure_ascii=False, allow_nan=False, default=str))
    return "\n".join(lines) + ("\n" if lines else "")


def _flatten_catalog(d: Any) -> dict[str, Any]:
    """A catalog's datasets as a flat context ``{documents: [...]}``; anything else is returned as-is (so an
    earlier context document or its JSON Lines works too)."""
    if isinstance(d, dict) and "projects" in d:
        docs: list[dict[str, Any]] = []
        for proj in d.get("projects", []):
            for x in proj.get("datasets", []):
                docs.append({**x, "file": proj.get("file"), "project": proj.get("name")})
        return {"documents": docs, "engine_version": d.get("engine_version"), "fingerprint": d.get("fingerprint")}
    return d


def _catalog_key(file: Any, d: dict[str, Any]) -> str:
    """A dataset's identity across a catalog: its project file and its step id (node ids repeat between
    projects, so the file is part of the key)."""
    return f"{file}#{d.get('id') or d.get('node')}"


def catalog_changes(catalog: dict[str, Any], previous: Any) -> dict[str, Any]:
    """``catalog`` reduced to the datasets whose content changed since ``previous`` (an earlier catalog, or its
    JSON Lines), grouped back under their projects. Reuses the context ``content_hash`` comparison."""
    prev = _flatten_catalog(_as_context(previous))
    if prev.get("documents"):
        prev = {**prev, "documents": [{**x, "id": _catalog_key(x.get("file"), x)} for x in prev["documents"]]}
    docs: list[dict[str, Any]] = []
    for proj in catalog.get("projects", []):
        for d in proj.get("datasets", []):
            docs.append({**d, "id": _catalog_key(proj.get("file"), d), "file": proj.get("file"),
                         "project": proj.get("name")})
    flat = {"documents": docs, "engine_version": catalog.get("engine_version"),
            "fingerprint": catalog.get("fingerprint")}
    delta = context_changes(flat, prev)
    keep = {str(d.get("id")) for d in delta.get("documents", [])}
    projects = []
    for proj in catalog.get("projects", []):
        kept = [d for d in proj.get("datasets", []) if _catalog_key(proj.get("file"), d) in keep]
        if kept:
            projects.append({**proj, "datasets": kept})
    out = dict(catalog)
    out["projects"] = projects
    out["count"] = sum(len(p["datasets"]) for p in projects)
    out["changes"] = delta.get("changes")
    return json_safe(out)


def context_text(ctx: dict[str, Any]) -> str:
    """A context document as something a person reads: a doc card per dataset, then the relations."""
    lines = [f'Project "{ctx["project"]["name"]}" — {len(ctx.get("tables", []))} dataset(s), context version {ctx.get("version")}']
    for d in ctx.get("documents", []):
        lines += ["", f"[{d['node']}] {d['title']}", "  " + d["text"]]
    if ctx.get("relations"):
        lines += ["", "Relations:"]
        for r in ctx["relations"]:
            bits = [r.get("kind", "")]
            if r.get("left_on"):
                bits.append(f"{r['left_on']} = {r.get('right_on')}")
            if r.get("match_pct") is not None:
                bits.append(f"{r['match_pct']}% match")
            lines.append("  " + " ↔ ".join(r.get("tables", [])) + "  (" + ", ".join(str(b) for b in bits if b) + ")")
    for nid, why in (ctx.get("skipped") or {}).items():
        lines.append(f"  could not read {nid}: {why}")
    if ctx.get("graph"):
        g = ctx["graph"]
        edges = g.get("edges", [])
        lines += ["", f"Cross-project relations ({len(edges)}):"]
        for e in edges:
            bits = [str(e.get("kind") or "")]
            if e.get("left_on"):
                bits.append(f"{e['left_on']} = {e.get('right_on')}")
            if e.get("match_pct"):
                bits.append(f"{e['match_pct']}% match")
            lines.append(f"  {e.get('left')} → {e.get('right')}  ({', '.join(b for b in bits if b)})")
        if g.get("neighbours"):
            lines.append("  neighbours: " + ", ".join(str(n.get("id")) for n in g["neighbours"]))
    if ctx.get("inputs"):
        lines += ["", "Inputs: " + ", ".join(f"{i['name']}={i['value']}" for i in ctx["inputs"])]
    return "\n".join(lines)
