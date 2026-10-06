"""What the command line and the MCP server share: finding steps, adding them, reporting them, charting and
reading results. Both front ends call these, so a step is reported the same way wherever it is asked for."""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import polars as pl

from ..core import Pipeline, PipelineError, registry
from ..core.dtypes import json_safe
from ..core.executor import Executor, NodeState, CODE_FINGERPRINT
from ..core.model import Node, FORMAT_VERSION
from ._atomic import write_text_atomic  # noqa: E402
from ._safety import output_files, source_files, unsafe_outputs, unsafe_write  # noqa: E402
from ._context import (  # noqa: E402 - re-exported for the CLI, MCP server and the SDK
    CONTEXT_DOC, CONTEXT_MAX_SAMPLE_ROWS, CONTEXT_VERSION, ROCRATE_COPY, build_catalog, build_context,
    catalog_changes, catalog_jsonl, context_changes, context_jsonl, context_text, engine_version,
    export_fair, find_pipelines, package_rocrate, parse_context,
)


class StepFailed(ValueError):
    """A step asked for could not be computed (the CLI exits 1, as for a failed run, not 2 for a usage error)."""



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


def node_public(p: Pipeline, node_id: str) -> dict[str, Any]:
    """A step's own definition, safe to show or hand back to a person or an agent: a ``secret`` setting
    (a database password, an API token) and credentials inside any other text are blanked. Use this wherever
    ``Node.to_dict()`` would otherwise echo settings; the saved file keeps the raw values."""
    from ..core.secrets import redact_params
    n = p.nodes[node_id]
    return {**n.to_dict(), "params": redact_params(registry.get(n.type), n.params)}


def run_record(p: Pipeline, ex: Executor, res: dict[str, NodeState], elapsed: float) -> dict[str, Any]:
    """How a run is reported by both front ends; `nodes` is keyed by step id. ``findings`` are what the steps
    said (a sentence each), most interesting first, and ``headline`` is the best of them."""
    from ..core.findings import collect, record as finding_record, headline as finding_headline
    failed = [nid for nid, s in res.items() if s.status == "failed"]
    found = collect(finding_record(p.nodes[nid].type, p.nodes[nid].title, s.report, s.status)
                    for nid, s in res.items())
    return {"ok": not failed, "failed": failed, "nodes": {nid: node_record(p, s) for nid, s in res.items()},
            "problems": p.problems(), "elapsed": elapsed, "cache_dir": str(ex.cache_dir),
            "findings": found, "headline": finding_headline(found)}


BATCH_EXT = ("csv", "tsv", "txt", "parquet", "xlsx", "geojson")
BATCH_DATA_EXT = {".csv", ".tsv", ".txt", ".dat", ".tab", ".log", ".xlsx", ".xlsm", ".xls", ".xlsb", ".ods",
                  ".parquet", ".pq", ".geojson", ".json", ".gpkg", ".shp"}


def expand_files(directory: Path, patterns: list[str] | None) -> list[Path]:
    """The data files a list of paths, folders and globs names, resolved and sorted (deterministic order)."""
    import glob as _glob
    found: list[Path] = []
    for pat in patterns or []:
        p = Path(str(pat)).expanduser()
        if not p.is_absolute():
            p = directory / p
        if any(ch in str(pat) for ch in "*?["):
            found += [Path(x) for x in _glob.glob(str(p), recursive=True)]
        elif p.is_dir():
            found += [f for f in sorted(p.iterdir()) if f.is_file()]
        elif p.is_file():
            found.append(p)
    return sorted({f.resolve() for f in found if f.is_file() and f.suffix.lower() in BATCH_DATA_EXT}, key=str)


def _batch_output_names(inputs: list[Path], folder: Path, ext: str, reserved: set[str]) -> dict[Path, str]:
    """A unique output file name for each batch input, decided up front so parallel workers cannot collide.

    A bare stem is used when it is unique (``run1.csv``); a stem that repeats, or that would shadow the
    combined table, is qualified with the file's folder path (``a__run.csv``); a still-repeating name gets a
    numeric suffix. The choice never depends on other workers, so the same batch always writes the same names."""
    names: dict[Path, str] = {}
    taken: set[str] = {f"{r}.{ext}".casefold() for r in reserved}
    for f in inputs:
        stem = f.stem
        if stem.casefold() in {r.casefold() for r in reserved} or f"{stem}.{ext}".casefold() in taken:
            try:
                rel = f.resolve().relative_to(folder)
                stem = "__".join(rel.with_suffix("").parts)
            except ValueError:                       # outside the project folder: keep the name, suffix later
                stem = f.stem
        name = f"{stem}.{ext}"
        if name.casefold() in taken:                  # a deeper collision: a stable numeric suffix
            i = 2
            while f"{stem}_{i}.{ext}".casefold() in taken:
                i += 1
            name = f"{stem}_{i}.{ext}"
        taken.add(name.casefold())
        names[f] = name
    return names


def _batch_source(p: Pipeline, loader: str | None) -> tuple[str, str]:
    """The source step whose file setting a batch rebinds, and the name of that setting."""
    from ..core.registry import registry
    sources = [nid for nid, n in p.nodes.items() if registry.get(n.type).kind == "source"]
    if loader:
        require_node(p, loader)
        if loader not in sources:
            raise ValueError(f"{p.nodes[loader].title} is not a step that brings data in. Try one of: {sources}")
        nid = loader
    elif len(sources) == 1:
        nid = sources[0]
    else:
        raise ValueError(f"Which step reads the files? Name one of: {sources}")
    nt = registry.get(p.nodes[nid].type)
    if nt.key == "load_folder":
        raise ValueError("This step already reads a whole folder; use it on its own instead of a batch")
    paths = [prm.name for prm in nt.params if prm.kind == "path"]
    if not paths:
        raise ValueError(f"{p.nodes[nid].title} does not read a file, so a batch cannot swap files into it")
    return nid, paths[0]


def run_batch(p: Pipeline, files: list[str] | None, *, target: str, out_dir: Path | str, loader: str | None = None,
              ext: str = "csv", jobs: int = 1, force: bool = False, combined: bool = True,
              manifest: Path | str | None = None, source_column: str = "source_file",
              on_event: Any = None) -> dict[str, Any]:
    """Apply a project to many files: for each file, rebind the source's path, run ``target``, and write its
    output to ``out_dir``; then write one combined table (with a column naming the source file). The project's
    cache is shared, so rerunning a batch serves unchanged files from cache. Returns a manifest of what happened.

    Nothing about the project on disk changes: each file runs on a private copy. Writes are confined to the
    project's folder, never over a data file the project reads and never into ``.dancr``."""
    import copy
    from concurrent.futures import ThreadPoolExecutor
    from datetime import datetime
    from ..core.registry import in_dancr_folder
    from ..core.model import Pipeline as _Pipeline, portable_path
    from ..core.nodes.outputs import write_table
    emit = on_event or (lambda e: None)
    folder = p.directory.resolve()
    if not target:
        order = p.topological_order()
        if not order:
            raise ValueError("The project has no steps to run")
        target = order[-1]
    require_node(p, target)
    loader_id, path_param = _batch_source(p, loader)
    ext = str(ext or "csv").lower().lstrip(".")
    if ext not in BATCH_EXT:
        raise ValueError(f"Save as one of: {', '.join(BATCH_EXT)}")
    inputs = expand_files(folder, files)
    if not inputs:
        raise ValueError("No data files matched. Name a folder, a glob or files")
    destination = Path(out_dir).expanduser()
    destination = (destination if destination.is_absolute() else folder / destination).resolve()
    if not destination.is_relative_to(folder):
        raise ValueError(f"Batch results must be written inside the project folder {folder}, not {destination}")
    if in_dancr_folder(destination, folder):
        raise ValueError(f"Won't write batch results into DANCR's own .dancr folder: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    names = _batch_output_names(inputs, folder, ext, reserved={"combined"})

    def one(f: Path) -> tuple[dict[str, Any], tuple[Path, str] | None]:
        clone = _Pipeline.from_dict(copy.deepcopy(p.to_dict()), p.path)
        clone.set_params(loader_id, **{path_param: portable_path(f, folder)})
        ex = Executor(clone)
        st = ex.run(targets=[target], force=force, sweep=False).get(target)
        rec: dict[str, Any] = {"file": str(f), "name": names[f], "status": (st.status if st else "idle")}
        if st is not None and st.status == "done" and st.output:
            out_file = destination / names[f]
            why = unsafe_write(clone, out_file, folder)
            if why:
                rec.update(status="failed", error=why)
                emit({"type": "batch_file", "file": f.name, "status": "failed"})
                return rec, None
            write_table(pl.scan_parquet(st.output), out_file)
            rec.update(rows=st.rows, output=str(out_file), elapsed=st.elapsed)
        else:
            rec["error"] = st.error if st is not None else "not run"
        emit({"type": "batch_file", "file": f.name, "status": rec["status"]})
        return rec, ((f, st.output) if rec["status"] == "done" and st and st.output else None)

    results: list[dict[str, Any]] = []
    ready: list[tuple[Path, str]] = []
    if int(jobs or 1) > 1:
        with ThreadPoolExecutor(max_workers=int(jobs)) as pool:
            for rec, out in pool.map(one, inputs):
                results.append(rec)
                if out:
                    ready.append(out)
    else:
        for f in inputs:
            rec, out = one(f)
            results.append(rec)
            if out:
                ready.append(out)
    results.sort(key=lambda r: r["file"])
    ready.sort(key=lambda pair: str(pair[0]))

    combined_path: Path | None = None
    if combined and ready:
        frames = []
        for f, output in ready:
            lf = pl.scan_parquet(output)
            if source_column:
                lf = lf.with_columns(pl.lit(f.name, dtype=pl.Utf8).alias(source_column))
            frames.append(lf)
        candidate = destination / f"combined.{ext}"
        why = unsafe_write(p, candidate, folder)
        if not why:
            write_table(pl.concat(frames, how="diagonal_relaxed"), candidate)
            combined_path = candidate

    record = json_safe({
        "kind": "dancr.batch",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "project": {"name": p.name, "file": p.path.name if p.path else None},
        "target": target, "loader": loader_id, "out_dir": str(destination), "ext": ext,
        "count": len(results), "ok": all(r.get("status") == "done" for r in results),
        "files": results, "combined": str(combined_path) if combined_path else None,
    })
    if manifest is not None:
        from ..core.fair import dump as _dump_json
        write_text_atomic(manifest, _dump_json(record))
        record["manifest"] = str(manifest)
    return record


def _watch_state(p: Pipeline, exclude: set[Path] | None = None) -> dict[str, list[int] | None]:
    """A snapshot of everything that should trigger a rerun: the project file and every source file (folder
    members included). Besides size and time it samples the content, as the executor's own fingerprint does,
    so a same-size rewrite that kept the modification time (``cp -p``, a restored backup) is still seen. A
    file that is gone is None, so deleting or moving one is a change too. ``exclude`` drops files a run itself
    writes (a batch's output folder), which must not look like a change."""
    from ..core.executor import _content_sample
    state: dict[str, list[int] | None] = {}

    def stamp(path: Path) -> list[int | str] | None:
        try:
            st = path.stat()
            return [st.st_size, st.st_mtime_ns, _content_sample(path, st.st_size)]
        except OSError:
            return None

    if p.path is not None:
        state[str(p.path)] = stamp(p.path)
    for f in source_files(p):
        if exclude and any(f.resolve() == e or e in f.resolve().parents for e in exclude):
            continue
        state[str(f)] = stamp(f)
    return state


def watch(pipeline: Path | str, *, node: str | None = None, batch: bool = False, files: list[str] | None = None,
          out_dir: Path | str | None = None, loader: str | None = None, interval: float = 2.0, once: bool = False,
          on_event: Any = None, stop: "threading.Event | None" = None) -> dict[str, Any]:
    """Watch a project and its data, and rerun it when anything changes. Runs once when it starts, then polls;
    a change is acted on only after a quiet poll (a file being written is not run mid-write). With ``batch`` it
    runs ``run_batch`` instead. Returns a small record of what happened. The caller may pass a ``stop`` event to
    end it (Ctrl+C on the command line).

    This is a long-running loop, so it has no MCP tool: an agent should call ``run_pipeline``/``run_batch``."""
    stop = stop or threading.Event()
    emit = on_event or (lambda e: None)
    path = Path(pipeline).expanduser().resolve()
    record: dict[str, Any] = {"kind": "dancr.watch", "path": str(path), "interval": float(interval),
                              "runs": 0, "batches": 0, "changed": [], "ok": True}

    def run_once(p: Pipeline) -> dict[str, Any]:
        if batch:
            rec = run_batch(p, files, target=node, out_dir=out_dir or "watched", loader=loader)
            record["batches"] += 1
        else:
            ex = Executor(p)
            t0 = time.perf_counter()
            res = ex.run(targets=[node] if node else None)
            rec = run_record(p, ex, res, time.perf_counter() - t0)
            record["runs"] += 1
        record["ok"] = record["ok"] and bool(rec.get("ok"))
        return rec

    try:
        p = Pipeline.load(path)
    except Exception as e:  # noqa: BLE001 - report plainly, do not loop forever on a broken file
        emit({"type": "watch_error", "error": str(e)})
        record["ok"] = False
        return record
    # a batch writing into a watched source folder must not see its own outputs as a fresh change
    exclude: set[Path] = set()
    if batch:
        dest = out_dir or "watched"
        dest = Path(dest).expanduser()
        exclude.add((dest if dest.is_absolute() else p.directory / dest).resolve())
    emit({"type": "watch_started", "path": str(path)})
    trigger = _watch_state(p, exclude)                  # what the very first run is based on
    rec = run_once(p)
    emit({"type": "watch_ran", "record": rec, "why": "start"})
    if once:
        return record

    # a change that landed while the first run executed is not lost: compare after it, and rerun on a quiet poll
    previous = _watch_state(p, exclude)
    pending = previous != trigger
    changed = _changed_files(previous, trigger) if pending else []
    while not stop.wait(max(0.1, float(interval))):
        try:
            p = Pipeline.load(path)                     # reload each poll: the project file itself may have changed
        except Exception as e:  # noqa: BLE001
            emit({"type": "watch_error", "error": str(e)})
            continue
        snap = _watch_state(p, exclude)
        if snap == previous:
            if pending:                                 # changed, then a quiet poll: act on it now
                pending = False
                trigger = previous
                try:
                    rec = run_once(p)
                    emit({"type": "watch_ran", "record": rec, "why": "changed", "files": changed})
                except Exception as e:  # noqa: BLE001
                    record["ok"] = False
                    pending = True                      # try the same change again on the next quiet poll
                    emit({"type": "watch_error", "error": str(e)})
                after = _watch_state(p, exclude)
                if after != trigger:                    # the data moved again while we were running: act on it too
                    changed = _changed_files(after, trigger)
                    record["changed"] = changed
                    pending = True
                previous = after
        else:
            changed = _changed_files(snap, previous)
            record["changed"] = changed
            emit({"type": "watch_changed", "files": changed})
            pending = True
            previous = snap
    record["stopped"] = True
    return record


def _changed_files(after: dict[str, Any], before: dict[str, Any]) -> list[str]:
    """The paths whose stamp differs between two snapshots, for a watch message."""
    return sorted({k for k in set(after) | set(before) if after.get(k) != before.get(k)})


def registry_get(p: Pipeline, node_id: str):
    from ..core.registry import registry
    return registry.get(p.nodes[node_id].type)


@contextmanager
def result_frame(p: Pipeline, ex: Executor, node_id: str, run: bool) -> Iterator[pl.LazyFrame]:
    """A step's output, running it first when `run` is set and it is not computed yet. The frame reads the stored
    result lazily, so the result is held (see ``Executor.hold``) until the block ends: another process's cache
    sweep must not delete it while it is being read."""
    require_node(p, node_id)
    # a step that stores nothing (a chart, a Save to file) is read from the result of a step above it: hold every
    # step it depends on, so a cache sweep elsewhere cannot delete that result while it is read
    held = {node_id} | (p.upstream_closure(node_id) if not registry_get(p, node_id).materialize else set())
    memo: dict[str, str] = {}
    ex.hold({n: ex.safe_hash(n, memo) for n in held})           # before looking for the result, as Executor.gc expects
    try:
        st = ex.state(node_id)
        if st.status != "done":
            if not run:
                raise ValueError(f"{node_id} hasn't been run yet (it's {st.status}). Run the project first, or ask for it to be run")
            res = ex.run(targets=[node_id])
            if res[node_id].status != "done":
                raise StepFailed(f"{node_id} failed: {res[node_id].error}")
        yield ex.frame(node_id)
    finally:
        ex.release()


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
    from ..core.samples import build_template as _build, write_sample, check_template
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



# ----------------------------------------------------------------- one writer at a time
LOCK_WAIT = 30.0          # seconds to wait for another program to finish changing a project file


class ProjectBusy(PipelineError):
    """Another program is changing the project file, or changed it while this was being done."""


def _try_lock(fd: int) -> bool:
    """True when the lock is ours, False when another holder has it. A folder that cannot lock at all (some
    network drives) is an error of its own, not a wait that ends in "being changed by another program"."""
    import errno
    busy = (errno.EAGAIN, errno.EWOULDBLOCK, errno.EACCES, getattr(errno, "EDEADLOCK", errno.EDEADLK))
    try:
        if sys.platform == "win32":
            import msvcrt
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as e:
        # Windows msvcrt.locking reports a held lock as winerror 33 (ERROR_LOCK_VIOLATION), which some builds
        # surface without a matching errno; treat it as busy rather than an unlockable folder.
        if e.errno in busy or getattr(e, "winerror", None) == 33:
            return False
        raise PipelineError(f"The project's folder can't lock files ({e.strerror or e}), so two programs could save over each other "
                            "there. Move the project to a local folder.") from e
    return True


def _unlock(fd: int) -> None:
    if sys.platform == "win32":
        import msvcrt
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(fd, fcntl.LOCK_UN)


def lock_file(path: Path) -> Path:
    path = Path(path).expanduser().resolve()
    return path.parent / ".dancr" / "locks" / f"{path.name}.lock"


_mine = threading.local()          # the project files whose lock this thread holds, so taking one again nests


@contextmanager
def project_lock(path: Path | str, wait: float | None = None) -> Iterator[None]:
    """Hold a project file's lock. The window, the command line and the MCP server each take it to read, change
    and write a project file, so none of them overwrites another's change. It is an operating-system lock on a
    file in the project's .dancr folder, so it goes away with the process however the process ends. A thread
    that already holds it (a save inside an edit) just carries on; other threads of the same process wait."""
    path = Path(path).expanduser().resolve()
    held = _mine.__dict__.setdefault("paths", set())
    if path in held:
        yield
        return
    held.add(path)
    try:
        with _os_lock(path, wait):
            yield
    finally:
        held.discard(path)


@contextmanager
def _os_lock(path: Path, wait: float | None) -> Iterator[None]:
    lock = lock_file(path)
    lock.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        deadline = time.monotonic() + (LOCK_WAIT if wait is None else wait)
        while not _try_lock(fd):
            if time.monotonic() >= deadline:
                raise ProjectBusy(f"{path.name} is being changed by another program (the DANCR window, the command "
                                  "line or an agent). Try again in a moment.")
            time.sleep(0.05)
        try:
            yield
        finally:
            _unlock(fd)
    finally:
        os.close(fd)


def read_project(path: Path) -> tuple[Pipeline, str]:
    """A project file and the exact text it was read from."""
    path = Path(path).expanduser().resolve()
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise PipelineError(f"{path.name} is not valid JSON: {e}") from e
    return Pipeline.from_dict(data, path), text


@contextmanager
def editing(path: Path | str) -> Iterator[Pipeline]:
    """Load a project file, let the caller change it, and save it if anything changed, holding the file's lock
    throughout. A program that does not take the lock (a text editor) may still change the file meanwhile:
    then nothing is saved and the caller is told to try again."""
    path = Path(path).expanduser().resolve()
    with project_lock(path):
        p, text = read_project(path)
        before = p.dumps()                  # text, not the dict: the dict shares its parts with the project
        yield p
        if p.dumps() == before:
            return
        if path.read_text(encoding="utf-8") != text:
            raise ProjectBusy(f"{path.name} was changed by another program while this was running, so nothing was "
                              "saved. Try again.")
        p.save(expected_text=text)          # re-checked immediately before the replace, closing the last window


def editing_deferred(path: Path | str, compute: Any) -> Any:
    """Like :func:`editing`, but runs ``compute(p)`` — which may take minutes (a model turn, an executor build
    over the whole data) — with the project file's lock *released*, then takes the lock only briefly to save.

    This is why a long agent call never blocks the window's autosave or another tool: the file is locked only
    for the read and the final write. If the file changed on disk while computing, nothing is saved and
    ProjectBusy is raised (the caller tries again), exactly as :func:`editing` does. Returns compute's result."""
    path = Path(path).expanduser().resolve()
    p, text = read_project(path)
    before = p.dumps()
    result = compute(p)
    with project_lock(path):
        if path.read_text(encoding="utf-8") != text:
            raise ProjectBusy(f"{path.name} was changed while this was running, so nothing was saved. Try again.")
        if p.dumps() != before:
            p.save(expected_text=text)
    return result


# ----------------------------------------------------------------- answers (shared by `dancr ask/suggest/answer` and MCP)
def inspect_file(path: str | Path, rows: int = 5) -> dict[str, Any]:
    """Columns, types and the first rows of any file DANCR can read — a table, or a bio format read by its own
    step. Relative paths are taken from the current folder."""
    from ..core.registry import Ctx
    from ..core.nodes.load import scan_file
    from ..core.nodes.bio import bio_node_for
    from ..core.nodes.document import doc_node_for
    fp = Path(str(path)).expanduser()
    if not fp.is_absolute():
        fp = Path.cwd() / fp
    fp = fp.resolve()
    ctx = Ctx(fp.parent, "inspect", "inspect", preview=True)
    special = doc_node_for(fp) or bio_node_for(fp)
    if special:
        res = registry.get(special).apply(ctx, {}, {"path": str(fp)})
        lf = getattr(res, "frame", res)
        messages = list(getattr(res, "messages", []) or [])
    else:
        lf, messages, _ = scan_file(ctx, {"path": str(fp), "has_header": True, "parse_dates": True})
    schema = lf.collect_schema()
    head = lf.head(max(1, min(int(rows), 100))).collect(engine="streaming")
    return {"columns": [{"name": k, "dtype": str(v)} for k, v in schema.items()],
            "messages": messages, "head": json.loads(head.write_json())}


def read_document(path: str | Path, *, what: str = "blocks", tier: str | None = None, pages: str | None = None,
                  allow_remote: bool = False, rows: int = 200, output_root: str | Path | None = None) -> dict[str, Any]:
    """Read a document through the 'Load document' step and return its rows: blocks (doc/page/block/type/text/
    locator) or a catalog of extracted tables. Tables are written next to the document (inside ``output_root``)."""
    from ..core.registry import Ctx
    fp = Path(str(path)).expanduser()
    if not fp.is_absolute():
        fp = Path.cwd() / fp
    fp = fp.resolve()
    params: dict[str, Any] = {"path": str(fp), "what": what, "allow_remote": allow_remote}
    if tier:
        params["tier"] = tier
    if pages:
        params["pages"] = pages
    ctx = Ctx(fp.parent, "read_document", "read_document",
              output_root=Path(output_root).resolve() if output_root else fp.parent)
    res = registry.get("load_document").apply(ctx, {}, params)
    lf = getattr(res, "frame", res)
    df = lf.head(max(1, int(rows))).collect(engine="streaming")
    return {"columns": [{"name": k, "dtype": str(v)} for k, v in df.schema.items()],
            "rows": json.loads(df.write_json()), "messages": list(getattr(res, "messages", []) or []),
            "report": getattr(res, "report", {}) or {}}


def add_files(p: Pipeline, files: list[str] | None) -> list[str]:
    """A load step for each file not loaded yet (a file already loaded keeps its step and its settings).
    Returns the load steps' ids, in the order given."""
    from ..core.registry import resolve_path
    out = []
    for f in files or []:
        target = resolve_path(p.directory, str(f)).resolve()
        if not target.exists():
            raise ValueError(f"File not found: {target}")
        from ..core.nodes.load import tables_in
        from ..core.nodes.bio import bio_node_for
        from ..core.nodes.document import doc_node_for
        try:
            rel = str(target.relative_to(p.directory.resolve()))
        except ValueError:
            rel = str(target)
        bio_type = doc_node_for(target) or bio_node_for(target)
        if bio_type:
            existing = next((nid for nid, n in p.nodes.items() if n.type == bio_type and n.params.get("path")
                             and resolve_path(p.directory, str(n.params["path"])).resolve() == target), None)
            if existing is None:
                existing = add_step(p, bio_type, {"path": rel}, title=target.stem).id
            out.append(existing)
            continue
        def part(n: dict[str, Any]) -> tuple[str, str, str]:
            # which table of a multi-table file a load step reads: a workbook sheet, a table on a sheet, a layer
            return (str(n.get("sheet") or ""), str(n.get("table") or ""), str(n.get("layer") or ""))

        for title, extra in tables_in(target):
            wanted = part(extra)
            existing = next((nid for nid, n in p.nodes.items() if n.type == "load_file" and n.params.get("path")
                             and resolve_path(p.directory, str(n.params["path"])).resolve() == target
                             and part(n.params) == wanted), None)
            if existing is None:
                existing = add_step(p, "load_file", {"path": rel, **extra}, title=title).id
            out.append(existing)
    return out


def data_model(p: Pipeline, deep: bool = True):
    from ..core.answers import model_for
    return model_for(p, Executor(p), deep=deep)


def suggestions(p: Pipeline, focus: str | None = None) -> list[dict[str, Any]]:
    from ..core.recipes import suggest
    from ..core.answers import model_for
    from ..core.understand import default_tables
    if focus:
        require_node(p, focus)
    nodes = default_tables(p) + ([focus] if focus and focus not in default_tables(p) else [])
    m = model_for(p, Executor(p), nodes=nodes)
    return [{"index": i, **s.to_dict()} for i, s in enumerate(suggest(m, focus))]


def build_answer(p: Pipeline, spec: dict[str, Any], answer_id: str | None = None) -> dict[str, Any]:
    """Build (or rebuild) an answer from a spec; the caller saves the project."""
    from ..core import answers
    m = answers.model_for(p, Executor(p), nodes=answers.tables_for(p, spec))
    a, plan = answers.build(p, m, spec, answer_id=answer_id)
    return {**answers.describe(p, a), "chips": plan.chips, "why": plan.why, "sentence": plan.sentence(),
            "set_aside": plan.set_aside, "note": answers.set_aside_note(plan.set_aside)}


def ask_question(p: Pipeline, text: str, build: bool = True) -> dict[str, Any]:
    """Read a question; with ``build`` add its answer to the project. ``ok`` is False (with a message and any
    "did you mean" hints) when the question could not be read. Words the project has learned to read
    (``core.memory``) are applied, and a spelling repair is remembered for next time."""
    from ..core import memory
    from ..core.ask import ask
    from ..core.answers import model_for
    m = model_for(p, Executor(p))
    asked = ask(m, text, aliases=memory.aliases(p))
    if asked.corrected:
        memory.remember_corrections(p, asked.corrected)
    memory.remember_question(p, text)
    out: dict[str, Any] = {"question": asked.to_dict()}
    if asked.ok and build:
        out["answer"] = build_answer(p, asked.spec)
    return out


def assistant_turn(p: Pipeline, text: str, *, allow_samples: bool = False, focus: str | None = None,
                   build: bool = False, provider: Any = None, settings: Any = None,
                   output_root: Path | str | None = None, check_output: Any = None) -> dict[str, Any]:
    """One Assistant turn against the project, headless. Runs the model over a snapshot and the engine's tools;
    the model only proposes. With ``build``, applies the proposal (its steps and an Answer) and runs it.

    The key and endpoint come from the environment (``ModelSettings.from_env``) unless ``settings`` is given;
    ``provider`` overrides the client (tests, or the scripted fake)."""
    from ..core.assistant import AssistantSession, load_thread
    from ..core.assistant.client import ModelSettings, OpenAIProvider, provider_for
    from ..core.assistant.store import save_thread
    from ..core import answers
    settings = settings or ModelSettings.from_env()
    ex = Executor(p)
    model = answers.model_for(p, ex)
    session = AssistantSession(ex.pipeline, ex, model, provider_for(settings, provider), settings,
                               allow_samples=allow_samples, focus=focus, thread=load_thread(p), run=True)
    reply = session.turn(text)
    session.record(text, reply)
    save_thread(p, session.thread)
    out: dict[str, Any] = {"kind": reply.kind, "text": reply.text, "proposal": reply.proposal,
                           "flags": reply.flags, "unverified": reply.unverified, "usage": reply.usage,
                           "tool_calls": reply.tool_calls}
    if isinstance(session.provider, OpenAIProvider) and settings.configured:
        # no window here, so the egress is stated rather than consented; the caller configured the key
        out["sent_to"] = {"endpoint": settings.base_url, "model": settings.model}
        import logging
        logging.getLogger("dancr.headless").info(
            "assistant turn sent a project profile to %s (%s)", settings.base_url, settings.model)
    if reply.kind in ("error", "paused"):
        out["error"] = reply.text or reply.error
        return out
    prop = reply.proposal or {}
    if not build or prop.get("kind") not in ("answer", "steps", "edits"):
        return out
    if prop.get("kind") == "edits":
        from ..core.assistant.edits import apply_edits
        edits = prop.get("edits") or []
        out["applied_edits"] = apply_edits(p, edits)
        if check_output is not None:
            for e in edits:                       # a repointed sink must not escape, just as a built step may not
                nid = e.get("node") if isinstance(e, dict) else None
                if nid and nid in p.nodes:
                    check_output(p, nid)
        return out
    if prop.get("kind") == "answer":
        a, _plan = answers.build(p, model, prop["spec"])
        terminal = a.terminal
        out["answer"] = answers.describe(p, a)
    else:
        terminal = None
        for s in prop.get("steps") or []:
            node = add_step(p, s["type"], s.get("params"), s.get("title"), None, s.get("after"), s.get("port"))
            if check_output is not None:          # a proposed sink step may not save outside the project folder
                check_output(p, node.id)
            terminal = node.id
    if not terminal:
        return out
    st = Executor(p, output_root=output_root).run(targets=[terminal])[terminal]
    finding = (st.report or {}).get("finding", {}).get("statement", "")
    out.update({"terminal": terminal, "status": st.status, "finding": finding, "error": st.error})
    last = session.thread.turns[-1] if session.thread.turns else None
    if last is not None and last.role == "assistant":     # fold the outcome in, so a reload shows it built
        last.node = terminal
        if isinstance(out.get("answer"), dict):
            last.answer = out["answer"].get("id")
        if finding:
            last.finding = finding
        save_thread(p, session.thread)
    return out


def connection_map(p: Pipeline, *, recompute: bool = False) -> dict[str, Any]:
    """How the project's tables relate (the engine's own links/stacks/alignments). Returns the map saved in the
    project, or computes and saves one when asked (``recompute``) or when none is saved yet."""
    from ..core.assistant import load_thread
    from ..core.assistant.store import save_thread
    thread = load_thread(p)
    if thread.connections and not recompute:
        return {"connections": thread.connections, "count": len(thread.connections), "source": "saved"}
    from ..core.answers import model_for
    from ..core.assistant.tools import ToolRunner
    ex = Executor(p)
    m = model_for(p, ex)
    out = ToolRunner(ex.pipeline, ex, m).call("list_connections", {}).content
    thread.connections = out.get("connections") or []
    save_thread(p, thread)
    return {**out, "source": "computed"}


def change_answer(p: Pipeline, answer_id: str, key: str | None = None, value: Any = None,
                  assumption: int | None = None, choice: int = 0) -> dict[str, Any]:
    """Change an answer by one chip (``key`` = ``value``) or by choosing an alternative of one of its
    assumptions (``assumption`` index, ``choice`` index), and rebuild it in place."""
    from ..core.recipes import apply_choice
    a = p.answer(answer_id)
    if a is None:
        raise ValueError(f"No answer called {answer_id!r}. Answers: {[x.id for x in p.answers]}")
    if assumption is not None:
        if not 0 <= assumption < len(a.assumptions):
            raise ValueError(f"The answer has {len(a.assumptions)} assumptions (numbered from 0)")
        choices = a.assumptions[assumption].get("choices") or []
        if not 0 <= choice < len(choices):
            raise ValueError(f"That assumption has {len(choices)} alternatives (numbered from 0)")
        spec = apply_choice(a.spec, "set", choices[choice]["set"])
    elif key:
        _check_choice(p, a, key, value)
        spec = apply_choice(a.spec, key, value)
    else:
        raise ValueError("Say what to change: a chip (key and value) or an assumption's alternative")
    return build_answer(p, spec, answer_id)


def _check_choice(p: Pipeline, a, key: str, value: Any) -> None:
    """A chip may only be set to one of its choices (what the window offers), so a typo never builds a broken answer."""
    from ..core.answers import model_for, tables_for
    from ..core.recipes import chips
    m = model_for(p, Executor(p), deep=False, nodes=tables_for(p, a.spec))
    cs = {c["key"]: c for c in chips(m, a.spec)}
    if key not in cs:
        raise ValueError(f"This answer has no choice called {key!r}. Choices: {', '.join(cs) or 'none'}")
    allowed = [ch["value"] for ch in cs[key]["choices"]]
    if value not in allowed:
        shown = ", ".join(json.dumps(v) for v in allowed[:12])
        raise ValueError(f"{key} cannot be {json.dumps(value)}. Choose one of: {shown}")


