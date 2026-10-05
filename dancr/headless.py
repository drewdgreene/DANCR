"""What the command line and the MCP server share: finding steps, adding them, reporting them, charting and
reading results. Both front ends call these, so a step is reported the same way wherever it is asked for."""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import polars as pl

from .core import Pipeline, PipelineError, registry
from .core.registry import in_dancr_folder
from .core.dtypes import json_safe
from .core.executor import Executor, NodeState, CODE_FINGERPRINT
from .core.model import Node, FORMAT_VERSION


class StepFailed(ValueError):
    """A step asked for could not be computed (the CLI exits 1, as for a failed run, not 2 for a usage error)."""


def write_text_atomic(path: Path | str, text: str, encoding: str = "utf-8") -> Path:
    """Write text to ``path`` atomically: a unique temp file beside it, then one replace. A crash or a full disk
    mid-write leaves the existing file untouched, never a half-written one (the project file and every exported
    table already write this way)."""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        tmp.write_text(text, encoding=encoding)
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)
    return target


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
    """How a run is reported by both front ends; `nodes` is keyed by step id. ``findings`` are what the steps
    said (a sentence each), most interesting first, and ``headline`` is the best of them."""
    from .core.findings import collect, record as finding_record, headline as finding_headline
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


def _batch_source(p: Pipeline, loader: str | None) -> tuple[str, str]:
    """The source step whose file setting a batch rebinds, and the name of that setting."""
    from .core.registry import registry
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
    from .core.registry import in_dancr_folder
    from .core.model import Pipeline as _Pipeline, portable_path
    from .core.nodes.outputs import write_table
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

    def one(f: Path) -> tuple[dict[str, Any], tuple[Path, str] | None]:
        clone = _Pipeline.from_dict(copy.deepcopy(p.to_dict()), p.path)
        clone.set_params(loader_id, **{path_param: portable_path(f, folder)})
        ex = Executor(clone)
        st = ex.run(targets=[target], force=force, sweep=False).get(target)
        rec: dict[str, Any] = {"file": str(f), "name": f.name, "status": (st.status if st else "idle")}
        if st is not None and st.status == "done" and st.output:
            out_file = destination / f"{f.stem}.{ext}"
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
        from .core.fair import dump as _dump_json
        write_text_atomic(manifest, _dump_json(record))
        record["manifest"] = str(manifest)
    return record


def _watch_state(p: Pipeline) -> dict[str, list[int] | None]:
    """A snapshot of everything that should trigger a rerun: the project file and every source file (folder
    members included). A file that is gone is None, so deleting or moving one is a change too."""
    state: dict[str, list[int] | None] = {}

    def stamp(path: Path) -> list[int] | None:
        try:
            st = path.stat()
            return [st.st_size, st.st_mtime_ns]
        except OSError:
            return None

    if p.path is not None:
        state[str(p.path)] = stamp(p.path)
    for f in source_files(p):
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
    emit({"type": "watch_started", "path": str(path)})
    rec = run_once(p)
    emit({"type": "watch_ran", "record": rec, "why": "start"})
    if once:
        return record

    previous = _watch_state(p)
    pending = False
    changed: list[str] = []
    while not stop.wait(max(0.1, float(interval))):
        try:
            p = Pipeline.load(path)                     # reload each poll: the project file itself may have changed
        except Exception as e:  # noqa: BLE001
            emit({"type": "watch_error", "error": str(e)})
            continue
        snap = _watch_state(p)
        if snap == previous:
            if pending:                                 # changed, then a quiet poll: act on it now
                pending = False
                try:
                    rec = run_once(p)
                    emit({"type": "watch_ran", "record": rec, "why": "changed", "files": changed})
                    previous = _watch_state(p)          # our own result files must not look like a change again
                except Exception as e:  # noqa: BLE001
                    record["ok"] = False
                    emit({"type": "watch_error", "error": str(e)})
        else:
            changed = sorted({k for k in set(snap) | set(previous) if snap.get(k) != previous.get(k)})
            record["changed"] = changed
            emit({"type": "watch_changed", "files": changed})
            pending = True
            previous = snap
    record["stopped"] = True
    return record


def registry_get(p: Pipeline, node_id: str):
    from .core.registry import registry
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


def _path_settings(p: Pipeline, node_id: str) -> list[Path]:
    """The files named by a step's path settings, resolved (symlinks and `..` included)."""
    node = p.nodes[node_id]
    nt = registry.get(node.type)
    out = []
    for prm in nt.params:
        v = node.params.get(prm.name)
        if prm.kind in ("path", "dir") and isinstance(v, str) and v.strip():
            target = Path(v).expanduser()
            out.append((target if target.is_absolute() else p.directory / target).resolve())
    if node.type == "report" and node.params.get("pdf", True):
        out += [t.with_suffix(".pdf") for t in out]           # the report's PDF sits next to it
    return out


def source_files(p: Pipeline) -> set[Path]:
    """Every file a source step of the project reads, including the members of a folder source (so an export or
    a batch never overwrites a file the project reads from)."""
    out: set[Path] = set()
    for nid, n in p.nodes.items():
        nt = registry.get(n.type)
        if nt.kind != "source":
            continue
        out |= set(_path_settings(p, nid))
        if nt.source_files is not None:
            try:
                out |= {Path(f).resolve() for f in nt.source_files(p.directory, n.params)}
            except Exception:  # noqa: BLE001 - an unreadable folder simply contributes nothing
                pass
    return out


def output_files(p: Pipeline, node_id: str) -> list[Path]:
    """The files a step saves (nothing for steps that do not save files)."""
    return _path_settings(p, node_id) if registry.get(p.nodes[node_id].type).kind == "sink" else []


def _same_file(a: Path, b: Path) -> bool:
    """One file under two spellings too: A.CSV and a.csv on a Mac or Windows disk, or a hard link."""
    if a == b:
        return True
    try:
        return os.path.samefile(a, b)
    except OSError:                        # either does not exist yet: then they are not the same file
        return False


def unsafe_write(p: Pipeline, target: Path, folder: Path) -> str | None:
    """Why writing `target` is refused (outside `folder`, or over a data file the project reads), else None."""
    target = target.resolve()
    if not target.is_relative_to(folder):
        return f"Can only save inside the project folder {folder}, not {target}"
    if in_dancr_folder(target, folder):
        return f"Won't save into DANCR's own .dancr folder: {target}"
    if any(_same_file(target, src) for src in source_files(p)):
        return f"Won't save over {target}, because the project reads its data from that file"
    return None


def unsafe_outputs(p: Pipeline, node_id: str, folder: Path) -> list[str]:
    """Why a step's files may not be written: each lands outside `folder` or over a file the project reads."""
    return [why for t in output_files(p, node_id) if (why := unsafe_write(p, t, folder))]


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
        if e.errno in busy:
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
        p.save()


# ----------------------------------------------------------------- answers (shared by `dancr ask/suggest/answer` and MCP)
def add_files(p: Pipeline, files: list[str] | None) -> list[str]:
    """A load step for each file not loaded yet (a file already loaded keeps its step and its settings).
    Returns the load steps' ids, in the order given."""
    from .core.registry import resolve_path
    out = []
    for f in files or []:
        target = resolve_path(p.directory, str(f)).resolve()
        if not target.exists():
            raise ValueError(f"File not found: {target}")
        from .core.nodes.load import tables_in
        try:
            rel = str(target.relative_to(p.directory.resolve()))
        except ValueError:
            rel = str(target)
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
    from .core.answers import model_for
    return model_for(p, Executor(p), deep=deep)


def suggestions(p: Pipeline, focus: str | None = None) -> list[dict[str, Any]]:
    from .core.recipes import suggest
    from .core.answers import model_for
    from .core.understand import default_tables
    if focus:
        require_node(p, focus)
    nodes = default_tables(p) + ([focus] if focus and focus not in default_tables(p) else [])
    m = model_for(p, Executor(p), nodes=nodes)
    return [{"index": i, **s.to_dict()} for i, s in enumerate(suggest(m, focus))]


def build_answer(p: Pipeline, spec: dict[str, Any], answer_id: str | None = None) -> dict[str, Any]:
    """Build (or rebuild) an answer from a spec; the caller saves the project."""
    from .core import answers
    m = answers.model_for(p, Executor(p), nodes=answers.tables_for(p, spec))
    a, plan = answers.build(p, m, spec, answer_id=answer_id)
    return {**answers.describe(p, a), "chips": plan.chips, "why": plan.why, "sentence": plan.sentence(),
            "set_aside": plan.set_aside, "note": answers.set_aside_note(plan.set_aside)}


def ask_question(p: Pipeline, text: str, build: bool = True) -> dict[str, Any]:
    """Read a question; with ``build`` add its answer to the project. ``ok`` is False (with a message and any
    "did you mean" hints) when the question could not be read. Words the project has learned to read
    (``core.memory``) are applied, and a spelling repair is remembered for next time."""
    from .core import memory
    from .core.ask import ask
    from .core.answers import model_for
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
                   build: bool = False, provider: Any = None, settings: Any = None) -> dict[str, Any]:
    """One Assistant turn against the project, headless. Runs the model over a snapshot and the engine's tools;
    the model only proposes. With ``build``, applies the proposal (its steps and an Answer) and runs it.

    The key and endpoint come from the environment (``ModelSettings.from_env``) unless ``settings`` is given;
    ``provider`` overrides the client (tests, or the scripted fake)."""
    from .core.assistant import AssistantSession, load_thread
    from .core.assistant.client import ModelSettings, provider_for
    from .core.assistant.store import save_thread
    from .core import answers
    settings = settings or ModelSettings.from_env()
    ex = Executor(p)
    model = answers.model_for(p, ex)
    session = AssistantSession(ex.pipeline, ex, model, provider_for(settings, provider), settings,
                               allow_samples=allow_samples, focus=focus, thread=load_thread(p), run=True)
    reply = session.turn(text)
    session.record(text, reply)
    save_thread(p, session.thread)
    out: dict[str, Any] = {"kind": reply.kind, "text": reply.text, "proposal": reply.proposal,
                           "flags": reply.flags, "usage": reply.usage, "tool_calls": reply.tool_calls}
    if reply.kind in ("error", "paused"):
        out["error"] = reply.text or reply.error
        return out
    prop = reply.proposal or {}
    if not build or prop.get("kind") not in ("answer", "steps"):
        return out
    if prop.get("kind") == "answer":
        a, _plan = answers.build(p, model, prop["spec"])
        terminal = a.terminal
        out["answer"] = answers.describe(p, a)
    else:
        terminal = None
        for s in prop.get("steps") or []:
            node = add_step(p, s["type"], s.get("params"), s.get("title"), None, s.get("after"), s.get("port"))
            terminal = node.id
    if not terminal:
        return out
    st = Executor(p).run(targets=[terminal])[terminal]
    out.update({"terminal": terminal, "status": st.status,
                "finding": (st.report or {}).get("finding", {}).get("statement", ""), "error": st.error})
    return out


def connection_map(p: Pipeline, *, recompute: bool = False) -> dict[str, Any]:
    """How the project's tables relate (the engine's own links/stacks/alignments). Returns the map saved in the
    project, or computes and saves one when asked (``recompute``) or when none is saved yet."""
    from .core.assistant import load_thread
    from .core.assistant.store import save_thread
    thread = load_thread(p)
    if thread.connections and not recompute:
        return {"connections": thread.connections, "count": len(thread.connections), "source": "saved"}
    from .core.answers import model_for
    from .core.assistant.tools import ToolRunner
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
    from .core.recipes import apply_choice
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
    from .core.answers import model_for, tables_for
    from .core.recipes import chips
    m = model_for(p, Executor(p), deep=False, nodes=tables_for(p, a.spec))
    cs = {c["key"]: c for c in chips(m, a.spec)}
    if key not in cs:
        raise ValueError(f"This answer has no choice called {key!r}. Choices: {', '.join(cs) or 'none'}")
    allowed = [ch["value"] for ch in cs[key]["choices"]]
    if value not in allowed:
        shown = ", ".join(json.dumps(v) for v in allowed[:12])
        raise ValueError(f"{key} cannot be {json.dumps(value)}. Choose one of: {shown}")


# ------------------------------------------------ knowledge-base context (schema, stats, samples, cards)
CONTEXT_VERSION = 2          # the context document's own format, independent of the pipeline format
CONTEXT_MAX_SAMPLE_ROWS = 100
CONTEXT_DOC = "dancr.table"


def engine_version() -> str:
    """The DANCR version that built or read something (the context document, a run manifest)."""
    from . import __version__
    return __version__


def _dataset_meta(p: Pipeline) -> dict[str, Any]:
    """The project's dataset-level metadata (creator, license, description…), kept in ``meta["dataset"]`` so an
    older DANCR opening the project preserves it (unknown top-level keys are dropped, unknown meta keys are not)."""
    meta = p.meta.get("dataset") if isinstance(getattr(p, "meta", None), dict) else None
    return dict(meta) if isinstance(meta, dict) else {}


def build_context(p: Pipeline, executor: Executor | None = None, *, nodes: list[str] | None = None,
                  deep: bool = True, stats: bool = False, samples: bool = False, sample_rows: int = 10,
                  run: bool | None = None) -> dict[str, Any]:
    """One document describing the project's datasets for a knowledge base to index: the compact profile
    (schemas, roles, relations) plus, optionally, per-column statistics and capped sample rows, and a prose
    *doc card* per table that a search index can match on. Nothing is changed; only the cache may be written
    (when ``run`` lets a table be computed for real statistics).

    ``stats``/``samples`` need a table's result; a source that has not run yet is computed when ``run`` is left
    unset (defaults on for either), and statistics are skipped, with a note, when ``run=False``. Samples fall
    back to a preview of the first rows for a table that has not been computed. The Assistant's own profile is
    a subset of this document (see :mod:`dancr.core.profile`)."""
    from .core.answers import model_for
    from .core.profile import clean, project_profile, table_card
    from .core.dtypes import json_safe
    ex = executor if executor is not None else Executor(p)
    model = model_for(p, ex, deep=deep, nodes=nodes)
    do_run = bool(stats or samples) if run is None else bool(run)
    node = nodes[0] if nodes and len(nodes) == 1 else None
    held: dict[str, str | None] = {}
    for nid in model.tables:
        try:
            held[nid] = ex.safe_hash(nid)
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
        return json_safe(out)
    finally:
        ex.release()


def export_fair(p: Pipeline, executor: Executor | None = None, *, fmt: str = "schema.org",
                nodes: list[str] | None = None, deep: bool = True, samples: bool = False,
                sample_rows: int = 10, out: Path | str | None = None, path_root: str = "",
                run: bool | None = None) -> dict[str, Any]:
    """A FAIR descriptor for the project's datasets: ``schema.org`` (JSON-LD), ``frictionless`` (Data Package),
    ``manifest`` (what produced the results) or ``rocrate`` (a metadata graph). Built from the context document
    and the project's dataset metadata; ``out`` writes it atomically. Nothing is changed but the cache."""
    from .core import fair
    ex = executor if executor is not None else Executor(p)
    ctx = build_context(p, ex, nodes=nodes, deep=deep, stats=fmt in ("manifest", "rocrate"),
                        samples=samples, sample_rows=sample_rows, run=run)
    meta = p.dataset_meta()
    manifest = fair.run_manifest(p, ex, meta=meta) if fmt.strip().lower().replace("_", ".") in (
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
    import os
    import shutil
    import zipfile
    from .core import fair
    ex = executor if executor is not None else Executor(p)
    folder = p.directory.resolve()
    meta = dict(meta if meta is not None else p.dataset_meta())
    mode = (copy or "metadata").lower()
    if mode not in ROCRATE_COPY:
        raise ValueError(f"copy must be one of: {', '.join(ROCRATE_COPY)}")
    ctx = build_context(p, ex, stats=False, samples=samples)
    man = manifest if manifest is not None else fair.run_manifest(p, ex, meta=meta)

    destination = Path(out).expanduser()
    destination = (destination if destination.is_absolute() else folder / destination).resolve()
    if not destination.is_relative_to(folder):
        raise ValueError(f"A crate must be written inside the project folder {folder}, not {destination}")
    if in_dancr_folder(destination, folder):
        raise ValueError(f"Won't write a crate into DANCR's own .dancr folder: {destination}")

    descriptors: dict[str, str] = {
        "dancr-pipeline.json": p.dumps(),
        "datapackage.json": fair.dump(fair.datapackage(ctx, meta)),
        "dataset.jsonld": fair.dump(fair.dataset_jsonld(ctx, meta)),
        "dancr-manifest.json": fair.dump(man),
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
            "context.jsonl": "Knowledge-base context"}.get(name, name)


def _table_stats(ex: Executor, node_id: str) -> list[dict[str, Any]] | None:
    """Per-column summary statistics (count, missing, mean, std, min, quartiles, max) of a computed table, or
    None when it has not been computed (statistics on a preview would be misleading)."""
    st = ex.state(node_id)
    if st.status != "done" or not st.output:
        return None
    from .core.dtypes import json_safe
    from .views.stats import column_summary
    return json_safe(column_summary(pl.scan_parquet(st.output)).to_dicts())


def _table_sample(ex: Executor, node_id: str, rows: int) -> tuple[list[dict[str, Any]], str]:
    """Up to ``rows`` rows of a table and where they came from: "cache" (a spread/head of a computed result)
    or "preview" (the first rows of a table not computed yet)."""
    from .core.dtypes import json_safe
    n = max(1, min(int(rows), CONTEXT_MAX_SAMPLE_ROWS))
    lf, kind = ex.sample_frame(node_id, n)
    return json_safe(lf.head(n).collect(engine="streaming").to_dicts()), ("cache" if kind in ("all", "spread") else "preview")


def context_jsonl(ctx: dict[str, Any]) -> str:
    """A context document as JSON Lines, one line per dataset — the unit a RAG index ingests. Each line carries
    the engine version, when it was generated and the dataset's ``content_hash``, so an index can spot a stale
    document and refresh only what changed (see ``context_changes``)."""
    from .core.dtypes import json_safe
    lines = []
    for d in ctx.get("documents", []):
        line = {"kind": CONTEXT_DOC, "version": ctx.get("version", CONTEXT_VERSION),
                "project": (ctx.get("project") or {}).get("name"),
                "engine_version": ctx.get("engine_version"),
                "generated_at": ctx.get("generated_at"), **d}
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
                  samples: bool = False, fair_format: str | None = None, jobs: int = 1) -> dict[str, Any]:
    """One document describing every DANCR project under ``root``: each project's datasets (the context export)
    and, optionally, a FAIR descriptor each. A broken project is listed under ``skipped`` with the reason, never
    raised. Statistics and samples need a table's result and compute it (each project's own cache)."""
    from datetime import datetime
    from .core import fair
    root = Path(root).expanduser().resolve()
    files = find_pipelines(root, pattern, recursive)

    def one(f: Path) -> tuple[dict[str, Any] | None, str | None, str | None]:
        try:
            p = Pipeline.load(f)
        except Exception as e:  # noqa: BLE001 - a broken project is reported, not fatal
            return None, str(f), f"cannot read: {e}"
        try:
            ex = Executor(p)
            ctx = build_context(p, ex, stats=stats, samples=samples)
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
    from .core.dtypes import json_safe
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
    if ctx.get("inputs"):
        lines += ["", "Inputs: " + ", ".join(f"{i['name']}={i['value']}" for i in ctx["inputs"])]
    return "\n".join(lines)
