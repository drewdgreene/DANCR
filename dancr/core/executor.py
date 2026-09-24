"""Runs a pipeline, materializing each node's output to Parquet in a cache.

Design rules:
- ``run()`` works on a frozen snapshot of the pipeline, never the live object.
- Plan hashes are recomputed per call (memo lives only inside one call).
- Cache writers use a unique temp file, validate it, then publish atomically. If
  another writer published the same hash first, theirs wins.
- Garbage collection only touches nodes that were just run and never fails a run.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import polars as pl

from .model import Pipeline, PipelineError
from .registry import registry, Ctx, NodeResult, NodeType, resolve_path
from .dtypes import json_safe

log = logging.getLogger("dancr.executor")

IMPL_VERSION = "4"     # bump to invalidate every cache


# engine modules that never shape a step's output (help text, reading tables and planning answers, sample
# data): editing them must not throw every cached result away
NOT_SEMANTIC = {"examples.py", "understand.py", "recipes.py", "ask.py", "answers.py", "planner.py", "samples.py"}


def _version_of(package: str) -> str:
    from importlib.metadata import version, PackageNotFoundError
    try:
        return version(package)
    except PackageNotFoundError:
        return "none"


def _code_fingerprint() -> str:
    """Hash of the code that can shape a step's output, so cached outputs are invalidated when it changes.

    Every engine module and every renderer (report/workbook/export steps write charts and tables),
    plus the Polars version. Listing files by hand is how a module gets forgotten, so nothing is listed."""
    import numpy
    here = Path(__file__).parent
    h = hashlib.sha1(IMPL_VERSION.encode())
    for lib in (pl.__version__, numpy.__version__, _version_of("fastexcel")):   # the libraries that compute and read
        h.update(lib.encode())
    semantic = [f for f in [*here.rglob("*.py"), *(here.parent / "views").rglob("*.py")] if f.name not in NOT_SEMANTIC]
    for f in sorted(semantic):
        try:
            h.update(f.relative_to(here.parent).as_posix().encode())
            h.update(f.read_bytes())
        except OSError:
            pass
    return h.hexdigest()[:12]


CODE_FINGERPRINT = _code_fingerprint()
_PROCESS_ID = uuid.uuid4().hex[:8]


@dataclass
class NodeState:
    node_id: str
    status: str = "idle"            # idle | stale | running | done | failed
    rows: int | None = None
    columns: list[dict[str, str]] = field(default_factory=list)
    elapsed: float | None = None
    error: str | None = None
    messages: list[str] = field(default_factory=list)
    report: dict[str, Any] = field(default_factory=dict)
    output: str | None = None       # parquet path
    hash: str | None = None
    finished_at: str | None = None
    from_cache: bool = False
    column_stats: dict[str, dict[str, Any]] = field(default_factory=dict)   # name -> {nulls, min, max, mean}
    files: dict[str, list[int]] = field(default_factory=dict)   # files a step wrote -> [size, mtime_ns] when written

    def to_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))

    @property
    def schema_names(self) -> list[str]:
        return [c["name"] for c in self.columns]


EventFn = Callable[[dict[str, Any]], None]


class ExecutionCancelled(Exception):
    pass


class PreviewUnavailable(PipelineError):
    """A node could not be computed on a *sample* of its inputs.

    Previews are best-effort, so this is an expected outcome (too few rows, a column absent from
    the sample, an unreadable upstream file), not a bug: callers show the message instead of a
    traceback, and the real run may still succeed on the full data."""


def default_cache_dir(pipeline: Pipeline) -> Path:
    if pipeline.path is not None:
        return (pipeline.path.parent / ".dancr" / "cache" / pipeline.path.stem).resolve()
    from ..logsetup import untitled_cache_root
    return untitled_cache_root() / f"untitled-{_PROCESS_ID}"


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        return _pid_alive_windows(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _pid_alive_windows(pid: int) -> bool:
    """On Windows ``os.kill(pid, 0)`` sends CTRL_C_EVENT instead of probing, so ask the kernel."""
    import ctypes
    from ctypes import wintypes
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    STILL_ACTIVE = 259
    ERROR_ACCESS_DENIED = 5
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    h = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not h:
        return ctypes.get_last_error() == ERROR_ACCESS_DENIED     # exists, owned by someone else
    try:
        code = wintypes.DWORD()
        if not k32.GetExitCodeProcess(h, ctypes.byref(code)):
            return True
        return code.value == STILL_ACTIVE
    finally:
        k32.CloseHandle(h)


def sweep_untitled_caches() -> int:
    """Delete result folders of unsaved projects whose DANCR process is gone. Returns how many were removed."""
    from ..logsetup import untitled_cache_root
    root = untitled_cache_root()
    if not root.exists():
        return 0
    removed = 0
    for d in root.iterdir():
        if not d.is_dir():
            continue
        try:
            pid = int((d / "owner.pid").read_text().strip())
            alive = _pid_alive(pid)
        except (OSError, ValueError):
            alive = False
        if not alive:
            shutil.rmtree(d, ignore_errors=True)
            removed += 1
    return removed


class CacheFull(OSError):
    """The disk that holds DANCR's results is out of space."""


def _file_stamp(path: Path) -> list[int] | None:
    try:
        st = path.stat()
    except OSError:
        return None
    return [st.st_size, st.st_mtime_ns]


BLANK_NOTE = "Blank or unreadable cells — "      # a source's note about blanks; reports leave it out
LIVE_DIR = ".live"
LEASE_FOREIGN_MAX_AGE = 7 * 86400
_HOST = __import__("socket").gethostname()      # per-process lists of the results each DANCR process is using (see Executor.hold)


class Executor:
    def __init__(self, pipeline: Pipeline, cache_dir: Path | str | None = None,
                 output_root: Path | str | None = None) -> None:
        self.pipeline = pipeline
        self.cache_dir = Path(cache_dir).resolve() if cache_dir is not None else default_cache_dir(pipeline)
        self.output_root = Path(output_root).resolve() if output_root is not None else None   # steps may only write inside it
        self._lease = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"

    # ------------------------------------------------------------ hashing
    def _source_fingerprint(self, node_type: NodeType, params: dict[str, Any]) -> list[Any]:
        fp: list[Any] = []
        if node_type.kind != "source":
            return fp
        for p in node_type.params:
            if p.kind == "path" and params.get(p.name):
                try:
                    path = resolve_path(self.pipeline.directory, str(params[p.name]))
                    st = path.stat()
                    fp.append([str(path.resolve()), st.st_size, st.st_mtime_ns])
                except (OSError, ValueError, TypeError):
                    fp.append([str(params[p.name]), "missing"])
        return fp

    def _resolved_paths(self, node_type: NodeType, params: dict[str, Any]) -> dict[str, str]:
        out: dict[str, str] = {}
        for p in node_type.params:
            if p.kind == "path" and isinstance(params.get(p.name), str) and params[p.name].strip():
                try:
                    out[p.name] = str(resolve_path(self.pipeline.directory, params[p.name]).resolve())
                except (OSError, ValueError, RuntimeError):
                    out[p.name] = params[p.name]
        return out

    def _sink_fingerprint(self, node_id: str, node_type: NodeType, params: dict[str, Any]) -> dict[str, Any] | None:
        """What a file-writing step reads besides its settings and inputs: the titles of the steps feeding it
        (sheet names, report headings) and the column labels and units (report and chart labels)."""
        if node_type.kind != "sink":
            return None
        titles = {port: [self.pipeline.nodes[s].title for s in srcs]
                  for port, srcs in sorted(self.pipeline.inputs_of(node_id).items())}
        return {"titles": titles, "columns": self.pipeline.columns}

    def inputs_used(self, node_id: str, params_json: str | None = None) -> dict[str, Any]:
        """Project inputs whose names appear in this node's settings (so only they affect its cache hash)."""
        vals = self.pipeline.input_values()
        if not vals:
            return {}
        # ensure_ascii=False: an escaped name ("D\\u00e9bit") would never match the input "Débit"
        text = (params_json or json.dumps(self.pipeline.nodes[node_id].params, sort_keys=True, default=str, ensure_ascii=False)).lower()
        return {k: v for k, v in vals.items() if re.search(r"(?<![\w])" + re.escape(k.lower()) + r"(?![\w])", text)}

    def plan_hash(self, node_id: str, memo: dict[str, str] | None = None) -> str:
        memo = {} if memo is None else memo
        if node_id in memo:
            return memo[node_id]
        node = self.pipeline.nodes[node_id]
        nt = registry.get(node.type)
        ins = self.pipeline.inputs_of(node_id)
        params_json = json.dumps(node.params, sort_keys=True, default=str, ensure_ascii=False)   # once: reused for the inputs scan
        paths = self._resolved_paths(nt, node.params)
        payload = {
            "v": CODE_FINGERPRINT,
            "type": node.type,
            "params": json.dumps({k: v for k, v in node.params.items() if k not in paths}, sort_keys=True, default=str, ensure_ascii=False),
            "paths": paths,            # resolved: the same file after Save As hashes the same, another file does not
            "inputs": {port: [self.plan_hash(s, memo) for s in srcs] for port, srcs in sorted(ins.items())},
            "src": self._source_fingerprint(nt, node.params),
            "values": self.inputs_used(node_id, params_json),
            "sink": self._sink_fingerprint(node_id, nt, node.params),
        }
        h = hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]
        memo[node_id] = h
        return h

    # ------------------------------------------------------------ paths
    def node_dir(self, node_id: str) -> Path:
        return self.cache_dir / node_id

    def _meta_path(self, node_id: str, h: str) -> Path:
        return self.node_dir(node_id) / f"{h}.json"

    def _failed_path(self, node_id: str, h: str) -> Path:
        return self.node_dir(node_id) / f"{h}.failed.json"

    def _parquet_path(self, node_id: str, h: str) -> Path:
        return self.node_dir(node_id) / f"{h}.parquet"

    def state(self, node_id: str, memo: dict[str, str] | None = None) -> NodeState:
        try:
            h = self.plan_hash(node_id, memo)
        except (KeyError, PipelineError, RecursionError, TypeError, ValueError):
            return NodeState(node_id, status="idle")
        meta = self._meta_path(node_id, h)
        if meta.exists():
            try:
                d = json.loads(meta.read_text())
                st = NodeState(**{k: v for k, v in d.items() if k in NodeState.__dataclass_fields__})
                st.status = "done"
                st.from_cache = True
                if st.output:
                    out = self.cache_dir / st.output        # stored relative, so a moved cache folder still works
                    if not out.exists():
                        return NodeState(node_id, status="stale", hash=h)
                    st.output = str(out)
                for f, stamp in st.files.items():            # a written file that was deleted or changed is written again
                    if _file_stamp(Path(f)) != stamp:
                        return NodeState(node_id, status="stale", hash=h)
                return st
            except Exception as e:  # noqa: BLE001 - a damaged record means the step runs again
                log.warning("result record %s is unreadable (%s); the step will run again", meta, e)
        failed = self._failed_path(node_id, h)
        if failed.exists():
            try:
                d = json.loads(failed.read_text())
                st = NodeState(**{k: v for k, v in d.items() if k in NodeState.__dataclass_fields__})
                st.status = "failed"
                return st
            except Exception:
                pass
        nd = self.node_dir(node_id)
        try:
            status = "stale" if nd.exists() and any(nd.glob("*.json")) else "idle"
        except OSError:
            status = "idle"
        return NodeState(node_id, status=status, hash=h)

    def states(self) -> dict[str, NodeState]:
        memo: dict[str, str] = {}
        return {nid: self.state(nid, memo) for nid in list(self.pipeline.nodes)}

    # ------------------------------------------------------------ frames
    PREVIEW_ROWS = 50_000

    def frame(self, node_id: str, sample_rows: int | None = None, _visiting: set[str] | None = None,
              memo: dict[str, str] | None = None) -> pl.LazyFrame:
        """A LazyFrame for this node's output.

        Cached outputs are scanned from Parquet. Anything not computed yet is composed from
        *samples* of its inputs (``sample_rows`` rows, default PREVIEW_ROWS), never from the
        whole upstream data, so schema lookups and previews stay cheap no matter how deep the
        pipeline is. The real run never uses this path. ``memo`` shares plan hashes down the recursion, so a
        deep pipeline is hashed once, not once per level.
        """
        memo = {} if memo is None else memo
        st = self.state(node_id, memo)
        if st.status == "done" and st.output:
            return pl.scan_parquet(st.output)
        rows = sample_rows or self.PREVIEW_ROWS
        node = self.pipeline.nodes[node_id]
        nt = registry.get(node.type)
        _visiting = _visiting or set()
        if node_id in _visiting:
            raise PipelineError("loop")
        # A path set, not a global seen-set: two branches that share one not-yet-run ancestor
        # (a diamond) are not a loop, so each branch descends with its own copy of the path.
        _visiting = _visiting | {node_id}
        inputs = {port: [self.sample_frame(s, rows, _visiting, memo)[0] for s in srcs]
                  for port, srcs in self.pipeline.inputs_of(node_id).items()}
        ctx = self._ctx(node_id, preview=True, memo=memo)
        res = nt.apply(ctx, inputs, node.params)
        lf = res.frame if isinstance(res, NodeResult) else res
        return lf.head(rows) if nt.kind == "source" else lf

    def schema(self, node_id: str) -> dict[str, pl.DataType] | None:
        try:
            return dict(self.frame(node_id, 2_000).collect_schema())
        except Exception:
            return None

    def input_schemas(self, node_id: str) -> dict[str, dict[str, pl.DataType]]:
        out: dict[str, dict[str, pl.DataType]] = {}
        for port, srcs in self.pipeline.inputs_of(node_id).items():
            if srcs:
                s = self.schema(srcs[0])
                if s is not None:
                    out[port] = s
        return out

    def sample_frame(self, node_id: str, rows: int, _visiting: set[str] | None = None,
                     memo: dict[str, str] | None = None) -> tuple[pl.LazyFrame, str]:
        """A sample of a node's output for previews: (frame, kind) where kind is "spread"
        (every k-th row of a cached output), "all" (small cached output) or "head"
        (composed from the first rows of not-yet-run sources)."""
        memo = {} if memo is None else memo
        st = self.state(node_id, memo)
        if st.status == "done" and st.output and st.rows is not None:
            lf = pl.scan_parquet(st.output)
            if st.rows <= rows:
                return lf, "all"
            k = -(-st.rows // rows)                  # rounded up: never more than `rows` rows
            return lf.gather_every(k), "spread"
        return self.frame(node_id, rows, _visiting, memo).head(rows), "head"

    def preview(self, node_id: str, rows: int = PREVIEW_ROWS) -> tuple[pl.DataFrame, NodeResult | None, str]:
        """Compute this node on a sample of each input. Returns (df, result, sample_kind).

        Best-effort: a node that cannot be computed on the sample raises PreviewUnavailable with a
        plain-English message, never a raw traceback, because a preview failing does not mean the
        real run will fail."""
        try:
            memo: dict[str, str] = {}
            node = self.pipeline.nodes[node_id]
            nt = registry.get(node.type)
            inputs: dict[str, list[pl.LazyFrame]] = {}
            kinds: set[str] = set()
            for port, srcs in self.pipeline.inputs_of(node_id).items():
                frames = []
                for s in srcs:
                    lf, kind = self.sample_frame(s, rows, memo=memo)
                    frames.append(lf)
                    kinds.add(kind)
                inputs[port] = frames
            ctx = self._ctx(node_id, preview=True, memo=memo)
            res = nt.apply(ctx, inputs, node.params)
            lf = res.frame if isinstance(res, NodeResult) else res
            kind = "head" if "head" in kinds else ("spread" if "spread" in kinds else ("all" if kinds else "head"))
            return lf.head(rows).collect(engine="streaming"), (res if isinstance(res, NodeResult) else None), kind
        except PreviewUnavailable:
            raise
        except Exception as e:  # noqa: BLE001 - a preview must never surface a traceback
            raise PreviewUnavailable(friendly_error(e)) from e

    def _ctx(self, nid: str, preview: bool, results: dict[str, NodeState] | None = None, memo: dict[str, str] | None = None) -> Ctx:
        node = self.pipeline.nodes[nid]
        ctx = Ctx(self.pipeline.directory, nid, node.title, preview=preview, cache_dir=self.cache_dir,
                  output_root=self.output_root)
        ctx.inputs = self.pipeline.input_values()
        ctx.columns = self.pipeline.columns
        meta: dict[str, list[dict]] = {}
        for port, srcs in self.pipeline.inputs_of(nid).items():
            lst = []
            for s in srcs:
                up_node = self.pipeline.nodes[s]
                up_state = (results or {}).get(s) or self.state(s, memo)
                lst.append({"node": s, "title": up_node.title, "node_type": up_node.type, "params": up_node.params,
                            "messages": [m for m in up_state.messages if not m.startswith(BLANK_NOTE)], "report": up_state.report,
                            "status": up_state.status})
            meta[port] = lst
        ctx.upstream_meta = meta
        ctx.item_meta = meta.get("items")
        return ctx

    # ------------------------------------------------------------ running
    def run(self, targets: list[str] | None = None, on_event: EventFn | None = None,
            cancel: threading.Event | None = None, force: bool = False) -> dict[str, NodeState]:
        emit = on_event or (lambda e: None)
        order = self.pipeline.topological_order(targets)
        memo: dict[str, str] = {}
        results: dict[str, NodeState] = {}
        self._claim_cache_dir()          # owner.pid before anything else: a starting window's sweep must see an owner
        run_lease = f"{self._lease}-run"
        self._write_lease(run_lease, {nid: self._safe_hash(nid, memo) for nid in order})
        try:
            return self._run(order, emit, cancel, force, memo, results)
        finally:
            self._remove_lease(run_lease)

    def _safe_hash(self, nid: str, memo: dict[str, str]) -> str | None:
        try:
            return self.plan_hash(nid, memo)
        except Exception:  # noqa: BLE001 - a broken node simply holds nothing
            return None

    def _run(self, order: list[str], emit: EventFn, cancel: threading.Event | None, force: bool,
             memo: dict[str, str], results: dict[str, NodeState]) -> dict[str, NodeState]:
        emit({"type": "run_started", "order": order})
        t_run = time.perf_counter()
        for i, nid in enumerate(order):
            if cancel is not None and cancel.is_set():
                emit({"type": "run_cancelled", "states": results})
                raise ExecutionCancelled()
            node = self.pipeline.nodes[nid]
            nt = registry.get(node.type)
            h = self.plan_hash(nid, memo)
            st = self.state(nid, memo)
            if st.status == "done" and not force:
                results[nid] = st
                emit({"type": "node_cached", "node": nid, "state": st, "index": i, "total": len(order)})
                continue
            ups = [s for srcs in self.pipeline.inputs_of(nid).values() for s in srcs]
            bad = [u for u in ups if (results[u] if u in results else self.state(u, memo)).status != "done"]
            if bad:
                st = NodeState(nid, status="failed", hash=h, error=f"Waiting on {', '.join(self.pipeline.nodes[b].title for b in bad)}")
                results[nid] = st
                emit({"type": "node_failed", "node": nid, "state": st, "index": i, "total": len(order)})
                continue
            emit({"type": "node_started", "node": nid, "index": i, "total": len(order)})
            st = self._run_node(nid, nt, h, results, memo, force)
            results[nid] = st
            emit({"type": "node_finished" if st.status == "done" else "node_failed", "node": nid, "state": st, "index": i, "total": len(order)})
        try:
            self.gc(only=order, memo=memo)
        except Exception as e:  # never fail a run because of housekeeping
            log.warning("cache gc failed: %s", e)
        emit({"type": "run_finished", "elapsed": time.perf_counter() - t_run, "states": results})
        return results

    def _run_node(self, nid: str, nt: NodeType, h: str, results: dict[str, NodeState], memo: dict[str, str],
                  force: bool = False) -> NodeState:
        node = self.pipeline.nodes[nid]
        t0 = time.perf_counter()
        ndir = self.node_dir(nid)
        ndir.mkdir(parents=True, exist_ok=True)
        st = NodeState(nid, status="running", hash=h)
        try:
            inputs: dict[str, list[pl.LazyFrame]] = {}
            upstream_output: str | None = None
            for port, srcs in self.pipeline.inputs_of(nid).items():
                frames = []
                for s in srcs:
                    up = results.get(s) or self.state(s, memo)
                    if not up.output:
                        raise PipelineError(f"{self.pipeline.nodes[s].title} has no output")
                    frames.append(pl.scan_parquet(up.output))
                    upstream_output = upstream_output or up.output
                inputs[port] = frames
            ctx = self._ctx(nid, preview=False, results=results, memo=memo)
            res = nt.apply(ctx, inputs, node.params)
            if not isinstance(res, NodeResult):
                res = NodeResult(res)
            lf = res.frame
            if nt.materialize:
                st.output = str(self._materialize(lf, nid, h, replace=force))
            else:
                st.output = upstream_output
            if st.output:
                scan = pl.scan_parquet(st.output)
                st.rows = int(scan.select(pl.len()).collect(engine="streaming")[0, 0])
                st.columns = [{"name": n, "dtype": str(d)} for n, d in scan.collect_schema().items()]
                st.column_stats = column_stats(scan)
            st.messages = list(res.messages)
            if nt.kind == "source" and st.output:
                st.messages += self._blank_report(st.output, st.column_stats)
            st.report = json_safe(dict(res.report))
            st.files = {str(f): stamp for f in res.files if (stamp := _file_stamp(Path(f))) is not None}
            st.status = "done"
            st.elapsed = time.perf_counter() - t0
            st.finished_at = datetime.now().isoformat(timespec="seconds")
            meta = st.to_dict()
            if st.output:
                meta["output"] = self._relative_output(st.output)
            self._write_json(self._meta_path(nid, h), meta)
            fp = self._failed_path(nid, h)
            if fp.exists():
                fp.unlink(missing_ok=True)
            return st
        except ExecutionCancelled:
            raise
        except Exception as e:
            st.status = "failed"
            st.error = friendly_error(e)
            st.elapsed = time.perf_counter() - t0
            st.finished_at = datetime.now().isoformat(timespec="seconds")
            log.debug("node %s failed:\n%s", nid, traceback.format_exc())
            try:
                self._write_json(self._failed_path(nid, h), st.to_dict())
            except OSError:
                pass
            return st

    MIN_FREE_BYTES = 512 * 1024 * 1024

    def _claim_cache_dir(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if self.pipeline.path is None:
            owner = self.cache_dir / "owner.pid"
            if not owner.exists():
                owner.write_text(str(os.getpid()))

    def _check_disk(self) -> None:
        try:
            free = shutil.disk_usage(self.cache_dir).free
        except OSError:
            return
        if free < self.MIN_FREE_BYTES:
            raise CacheFull(f"The disk that holds DANCR's results ({self.cache_dir}) has only {free / 1e9:.1f} GB left. "
                            "Free some space, or use Run → Clear cached results, and run again.")

    def _relative_output(self, output: str) -> str:
        """Outputs are recorded relative to the cache folder, which moves with the project on Save As."""
        try:
            return Path(output).relative_to(self.cache_dir).as_posix()
        except ValueError:
            return output

    def _materialize(self, lf: pl.LazyFrame, nid: str, h: str, replace: bool = False) -> Path:
        """Stream the result to Parquet. Every step streams (spilling to disk when memory is short); a plan
        that cannot be streamed is a bug in that step and is reported as such, never worked around in memory.

        Two writers of the same hash normally compute the same thing, so the first one published wins.
        ``replace`` (a forced run) publishes this result regardless: forcing exists to repair a cached file."""
        out = self._parquet_path(nid, h)
        tmp = out.with_name(f"{h}.{os.getpid()}.{uuid.uuid4().hex[:6]}.tmp.parquet")
        self._claim_cache_dir()
        self._check_disk()
        try:
            try:
                lf.sink_parquet(tmp, compression="zstd", statistics=True, maintain_order=True)
            except Exception as e:
                msg = str(e)
                if _disk_full(msg):
                    raise CacheFull(f"The disk that holds DANCR's results ({self.cache_dir}) is full. "
                                    "Free some space, or use Run → Clear cached results, and run again.") from e
                log.error("step %s could not be streamed: %s", nid, msg)
                raise RuntimeError(f"'{self.pipeline.nodes[nid].title}' could not be computed as a streaming step "
                                   f"({msg.splitlines()[0] if msg else type(e).__name__}). This is a DANCR bug; please report it with the log file.") from e
            _validate_parquet(tmp)
            if not replace and out.exists() and _parquet_ok(out):
                tmp.unlink(missing_ok=True)      # someone else published the same result first
            else:
                try:
                    os.replace(tmp, out)
                except PermissionError as e:     # Windows: another program has the old result open
                    raise RuntimeError(f"The earlier result of '{self.pipeline.nodes[nid].title}' is open in another "
                                       "program or DANCR window, so it cannot be replaced. Close it and run again.") from e
        finally:
            tmp.unlink(missing_ok=True)
        return out

    @staticmethod
    def _write_json(path: Path, data: dict[str, Any]) -> None:
        tmp = path.with_name(f"{path.stem}.{os.getpid()}.{uuid.uuid4().hex[:6]}.tmp.json")
        tmp.write_text(json.dumps(data, default=str, indent=1))
        os.replace(tmp, path)

    @staticmethod
    def _blank_report(output: str, stats: dict[str, dict[str, Any]] | None = None) -> list[str]:
        """Messages about blank cells in a source's output, so bad rows are not invisible."""
        try:
            scan = pl.scan_parquet(output)
            schema = scan.collect_schema()
            if stats:
                counts = {c: int(stats.get(c, {}).get("nulls", 0)) for c in schema}
            else:
                counts = scan.select([pl.col(c).null_count().alias(c) for c in schema]).collect(engine="streaming").row(0, named=True)
        except Exception:
            return []
        parts = [f"{c}: {n:,}" for c, n in counts.items() if n]
        if not parts:
            return []
        note = BLANK_NOTE + ", ".join(parts[:6]) + (" …" if len(parts) > 6 else "")
        time_cols = [c for c, dt in schema.items() if isinstance(dt, (pl.Datetime, pl.Date)) and counts.get(c)]
        if time_cols:
            note += f". Rows with a blank {time_cols[0]} are skipped by time-based steps."
        return [note]

    # ------------------------------------------------------------ cache mgmt
    def hold(self, hashes: dict[str, str | None]) -> None:
        """Record which result of each node this process is using, so another process's cache sweep
        (the CLI or an agent running the saved file while the window shows unsaved edits) keeps them."""
        if self.pipeline.path is not None:           # an unsaved project's cache is private to this process
            self._write_lease(self._lease, hashes)

    def release(self) -> None:
        self._remove_lease(self._lease)

    def _write_lease(self, name: str, hashes: dict[str, str | None]) -> None:
        try:
            d = self.cache_dir / LIVE_DIR
            d.mkdir(parents=True, exist_ok=True)
            self._write_json(d / f"{name}.json", {"pid": os.getpid(), "host": _HOST,
                                                  "hashes": {k: v for k, v in hashes.items() if v}})
        except OSError as e:
            log.warning("could not record the results in use in %s: %s", self.cache_dir, e)

    def _remove_lease(self, name: str) -> None:
        try:
            (self.cache_dir / LIVE_DIR / f"{name}.json").unlink(missing_ok=True)
        except OSError:
            pass

    def _held(self) -> dict[str, set[str]]:
        """Hashes that live DANCR processes hold, per node. Leases of processes that are gone are removed."""
        held: dict[str, set[str]] = {}
        d = self.cache_dir / LIVE_DIR
        try:
            leases = [f for f in d.glob("*.json") if ".tmp." not in f.name]
        except OSError:
            return held
        for f in leases:
            try:
                data = json.loads(f.read_text())
                pid = int(data["pid"])
                age = time.time() - f.stat().st_mtime
            except (OSError, ValueError, KeyError, TypeError):
                continue                             # being written right now, or unreadable: ignore this pass
            if data.get("host", _HOST) != _HOST:
                # another machine sharing the folder: its processes cannot be checked from here, so its lease
                # is trusted until it is a week old (a live window rewrites it whenever what it shows changes)
                if age > LEASE_FOREIGN_MAX_AGE:
                    f.unlink(missing_ok=True)
                    continue
            elif not _pid_alive(pid):
                f.unlink(missing_ok=True)
                continue
            for nid, h in (data.get("hashes") or {}).items():
                held.setdefault(nid, set()).add(h)
        return held

    def gc(self, only: list[str] | None = None, keep_per_node: int = 1, grace_seconds: float = 900.0,
           memo: dict[str, str] | None = None) -> None:
        """Delete stale cached outputs of the given nodes (default: all nodes of this pipeline).
        Older versions are kept while younger than `grace_seconds` so a quick undo stays instant.
        Never raises."""
        if not self.cache_dir.exists():
            return
        memo = {} if memo is None else memo
        nodes = only if only is not None else list(self.pipeline.nodes)
        held = self._held()
        for nid in nodes:
            nd = self.node_dir(nid)
            try:
                live = {self.plan_hash(nid, memo)} | held.get(nid, set())
            except Exception:
                continue
            # a result is a record ({hash}.json) with, for steps that keep a table, {hash}.parquet
            now = time.time()
            results: dict[str, float] = {}
            try:
                for f in nd.iterdir():
                    name, mtime = f.name, f.stat().st_mtime
                    if ".tmp." in name:
                        if now - mtime > 3600:          # left by a writer that died
                            f.unlink(missing_ok=True)
                    elif name.endswith((".parquet", ".json")) and not name.endswith(".failed.json"):
                        stem = name.split(".")[0]
                        results[stem] = max(results.get(stem, 0.0), mtime)
            except OSError:
                continue
            kept = 0
            for mtime, stem in sorted(((m, st) for st, m in results.items()), reverse=True):
                if stem in live:
                    continue
                if kept < keep_per_node and now - mtime < grace_seconds:
                    kept += 1
                    continue
                for f in (nd / f"{stem}.parquet", nd / f"{stem}.json"):
                    try:
                        f.unlink(missing_ok=True)
                    except OSError:
                        pass
            try:
                for j in nd.glob("*.failed.json"):
                    if j.name.split(".")[0] not in live:
                        j.unlink(missing_ok=True)
            except OSError:
                pass
        if only is None:
            # directories of nodes that no longer exist
            try:
                for nd in self.cache_dir.iterdir():
                    if nd.is_dir() and nd.name not in self.pipeline.nodes and not nd.name.startswith(".") \
                            and nd.name not in held:
                        shutil.rmtree(nd, ignore_errors=True)
            except OSError:
                pass

    def clear_cache(self) -> None:
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def cache_size(self) -> int:
        try:
            return sum(p.stat().st_size for p in self.cache_dir.rglob("*.parquet")) if self.cache_dir.exists() else 0
        except OSError:
            return 0


def column_stats(scan: pl.LazyFrame) -> dict[str, dict[str, Any]]:
    """Cheap per-column facts for header badges and tooltips: nulls, and min/max/mean for numbers and dates.
    One streaming pass; never loads the table."""
    try:
        schema = scan.collect_schema()
        exprs: list[pl.Expr] = []
        for c, dt in schema.items():
            exprs.append(pl.col(c).null_count().alias(f"{c}\x00n"))
            if dt.is_numeric():
                f = pl.col(c).cast(pl.Float64)
                exprs += [f.min().alias(f"{c}\x00min"), f.max().alias(f"{c}\x00max"), f.mean().alias(f"{c}\x00mean")]
            elif isinstance(dt, (pl.Datetime, pl.Date)):
                exprs += [pl.col(c).min().cast(pl.Utf8).alias(f"{c}\x00min"), pl.col(c).max().cast(pl.Utf8).alias(f"{c}\x00max")]
        row = scan.select(exprs).collect(engine="streaming").row(0, named=True)
        out: dict[str, dict[str, Any]] = {}
        for k, v in row.items():
            c, key = k.split("\x00")
            out.setdefault(c, {})["nulls" if key == "n" else key] = json_safe(v)
        return out
    except Exception as e:  # noqa: BLE001 - facts for badges and tooltips; the result itself is fine
        log.warning("column statistics could not be computed: %s", e)
        return {}


def _validate_parquet(path: Path) -> None:
    """Read the first and last row group so a truncated/overwritten file is caught before publishing."""
    lf = pl.scan_parquet(path)
    lf.head(1).collect(engine="streaming")
    lf.tail(1).collect(engine="streaming")


def _parquet_ok(path: Path) -> bool:
    try:
        _validate_parquet(path)
        return True
    except Exception:
        return False


def _disk_full(msg: str) -> bool:
    m = msg.lower()
    return "no space left" in m or "disk quota" in m or "os error 28" in m or "os error 122" in m


def friendly_error(e: BaseException) -> str:
    """Turn Polars/IO exceptions into something a person can act on."""
    msg = str(e).strip()
    name = type(e).__name__
    if isinstance(e, CacheFull):
        return msg
    if _disk_full(msg):
        return "The disk is full, so the file could not be written. Free some space and run again."
    if isinstance(e, FileNotFoundError):
        return msg or "File not found"
    if isinstance(e, PermissionError):
        return f"No permission to access {e.filename or 'the file'}"
    if isinstance(e, IsADirectoryError):
        return f"{e.filename or 'That path'} is a folder, not a file"
    if isinstance(e, MemoryError):
        return "DANCR ran out of memory in this step, which should not happen: every step is meant to stream. Please report it with the log file (Help → Show log file)."
    if "ColumnNotFoundError" in name or ("not found" in msg and "column" in msg.lower()):
        first = msg.splitlines()[0] if msg else "column not found"
        return f"Column not found: {first}"
    if "DuplicateError" in name:
        return f"Two columns would have the same name: {msg.splitlines()[0]}"
    if "NoDataError" in name:
        return "The file is empty"
    if "SchemaError" in name or "ComputeError" in name or "InvalidOperationError" in name:
        first = msg.splitlines()[0] if msg else msg
        return first[:400]
    if isinstance(e, (ValueError, PipelineError, KeyError)):
        return (msg.strip("'\"") if isinstance(e, KeyError) else msg)[:600]
    first = msg.splitlines()[0] if msg else name
    return f"{name}: {first}"[:600]
