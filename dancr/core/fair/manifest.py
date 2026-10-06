"""The run manifest: provenance of exactly what produced the project's results."""
from __future__ import annotations

from typing import Any

from ._common import MANIFEST_KIND, MANIFEST_VERSION, now as _now, version as _version


def _source_fingerprint(pipe, node_id: str) -> list[dict[str, Any]]:
    """What a source step reads, with the size/time/content sample the executor fingerprints it by."""
    from ..registry import registry, resolve_path
    from ..executor import _content_sample
    node = pipe.nodes[node_id]
    nt = registry.get(node.type)
    out: list[dict[str, Any]] = []
    for prm in nt.params:
        if prm.kind != "path":
            continue
        v = node.params.get(prm.name)
        if not (isinstance(v, str) and v.strip()):
            continue
        try:
            target = resolve_path(pipe.directory, v)
            st = target.stat()
            out.append({"path": str(target.resolve()), "size": st.st_size, "mtime_ns": st.st_mtime_ns,
                        "sample": _content_sample(target, st.st_size)})
        except (OSError, ValueError, TypeError):
            out.append({"path": str(v), "missing": True})
    return out


def run_manifest(pipe, executor=None, states: dict[str, Any] | None = None,
                 meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """Exactly what produced the project's results: the engine and library versions, the code fingerprint, the
    source files (with the stamp they are fingerprinted by), and every step's plan hash, row count and elapsed
    time. This is the provenance half of a FAIR record, and it is already what the cache keys on."""
    from ..executor import engine_versions, CODE_FINGERPRINT, IMPL_VERSION
    if executor is None:
        from ..executor import Executor
        executor = Executor(pipe)
    if states is None:
        states = executor.states()
    dataset_meta = dict(meta if meta is not None else pipe.dataset_meta())

    nodes = []
    for nid in pipe.topological_order():
        node = pipe.nodes[nid]
        st = states.get(nid)
        try:
            h = executor.plan_hash(nid)
        except Exception:  # noqa: BLE001 - a step whose hash cannot be worked out simply reports none
            h = None
        entry: dict[str, Any] = {"id": nid, "type": node.type, "title": node.title, "hash": h}
        if st is not None:
            entry["status"] = st.status
            entry["rows"] = st.rows
            entry["columns"] = list(st.columns or [])
            entry["elapsed"] = st.elapsed
            entry["finished_at"] = st.finished_at
            if st.files:
                entry["files"] = sorted(st.files)
        nodes.append(entry)

    sources = []
    from ..registry import registry
    for nid, node in pipe.nodes.items():
        if registry.get(node.type).kind == "source":
            sources.append({"node": nid, "title": node.title, "files": _source_fingerprint(pipe, nid)})

    return {
        "kind": MANIFEST_KIND,
        "version": MANIFEST_VERSION,
        "generated_at": _now(),
        "engine": {"name": "DANCR", "version": _version(), "impl_version": IMPL_VERSION,
                   "fingerprint": CODE_FINGERPRINT},
        "libraries": engine_versions(),
        "project": {"name": pipe.name, "file": pipe.path.name if pipe.path else None},
        "dataset": dataset_meta,
        "sources": sources,
        "nodes": nodes,
    }
