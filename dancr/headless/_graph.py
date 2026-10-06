"""Build, load and query the repository-level entity graph (roadmap A1).

The pure model is :mod:`dancr.core.graph`; this module is where real projects are
read. It walks a repository's project files (bounded by ``paths`` when given),
derives each project's tables and relations with :mod:`dancr.core.understand`,
and writes one atomic SQLite database under ``<root>/.dancr/graph/``.

Incremental: a project whose file bytes and source-file stamps have not changed
since the last build is carried over from the stored graph without being read
again (its plan-hash versions are preserved). The build never touches a project
file, so it takes only the repository lock (docs/adr/0003).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..core import Pipeline, registry
from ..core.executor import CODE_FINGERPRINT, Executor
from ..core.graph import (GRAPH_VERSION, SENSITIVITY_LEVELS, Column, Dataset, Edge, Graph, Project,
                          Source, is_restricted, load_graph)
from ..core.events import EventLog
from ..core.identity import (IDENTITY_VERSION, dataset_column_id, dataset_id, digest, file_digest, project_id,
                             source_id)
from ..core.repo import Repo, write_json_atomic
from ..core.understand import default_tables, understand

log = logging.getLogger("dancr.graph")


def _sensitivity(pipe: Pipeline, node_id: str, project_level: str) -> str:
    """A dataset's sensitivity: the project's declared level, or a 'Label sensitivity' step's own level for its
    output. Anything else is public."""
    node = pipe.nodes.get(node_id)
    if node is not None and node.type == "label_sensitivity":
        level = str(node.params.get("level") or "").lower()
        if level in SENSITIVITY_LEVELS:
            return level
    return project_level if project_level in SENSITIVITY_LEVELS else "public"


def _source_stamp(root: Path, pipe: Pipeline) -> list[Any]:
    """A cheap stamp of every file the project reads (size, modification time, a content sample), the same idea
    as the executor's cache fingerprint, so a source that changed without the project file changing is noticed."""
    from ..core.executor import _content_sample
    from ._safety import source_files
    parts: list[Any] = []
    for fp in sorted(source_files(pipe), key=str):
        try:
            st = fp.stat()
            parts.append([str(fp), st.st_size, st.st_mtime_ns, _content_sample(fp, st.st_size)])
        except OSError:
            parts.append([str(fp), "missing"])
    return parts


def change_key(project_path: Path, pipe: Pipeline) -> str:
    """What says a project has not moved since a stored graph: the project file's bytes and every source file's
    stamp. Deterministic, so an unchanged project is always reused."""
    return digest({"project": file_digest(project_path), "sources": _source_stamp(project_path.parent, pipe)})


def _project_graph(root: Path, path: Path, pipe: Pipeline) -> tuple[Project, list[Dataset], list[Edge]]:
    """Derive one project's vertices and edges. ``understand`` reads a bounded sample of each source table; the
    plan hash is the dataset's version."""
    pid = project_id(root, path)
    project_level = str((pipe.meta or {}).get("sensitivity") or "public").lower()
    proj = Project(pid, pipe.name, Path(pid).name, change_key(path, pipe), CODE_FINGERPRINT, project_level)
    ex = Executor(pipe)
    model = understand(pipe, ex, nodes=default_tables(pipe))
    datasets: list[Dataset] = []
    version_of: dict[str, str] = {}
    sens_of: dict[str, str] = {}
    memo: dict[str, str] = {}
    for t in model.tables.values():
        dsid = dataset_id(pid, t.node)
        version = ex.safe_hash(t.node, memo) or ""
        version_of[dsid] = version
        sens = _sensitivity(pipe, t.node, project_level)
        sens_of[dsid] = sens
        ds = Dataset(dsid, pid, t.node, t.title, shape=t.shape, rows=t.rows, rows_exact=t.rows_exact,
                     time_column=t.time or "", source=Path(t.source).name if t.source else "",
                     version=version, sensitivity=sens)
        for c in t.columns:
            ds.columns.append(Column(dataset_column_id(dsid, c.name), dsid, c.name, c.role, c.kind, c.unit or ""))
        nt = registry.get(pipe.nodes[t.node].type)
        if nt.kind == "source" and t.source:
            fp = Path(t.source)
            full = fp if fp.is_absolute() else (pipe.directory / fp)
            size, sample = _file_size_sample(full)
            ds.sources.append(Source(source_id(pid, t.node, fp), dsid, Path(str(fp)).name, size, sample))
        datasets.append(ds)

    edges: list[Edge] = []
    for r in model.relations:
        left, right = dataset_id(pid, r.tables[0]), dataset_id(pid, r.tables[1])
        if left not in version_of or right not in version_of:
            continue
        eid = f"{pid}#{r.id}"
        edges.append(Edge(eid, r.kind, left, right, r.left_on or "", r.right_on or "",
                          r.cardinality or "", float(r.match_pct or 0.0), float(r.score or 0.0),
                          getattr(r, "why", "") or "", version_of[left], version_of[right]))
    # a link's key columns are resolved entities: mark them on the dataset they belong to
    col_by_dataset = {d.id: {c.name: c for c in d.columns} for d in datasets}
    for e in edges:
        if e.kind == "link":
            if (c := col_by_dataset.get(e.left, {}).get(e.left_on)) is not None:
                c.is_entity = True
            if (c := col_by_dataset.get(e.right, {}).get(e.right_on)) is not None:
                c.is_entity = True
    return proj, datasets, edges


def _file_size_sample(path: Path) -> tuple[int | None, str]:
    from ..core.executor import _content_sample
    try:
        st = path.stat()
    except OSError:
        return None, ""
    return st.st_size, _content_sample(path, st.st_size)


def _carry_over(src: Graph, dst: Graph, project_id_: str) -> None:
    """Copy a project and everything hanging off its datasets from the stored graph into the new one."""
    if (proj := src.projects.get(project_id_)) is not None:
        dst.add_project(proj)
    ids = [d.id for d in src.datasets.values() if d.project == project_id_]
    idset = set(ids)
    for did in sorted(ids):
        dst.add_dataset(src.datasets[did])
    for e in sorted(src.edges.values(), key=lambda x: x.id):
        if e.left in idset or e.right in idset:
            dst.add_edge(e)


def add_cross_project_keys(graph: Graph) -> int:
    """Add the deterministic cross-project links the entity columns imply: two datasets in *different* projects
    whose resolved key columns have the same normalised name are the same key. No model and no value scan: the
    match percentage is left unmeasured (0) and the confidence is a fixed, documented value, so the edge is a
    *proposal* the person can check, never a silent join. Returns how many edges were added."""
    from itertools import combinations
    from ..core.identity import key_norm
    by_key: dict[str, list[tuple[str, str, str, str]]] = {}
    for d in sorted(graph.datasets.values(), key=lambda x: x.id):
        for c in sorted(d.columns, key=lambda x: x.id):
            if c.is_entity:
                by_key.setdefault(key_norm(c.name), []).append((d.id, c.name, d.project, d.version))
    added = 0
    for norm, items in sorted(by_key.items()):
        if len({p for *_, p in items}) < 2:
            continue
        seen: set[tuple[str, str]] = set()
        for (da, na, pa, va), (db, nb, pb, vb) in combinations(sorted(items, key=lambda t: (t[2], t[0], t[1])), 2):
            if pa == pb:
                continue
            pair = (da, db) if da <= db else (db, da)
            if pair in seen:
                continue
            seen.add(pair)
            left = da if da <= db else db
            right = db if da <= db else da
            lon = next(n for i, n, *_ in items if i == left)
            ron = next(n for i, n, *_ in items if i == right)
            eid = f"xlink:{left}>{right}:{norm}"
            graph.add_edge(Edge(eid, "link", left, right, lon, ron, "", 0.0, 0.8,
                                f"{lon} and {ron} name the same key (across projects); the match has not been measured",
                                next(v for i, *_, v in items if i == left),
                                next(v for i, *_, v in items if i == right)))
            added += 1
    return added


def build_graph(root: Path | str, *, projects: list[str] | None = None, force: bool = False,
                jobs: int = 1) -> dict[str, Any]:
    """Build (or refresh) the repository graph and write it atomically. Returns a summary.

    ``projects`` restricts the walk to those project files (relative to ``root`` or absolute). ``force``
    rebuilds every included project instead of carrying unchanged ones over. ``jobs`` builds several projects
    at once (the results are merged in sorted order, so the graph is the same either way)."""
    from . import repo_lock
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"{root} is not a folder")
    repo = Repo(root).ensure()
    with repo_lock(root):
        wanted = _project_files(root, projects)
        existing = None if force else load_graph(root)
        reusable = (existing is not None and existing.meta.get("code_fingerprint") == CODE_FINGERPRINT
                    and existing.meta.get("identity_version") == IDENTITY_VERSION
                    and existing.meta.get("graph_version") == GRAPH_VERSION)
        graph = Graph()
        graph.meta = {"identity_version": IDENTITY_VERSION, "graph_version": GRAPH_VERSION,
                      "code_fingerprint": CODE_FINGERPRINT}
        rebuilt: list[str] = []
        reused: list[str] = []
        skipped: dict[str, str] = {}

        def one(path: Path) -> tuple[str, Any]:
            pid = project_id(root, path)
            try:
                pipe = Pipeline.load(path)
            except Exception as e:  # noqa: BLE001 - a broken project is skipped, not fatal
                return "error", (pid, str(e))
            old = existing.projects.get(pid) if reusable else None
            if old is not None and old.content_hash == change_key(path, pipe):
                return "reuse", pid
            try:
                return "built", (pid, _project_graph(root, path, pipe))
            except Exception as e:  # noqa: BLE001
                return "error", (pid, str(e))

        outcomes: list[tuple[str, Any]] = []
        if int(jobs or 1) > 1 and len(wanted) > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=int(jobs)) as pool:
                outcomes = list(pool.map(one, wanted))
        else:
            outcomes = [one(p) for p in wanted]

        for kind, payload in outcomes:
            if kind == "reuse":
                _carry_over(existing, graph, payload)
                reused.append(payload)
            elif kind == "built":
                pid, (proj, datasets, edges) = payload
                graph.add_project(proj)
                for d in datasets:
                    graph.add_dataset(d)
                for e in edges:
                    graph.add_edge(e)
                rebuilt.append(pid)
            else:
                pid, why = payload
                skipped[pid] = why

        generated = _now()
        cross = add_cross_project_keys(graph)
        counts = graph.summary()
        graph.meta.update({"built_at": generated, "cross_project_edges": cross,
                           "counts": {k: counts[k] for k in
                          ("projects", "datasets", "columns", "sources", "edges")}})
        graph.write(repo.graph_db())
        write_json_atomic(repo.graph_meta(), {**graph.meta, "rebuilt": sorted(rebuilt),
                                              "reused": sorted(reused), "skipped": dict(sorted(skipped.items()))})
        EventLog(repo.events_log()).append(
            "graph_updated", root=str(root), rebuilt=sorted(rebuilt), reused=sorted(reused),
            skipped=sorted(skipped), counts=graph.meta["counts"])
    return {"kind": "dancr.graph", "root": str(root), "path": str(repo.graph_db()),
            "rebuilt": sorted(rebuilt), "reused": sorted(reused), "skipped": dict(sorted(skipped.items())),
            "cross_project_edges": cross, **counts}


def _now() -> str:
    from datetime import datetime
    return datetime.now().isoformat(timespec="seconds")


def _project_files(root: Path, projects: list[str] | None) -> list[Path]:
    """The project files to build: the named ones (resolved under root), else every project found by the
    catalog walk. Sorted, so the build order is deterministic."""
    if projects:
        out = []
        for name in projects:
            p = Path(name).expanduser()
            p = (p if p.is_absolute() else root / p).resolve()
            if not p.is_file():
                raise ValueError(f"No project file at {p}")
            out.append(p)
        return sorted(set(out), key=str)
    from ._context import find_pipelines
    return find_pipelines(root)


def _require_graph(root: Path | str) -> tuple[Path, Graph]:
    root = Path(root).expanduser().resolve()
    graph = load_graph(root)
    if graph is None:
        raise ValueError(f"No graph has been built for {root}. Run: dancr graph build {root}")
    return root, graph


def graph_summary(root: Path | str) -> dict[str, Any]:
    """Counts and metadata of the stored graph."""
    repo_root, graph = _require_graph(root)
    return {"kind": "dancr.graph.summary", "root": str(repo_root), **graph.summary()}


def graph_query(root: Path | str, *, kind: str | None = None, project: str | None = None,
                text: str | None = None, allow_restricted: bool = False, limit: int | None = None) -> dict[str, Any]:
    """Datasets and edges matching the filters (see :meth:`dancr.core.graph.Graph.query`)."""
    repo_root, graph = _require_graph(root)
    out = graph.query(kind=kind, project=project, text=text, allow_restricted=allow_restricted, limit=limit)
    return {"kind": "dancr.graph.query", "root": str(repo_root), **out}


def graph_neighbors(root: Path | str, dataset: str, *, allow_restricted: bool = False) -> dict[str, Any]:
    repo_root, graph = _require_graph(root)
    return {"kind": "dancr.graph.neighbors", "root": str(repo_root),
            **graph.neighbors(dataset, allow_restricted=allow_restricted)}


def graph_path(root: Path | str, source: str, target: str, *, allow_restricted: bool = False) -> dict[str, Any]:
    repo_root, graph = _require_graph(root)
    return {"kind": "dancr.graph.path", "root": str(repo_root),
            **graph.path(source, target, allow_restricted=allow_restricted)}


def graph_shared_keys(root: Path | str, *, allow_restricted: bool = False) -> dict[str, Any]:
    repo_root, graph = _require_graph(root)
    keys = graph.shared_keys(allow_restricted=allow_restricted)
    return {"kind": "dancr.graph.shared_keys", "root": str(repo_root), "keys": keys, "count": len(keys)}


def graph_slice(root: Path | str, *, allow_restricted: bool = False) -> dict[str, Any]:
    """The whole graph as a JSON-able document (for a context/FAIR export or an agent). Restricted datasets
    and the edges touching them are withheld unless ``allow_restricted``."""
    repo_root, graph = _require_graph(root)
    datasets = graph.all_datasets(allow_restricted=allow_restricted)
    edges = graph.all_edges(allow_restricted=allow_restricted)
    return {"kind": "dancr.graph.slice", "root": str(repo_root), "version": GRAPH_VERSION,
            "meta": dict(sorted(graph.meta.items(), key=lambda kv: kv[0])),
            "projects": [p.to_dict() for p in sorted(graph.projects.values(), key=lambda x: x.id)],
            "datasets": [d.to_dict() for d in datasets], "edges": [e.to_dict() for e in edges]}


def cross_ask(root: Path | str, question: str, *, allow_restricted: bool = False) -> dict[str, Any]:
    """Answer a structural cross-project question against the stored graph, citing edge evidence. No execution
    (docs/adr/0005-cross-project-scope.md)."""
    from ..core.crossask import ask
    repo_root, graph = _require_graph(root)
    return {"root": str(repo_root), **ask(graph, question, allow_restricted=allow_restricted)}


def cross_suggest(root: Path | str, *, allow_restricted: bool = False) -> dict[str, Any]:
    """The structural questions the stored graph can answer, best first."""
    from ..core.crossask import suggest
    repo_root, graph = _require_graph(root)
    return {"kind": "dancr.crossask.suggest", "root": str(repo_root),
            "suggestions": suggest(graph, allow_restricted=allow_restricted)}
