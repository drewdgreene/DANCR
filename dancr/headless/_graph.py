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

import json
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
from ..core.names import looks_like_key
from ..core.repo import Repo, write_json_atomic
from ..core.understand import ID as ID_ROLE
from ..core.understand import _vhash, default_tables, understand

log = logging.getLogger("dancr.graph")


def _sensitivity(pipe: Pipeline, node_id: str, project_level: str, restricted: frozenset[str] = frozenset()) -> str:
    """A dataset's sensitivity: the project's declared level, a 'Label sensitivity' step's own level (or, for a
    per-row label column, ``confidential``), and anything derived from a restricted table. Anything else public."""
    if node_id in restricted:
        node = pipe.nodes.get(node_id)
        if node is not None and node.type == "label_sensitivity" and not str(node.params.get("column") or "").strip():
            level = str(node.params.get("level") or "").lower()
            if level in SENSITIVITY_LEVELS:
                return level                       # keep the declared level for a whole-table label
        return "confidential"
    return project_level if project_level in SENSITIVITY_LEVELS else "public"


def _source_stamp(pipe: Pipeline) -> list[Any]:
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
    return digest({"project": file_digest(project_path), "sources": _source_stamp(pipe)})


def _project_graph(root: Path, path: Path, pipe: Pipeline) -> tuple[Project, list[Dataset], list[Edge]]:
    """Derive one project's vertices and edges. ``understand`` reads a bounded sample of each source table; the
    plan hash is the dataset's version."""
    pid = project_id(root, path)
    project_level = str((pipe.meta or {}).get("sensitivity") or "public").lower()
    proj = Project(pid, pipe.name, Path(pid).name, change_key(path, pipe), CODE_FINGERPRINT, project_level)
    ex = Executor(pipe)
    model = understand(pipe, ex, nodes=default_tables(pipe))
    from ..core.graph import restricted_nodes
    restricted = frozenset(restricted_nodes(pipe, list(model.tables)))
    datasets: list[Dataset] = []
    version_of: dict[str, str] = {}
    sens_of: dict[str, str] = {}
    understood: dict[str, Any] = {}          # graph column id -> the understand Column (for its key sketch)
    memo: dict[str, str] = {}
    for t in model.tables.values():
        dsid = dataset_id(pid, t.node)
        version = ex.safe_hash(t.node, memo) or ""
        version_of[dsid] = version
        sens = _sensitivity(pipe, t.node, project_level, restricted)
        sens_of[dsid] = sens
        ds = Dataset(dsid, pid, t.node, t.title, shape=t.shape, rows=t.rows, rows_exact=t.rows_exact,
                     time_column=t.time or "", source=Path(t.source).name if t.source else "",
                     version=version, sensitivity=sens)
        for c in t.columns:
            cid = dataset_column_id(dsid, c.name)
            col = Column(cid, dsid, c.name, c.role, c.kind, c.unit or "")
            # a key-like column is an entity candidate even when no intra-project link used it, so two
            # single-table projects that share a key can still be compared across projects
            col.is_entity = bool(c.link_candidate and c._keys and (c.role == ID_ROLE or looks_like_key(c.name)))
            ds.columns.append(col)
            understood[cid] = c
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
            left_col = col_by_dataset.get(e.left, {}).get(e.left_on)
            if left_col is not None:
                left_col.is_entity = True
            right_col = col_by_dataset.get(e.right, {}).get(e.right_on)
            if right_col is not None:
                right_col.is_entity = True
    # every entity column carries a capped key sketch, so a later build can measure cross-project overlap
    # without re-reading the project (the incremental-build invariant)
    for d in datasets:
        for col in d.columns:
            understood_col = understood.get(col.id)
            if col.is_entity and understood_col is not None and understood_col._keys:
                col.sketch = json.dumps({"keys": sorted(understood_col._keys), "cut": understood_col._key_cut,
                                         "unique": bool(understood_col.unique)})
    return proj, datasets, edges


def _file_size_sample(path: Path) -> tuple[int | None, str]:
    from ..core.executor import _content_sample
    try:
        st = path.stat()
    except OSError:
        return None, ""
    return st.st_size, _content_sample(path, st.st_size)


def _carry_over(src: Graph, dst: Graph, project_id_: str) -> None:
    """Copy a project and everything hanging off its datasets from the stored graph into the new one. Cross-project
    edges (``xlink:``) are *not* carried: they depend on other projects too and are regenerated every build, so
    carrying one over could keep a stale link after another project's key changed."""
    if (proj := src.projects.get(project_id_)) is not None:
        dst.add_project(proj)
    ids = [d.id for d in src.datasets.values() if d.project == project_id_]
    idset = set(ids)
    for did in sorted(ids):
        dst.add_dataset(src.datasets[did])
    for e in sorted(src.edges.values(), key=lambda x: x.id):
        if (e.left in idset or e.right in idset) and not e.id.startswith("xlink:"):
            dst.add_edge(e)


def _sketch(sketch: str) -> tuple[list[str], int | None, bool]:
    """A stored key sketch as (values, hash cut, the key is unique in its table). A damaged sketch is empty."""
    try:
        d = json.loads(sketch or "{}")
        return list(d.get("keys") or []), d.get("cut"), bool(d.get("unique"))
    except (ValueError, TypeError):
        return [], None, False


def _comparable(a: tuple[list[str], int | None, bool], b: tuple[list[str], int | None, bool]) -> tuple[set[str], set[str]]:
    """The two key sets on a consistent sample: if either was capped by hash, keep only values below the smaller
    cut on both sides, so a big table cannot out-vote a small one (the same rule as intra-project links)."""
    a_keys, a_cut, _ = a
    b_keys, b_cut, _ = b
    cuts = [c for c in (a_cut, b_cut) if c is not None]
    if not cuts:
        return set(a_keys), set(b_keys)
    cut = min(cuts)
    return {v for v in a_keys if _vhash(v) <= cut}, {v for v in b_keys if _vhash(v) <= cut}


def _overlap(a: set[str], b: set[str]) -> float:
    """Half containment (the smaller side found in the larger) and half Jaccard, so a tiny set that happens to
    sit inside a big one does not score as a match. The same measure the intra-project links use."""
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return 0.5 * inter / min(len(a), len(b)) + 0.5 * inter / len(a | b)


def _cardinality(left_unique: bool, right_unique: bool) -> str:
    """How two keys relate, from whether each side's values are unique in its own table: a unique key is one row
    per value, so it is the standard one-to-one / one-to-many / many-to-one / many-to-many."""
    if left_unique and right_unique:
        return "one-to-one"
    if left_unique:
        return "one-to-many"
    if right_unique:
        return "many-to-one"
    return "many-to-many"


def add_cross_project_keys(graph: Graph) -> int:
    """Add the cross-project links the entity columns imply, **measured** from the key sketches each column
    carries.

    Two datasets in *different* projects whose resolved key columns have the same normalised name and whose
    sampled key values actually overlap get a ``link`` edge with a measured match percentage and a cardinality,
    and a confidence derived from the overlap. A same-named key with no shared values is not a link. No model
    and no re-read: the sketches were stored when the projects were built, so a carried-over project is compared
    without being opened (the incremental-build invariant). Deterministic. Returns how many edges were added."""
    from itertools import combinations
    from ..core.identity import key_norm
    by_key: dict[str, list[tuple[str, str, str, str, str]]] = {}
    for d in sorted(graph.datasets.values(), key=lambda x: x.id):
        for c in sorted(d.columns, key=lambda x: x.id):
            if c.is_entity and c.sketch:
                by_key.setdefault(key_norm(c.name), []).append((d.id, c.name, d.project, d.version, c.sketch))
    added = 0
    for norm, items in sorted(by_key.items()):
        if len({t[2] for t in items}) < 2:                 # need two different projects to compare
            continue
        seen: set[tuple[str, str]] = set()
        for a, b in combinations(sorted(items, key=lambda t: (t[2], t[0], t[1])), 2):
            (da, na, pa, va, sa), (db, nb, pb, vb, sb) = a, b
            if pa == pb:
                continue
            if da <= db:
                left, lon, lv, right, ron, rv, ls, rs = da, na, va, db, nb, vb, sa, sb
            else:
                left, lon, lv, right, ron, rv, ls, rs = db, nb, vb, da, na, va, sb, sa
            pair = (left, right)
            if pair in seen:
                continue
            seen.add(pair)
            la, ra = _sketch(ls), _sketch(rs)
            lset, rset = _comparable(la, ra)
            score = _overlap(lset, rset)
            if score <= 0:
                continue                                   # same name, no shared values: not a link
            pct = round(score * 100, 1)
            card = _cardinality(la[2], ra[2])
            eid = f"xlink:{left}>{right}:{norm}"
            graph.add_edge(Edge(eid, "link", left, right, lon, ron, card, pct, round(score, 3),
                                f"{lon} and {ron} name the same key across projects and share {pct}% of their "
                                f"sampled values ({card})", lv, rv))
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
        existing = load_graph(root)                  # kept both for reuse and to carry over the projects not named now
        reusable = (not force and existing is not None
                    and existing.meta.get("code_fingerprint") == CODE_FINGERPRINT
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
            old = existing.projects.get(pid) if (reusable and existing is not None) else None
            if old is not None and old.content_hash == change_key(path, pipe):
                return "reuse", pid
            try:
                return "built", (pid, _project_graph(root, path, pipe))
            except Exception as e:  # noqa: BLE001
                return "error", (pid, str(e))

        outcomes: list[tuple[str, Any]] = []
        if int(jobs or 1) > 1 and len(wanted) > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=max(1, min(int(jobs), 32))) as pool:
                outcomes = list(pool.map(one, wanted))
        else:
            outcomes = [one(p) for p in wanted]

        for kind, payload in outcomes:
            if kind == "reuse":
                assert existing is not None          # only produced when an existing graph was reused
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

        # Building a *subset* of projects (``projects=[…]``) must not drop the rest of the repository from the
        # graph: carry over every stored project the build did not name, so the result is the whole repository.
        if existing is not None:
            wanted_ids = {project_id(root, p) for p in wanted}
            for pid in sorted(existing.projects):
                if pid not in wanted_ids and pid not in graph.projects:
                    _carry_over(existing, graph, pid)
                    reused.append(pid)

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
    # A project's own name and relative path are identity too, so a restricted project (or one whose every
    # dataset is withheld) must not appear in the slice either.
    visible = {d.project for d in datasets}
    projects = [p for p in graph.projects.values()
                if p.id in visible and (allow_restricted or not is_restricted(p.sensitivity))]
    return {"kind": "dancr.graph.slice", "root": str(repo_root), "version": GRAPH_VERSION,
            "meta": dict(sorted(graph.meta.items(), key=lambda kv: kv[0])),
            "projects": [p.to_dict() for p in sorted(projects, key=lambda x: x.id)],
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


def infer_repo_root(project_path: Path | str) -> Path | None:
    """The repository a project belongs to, from a graph already built near it: the nearest ancestor folder
    (the project's own folder first) that holds ``.dancr/graph/graph.db``, or None when there is no graph."""
    p = Path(project_path).expanduser().resolve()
    for base in [p.parent, *p.parent.parents]:
        if (base / ".dancr" / "graph" / "graph.db").is_file():
            return base
    return None


def graph_context_for(pipe: Pipeline, *, root: Path | str | None = None,
                      allow_restricted: bool = False) -> dict[str, Any] | None:
    """The cross-project graph block for a project's knowledge-base context: the edges incident to this
    project's datasets (with their evidence) plus the neighbouring datasets (id, title, shape, project), and a
    per-dataset slice for each document. Deterministic (edges and neighbours by id). None when there is no
    graph, or the project has no datasets in it — the context export then simply omits the block.

    A dataset's ``meta["sensitivity"]`` (or a Label-sensitivity step) marks it restricted; restricted datasets
    and the edges touching them are withheld unless ``allow_restricted``."""
    if pipe.path is None:
        return None
    base = Path(root).expanduser().resolve() if root else infer_repo_root(pipe.path)
    if base is None:
        return None
    graph = load_graph(base)
    if graph is None:
        return None
    pid = project_id(base, pipe.path)
    own = {d.id for d in graph.datasets.values() if d.project == pid}
    if not own:
        return None
    edges = sorted((e for e in graph.all_edges(allow_restricted=allow_restricted)
                    if e.left in own or e.right in own), key=lambda e: e.id)
    if not edges:
        return None
    neighbour_ids = sorted({(e.right if e.left in own else e.left) for e in edges})
    neighbours = []
    for nid in neighbour_ids:
        d = graph.datasets.get(nid)
        if d is None or (is_restricted(d.sensitivity) and not allow_restricted):
            continue
        neighbours.append({"id": d.id, "title": d.title, "shape": d.shape, "project": d.project})
    per: dict[str, dict[str, Any]] = {}
    for d in sorted((x for x in graph.datasets.values() if x.project == pid), key=lambda x: x.node):
        mine = [e for e in edges if e.left == d.id or e.right == d.id]
        if not mine:
            continue
        nbr_ids = sorted({(e.right if e.left == d.id else e.left) for e in mine})
        per[d.node] = {"edges": [e.to_dict() for e in mine],
                       "neighbours": [graph.datasets[x].to_dict() for x in nbr_ids
                                      if x in graph.datasets and (allow_restricted or not is_restricted(graph.datasets[x].sensitivity))]}
    block = {"kind": "dancr.graph.context", "version": GRAPH_VERSION, "root": str(base), "project": pid,
             "edges": [e.to_dict() for e in edges], "neighbours": neighbours, "per_dataset": per}
    block["content_hash"] = digest({"edges": block["edges"], "neighbours": block["neighbours"]})
    return block
