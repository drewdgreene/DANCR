"""The cross-project entity graph: vertices and edges with their evidence.

A graph is a *projection* of hashed project state, rebuilt from the repository and
never a source of truth. This module is pure: it defines the schema, an in-memory
model, deterministic queries, and read/write to a SQLite database
(``<root>/.dancr/graph/graph.db``) via :mod:`dancr.core.repo`. Building the graph
from real projects lives in :mod:`dancr.headless._graph`, so the core stays free
of execution.

Identity and versioning follow ``docs/adr/0001-graph-identity.md``. Every list a
query returns is sorted by id, so the same repository always answers the same way.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .dtypes import json_safe
from .identity import IDENTITY_VERSION, key_norm, split_dataset_id
from .repo import Repo, write_sqlite_atomic

GRAPH_VERSION = 2
SENSITIVITY_LEVELS = ("public", "internal", "confidential", "restricted")
RESTRICTED_LEVELS = ("confidential", "restricted")
_ORDER = {name: i for i, name in enumerate(SENSITIVITY_LEVELS)}


def sensitivity_at_least(level: str, threshold: str = "confidential") -> bool:
    """Whether ``level`` is at least as sensitive as ``threshold`` (ranked public < internal < confidential < restricted)."""
    return _ORDER.get((level or "public").lower(), 0) >= _ORDER.get(threshold.lower(), 2)


def is_restricted(level: str | None) -> bool:
    """Whether a dataset at this level is withheld from a default export or query."""
    return (level or "public").lower() in RESTRICTED_LEVELS


@dataclass
class Project:
    id: str
    name: str
    file: str                          # relative path under the repo root
    content_hash: str = ""             # a digest of the project file's bytes (incremental skip)
    engine_fingerprint: str = ""
    sensitivity: str = "public"

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self.__dict__)


@dataclass
class Column:
    id: str
    dataset: str
    name: str
    role: str = ""
    kind: str = ""
    unit: str = ""
    is_entity: bool = False            # an endpoint of a link: a resolved key

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self.__dict__)


@dataclass
class Source:
    id: str
    dataset: str
    path: str = ""
    size: int | None = None
    sample: str = ""

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self.__dict__)


@dataclass
class Dataset:
    id: str
    project: str
    node: str
    title: str
    shape: str = ""
    rows: int | None = None
    rows_exact: bool = False
    time_column: str = ""
    source: str = ""
    version: str = ""                  # the step's plan hash (the dataset's version)
    sensitivity: str = "public"
    columns: list[Column] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)

    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    def to_dict(self) -> dict[str, Any]:
        return json_safe({"id": self.id, "project": self.project, "node": self.node, "title": self.title,
                          "shape": self.shape, "rows": self.rows, "rows_exact": self.rows_exact,
                          "time_column": self.time_column, "source": self.source, "version": self.version,
                          "sensitivity": self.sensitivity,
                          "columns": [c.to_dict() for c in self.columns],
                          "sources": [s.to_dict() for s in self.sources]})


@dataclass
class Edge:
    id: str
    kind: str                          # link | stack | align | near | containment
    left: str                          # dataset id
    right: str                         # dataset id
    left_on: str = ""
    right_on: str = ""
    cardinality: str = ""
    match_pct: float = 0.0
    confidence: float = 0.0
    evidence: str = ""
    left_version: str = ""
    right_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self.__dict__)

    @property
    def endpoints(self) -> tuple[str, str]:
        return (self.left, self.right)


class Graph:
    """An in-memory graph. Add vertices and edges, then query or persist."""

    def __init__(self) -> None:
        self.projects: dict[str, Project] = {}
        self.datasets: dict[str, Dataset] = {}
        self.edges: dict[str, Edge] = {}
        self.meta: dict[str, Any] = {"identity_version": IDENTITY_VERSION, "graph_version": GRAPH_VERSION}

    # ---------------------------------------------------------------- build
    def add_project(self, project: Project) -> Project:
        self.projects[project.id] = project
        return project

    def add_dataset(self, dataset: Dataset) -> Dataset:
        self.datasets[dataset.id] = dataset
        return dataset

    def add_edge(self, edge: Edge) -> Edge:
        self.edges[edge.id] = edge
        return edge

    # ---------------------------------------------------------------- views
    def datasets_of(self, project_id: str, *, allow_restricted: bool = False) -> list[Dataset]:
        out = [d for d in self.datasets.values() if d.project == project_id]
        if not allow_restricted:
            out = [d for d in out if not is_restricted(d.sensitivity)]
        return sorted(out, key=lambda d: d.id)

    def edges_of(self, dataset_id: str, *, allow_restricted: bool = False) -> list[Edge]:
        hidden = self._hidden(allow_restricted)
        return sorted((e for e in self.edges.values()
                       if dataset_id in e.endpoints and e.left not in hidden and e.right not in hidden),
                      key=lambda e: e.id)

    def _hidden(self, allow_restricted: bool) -> set[str]:
        if allow_restricted:
            return set()
        return {d.id for d in self.datasets.values() if is_restricted(d.sensitivity)}

    def all_edges(self, *, allow_restricted: bool = False, kind: str | None = None) -> list[Edge]:
        hidden = self._hidden(allow_restricted)
        out = [e for e in self.edges.values() if e.left not in hidden and e.right not in hidden]
        if kind:
            out = [e for e in out if e.kind == kind]
        return sorted(out, key=lambda e: e.id)

    def all_datasets(self, *, allow_restricted: bool = False) -> list[Dataset]:
        out = list(self.datasets.values())
        if not allow_restricted:
            out = [d for d in out if not is_restricted(d.sensitivity)]
        return sorted(out, key=lambda d: d.id)

    # ---------------------------------------------------------------- queries
    def query(self, *, kind: str | None = None, project: str | None = None, text: str | None = None,
              allow_restricted: bool = False, limit: int | None = None) -> dict[str, Any]:
        """Datasets and edges matching the filters. ``text`` matches a dataset's id, title, node, a column name
        or a source file name, case-insensitively."""
        datasets = self.all_datasets(allow_restricted=allow_restricted)
        edges = self.all_edges(allow_restricted=allow_restricted, kind=kind)
        if project:
            datasets = [d for d in datasets if d.project == project or d.id.startswith(project)]
            edges = [e for e in edges if split_dataset_id(e.left)[0] == project or split_dataset_id(e.right)[0] == project]
        if text:
            q = key_norm(text)

            def hit(d: Dataset) -> bool:
                hay = [d.id, d.title, d.node, d.source] + d.column_names()
                return any(q in key_norm(h) for h in hay if h)
            datasets = [d for d in datasets if hit(d)]
            keep = {d.id for d in datasets}
            edges = [e for e in edges if e.left in keep or e.right in keep]
        total = len(datasets)
        if limit is not None and limit >= 0:
            datasets = datasets[:limit]
        return {"datasets": [d.to_dict() for d in datasets], "edges": [e.to_dict() for e in edges],
                "count": len(datasets), "total": total}

    def neighbors(self, dataset_id: str, *, allow_restricted: bool = False) -> dict[str, Any]:
        """A dataset's immediate neighbours, with the edge that connects each: its links, stacks, alignments,
        nearest-place and containment relations. A restricted dataset is withheld unless allowed."""
        if dataset_id not in self.datasets:
            raise ValueError(f"No dataset {dataset_id!r} in the graph. Use graph_query to find one.")
        if is_restricted(self.datasets[dataset_id].sensitivity) and not allow_restricted:
            raise ValueError(f"{dataset_id} is restricted; pass allow_restricted to use it")
        edges = self.edges_of(dataset_id, allow_restricted=allow_restricted)
        out = []
        for e in edges:
            other = e.right if e.left == dataset_id else e.left
            ds = self.datasets.get(other)
            out.append({"dataset": ds.to_dict() if ds else {"id": other}, "edge": e.to_dict()})
        return {"dataset": self.datasets[dataset_id].to_dict(), "neighbors": out, "count": len(out)}

    def path(self, a: str, b: str, *, allow_restricted: bool = False) -> dict[str, Any]:
        """The shortest chain of datasets (and the edges between them) joining two datasets, if any. Ties are
        broken by id, so the answer is deterministic."""
        for nid in (a, b):
            if nid not in self.datasets:
                raise ValueError(f"No dataset {nid!r} in the graph")
        hidden = self._hidden(allow_restricted)
        if a in hidden or b in hidden:
            raise ValueError("One of those datasets is restricted; pass allow_restricted to use it")
        # deterministic BFS: neighbours visited in edge-id order
        adj: dict[str, list[tuple[str, str]]] = {}
        for e in sorted(self.edges.values(), key=lambda x: x.id):
            if e.left in hidden or e.right in hidden:
                continue
            adj.setdefault(e.left, []).append((e.id, e.right))
            adj.setdefault(e.right, []).append((e.id, e.left))
        from collections import deque
        prev: dict[str, tuple[str, str]] = {}
        seen = {a}
        q = deque([a])
        while q:
            cur = q.popleft()
            if cur == b:
                break
            for eid, nxt in sorted(adj.get(cur, []), key=lambda t: (t[0], t[1])):
                if nxt not in seen:
                    seen.add(nxt)
                    prev[nxt] = (cur, eid)
                    q.append(nxt)
        if b not in seen:
            return {"found": False, "from": a, "to": b, "datasets": [], "edges": [], "hops": None}
        nodes, edges = [b], []
        cur = b
        while cur != a:
            parent, eid = prev[cur]
            edges.append(eid)
            nodes.append(parent)
            cur = parent
        nodes.reverse()
        edges.reverse()
        return {"found": True, "from": a, "to": b, "datasets": nodes, "edges": edges, "hops": len(edges)}

    def shared_keys(self, *, allow_restricted: bool = False) -> list[dict[str, Any]]:
        """Every link between datasets of *different projects*: the same key reaching across a repository. Sorted
        by key name then dataset, deterministic."""
        out = []
        for e in self.all_edges(allow_restricted=allow_restricted, kind="link"):
            pl, _ = split_dataset_id(e.left)
            pr, _ = split_dataset_id(e.right)
            if pl == pr:
                continue
            out.append({"left": e.left, "right": e.right, "left_on": e.left_on, "right_on": e.right_on,
                        "cardinality": e.cardinality, "match_pct": e.match_pct, "confidence": e.confidence,
                        "evidence": e.evidence})
        out.sort(key=lambda r: (key_norm(r["left_on"]), r["left"], r["right"]))
        return out

    def summary(self) -> dict[str, Any]:
        kinds: dict[str, int] = {}
        for e in self.edges.values():
            kinds[e.kind] = kinds.get(e.kind, 0) + 1
        shapes: dict[str, int] = {}
        for d in self.datasets.values():
            shapes[d.shape] = shapes.get(d.shape, 0) + 1
        withheld = sum(1 for d in self.datasets.values() if is_restricted(d.sensitivity))
        return {"projects": len(self.projects), "datasets": len(self.datasets), "columns": sum(len(d.columns) for d in self.datasets.values()),
                "sources": sum(len(d.sources) for d in self.datasets.values()), "edges": len(self.edges),
                "edge_kinds": dict(sorted(kinds.items())), "shapes": dict(sorted(shapes.items())),
                "withheld_restricted": withheld,
                "meta": dict(sorted(self.meta.items(), key=lambda kv: kv[0]))}

    # ---------------------------------------------------------------- persist
    def to_sqlite(self, conn: sqlite3.Connection) -> None:
        """Write the whole graph into an empty SQLite database (used as the ``build`` callback of
        :func:`dancr.core.repo.write_sqlite_atomic`). Rows go in id order, so the file is deterministic."""
        conn.executescript(
            """
            CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE projects(id TEXT PRIMARY KEY, name TEXT, file TEXT, content_hash TEXT,
                                  engine_fingerprint TEXT, sensitivity TEXT);
            CREATE TABLE datasets(id TEXT PRIMARY KEY, project TEXT, node TEXT, title TEXT, shape TEXT,
                                  rows INTEGER, rows_exact INTEGER, time_column TEXT, source TEXT,
                                  version TEXT, sensitivity TEXT);
            CREATE TABLE columns(id TEXT PRIMARY KEY, dataset TEXT, name TEXT, role TEXT, kind TEXT,
                                 unit TEXT, is_entity INTEGER);
            CREATE TABLE sources(id TEXT PRIMARY KEY, dataset TEXT, path TEXT, size INTEGER, sample TEXT);
            CREATE TABLE edges(id TEXT PRIMARY KEY, kind TEXT, left_dataset TEXT, right_dataset TEXT,
                               left_on TEXT, right_on TEXT, cardinality TEXT, match_pct REAL,
                               confidence REAL, evidence TEXT, left_version TEXT, right_version TEXT);
            """
        )
        import json
        for k, v in sorted(self.meta.items(), key=lambda kv: kv[0]):
            conn.execute("INSERT INTO meta(key, value) VALUES (?, ?)", (str(k), json.dumps(json_safe(v), sort_keys=True)))
        for p in sorted(self.projects.values(), key=lambda x: x.id):
            conn.execute("INSERT INTO projects VALUES (?,?,?,?,?,?)",
                         (p.id, p.name, p.file, p.content_hash, p.engine_fingerprint, p.sensitivity))
        for d in sorted(self.datasets.values(), key=lambda x: x.id):
            conn.execute("INSERT INTO datasets VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                         (d.id, d.project, d.node, d.title, d.shape, d.rows, 1 if d.rows_exact else 0,
                          d.time_column, d.source, d.version, d.sensitivity))
            for c in sorted(d.columns, key=lambda x: x.id):
                conn.execute("INSERT INTO columns VALUES (?,?,?,?,?,?,?)",
                             (c.id, c.dataset, c.name, c.role, c.kind, c.unit, 1 if c.is_entity else 0))
            for s in sorted(d.sources, key=lambda x: x.id):
                conn.execute("INSERT INTO sources VALUES (?,?,?,?,?)", (s.id, s.dataset, s.path, s.size, s.sample))
        for e in sorted(self.edges.values(), key=lambda x: x.id):
            conn.execute("INSERT INTO edges VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                         (e.id, e.kind, e.left, e.right, e.left_on, e.right_on, e.cardinality,
                          e.match_pct, e.confidence, e.evidence, e.left_version, e.right_version))

    def write(self, path: Path | str) -> Path:
        return write_sqlite_atomic(path, self.to_sqlite)

    @classmethod
    def from_sqlite(cls, path: Path | str) -> Graph:
        """Load a graph written by :meth:`write`. Columns and sources are reattached to their dataset."""
        import json
        g = cls()
        conn = sqlite3.connect(str(path))
        conn.row_factory = sqlite3.Row
        try:
            g.meta = {}
            for row in conn.execute("SELECT key, value FROM meta"):
                try:
                    g.meta[row["key"]] = json.loads(row["value"])
                except ValueError:
                    g.meta[row["key"]] = row["value"]
            for r in conn.execute("SELECT * FROM projects"):
                g.add_project(Project(r["id"], r["name"], r["file"], r["content_hash"],
                                      r["engine_fingerprint"], r["sensitivity"]))
            for r in conn.execute("SELECT * FROM datasets"):
                g.add_dataset(Dataset(r["id"], r["project"], r["node"], r["title"], r["shape"], r["rows"],
                                      bool(r["rows_exact"]), r["time_column"], r["source"], r["version"],
                                      r["sensitivity"]))
            for r in conn.execute("SELECT * FROM columns ORDER BY id"):
                d = g.datasets.get(r["dataset"])
                if d is not None:
                    d.columns.append(Column(r["id"], r["dataset"], r["name"], r["role"], r["kind"],
                                            r["unit"], bool(r["is_entity"])))
            for r in conn.execute("SELECT * FROM sources ORDER BY id"):
                d = g.datasets.get(r["dataset"])
                if d is not None:
                    d.sources.append(Source(r["id"], r["dataset"], r["path"], r["size"], r["sample"]))
            for r in conn.execute("SELECT * FROM edges"):
                g.add_edge(Edge(r["id"], r["kind"], r["left_dataset"], r["right_dataset"], r["left_on"],
                                r["right_on"], r["cardinality"], r["match_pct"], r["confidence"],
                                r["evidence"], r["left_version"], r["right_version"]))
        finally:
            conn.close()
        return g


def load_graph(root: Path | str) -> Graph | None:
    """The repository's stored graph, or None when none has been built."""
    db = Repo(root).graph_db()
    if not db.is_file():
        return None
    try:
        return Graph.from_sqlite(db)
    except (sqlite3.Error, OSError, KeyError, IndexError, ValueError):
        return None                       # a damaged or differently-shaped database is rebuilt, never fatal


def graph_fingerprint(graph: Graph) -> str:
    """A short digest of a graph's content (meta excluded except versions), for a quick change check."""
    from .identity import digest
    payload = {"projects": [p.to_dict() for p in sorted(graph.projects.values(), key=lambda x: x.id)],
               "datasets": [d.to_dict() for d in sorted(graph.datasets.values(), key=lambda x: x.id)],
               "edges": [e.to_dict() for e in sorted(graph.edges.values(), key=lambda x: x.id)]}
    return digest(payload)
