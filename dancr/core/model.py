"""The pipeline document: nodes, edges, notes. Pure data, JSON on disk."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterator

from .registry import registry

FORMAT_VERSION = 2


class PipelineError(Exception):
    pass


@dataclass
class Node:
    id: str
    type: str
    title: str
    x: float = 0.0
    y: float = 0.0
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "type": self.type, "title": self.title,
                "x": round(self.x, 1), "y": round(self.y, 1), "params": self.params}


@dataclass
class Edge:
    source: str
    target: str
    port: str = "in"

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "target": self.target, "port": self.port}

    def key(self) -> tuple[str, str, str]:
        return (self.source, self.target, self.port)


@dataclass
class Note:
    id: str
    x: float
    y: float
    text: str
    width: float = 220.0
    height: float = 120.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_\-]*$")


def _coerce_input(v: Any) -> Any:
    """Inputs are numbers when they look like numbers, otherwise text."""
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, float)):
        return v
    s = str(v).strip()
    try:
        f = float(s.replace(",", ""))
        return int(f) if f.is_integer() and "." not in s and "e" not in s.lower() else f
    except ValueError:
        return s


def _num(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
        return f if f == f else default
    except (TypeError, ValueError):
        return default


@dataclass
class Input:
    """A named value people can use in formulas: a maximum allowed, a conversion factor, a threshold."""
    name: str
    value: Any
    unit: str = ""
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "value": self.value, "unit": self.unit, "note": self.note}


_INPUT_RE = re.compile(r"^[^\W\d][\w ]*$")


class Pipeline:
    def __init__(self, name: str = "Untitled") -> None:
        self.name = name
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        self.notes: list[Note] = []
        self.path: Path | None = None
        self.meta: dict[str, Any] = {}
        self.columns: dict[str, dict[str, Any]] = {}     # column name -> {"label": ..., "unit": ...}
        self.inputs: list[Input] = []

    # ---------------------------------------------------------------- inputs
    def input_values(self) -> dict[str, Any]:
        return {i.name: i.value for i in self.inputs}

    def set_input(self, name: str, value: Any = None, unit: str | None = None, note: str | None = None) -> Input:
        name = str(name).strip()
        if not _INPUT_RE.match(name):
            raise PipelineError(f"Input names must start with a letter and contain only letters, digits, spaces and underscores: {name!r}")
        for i in self.inputs:
            if i.name.lower() == name.lower():
                if value is not None:
                    i.value = _coerce_input(value)
                if unit is not None:
                    i.unit = unit
                if note is not None:
                    i.note = note
                return i
        inp = Input(name, _coerce_input(value), unit or "", note or "")
        self.inputs.append(inp)
        return inp

    def remove_input(self, name: str) -> None:
        self.inputs = [i for i in self.inputs if i.name.lower() != name.lower()]

    # ---------------------------------------------------------------- column registry
    def column_label(self, name: str) -> str:
        return (self.columns.get(name) or {}).get("label") or name

    def column_unit(self, name: str) -> str:
        return (self.columns.get(name) or {}).get("unit") or ""

    def column_title(self, name: str) -> str:
        """Display name with unit, e.g. 'Value (units)'."""
        u = self.column_unit(name)
        return f"{self.column_label(name)} ({u})" if u else self.column_label(name)

    def set_column_meta(self, name: str, label: str | None = None, unit: str | None = None) -> None:
        entry = dict(self.columns.get(name) or {})
        if label is not None:
            entry["label"] = label.strip()
        if unit is not None:
            entry["unit"] = unit.strip()
        entry = {k: v for k, v in entry.items() if v}
        if entry:
            self.columns[name] = entry
        else:
            self.columns.pop(name, None)

    # ------------------------------------------------------------------ ids
    def new_id(self, type_key: str) -> str:
        base = type_key
        n = 1
        while f"{base}_{n}" in self.nodes:
            n += 1
        return f"{base}_{n}"

    # ---------------------------------------------------------------- nodes
    def add_node(self, type_key: str, title: str | None = None, params: dict[str, Any] | None = None,
                 x: float = 0.0, y: float = 0.0, id: str | None = None, strict: bool = True) -> Node:
        nt = registry.get(type_key)
        node_id = id or self.new_id(type_key)
        if not _ID_RE.match(node_id):
            raise PipelineError(f"Invalid node id {node_id!r}")
        if node_id in self.nodes:
            raise PipelineError(f"Node id {node_id!r} already exists")
        node = Node(id=node_id, type=type_key, title=title or nt.label, x=x, y=y,
                    params=nt.normalize_params(params or {}, strict=strict))
        self.nodes[node_id] = node
        return node

    def remove_node(self, node_id: str) -> list[Edge]:
        self._require(node_id)
        removed = [e for e in self.edges if e.source == node_id or e.target == node_id]
        self.edges = [e for e in self.edges if e not in removed]
        del self.nodes[node_id]
        return removed

    def set_params(self, node_id: str, **changes: Any) -> dict[str, Any]:
        node = self._require(node_id)
        nt = registry.get(node.type)
        merged = dict(node.params)
        valid = {p.name for p in nt.params}
        for k, v in changes.items():
            if k not in valid:
                raise PipelineError(f"{nt.label}: unknown setting {k!r}. Valid: {sorted(valid)}")
            merged[k] = nt.param(k).coerce(v)      # strict for the keys being changed
        old = node.params
        node.params = nt.normalize_params(merged, strict=False)   # lenient for untouched keys
        return old

    def rename_node(self, node_id: str, title: str) -> None:
        self._require(node_id).title = title

    def _require(self, node_id: str) -> Node:
        if node_id not in self.nodes:
            raise PipelineError(f"No node with id {node_id!r}. Nodes: {list(self.nodes)}")
        return self.nodes[node_id]

    # ---------------------------------------------------------------- edges
    def connect(self, source: str, target: str, port: str | None = None) -> Edge:
        src = self._require(source)
        dst = self._require(target)
        if source == target:
            raise PipelineError("A node cannot feed itself")
        dst_type = registry.get(dst.type)
        if dst_type.kind == "source" or not dst_type.inputs:
            raise PipelineError(f"{dst.title} does not take inputs")
        if port is None:
            taken = self.inputs_of(target)
            port = dst_type.route(src.type, taken) if dst_type.route else None
            if port is None:            # first port that still has room
                port = next((i.name for i in dst_type.inputs if i.multiple or not taken.get(i.name)), dst_type.inputs[0].name)
        spec = next((i for i in dst_type.inputs if i.name == port), None)
        if spec is None:
            raise PipelineError(f"{dst.title} has no input called {port!r}. Inputs: {[i.name for i in dst_type.inputs]}")
        if not spec.multiple:
            for e in self.edges:
                if e.target == target and e.port == port:
                    raise PipelineError(f"{dst.title}.{port} is already connected to {e.source}. Disconnect it first.")
        edge = Edge(source, target, port)
        if edge.key() in {e.key() for e in self.edges}:
            return edge
        if target in self.upstream_closure(source):
            raise PipelineError("That connection would create a loop")
        self.edges.append(edge)
        return edge

    def disconnect(self, source: str, target: str, port: str | None = None) -> None:
        before = len(self.edges)
        self.edges = [e for e in self.edges
                      if not (e.source == source and e.target == target and (port is None or e.port == port))]
        if len(self.edges) == before:
            raise PipelineError(f"No connection {source} -> {target}")

    def inputs_of(self, node_id: str) -> dict[str, list[str]]:
        """{port: [source ids in connection order]}"""
        out: dict[str, list[str]] = {}
        for e in self.edges:
            if e.target == node_id:
                out.setdefault(e.port, []).append(e.source)
        return out

    def outputs_of(self, node_id: str) -> list[str]:
        return [e.target for e in self.edges if e.source == node_id]

    def upstream_closure(self, node_id: str) -> set[str]:
        seen: set[str] = set()
        stack = [s for ins in self.inputs_of(node_id).values() for s in ins]
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            stack.extend(s for ins in self.inputs_of(n).values() for s in ins)
        return seen

    def downstream_closure(self, node_id: str) -> set[str]:
        seen: set[str] = set()
        stack = self.outputs_of(node_id)
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            stack.extend(self.outputs_of(n))
        return seen

    def topological_order(self, targets: list[str] | None = None) -> list[str]:
        """Kahn's algorithm. If targets given, only nodes needed to compute them."""
        wanted: set[str] | None = None
        if targets is not None:
            unknown = [t for t in targets if t not in self.nodes]
            if unknown:
                raise PipelineError(f"No node called {unknown[0]!r}. Nodes: {list(self.nodes)}")
            wanted = set(targets)
            for t in targets:
                wanted |= self.upstream_closure(t)
        indeg = {n: 0 for n in self.nodes if wanted is None or n in wanted}
        for e in self.edges:
            if e.target in indeg and e.source in indeg:
                indeg[e.target] += 1
        ready = [n for n in self.nodes if n in indeg and indeg[n] == 0]  # insertion order = stable
        order: list[str] = []
        while ready:
            n = ready.pop(0)
            order.append(n)
            for e in self.edges:
                if e.source == n and e.target in indeg:
                    indeg[e.target] -= 1
                    if indeg[e.target] == 0:
                        ready.append(e.target)
        if len(order) != len(indeg):
            raise PipelineError("The pipeline contains a loop")
        return order

    # ---------------------------------------------------------------- notes
    def add_note(self, text: str, x: float = 0, y: float = 0, id: str | None = None) -> Note:
        nid = id or f"note_{len(self.notes) + 1}"
        while any(n.id == nid for n in self.notes):
            nid += "_"
        note = Note(nid, x, y, text)
        self.notes.append(note)
        return note

    # ----------------------------------------------------------- validation
    def problems(self) -> list[str]:
        """Human-readable configuration problems (missing inputs, missing required settings)."""
        out: list[str] = []
        for node in self.nodes.values():
            nt = registry.get(node.type)
            ins = self.inputs_of(node.id)
            for spec in nt.inputs:
                if not spec.optional and not ins.get(spec.name):
                    out.append(f"{node.title}: nothing is connected to its {spec.label.lower()}")
            for msg in nt.param_problems(node.params):
                out.append(f"{node.title}: {msg}")
        return out

    # ------------------------------------------------------------- (de)serialize
    def to_dict(self) -> dict[str, Any]:
        return {
            "dancr": FORMAT_VERSION,
            "name": self.name,
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges],
            "notes": [n.to_dict() for n in self.notes],
            "inputs": [i.to_dict() for i in self.inputs],
            "columns": self.columns,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: Any, path: Path | None = None) -> "Pipeline":
        if not isinstance(data, dict) or data.get("dancr") != FORMAT_VERSION:
            raise PipelineError("This file is not a DANCR 2 pipeline (missing \"dancr\": 2)")
        try:
            return cls._from_dict(data, path)
        except PipelineError:
            raise
        except (KeyError, TypeError, AttributeError, ValueError) as e:
            raise PipelineError(f"This pipeline file is damaged: {type(e).__name__}: {e}") from e

    @classmethod
    def _from_dict(cls, data: dict[str, Any], path: Path | None) -> "Pipeline":
        p = cls(str(data.get("name") or "Untitled"))
        p.path = path
        meta = data.get("meta") or {}
        p.meta = dict(meta) if isinstance(meta, dict) else {}
        for nd in data.get("nodes") or []:
            if not isinstance(nd, dict) or "id" not in nd or "type" not in nd:
                raise PipelineError("A step in the file has no id or type")
            if not registry.has(nd["type"]):
                raise PipelineError(f"Unknown step type {nd['type']!r} (is this file from a newer DANCR?)")
            params = nd.get("params") or {}
            if not isinstance(params, dict):
                raise PipelineError(f"Step {nd['id']} has malformed settings")
            p.add_node(nd["type"], title=str(nd.get("title") or "") or None, params=params,
                       x=_num(nd.get("x")), y=_num(nd.get("y")), id=str(nd["id"]), strict=False)
        for ed in data.get("edges") or []:
            if not isinstance(ed, dict) or "source" not in ed or "target" not in ed:
                raise PipelineError("A connection in the file is malformed")
            p.connect(str(ed["source"]), str(ed["target"]), ed.get("port"))
        for nd in data.get("notes") or []:
            if not isinstance(nd, dict):
                continue
            n = p.add_note(str(nd.get("text", "")), _num(nd.get("x")), _num(nd.get("y")), id=nd.get("id"))
            n.width = _num(nd.get("width"), n.width)
            n.height = _num(nd.get("height"), n.height)
        for i in data.get("inputs") or []:
            if isinstance(i, dict) and i.get("name"):
                try:
                    p.set_input(str(i["name"]), i.get("value"), str(i.get("unit") or ""), str(i.get("note") or ""))
                except PipelineError:
                    continue
        cols = data.get("columns") or {}
        if isinstance(cols, dict):
            p.columns = {str(k): {kk: str(vv) for kk, vv in v.items() if kk in ("label", "unit") and vv}
                         for k, v in cols.items() if isinstance(v, dict)}
        return p

    def dumps(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n"

    def save(self, path: Path | str | None = None) -> Path:
        target = Path(path).expanduser().resolve() if path is not None else self.path
        if target is None:
            raise PipelineError("No file path to save to")
        if target.is_dir():
            raise PipelineError(f"{target} is a folder; choose a file name")
        old_path = self.path
        self.path = target                      # so relative paths serialise against the new folder
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            text = self.dumps()
            if target.exists():
                try:
                    if target.read_text(encoding="utf-8") != text:
                        self._keep_version(target)
                except OSError:
                    pass
            tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, target)
        except Exception:
            self.path = old_path
            raise
        return target

    # ---------------------------------------------------------------- versions
    MAX_VERSIONS = 50

    @staticmethod
    def versions_dir(path: Path) -> Path:
        return path.parent / ".dancr" / "versions" / path.stem

    @classmethod
    def _keep_version(cls, target: Path) -> None:
        """Copy the current file into the versions folder before overwriting it."""
        from datetime import datetime
        import shutil
        vdir = cls.versions_dir(target)
        vdir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.fromtimestamp(target.stat().st_mtime).strftime("%Y-%m-%d_%H-%M-%S")
        dest = vdir / f"{stamp}.json"
        if not dest.exists():
            shutil.copy2(target, dest)
        old = sorted(vdir.glob("*.json"))
        for f in old[:-cls.MAX_VERSIONS]:
            f.unlink(missing_ok=True)

    def versions(self) -> list[Path]:
        """Saved earlier versions of this file, newest first."""
        if self.path is None:
            return []
        vdir = self.versions_dir(self.path)
        return sorted(vdir.glob("*.json"), reverse=True) if vdir.exists() else []

    @classmethod
    def load(cls, path: Path | str) -> "Pipeline":
        path = Path(path).expanduser().resolve()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise PipelineError(f"{path.name} is not valid JSON: {e}") from e
        return cls.from_dict(data, path)

    @property
    def directory(self) -> Path:
        return self.path.parent if self.path else Path.cwd()

    def __iter__(self) -> Iterator[Node]:
        return iter(self.nodes.values())

    def __len__(self) -> int:
        return len(self.nodes)
