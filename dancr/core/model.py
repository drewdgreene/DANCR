"""The pipeline document: nodes, edges, notes. Pure data, JSON on disk."""
from __future__ import annotations

import json
import os
import re
import uuid
from collections import deque
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


@dataclass
class Answer:
    """A question and the steps that answer it.

    An Answer is a bookmark, not a pipeline step: it is *not* connected to the dataflow and the executor
    ignores it entirely. It keeps the question as a spec (see ``recipes.py``), what was assumed on the
    person's behalf, and which steps it built, so the question can be changed later without disturbing
    steps the person edited by hand. The canvas draws it as an unconnected card; the rail lists it.
    """
    id: str
    title: str
    x: float
    y: float
    terminal: str                      # node id whose output answers the question
    view: str = "chart"                # how to show it: chart | table
    spec: dict[str, Any] = field(default_factory=dict)          # the question
    steps: dict[str, dict[str, Any]] = field(default_factory=dict)   # plan key -> {"node", "made", "title"}
    assumptions: list[dict[str, Any]] = field(default_factory=list)
    rules: int = 0                     # the recipes' RULES_VERSION this was planned with

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "title": self.title, "x": round(self.x, 1), "y": round(self.y, 1),
                "terminal": self.terminal, "view": self.view, "spec": self.spec, "steps": self.steps,
                "assumptions": self.assumptions, "rules": self.rules}

    @property
    def nodes(self) -> list[str]:
        """The steps this answer built (not the tables it started from)."""
        return [s["node"] for s in self.steps.values() if s.get("node")]


_ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_\-]*$")


def _version_number(f: Path) -> int | None:
    m = re.match(r"^(\d{6,})_", f.name)
    return int(m.group(1)) if m else None


def _coerce_input(v: Any) -> Any:
    """Inputs are numbers when they look like numbers (read as everywhere else: '1,5' is 1.5, whole numbers
    stay exact), otherwise text. Lists and objects are refused rather than stored as their Python text."""
    from .dtypes import typed_value
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
        raise PipelineError(f"An input must be a finite number or text, not {v!r}")
    if isinstance(v, (int, float)):
        return v
    if isinstance(v, (list, dict, tuple, set)):
        raise PipelineError("An input holds one number or one piece of text, not a list or an object")
    s = str(v).strip()
    n = typed_value(s)
    return s if n is None else n


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


def rebase_params(node_type: str, params: dict[str, Any], old_dir: Path, new_dir: Path) -> dict[str, Any]:
    """``params`` with every path setting rewritten for a project moving from ``old_dir`` to ``new_dir``,
    so it still points at the same file (relative when that file is inside ``new_dir``, else absolute)."""
    try:
        nt = registry.get(node_type)
    except KeyError:
        return params
    out = params
    for prm in nt.params:
        v = params.get(prm.name)
        if prm.kind != "path" or not isinstance(v, str) or not v.strip():
            continue
        p = Path(v).expanduser()
        full = p if p.is_absolute() else Path(os.path.normpath(old_dir.resolve() / p))
        out = {**out, prm.name: portable_path(full, new_dir)}
    return out


def portable_path(path: Path, base: Path) -> str:
    """How a file is written into a path setting: relative when it is inside the project folder
    (so the folder can be moved or shared), else absolute."""
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return str(path)


_INPUT_RE = re.compile(r"^[^\W\d][\w ]*$")


class Pipeline:
    def __init__(self, name: str = "Untitled") -> None:
        self.name = name
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        self.notes: list[Note] = []
        self.answers: list[Answer] = []
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
        edge = self.plan_connect(source, target, port)
        if edge.key() not in {e.key() for e in self.edges}:
            self.edges.append(edge)
        return edge

    def plan_connect(self, source: str, target: str, port: str | None = None, ignoring: Edge | None = None) -> Edge:
        """The connection `connect` would make, checked (raises PipelineError) but not made. ``ignoring``: judge it
        as if that connection were already gone (moving a connection from one input to another)."""
        edges = [e for e in self.edges if ignoring is None or e.key() != ignoring.key()]
        src = self._require(source)
        dst = self._require(target)
        if source == target:
            raise PipelineError("A node cannot feed itself")
        dst_type = registry.get(dst.type)
        if dst_type.kind == "source" or not dst_type.inputs:
            raise PipelineError(f"{dst.title} does not take inputs")
        if port is None:
            taken = {}
            for e in edges:
                if e.target == target:
                    taken.setdefault(e.port, []).append(e.source)
            port = dst_type.route(src.type, taken) if dst_type.route else None
            if port is None:            # first port that still has room
                port = next((i.name for i in dst_type.inputs if i.multiple or not taken.get(i.name)), dst_type.inputs[0].name)
        spec = next((i for i in dst_type.inputs if i.name == port), None)
        if spec is None:
            raise PipelineError(f"{dst.title} has no input called {port!r}. Inputs: {[i.name for i in dst_type.inputs]}")
        if not spec.multiple:
            for e in edges:
                if e.target == target and e.port == port:
                    raise PipelineError(f"{dst.title}.{port} is already connected to {e.source}. Disconnect it first.")
        edge = Edge(source, target, port)
        if edge.key() in {e.key() for e in edges}:
            return edge
        if target in self.upstream_closure(source):
            raise PipelineError("That connection would create a loop")
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
        into: dict[str, list[str]] = {}                      # one pass over the edges, not one per step visited
        for e in self.edges:
            into.setdefault(e.target, []).append(e.source)
        seen: set[str] = set()
        stack = list(into.get(node_id, []))
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            stack.extend(into.get(n, []))
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
        out: dict[str, list[str]] = {}                       # one pass over the edges, not one per node
        for e in self.edges:
            if e.target in indeg and e.source in indeg:
                indeg[e.target] += 1
                out.setdefault(e.source, []).append(e.target)
        ready = deque(n for n in self.nodes if n in indeg and indeg[n] == 0)  # insertion order = stable
        order: list[str] = []
        while ready:
            n = ready.popleft()
            order.append(n)
            for t in out.get(n, []):
                indeg[t] -= 1
                if indeg[t] == 0:
                    ready.append(t)
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

    # ---------------------------------------------------------------- answers
    def add_answer(self, title: str, terminal: str, x: float = 0.0, y: float = 0.0, view: str = "chart",
                   spec: dict[str, Any] | None = None, id: str | None = None, steps: dict | None = None,
                   assumptions: list | None = None, rules: int = 0) -> Answer:
        aid = id or self._new_answer_id()
        if not _ID_RE.match(aid):
            raise PipelineError(f"Invalid answer id {aid!r}")
        if any(a.id == aid for a in self.answers):
            raise PipelineError(f"Answer id {aid!r} already exists")
        answer = Answer(aid, title or "Answer", x, y, terminal, view or "chart", dict(spec or {}),
                        dict(steps or {}), list(assumptions or []), int(rules or 0))
        self.answers.append(answer)
        return answer

    def remove_answer(self, answer_id: str) -> Answer | None:
        for a in self.answers:
            if a.id == answer_id:
                self.answers.remove(a)
                return a
        return None

    def answer(self, answer_id: str) -> Answer | None:
        return next((a for a in self.answers if a.id == answer_id), None)

    def _new_answer_id(self) -> str:
        taken = {a.id for a in self.answers}
        n = 1
        while f"answer_{n}" in taken:
            n += 1
        return f"answer_{n}"

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
            "answers": [a.to_dict() for a in self.answers],
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
        for ad in data.get("answers") or []:
            if not isinstance(ad, dict) or not ad.get("id") or not ad.get("terminal") or not isinstance(ad.get("spec"), dict):
                continue
            steps = {str(k): dict(v) for k, v in (ad.get("steps") or {}).items() if isinstance(v, dict) and v.get("node")}
            try:
                p.add_answer(str(ad.get("title") or "Answer"), str(ad["terminal"]), _num(ad.get("x")), _num(ad.get("y")),
                             str(ad.get("view") or "chart"), dict(ad["spec"]), id=str(ad["id"]), steps=steps,
                             assumptions=[a for a in ad.get("assumptions") or [] if isinstance(a, dict)],
                             rules=int(ad.get("rules") or 0))
            except (PipelineError, TypeError, ValueError):
                continue
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

    def _rebase_paths(self, old_dir: Path, new_dir: Path) -> None:
        """Save As to another folder: every path setting keeps pointing at the same file (relative when
        that file is inside the new folder, absolute otherwise)."""
        for node in self.nodes.values():
            node.params = rebase_params(node.type, node.params, old_dir, new_dir)

    def dumps(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n"

    def save(self, path: Path | str | None = None, auto: bool = False) -> Path:
        """Write the project atomically, keeping the file it replaces as an earlier version. ``auto`` marks
        an autosave, whose versions are kept apart so minute-by-minute copies never push out real saves."""
        target = Path(path).expanduser().resolve() if path is not None else self.path
        if target is None:
            raise PipelineError("No file path to save to")
        if target.is_dir():
            raise PipelineError(f"{target} is a folder; choose a file name")
        old_path = self.path
        old_dir = self.directory
        before = ({nid: n.params for nid, n in self.nodes.items()}, dict(self.meta))   # restored if the write fails
        if auto:
            self.meta["autosaved"] = True
        else:
            self.meta.pop("autosaved", None)
        self.path = target
        if old_dir.resolve() != target.parent:
            self._rebase_paths(old_dir, target.parent)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            text = self.dumps()
            if target.exists():
                try:
                    if target.read_text(encoding="utf-8") != text:
                        self._keep_version(target)
                except OSError:
                    pass
            tmp = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")   # unique per writer
            try:
                tmp.write_text(text, encoding="utf-8")
                os.replace(tmp, target)
            finally:
                tmp.unlink(missing_ok=True)
        except Exception:
            self.path = old_path
            params, self.meta = before
            for nid, prm in params.items():
                self.nodes[nid].params = prm
            raise
        return target

    # ---------------------------------------------------------------- versions
    MAX_VERSIONS = 50          # copies of files you saved
    MAX_AUTO_VERSIONS = 30     # copies of files autosave wrote, kept apart

    @staticmethod
    def versions_dir(path: Path) -> Path:
        return path.parent / ".dancr" / "versions" / path.stem

    @classmethod
    def _keep_version(cls, target: Path) -> None:
        """Copy the current file into the versions folder before overwriting it. Each copy is numbered one after
        the last, so the order never depends on clocks: two saves in the same second (or on a drive that keeps
        times to two seconds), or either side of the clocks going back, still list newest first."""
        from datetime import datetime
        import shutil
        vdir = cls.versions_dir(target)
        vdir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.fromtimestamp(target.stat().st_mtime).strftime("%Y-%m-%d_%H-%M-%S")
        try:
            auto = bool((json.loads(target.read_text(encoding="utf-8")).get("meta") or {}).get("autosaved"))
        except (OSError, ValueError, AttributeError):
            auto = False
        kept = cls._numbered(vdir)
        if kept and kept[-1].read_bytes() == target.read_bytes():
            return                                   # this very copy is already the latest version
        seq = _version_number(kept[-1]) + 1 if kept else 1
        shutil.copy2(target, vdir / f"{seq:06d}_{stamp}{'.auto' if auto else ''}.json")
        kept = cls._numbered(vdir)
        autos = [f for f in kept if f.name.endswith(".auto.json")]
        saved = [f for f in kept if not f.name.endswith(".auto.json")]
        for f in saved[:-cls.MAX_VERSIONS] + autos[:-cls.MAX_AUTO_VERSIONS]:
            f.unlink(missing_ok=True)

    @staticmethod
    def _numbered(vdir: Path) -> list[Path]:
        """The kept versions, oldest first."""
        return sorted((f for f in vdir.glob("*.json") if _version_number(f) is not None), key=_version_number)

    def versions(self) -> list[Path]:
        """Saved earlier versions of this file, newest first."""
        if self.path is None:
            return []
        vdir = self.versions_dir(self.path)
        return list(reversed(self._numbered(vdir))) if vdir.exists() else []

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
