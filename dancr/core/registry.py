"""Node type registry.

A :class:`NodeType` bundles a plain-English label, input ports, declared
params and an ``apply`` function that turns input LazyFrames into an output
LazyFrame. Nothing here touches Qt.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import polars as pl

from .params import Param

log = logging.getLogger("dancr")


@dataclass
class InputSpec:
    name: str = "in"
    label: str = "Input"
    multiple: bool = False     # accepts many upstream nodes on this port
    optional: bool = False


@dataclass
class NodeResult:
    frame: pl.LazyFrame
    report: dict[str, Any] = field(default_factory=dict)   # extra findings (fit coefficients, gap counts...)
    messages: list[str] = field(default_factory=list)      # human-readable notes for the log


@dataclass
class Ctx:
    """What a node's apply() gets besides its inputs."""
    pipeline_dir: Path
    node_id: str
    node_title: str
    preview: bool = False           # True when computing a quick preview on a sample
    cache_dir: Path | None = None
    logger: logging.Logger = log
    item_meta: list | None = None      # set by the executor for the report node
    inputs: dict[str, Any] | None = None            # named project inputs (belt area, permit limit, ...)
    upstream_meta: dict[str, list[dict]] | None = None   # {port: [{"title","node_type","params","messages","report"}]}
    columns: dict[str, dict] | None = None          # column registry: name -> {"label", "unit"}

    def resolve(self, path: str) -> Path:
        p = Path(path).expanduser()
        return p if p.is_absolute() else (self.pipeline_dir / p)


ApplyFn = Callable[[Ctx, dict[str, list[pl.LazyFrame]], dict[str, Any]], "pl.LazyFrame | NodeResult"]


@dataclass
class NodeType:
    key: str
    label: str
    category: str
    description: str
    apply: ApplyFn
    inputs: list[InputSpec] = field(default_factory=lambda: [InputSpec()])
    params: list[Param] = field(default_factory=list)
    kind: str = "transform"        # "source" | "transform" | "sink"
    summary: Callable[[dict[str, Any]], str] | None = None   # short subtitle for the canvas
    icon: str = ""                 # single glyph/emoji for palette
    materialize: bool = True       # write output to the cache; False for pure pass-through nodes
    route: Callable[[str, dict[str, list[str]]], str | None] | None = None   # (source node type, taken ports) -> port to use when none is given
    help_md: str = ""

    def param(self, name: str) -> Param:
        for p in self.params:
            if p.name == name:
                return p
        raise KeyError(name)

    def defaults(self) -> dict[str, Any]:
        return {p.name: p.default_value() for p in self.params}

    def normalize_params(self, params: dict[str, Any], strict: bool = True) -> dict[str, Any]:
        """Fill defaults and coerce types. Unknown keys raise. With strict=False,
        values that fail to coerce are kept as-is (problems() reports them)."""
        out = self.defaults()
        for k, v in params.items():
            if k not in out:
                if strict:
                    raise ValueError(f"{self.label}: unknown setting {k!r}. Valid: {list(out)}")
                continue
            try:
                out[k] = self.param(k).coerce(v)
            except ValueError:
                if strict:
                    raise
                out[k] = v
        return out

    def param_problems(self, params: dict[str, Any]) -> list[str]:
        out = []
        for p in self.params:
            if not p.is_visible(params):
                continue
            v = params.get(p.name)
            try:
                p.coerce(v)
            except ValueError as e:
                out.append(str(e))
                continue
            if p.required and v in (None, "", [], {}):
                out.append(f"{p.label} is not set")
        return out

    def summarize(self, params: dict[str, Any]) -> str:
        if self.summary:
            try:
                return self.summary(params)
            except Exception:
                return ""
        return ""

    def to_json(self) -> dict[str, Any]:
        return {
            "key": self.key, "label": self.label, "category": self.category,
            "description": self.description, "kind": self.kind,
            "inputs": [{"name": i.name, "label": i.label, "multiple": i.multiple, "optional": i.optional}
                       for i in self.inputs],
            "params": [p.to_json() for p in self.params],
        }


class Registry:
    def __init__(self) -> None:
        self._types: dict[str, NodeType] = {}
        self._loaded = False
        self._lock = threading.RLock()

    def register(self, nt: NodeType) -> NodeType:
        if nt.key in self._types:
            raise ValueError(f"Node type {nt.key!r} already registered")
        self._types[nt.key] = nt
        return nt

    def _ensure(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if not self._loaded:
                from . import nodes  # noqa: F401  (registers everything)
                self._loaded = True

    def get(self, key: str) -> NodeType:
        self._ensure()
        try:
            return self._types[key]
        except KeyError:
            raise KeyError(f"Unknown node type {key!r}. Known: {sorted(self._types)}") from None

    def has(self, key: str) -> bool:
        self._ensure()
        return key in self._types

    def all(self) -> list[NodeType]:
        self._ensure()
        return list(self._types.values())

    def by_category(self) -> dict[str, list[NodeType]]:
        out: dict[str, list[NodeType]] = {}
        for nt in self.all():
            out.setdefault(nt.category, []).append(nt)
        return out

    def to_json(self) -> list[dict[str, Any]]:
        return [nt.to_json() for nt in self.all()]


registry = Registry()
