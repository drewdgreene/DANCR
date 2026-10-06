"""FAIR descriptors for a project's datasets: schema.org/Dataset, a Frictionless Data Package, a run manifest
(provenance), and a minimal RO-Crate metadata graph.

These are *descriptors*, deterministic and standard-shaped, built from what DANCR already knows — the
knowledge-base context document (:func:`dancr.headless.build_context`) and the executor's plan hashes. They
make a project's tables findable (Google Dataset Search, any schema.org consumer), interoperable (Frictionless
consumers, generic metadata), and reusable (who made it, under what license, how to cite it, and exactly what
was run to produce it). Nothing is inferred by a model and nothing here computes data: it is a mapping.

Pure core: no Qt, no execution. The dataset-level fields come from ``Pipeline.dataset_meta()`` (kept in
``meta["dataset"]``); the schema, ranges and relations come from the context document.

The four serializers live in their own modules (``schemaorg``, ``frictionless``, ``manifest``, ``rocrate``);
this module re-exports them and holds the by-name dispatcher.
"""
from __future__ import annotations

import json
from typing import Any

from .frictionless import datapackage
from .manifest import run_manifest
from .rocrate import ro_crate, ro_crate_graph
from .schemaorg import dataset_jsonld

__all__ = ["dataset_jsonld", "datapackage", "run_manifest", "ro_crate", "ro_crate_graph",
           "fair_document", "dump", "FORMATS"]

FORMATS = ("schema.org", "frictionless", "manifest", "rocrate")


def fair_document(ctx: dict[str, Any], fmt: str, *, pipe=None, executor=None, manifest: dict[str, Any] | None = None,
                  meta: dict[str, Any] | None = None, pipeline_file: str | None = None) -> Any:
    """One descriptor by name: ``schema.org`` | ``frictionless`` | ``manifest`` | ``rocrate``."""
    key = (fmt or "").strip().lower().replace("_", ".")
    if key in ("schema.org", "schemaorg", "schema", "jsonld", "dataset"):
        return dataset_jsonld(ctx, meta)
    if key in ("frictionless", "datapackage", "data-package"):
        return datapackage(ctx, meta)
    if key in ("manifest", "provenance", "run"):
        if manifest is not None:
            return manifest
        if pipe is None:
            raise ValueError("A run manifest needs the project")
        return run_manifest(pipe, executor, meta=meta)
    if key in ("rocrate", "ro-crate"):
        return ro_crate(ctx, manifest, meta, pipeline_file=pipeline_file)
    raise ValueError(f"Unknown format {fmt!r}. Choose one of: {', '.join(FORMATS)}")


def dump(document: Any) -> str:
    """A descriptor as indented JSON text."""
    from ..dtypes import json_safe
    return json.dumps(json_safe(document), indent=2, ensure_ascii=False, default=str) + "\n"
