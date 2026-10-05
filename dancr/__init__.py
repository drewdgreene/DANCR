"""DANCR: Data Analysis Node-based Canvas for Research.

The same engine the window, the command line and the MCP server use is importable as a Python API:

    import dancr

    project = dancr.read_project("shop.json")
    with dancr.editing("shop.json") as p:        # changes are saved under the project file's lock
        dancr.add_step(p, "load_file", {"path": "orders.csv"}, node_id="orders")
    context = dancr.build_context(project)        # a knowledge-base document for a RAG to index

For an AI agent, the MCP server (``dancr mcp``) is the preferred surface; this package is for scripts and
notebooks. Names are imported on first use, so ``import dancr`` stays light and does not pull in Qt.
"""
from __future__ import annotations

import importlib
from typing import Any

__version__ = "2.1.0"

# The stable, documented public surface. Every name maps to (module, attribute) and is imported on first use,
# so `import dancr` imports nothing heavy. `__all__` lists the same names for `from dancr import *` and for docs.
_PUBLIC: dict[str, tuple[str, str]] = {
    # the pipeline document and the runtime
    "Pipeline": ("dancr.core", "Pipeline"),
    "Node": ("dancr.core", "Node"),
    "Edge": ("dancr.core", "Edge"),
    "Note": ("dancr.core", "Note"),
    "Input": ("dancr.core", "Input"),
    "PipelineError": ("dancr.core", "PipelineError"),
    "Executor": ("dancr.core.executor", "Executor"),
    "NodeState": ("dancr.core.executor", "NodeState"),
    "registry": ("dancr.core", "registry"),
    "NodeType": ("dancr.core", "NodeType"),
    "InputSpec": ("dancr.core", "InputSpec"),
    "Ctx": ("dancr.core", "Ctx"),
    "NodeResult": ("dancr.core", "NodeResult"),
    "Param": ("dancr.core", "Param"),
    # project files, and one writer at a time
    "read_project": ("dancr.headless", "read_project"),
    "editing": ("dancr.headless", "editing"),
    "project_lock": ("dancr.headless", "project_lock"),
    "write_text_atomic": ("dancr.headless", "write_text_atomic"),
    "ProjectBusy": ("dancr.headless", "ProjectBusy"),
    "StepFailed": ("dancr.headless", "StepFailed"),
    "LOCK_WAIT": ("dancr.headless", "LOCK_WAIT"),
    # understanding, asking, running
    "data_model": ("dancr.headless", "data_model"),
    "suggestions": ("dancr.headless", "suggestions"),
    "build_answer": ("dancr.headless", "build_answer"),
    "ask_question": ("dancr.headless", "ask_question"),
    "change_answer": ("dancr.headless", "change_answer"),
    "connection_map": ("dancr.headless", "connection_map"),
    "assistant_turn": ("dancr.headless", "assistant_turn"),
    "add_step": ("dancr.headless", "add_step"),
    "run_batch": ("dancr.headless", "run_batch"),
    "watch": ("dancr.headless", "watch"),
    "build_template": ("dancr.headless", "build_template"),
    "result_frame": ("dancr.headless", "result_frame"),
    "run_record": ("dancr.headless", "run_record"),
    "node_record": ("dancr.headless", "node_record"),
    # the knowledge-base context document (schema + stats + samples + doc cards)
    "build_context": ("dancr.headless", "build_context"),
    "context_jsonl": ("dancr.headless", "context_jsonl"),
    "context_text": ("dancr.headless", "context_text"),
    "context_changes": ("dancr.headless", "context_changes"),
    "parse_context": ("dancr.headless", "parse_context"),
    "build_catalog": ("dancr.headless", "build_catalog"),
    "catalog_jsonl": ("dancr.headless", "catalog_jsonl"),
    "find_pipelines": ("dancr.headless", "find_pipelines"),
    "CONTEXT_VERSION": ("dancr.headless", "CONTEXT_VERSION"),
    # FAIR descriptors (schema.org, Frictionless, run manifest, RO-Crate)
    "export_fair": ("dancr.headless", "export_fair"),
    "package_rocrate": ("dancr.headless", "package_rocrate"),
    "dataset_jsonld": ("dancr.core.fair", "dataset_jsonld"),
    "datapackage": ("dancr.core.fair", "datapackage"),
    "run_manifest": ("dancr.core.fair", "run_manifest"),
    "ro_crate": ("dancr.core.fair", "ro_crate"),
    # the compact profile the Assistant reads (the schema half of the context document)
    "project_profile": ("dancr.core.profile", "project_profile"),
    "table_card": ("dancr.core.profile", "table_card"),
}

__all__ = ["__version__", *_PUBLIC]


def __getattr__(name: str) -> Any:
    target = _PUBLIC.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(target[0]), target[1])


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_PUBLIC))
