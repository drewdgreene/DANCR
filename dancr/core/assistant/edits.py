"""Apply a batch of Assistant project edits (rename, set_params, set_input, column_label) to a pipeline.

The edits are validated first (``core.assistant.tools``), but this is the single place that changes the
pipeline, so the window (inside one undo step) and a headless caller (``dancr assistant --build``, the MCP
``assistant`` tool, or the MCP ``apply_edits`` tool) behave identically.
"""
from __future__ import annotations

from typing import Any


def apply_edits(pipeline: Any, edits: list[dict[str, Any]]) -> list[str]:
    """Apply each edit in turn; return one plain summary per edit.

    A bad edit raises (from the Pipeline methods); the caller owns any rollback (the window wraps the whole
    call in an undo macro)."""
    summaries: list[str] = []
    for e in edits or []:
        op = str(e.get("op") or "")
        if op == "rename":
            pipeline.rename_node(str(e["node"]), str(e["title"]))
            summaries.append(f"renamed {e['node']} to “{e['title']}”")
        elif op == "set_params":
            pipeline.set_params(str(e["node"]), **(e.get("params") or {}))
            summaries.append(f"changed settings on {e['node']}")
        elif op == "set_input":
            pipeline.set_input(str(e["name"]), e.get("value"), e.get("unit"), e.get("note"))
            summaries.append(f"set input {e['name']}")
        elif op == "column_label":
            pipeline.set_column_meta(str(e["column"]), e.get("label"), e.get("unit"))
            summaries.append(f"labelled column {e['column']}")
        else:
            raise ValueError(f"Unknown edit op {op!r}. Use rename, set_params, set_input or column_label")
    return summaries
