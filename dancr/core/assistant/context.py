"""What the model is allowed to see.

The model never reads the data. It reads a compact *profile* of the project's tables — column names,
roles, ranges, counts, categories — built from DANCR's own data model (``core.understand``), and it asks
the engine for anything else through tools. Profiles are capped and sanitised, and always placed inside a
clearly delimited DATA block, so a value in the data can never be read as an instruction.

The profile builder itself lives in :mod:`dancr.core.profile`, so the same schema description is shared with
the knowledge-base export (:func:`dancr.headless.build_context`). This module re-exports it for the Assistant.
"""
from __future__ import annotations

from ..profile import (  # noqa: F401
    CONTROL,
    MAX_TEXT,
    MAX_VALUES,
    clean,
    data_block,
    profile_json,
    project_profile,
    table_card,
    table_profile,
)

__all__ = [
    "CONTROL", "MAX_TEXT", "MAX_VALUES",
    "clean", "data_block", "profile_json", "project_profile", "table_card", "table_profile",
]
