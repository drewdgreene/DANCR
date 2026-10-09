"""Row-level sensitivity: the ``sensitivity`` column a Label-sensitivity step adds.

The label is an ordinary column, so it travels with the rows through joins, filters and exports. Restricted
rows are **withheld** from the surfaces that leave the project — search, an agent's reads over MCP, the
command line, the Python SDK, and shared exports — unless the caller explicitly allows them. The person's own
window (their own data, on their own machine) is not gated.

A blank or missing label is not restricted: only rows explicitly labelled ``confidential`` or ``restricted``
are withheld. Pure core; no Qt, no network.
"""
from __future__ import annotations

import polars as pl

LEVELS = ("public", "internal", "confidential", "restricted")
RESTRICTED = ("confidential", "restricted")
COLUMN = "sensitivity"


def is_restricted(level: str | None) -> bool:
    """Whether a level string is confidential or restricted."""
    return (level or "public").strip().lower() in RESTRICTED


def restricted_expr(column: str = COLUMN, levels: tuple[str, ...] = RESTRICTED) -> pl.Expr:
    """A boolean expression that is true only where the column *explicitly* names a restricted level.

    A null label compares to null and ``~null`` is null, which a filter would drop — so the ``is_not_null``
    guard is what keeps an unlabelled row visible."""
    level = pl.col(column).cast(pl.Utf8).str.strip_chars().str.to_lowercase()
    return level.is_not_null() & level.is_in([s.lower() for s in levels])


def has_labels(schema: pl.Schema | dict | None) -> bool:
    """Whether a schema carries a sensitivity column."""
    try:
        return COLUMN in (schema or {})
    except TypeError:
        return False


def withhold(lf: pl.LazyFrame, *, allow_restricted: bool = False, column: str = COLUMN,
             levels: tuple[str, ...] = RESTRICTED) -> pl.LazyFrame:
    """The frame without restricted rows, unless ``allow_restricted``. A frame with no sensitivity column is
    returned unchanged, so a project that never labels anything is untouched."""
    if allow_restricted:
        return lf
    try:
        if column not in lf.collect_schema().names():
            return lf
    except Exception:  # noqa: BLE001 - a schema that cannot be read leaves the frame as it is
        return lf
    return lf.filter(~restricted_expr(column, levels))
