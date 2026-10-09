"""Governance steps: mark how sensitive a table is, and make a shareable (redacted) copy.

A label travels as an ordinary column (``sensitivity``) and, being a column, flows through joins, exports and
the search index — so a restricted set is not just a note in the metadata, it is carried with the rows. The
retrieval step (``search_knowledge`` / 'Search index') reads that column and withholds restricted passages by
default. ``redact`` masks or pseudonymises columns so a derivative can be shared.
"""
from __future__ import annotations

from typing import Any

import polars as pl

from ..findings import finding, plural
from ..params import Param
from ..registry import NodeType, InputSpec, Ctx, NodeResult, registry
from ._common import first_input, schema_of, require_column

LEVELS = [("public", "Public"), ("internal", "Internal"), ("confidential", "Confidential"), ("restricted", "Restricted")]


def _label(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any]) -> NodeResult:
    lf = first_input(inputs)
    schema = schema_of(lf)
    name = (params.get("name") or "sensitivity").strip() or "sensitivity"
    col = (params.get("column") or "").strip()
    level = str(params.get("level") or "internal")
    if col:
        require_column(schema, col, "label column")
        # always cast to text, even when the label reuses the source column name, so the sensitivity column is
        # the plain-text labels the withholding and graph checks expect
        out = lf.with_columns(pl.col(col).cast(pl.Utf8).alias(name))
        said = f"Labelled rows from '{col}' into a '{name}' column"
        detail: dict[str, Any] = {"column": col, "label": name}
    else:
        out = lf.with_columns(pl.lit(level).alias(name))
        said = f"Labelled every row '{level}' in a '{name}' column"
        detail = {"level": level, "label": name}
    report = {**detail, "finding": finding("summary", said, exact=True)}
    return NodeResult(out, report=report, messages=[said])


registry.register(NodeType(
    key="label_sensitivity", label="Label sensitivity", category="Governance", icon="⚑",
    description="Mark a table (or the values of a column) as public, internal, confidential or restricted. The "
                "label is a column that travels with the rows, so search can withhold sensitive passages by default.",
    apply=_label,
    summary=lambda p: (f"label from {p.get('column')}" if p.get("column") else f"label '{p.get('level') or 'internal'}'"),
    params=[
        Param("level", "Sensitivity", "choice", default="internal", choices=LEVELS, visible_when={"column": ["", None]}),
        Param("column", "…or take the label from a column", "column", default="", column_group="string",
              help="Use this when different rows have different sensitivities"),
        Param("name", "Label column name", "text", default="sensitivity", advanced=True),
    ],
))


def _redact(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any]) -> NodeResult:
    lf = first_input(inputs)
    schema = schema_of(lf)
    cols = [require_column(schema, c, "column to redact") for c in (params.get("columns") or [])]
    if not cols:
        raise ValueError("Choose the column(s) to redact")
    method = params.get("method") or "mask"
    keep = int(params.get("keep") or 0)
    mask_char = (str(params.get("mask") or "*")[:1]) or "*"
    exprs = []
    for c in cols:
        base = pl.col(c).cast(pl.Utf8)
        if method == "hash":
            exprs.append(base.hash(seed=0).cast(pl.Utf8).str.slice(0, 12).alias(c))
        elif method == "drop":
            continue
        else:
            lead = base.str.slice(0, keep) if keep > 0 else pl.lit("")
            exprs.append((lead + pl.lit(mask_char * 4)).alias(c))
    out = lf.with_columns(exprs) if exprs else lf
    if method == "drop":
        out = out.drop(cols)
    said = f"Redacted {plural(len(cols), 'column')} ({method})"
    report = {"redacted": cols, "method": method, "finding": finding("summary", said, magnitude=float(len(cols)), exact=True)}
    return NodeResult(out, report=report, messages=[said])


registry.register(NodeType(
    key="redact", label="Redact columns", category="Governance", icon="▩",
    description="Make a shareable copy: mask a column (keep a few leading characters), replace it with a stable "
                "pseudonym (hash), or drop it. The rest of the table is unchanged.",
    apply=_redact,
    summary=lambda p: f"{p.get('method') or 'mask'} {len(p.get('columns') or [])} column(s)",
    params=[
        Param("columns", "Columns to redact", "columns", default=[], required=True),
        Param("method", "How", "choice", default="mask", choices=[
            ("mask", "Mask (keep the first few characters)"), ("hash", "Stable pseudonym (hash)"), ("drop", "Remove the column")]),
        Param("keep", "Characters to keep (mask)", "int", default=0, min=0, max=20, visible_when={"method": "mask"}),
        Param("mask", "Mask character", "text", default="*", visible_when={"method": "mask"}),
    ],
))
