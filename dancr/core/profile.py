"""A compact, safe *profile* of a project's tables: column names, roles, ranges, counts, categories and
relations, built from :mod:`dancr.core.understand`.

This is what leaves the machine to a model — the Assistant reads it instead of the rows — and it is also the
schema half of the knowledge-base export (:func:`dancr.headless.build_context`). Values are stripped of
control characters and capped, and :func:`data_block` wraps a payload so a value in the data can never close
its own delimiter or be read as an instruction.

Pure core: no Qt, no execution of its own.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .understand import Column, DataModel, Relation, Table, CATEGORY, CONSTANT, BLANK, FLAG, ID, MEASURE, TIME_ROLE

MAX_VALUES = 15             # category values kept per column
MAX_TEXT = 120              # characters kept from any one string
MAX_CARD_COLUMNS = 30       # columns named in a table's prose card (the structured profile keeps them all)
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean(value: Any, limit: int = MAX_TEXT) -> Any:
    """Any value as JSON-safe text: control characters stripped, length capped. Then a value can never smuggle
    a delimiter or a new instruction into the profile."""
    if value is None or isinstance(value, (int, float, bool)):
        return value
    s = CONTROL.sub(" ", str(value)).strip()
    return s[:limit] + ("…" if len(s) > limit else "")


def _column(c: Column) -> dict[str, Any]:
    d: dict[str, Any] = {"name": clean(c.name), "dtype": c.dtype, "role": c.role, "kind": c.kind}
    if c.label and c.label != c.name:
        d["label"] = clean(c.label)
    if c.unit:
        d["unit"] = clean(c.unit)
    if c.quantity:
        d["quantity"] = clean(c.quantity)
    if c.null_pct:
        d["null_pct"] = round(c.null_pct, 2)
    d["distinct"] = c.distinct
    if c.unique:
        d["unique"] = True
    if c.minimum is not None:
        d["min"] = clean(c.minimum)
    if c.maximum is not None:
        d["max"] = clean(c.maximum)
    if c.cadence:
        d["seconds_between_rows"] = round(c.cadence, 3)
    if c.values:
        d["values"] = [clean(v) for v in c.values[:MAX_VALUES]]
    if c.derived and c.derived.get("words"):
        d["calculated"] = clean(c.derived["words"], 200)
    if c.geo:
        d["coordinate"] = c.geo
    return d


def table_profile(t: Table) -> dict[str, Any]:
    d: dict[str, Any] = {"node": t.node, "title": clean(t.title), "shape": t.shape}
    if t.source:
        d["file"] = clean(Path(t.source).name)          # the name only: the folder may itself be sensitive
    if t.rows is not None:
        d["rows"] = t.rows
        d["rows_exact"] = t.rows_exact
    if t.time:
        d["time_column"] = clean(t.time)
        if t.start is not None:
            d["from"] = clean(t.start)
            d["to"] = clean(t.end)
    if t.geo:
        d["place"] = {k: clean(v) for k, v in t.geo.items()}
    d["columns"] = [_column(c) for c in t.columns]
    return d


def _relation(r: Relation) -> dict[str, Any]:
    d: dict[str, Any] = {"kind": r.kind, "tables": [clean(t) for t in r.tables], "why": clean(r.why, 200)}
    for k in ("left_on", "right_on", "cardinality", "tolerance"):
        if getattr(r, k, None):
            d[k] = clean(getattr(r, k))
    if r.match_pct:
        d["match_pct"] = round(r.match_pct, 1)
    if r.shared:
        d["shared_columns"] = [clean(s) for s in r.shared[:20]]
    if r.pairs:
        d["pairs"] = {clean(k): clean(v) for k, v in list(r.pairs.items())[:20]}
    if r.labels:
        d["labels"] = [clean(x) for x in r.labels[:20]]
    if r.geo:
        d["geo"] = {k: clean(v) for k, v in r.geo.items()}
    return d


def project_profile(pipe, model: DataModel, *, node: str | None = None) -> dict[str, Any]:
    """A compact profile of the project's tables (or just ``node`` when given), for the model to reason over."""
    tables = [model.tables[n] for n in model.tables if node is None or n == node]
    rels = [r for r in model.relations if node is None or node in r.tables]
    out: dict[str, Any] = {"project": clean(getattr(pipe, "name", "") or "Untitled"),
                           "tables": [table_profile(t) for t in tables],
                           "relations": [_relation(r) for r in rels]}
    if model.skipped:
        out["tables_that_could_not_be_read"] = {clean(k): clean(v) for k, v in model.skipped.items()}
    if pipe.inputs:
        out["inputs"] = [{"name": clean(i.name), "value": clean(i.value), "unit": clean(i.unit)} for i in pipe.inputs]
    return out


def profile_json(profile: dict[str, Any]) -> str:
    """The profile as the model reads it. Kept out of the instruction region by :func:`data_block`."""
    return json.dumps(profile, ensure_ascii=False, separators=(",", ":"), default=str)


def data_block(tag: str, payload: str) -> str:
    """Wrap data-derived text so the model can tell it from instructions. The tag names the source.

    A payload cannot close its own block: the closing form of the tag's *name* (its first word, so a tag
    carrying attributes such as ``tool_result name="…"`` is still safe) is escaped wherever it appears — matched
    case-insensitively and across stray spaces, so ``</TOOL_RESULT>`` and ``</ tool_result >`` cannot break out."""
    name = (tag.split() or [tag])[0]
    safe = re.sub(rf"</\s*{re.escape(name)}\b[^>]*>", f"<\\/{name}>", payload, flags=re.IGNORECASE)
    return f"<{tag}>\n{safe}\n</{tag}>"


# ----------------------------------------------------------------- prose "doc cards"
_SHAPE_WORDS = {"series": "time series", "lookup": "lookup table", "events": "event log"}


def _number(v: Any) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(clean(v))
    return f"{f:g}"


def _cadence(seconds: float) -> str:
    for unit, size in (("d", 86400.0), ("h", 3600.0), ("m", 60.0)):
        if seconds >= size:
            return f"{_number(seconds / size)}{unit}"
    return f"{_number(seconds)}s"


def _column_phrase(c: Column) -> str:
    """One clause naming a column and the fact that best identifies it, deterministic and plain."""
    name = clean(c.name)
    u = f" ({clean(c.unit)})" if c.unit else ""
    if c.role == MEASURE:
        rng = f", {_number(c.minimum)} to {_number(c.maximum)}" if c.minimum is not None and c.maximum is not None else ""
        return f"{name}{u} (measure{rng})"
    if c.role == CATEGORY:
        vals = ", ".join(str(clean(v)) for v in c.values[:5])
        return f"{name} (category: {vals})" if vals else f"{name} (category)"
    if c.role == TIME_ROLE:
        cad = f", every {_cadence(c.cadence)}" if c.cadence else ""
        return f"{name} (time{cad})"
    if c.role == ID:
        return f"{name} (id{', unique' if c.unique else ''})"
    if c.role == FLAG:
        return f"{name} (true/false)"
    if c.role == CONSTANT:
        return f"{name} (one value)"
    if c.role == BLANK:
        return f"{name} (blank)"
    return f"{name}{u} ({clean(c.role) or 'text'})"


def relation_phrase(r: Relation) -> str:
    """A relation as one plain clause, for a doc card or a human summary."""
    if r.kind == "link":
        match = f", {r.match_pct:.0f}% of keys match" if r.match_pct else ""
        card = f", {r.cardinality}" if r.cardinality else ""
        return f"{r.tables[0]} links to {r.tables[1]} on {r.left_on} = {r.right_on} ({match.strip(', ') or 'by key'}{card})"
    if r.kind == "stack":
        labels = f" (labels: {', '.join(clean(x) for x in r.labels)})" if r.labels else ""
        return "stackable tables: " + ", ".join(r.tables) + labels
    if r.kind == "align":
        tol = f" (tolerance {r.tolerance})" if r.tolerance else ""
        return f"{r.tables[0]} and {r.tables[1]} align in time{tol}"
    return clean(r.why, 200)


def table_card(t: Table, relations: list[Relation] | None = None) -> str:
    """A one-paragraph, deterministic description of a table: shape, size, source, time span, its columns
    (grouped by role) and how it relates to the project's other tables. Meant to be embedded as prose, so a
    search index has language to match, while the exact numbers stay in the structured profile."""
    shape = _SHAPE_WORDS.get(t.shape, "table")
    article = "an" if shape[:1] in "aeiou" else "a"
    size = f"{t.rows:,} rows" if t.rows is not None else f"{t.sampled:,}+ rows (sampled)"
    src = f" recorded from {clean(Path(t.source).name)}" if t.source else ""
    when = f", spanning {clean(t.start)} to {clean(t.end)} (by {clean(t.time)})" if t.time and t.start is not None else ""
    head = f'"{clean(t.title)}" is {article} {shape} of {size}{src}{when}.'
    shown = t.columns[:MAX_CARD_COLUMNS]
    body = " Columns: " + "; ".join(_column_phrase(c) for c in shown) + "."
    if len(t.columns) > len(shown):
        body += f" ({len(t.columns) - len(shown)} more columns.)"
    if relations:
        related = " ".join(f"Related: {relation_phrase(r)}." for r in relations if t.node in r.tables)
        if related:
            body += " " + related
    return head + body
