"""Shared helpers over the data model: labels, groups, links, buckets and accumulated totals."""
from __future__ import annotations

import copy
import json
import math
from collections import deque
from datetime import datetime
from dataclasses import dataclass
from typing import Any

from ..planner import Plan, PlanStep
from ..understand import (DataModel, Table, Column, Relation, ID, TEXT, SERIES, LOOKUP, bucket_for, norm, name_words)
from ..timeutil import parse_bucket
from ._constants import (RULES_VERSION, STAT_WORDS, STAT_CHOICES, EVERY_CHOICES, EVERY_WORDS, TOP_CHOICES, RECIPES, WEIGHT, NEAR_DEFAULT, EXPERIMENT_ROWS, TEST_CHOICES, GROUP_MAX, MAX_SUGGESTIONS, PlanError, Suggestion, AMOUNT_WORDS, READING_WORDS, MONEY_WORDS, CURRENCY_UNITS, PART_WORDS, QUANTITY_NAMES, PAIRED_WORDS, CELL_CHOICES, NEAR_CHOICES)

def _col(model: DataModel, ref: list | None) -> Column | None:
    if not ref:
        return None
    t = model.table(ref[0])
    return t.column(ref[1]) if t else None


def plural(name: str) -> str:
    """customer -> customers, category -> categories; a name that is not a plain word (product_id) stays."""
    head, _, word = name.rpartition(" ")
    if not word.isalpha() or word.endswith("s"):
        return name
    if word.endswith("y") and word[-2:-1] not in "aeiou":
        word = word[:-1] + "ies"
    elif word.endswith(("lf", "eaf", "oaf")):
        word = word[:-1] + "ves"                          # half, shelf, wolf -> halves, shelves, wolves (not roofs)
    elif word.endswith(("ch", "sh", "x", "z")):
        word += "es"
    else:
        word += "s"
    return f"{head} {word}" if head else word


def stat_title(stat: str, what: str) -> str:
    """'Average' + 'Avg Time on Page (s)' reads 'Average time on page (s)', not 'Average Avg Time on Page (s)'."""
    import re
    word = STAT_WORDS.get(stat, stat.title())
    if stat == "mean":
        what = re.sub(r"^(avg|average|mean)[\s_.:-]+", "", what, flags=re.IGNORECASE) or what
    return f"{word} {what}"


def label(model: DataModel, ref: list | None) -> str:
    """How a column is named for people: its display label."""
    c = _col(model, ref)
    if c is not None:
        return c.label or c.name
    return str(ref[1]) if ref else ""


def _tlabel(model: DataModel, node: str) -> str:
    t = model.table(node)
    return t.title if t else node


def groupables(model: DataModel, table: str) -> list[list]:
    """Columns a table's rows can be grouped by: its own categories, the categories of tables it links to, and
    names in lookup tables (customer names, product names), in that order; plus 'which table' for a stack."""
    t = model.table(table)
    if t is None:
        return []
    out: list[list] = [[table, c.name] for c in t.categories]
    st = model.stack_of(table)
    if st is not None:
        out.append([st.id, "source"])
    for node in reachable(model, table):
        u = model.table(node)
        out += [[node, c.name] for c in u.categories if [node, c.name] not in out]
        if u.shape == LOOKUP:
            out += [[node, c.name] for c in u.columns if c.kind == "text" and c.role in (ID, TEXT) and not _is_key(model, node, c.name)]
    return out


def _is_key(model: DataModel, node: str, col: str) -> bool:
    return any(r.kind == "link" and ((r.tables[1] == node and r.right_on == col) or (r.tables[0] == node and r.left_on == col))
               for r in model.relations)


def _usable_links(model: DataModel, node: str, avoid: set[str]) -> list[Relation]:
    return [r for r in model.relations if r.kind == "link" and r.tables[0] == node and r.id not in avoid
            and r.cardinality != "many-to-many" and r.match_pct > 0]


def reachable(model: DataModel, table: str, avoid: set[str] | None = None) -> list[str]:
    """Tables whose columns can be brought next to ``table``'s rows without multiplying them, nearest first."""
    starts = _members(model, table)
    seen, order = set(starts), []
    queue = deque(starts)
    while queue:
        n = queue.popleft()
        for r in _usable_links(model, n, avoid or set()):
            nxt = r.tables[1]
            if nxt not in seen:
                seen.add(nxt); order.append(nxt); queue.append(nxt)
    return order


def _members(model: DataModel, table: str) -> list[str]:
    st = model.stack_of(table)
    return list(st.tables) if st is not None else [table]


def _path(model: DataModel, starts: list[str], goal: str, avoid: set[str]) -> list[Relation] | None:
    """The shortest chain of links from any of ``starts`` to ``goal`` (breadth first, links in model order)."""
    prev: dict[str, tuple[str, Relation] | None] = {s: None for s in starts}
    queue = deque(starts)
    while queue:
        n = queue.popleft()
        if n == goal:
            chain: list[Relation] = []
            while prev[n] is not None:
                p, r = prev[n]
                chain.append(r); n = p
            return list(reversed(chain))
        for r in _usable_links(model, n, avoid):
            if r.tables[1] not in prev:
                prev[r.tables[1]] = (n, r)
                queue.append(r.tables[1])
    return None


def ordered_measures(t: Table) -> list[Column]:
    """The numbers most worth answering about first: amounts (sales, visits, quantity), then numbers with a unit
    (what a logger measures), then the rest, each in column order."""
    def rank(c: Column) -> tuple:
        words = set(name_words(c.name)) | set(name_words(c.label or ""))
        money = (c.unit or "").strip().lower() in CURRENCY_UNITS or words & MONEY_WORDS
        where = words & {"lat", "lon", "lng", "latitude", "longitude", "x", "y", "easting", "northing"}
        return (4 if where else 0 if money else 1 if words & AMOUNT_WORDS else (2 if c.unit else 3), t.columns.index(c))
    return sorted(t.measures, key=rank)



def default_stat(model: DataModel, table: str, measure: list | None = None) -> str:
    """Amounts are added up (the total sales per month); readings are averaged (the mean pressure per hour).
    Decided by what the number is — its name and unit — and, when that says nothing, by its table: a logger's
    numbers are readings, a lookup's numbers describe its rows (a product's unit cost), anything else adds up."""
    col = _col(model, measure) if measure and not measure[0].startswith("stack:") else None
    wide = model.table(measure[0]) if measure and not measure[0].startswith("stack:") else None
    if col is not None and wide is not None and wide.wide and col.name == "value":
        return "sum"                                   # the cells of a month-per-column sheet are amounts to add up
    if col is not None:
        words = set(name_words(col.name)) | set(name_words(col.label or ""))
        unit = (col.unit or "").strip().lower()
        if words & AMOUNT_WORDS and not words & {"price", "rate", "average", "mean", "ratio", "percent", "pct"}:
            return "sum"
        if unit in CURRENCY_UNITS:
            return "sum"
        if words & READING_WORDS or unit or (len(name_words(col.name)) > 1 and name_words(col.name)[-1] in ("f", "c", "k", "degc", "degf")):
            return "mean"                                  # tmax_F, temp_C: a reading in degrees
    t = model.table(measure[0] if measure and not measure[0].startswith("stack:") else table)
    return "mean" if t is not None and t.shape in (SERIES, LOOKUP) else "sum"


# =================================================================== suggestions

def _tidy_number(v: float) -> float:
    """54.099999999999994 as the 54.1 it is."""
    return float(f"{v:.10g}")


def _safe_spellings(sp: dict[str, str]) -> bool:
    return all('"' not in x and "\\" not in x for pair in sp.items() for x in pair)


def _spelling_formula(col: str, sp: dict[str, str]) -> str:
    """IF(TRIM(LOWER([c])) = "north", "North", …, [c]): each group of variants written the way most rows write it."""
    canon: dict[str, str] = {}
    for v, target in sp.items():
        canon[target.strip().lower()] = target
    expr = f"[{col}]"
    for key in sorted(canon, reverse=True):
        expr = f'IF(LOWER(TRIM([{col}])) = "{key}", "{canon[key]}", {expr})'
    return expr


def _free_name(base: str, taken: list[str]) -> str:
    name, k = base, 2
    while name in taken:
        name = f"{base}_{k}"; k += 1
    return name


def filter_text(model: DataModel, filters: list[dict]) -> str:
    words = {"eq": "is", "ne": "is not", "gt": "above", "lt": "below", "ge": "at least", "le": "at most",
             "between": "between", "outside": "not between", "contains": "contains", "in": "is one of"}
    words.update({"year": "in", "month": "in"})
    parts = []
    for f in filters:
        if f.get("text"):                              # as the person typed it: "in March", "after 2024-06-01"
            parts.append(f"{label(model, f['column'])} {f['text']}")
            continue
        v = f.get("value", "")
        if f["op"] == "in":
            v = " or ".join(str(x) for x in v) if isinstance(v, list) else v
            parts.append(f"{label(model, f['column'])} is {v}")
            continue
        if f["op"] in ("between", "outside"):
            v = f"{v} and {f.get('value2', '')}"
        if f.get("column2"):
            v = label(model, f["column2"])
        parts.append(f"{label(model, f['column'])} {words.get(f['op'], f['op'])} {v}".strip())
    return ", ".join(parts)



def _rule(f: dict, column: str) -> dict:
    rule = {"column": column, "op": f["op"], "value": f.get("value", "")}
    if f.get("value2") not in (None, ""):
        rule["value2"] = f["value2"]
    return rule


def auto_every(model: DataModel, t: Table, filters: list[dict] | None = None) -> str:
    """Readings: a few hundred points across the span. Events (sales, visits): a few dozen totals, since each
    point is a sum people read one by one. The span is the period asked about (a month, a year) when there is one."""
    tc = t.column(t.time) if t.time else None
    return bucket_for(_asked_span(t, filters) or t.span_seconds, tc.cadence if tc else None, target=400 if t.shape == SERIES else 60)


def _asked_span(t: Table, filters: list[dict] | None) -> float | None:
    spans = []
    for f in filters or []:
        if not t.time or f["column"][1] != t.time:
            continue
        if f["op"] == "month":
            spans.append(31 * 86400.0)
        elif f["op"] == "year":
            spans.append(365 * 86400.0)
        elif f["op"] in ("gt", "ge") and t.end is not None:          # "since March", "in the last 7 days"
            try:
                spans.append(max(86400.0, (t.end - datetime.fromisoformat(str(f["value"])[:19])).total_seconds()))
            except (ValueError, TypeError):
                pass
        elif f["op"] == "between":
            try:
                a, b = (datetime.fromisoformat(str(f[k])[:19]) for k in ("value", "value2"))
                spans.append(max(86400.0, (b - a).total_seconds()))
            except (ValueError, KeyError):
                pass
    return min(spans) if spans else None


def _bucket_stats(measures: list[list], stat: str, time_col: str) -> dict:
    if stat == "count" or not measures:
        return {"columns": [], "aggregations": [{"column": time_col, "stats": ["rows"], "alias": "rows"}]}
    return {"columns": [r[1] for r in measures], "default_stats": [stat]}


def _bucket_name(col: str, stat: str) -> str:
    return "rows" if stat == "count" else col


def _y_label(m: DataModel, ref: list | None, stat: str) -> str:
    if ref is None or stat == "count":
        return "rows"
    c = _col(m, ref)
    unit = c.unit if c is not None else ""
    name = label(m, ref)
    import re
    said = unit and re.search(rf"(^|[\s(\[]){re.escape(unit)}([\s)\]]|$)", name)
    return f"{name} ({unit})" if unit and not said else name


def _span_text(secs: float | None) -> str:
    from ..timeutil import format_seconds
    return format_seconds(secs) if secs else "unknown span"


def group_label(model: DataModel, ref: list | None, base: str | None = None, spoken: str | None = None) -> str:
    """A group as people say it: the word they used for it; 'customers' for a customer's name in a linked lookup;
    else the column's label."""
    if not ref:
        return ""
    if spoken:
        return spoken
    if ref[0].startswith("stack:"):
        rel = model.relation(ref[0])
        common = [w for w in name_words(model.tables[rel.tables[0]].title)
                  if all(w in name_words(model.tables[n].title) for n in rel.tables)] if rel else []
        return common[0] if common else "table"          # device_1, device_2: "by device"
    t = model.table(ref[0])
    c = _col(model, ref)
    if t is not None and t.shape == LOOKUP and c is not None and c.role in (ID, TEXT) and c.kind == "text" and ref[0] != base:
        return t.title
    return label(model, ref)


def _compare_every(t: Table) -> str:
    """A sensible period to compare two of: a couple of dozen across the span (a day for a week, a month for years)."""
    tc = t.column(t.time) if t.time else None
    every = bucket_for(t.span_seconds, tc.cadence if tc else None, target=24)
    return "3mo" if every.endswith("q") else every



def _part_formula(part: str, col: str) -> str:
    if part == "weekday":                             # "1 Mon" … "7 Sun": readable and in order
        names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        expr = '""'
        for i in range(7, 0, -1):
            expr = f'IF(WEEKDAY([{col}], 2) = {i}, "{i} {names[i - 1]}", {expr})'
        return expr
    return {"hour": f"HOUR([{col}])", "month": f"MONTH([{col}])", "day": f"DAY([{col}])"}[part]


def top_n(spec: dict) -> int:
    """How many rows a top or toprows answer keeps: ten unless the spec says; a whole number of 1 or more."""
    n = spec.get("n")
    if n is None:
        return 10
    if isinstance(n, bool) or not isinstance(n, (int, float)) or n != int(n) or n < 1:
        raise PlanError(f"top {n} is not a number of rows. Say a whole number of 1 or more, as in top 5")
    return int(n)



def row_noun(t: Table) -> str:
    """What a table's rows are, from a column that numbers them: a sample number names each sample."""
    for c in t.by_role(ID):
        w = name_words(c.name)
        if len(w) > 1 and w[0] in ("n", "no", "nr", "num", "number"):
            return plural(" ".join(w[1:]))
    return ""



def same_kind(model: DataModel, refs: list[list]) -> bool:
    """Columns that hold the same kind of number (one unit, or none, on a similar scale): they can be compared as
    groups (Control and Treated, Before and After), not related to each other as x and y."""
    cols = [_col(model, r) for r in refs]
    if len(cols) < 2 or any(c is None or c.role != "measure" for c in cols) or len({(c.unit or "").lower() for c in cols}) > 1:
        return False
    if not cols[0].unit and any(set(name_words(c.name)) & QUANTITY_NAMES for c in cols):
        return False                                       # length and weight: two quantities, even without units
    try:
        lo, hi = (sorted(abs(float(c.maximum)) + abs(float(c.minimum)) for c in cols)[i] for i in (0, -1))
    except (TypeError, ValueError):
        return False
    return lo > 0 and hi / lo <= 10


def _paired_names(names: list[str]) -> bool:
    """Columns named for moments of the same things (Before, After; Week 0, Week 4): their rows are pairs."""
    return all(set(name_words(n)) & PAIRED_WORDS for n in names)



def _default_cell(t: Table) -> str:
    """A grid cell size that splits a table's points into a few dozen cells, so a density map is readable."""
    la, lo = t.column(t.geo["lat"]), t.column(t.geo["lon"])
    try:
        span = max(abs(float(la.maximum) - float(la.minimum)), abs(float(lo.maximum) - float(lo.minimum)))
    except (TypeError, ValueError):
        return "0.1"
    if span <= 0:
        return "0.01"
    for size in (0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0):
        if span / size <= 40:
            return f"{size:g}"
    return "10"


