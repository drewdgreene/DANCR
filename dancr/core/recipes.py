"""Answers DANCR can give on its own: a catalogue of recipes over the data model.

A *spec* is a question written down as data, for example
``{"recipe": "breakdown", "table": "orders", "measure": ["orders", "qty"], "stat": "sum", "by": ["customers", "region"]}``.
Column references are ``[table node id, column name]`` pairs, so a question can name a column of another table
and the planner finds the link that brings it in.

- ``suggest(model)`` lists the answers worth offering for the tables in a model, best first.
- ``plan(model, spec)`` turns a spec into a :class:`~dancr.core.planner.Plan`: the steps to add, what the answer
  assumed on the person's behalf (each with the alternatives), and the choices shown as chips.

Deterministic: the same model and spec always give the same plan, and ties are broken by project order,
column order and the order of the recipes below. Pure core, no Qt.
"""
from __future__ import annotations

import copy
import json
import math
from collections import deque
from datetime import datetime
from dataclasses import dataclass
from typing import Any

from .planner import Plan, PlanStep
from .understand import (DataModel, Table, Column, Relation, ID, TEXT, SERIES, LOOKUP, bucket_for, norm, name_words)
from .timeutil import parse_bucket

RULES_VERSION = 5           # bump when a change to these rules would build a different plan from the same spec

STAT_WORDS = {"sum": "Total", "mean": "Average", "count": "Number of rows", "max": "Highest", "min": "Lowest",
              "median": "Median", "std": "Standard deviation of"}
STAT_CHOICES = ["sum", "mean", "median", "min", "max", "std", "count"]
EVERY_CHOICES = ["1m", "15m", "1h", "1d", "1w", "1mo", "1q", "1y"]
EVERY_WORDS = {"1s": "second", "10s": "10 seconds", "30s": "30 seconds", "1m": "minute", "5m": "5 minutes",
               "15m": "15 minutes", "30m": "30 minutes", "1h": "hour", "6h": "6 hours", "1d": "day", "1w": "week",
               "1mo": "month", "1q": "quarter", "1y": "year", "1ms": "millisecond", "10ms": "10 ms", "50ms": "50 ms",
               "100ms": "100 ms", "500ms": "half second", "5s": "5 seconds"}
TOP_CHOICES = [5, 10, 20, 50]

# the recipes, in the order that breaks ties
RECIPES = ["compare", "groups", "trend", "breakdown", "top", "toprows", "relationship", "gaps", "outliers", "single",
           "distribution", "linked", "stacked", "rows", "describe", "change", "explain", "drivers", "forecast", "quality",
           "nearest", "map", "density", "place"]
WEIGHT = {"compare": 100, "groups": 62, "trend": 95, "breakdown": 90, "top": 75, "relationship": 60, "gaps": 65,
          "outliers": 55, "single": 30, "distribution": 45, "linked": 50, "stacked": 60, "rows": 25, "toprows": 40,
          "describe": 20, "change": 88, "explain": 86, "drivers": 68, "forecast": 58, "quality": 18,
          "nearest": 72, "map": 56, "density": 46, "place": 66}
NEAR_DEFAULT = "10km"       # a proximity match with no distance asked for: near enough to mean something
EXPERIMENT_ROWS = 5_000     # a table this small, with groups and no dates, is a study: its groups are compared first
TEST_CHOICES = [("auto", "Test: chosen for me"), ("welch", "Welch's t-test"), ("student", "Student's t-test"),
                ("rank", "Rank test (Mann–Whitney)")]
GROUP_MAX = 12              # a group with more values than this is a "top N" question rather than a breakdown
MAX_SUGGESTIONS = 8


class PlanError(ValueError):
    """A question that cannot be answered from these tables, with the reason in plain English."""


@dataclass
class Suggestion:
    spec: dict[str, Any]
    title: str
    recipe: str
    score: float
    why: str = ""
    view: str = "chart"

    def to_dict(self) -> dict[str, Any]:
        return {"spec": self.spec, "title": self.title, "recipe": self.recipe, "score": self.score,
                "why": self.why, "view": self.view}


# =================================================================== helpers over the model
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


AMOUNT_WORDS = {"sales", "sale", "revenue", "amount", "amounts", "cost", "costs", "spend", "spent", "profit", "income",
                "qty", "quantity", "quantities", "units", "unit", "count", "counts", "total", "sum", "volume", "orders",
                "items", "visits", "hours", "minutes", "calls", "tickets", "turnover", "paid", "payment", "payments",
                "sold", "bookings", "downloads", "clicks", "views", "impressions", "rainfall", "precipitation", "energy"}
READING_WORDS = {"tmax", "tmin", "tavg", "dewpoint", "dew", "wind", "windspeed", "gust", "kmh", "mph", "lat", "lon",
                 "salary", "salaries", "wage", "wages", "pay", "battery", "bounce", "duration", "time", "temperature", "temp", "pressure", "humidity", "speed", "velocity", "level", "depth", "height",
                 "voltage", "current", "rate", "ratio", "percent", "pct", "percentage", "price", "score", "age", "ph",
                 "conductivity", "salinity", "concentration", "density", "flow", "rating", "latitude", "longitude",
                 "lat", "lon", "lng", "altitude", "elevation", "weight", "mass", "size", "length", "width", "psi",
                 "psia", "bar", "tension", "load", "signal", "strength", "frequency", "reading", "average",
                 "mean", "median", "index", "margin", "utilisation", "utilization", "occupancy", "efficiency"}
MONEY_WORDS = {"amount", "sales", "revenue", "turnover", "income", "cost", "costs", "spend", "profit", "paid",
               "payment", "payments", "debit", "credit", "balance", "value", "total"}
CURRENCY_UNITS = {"$", "€", "£", "¥", "usd", "eur", "gbp", "jpy", "chf", "aud", "cad", "nok", "sek", "dkk", "k$", "m$"}


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
def suggest(model: DataModel, focus: str | None = None, limit: int = MAX_SUGGESTIONS) -> list[Suggestion]:
    """The answers worth offering, best first, one of each kind before a second of any kind. ``focus`` limits
    them to one table (a step the person selected)."""
    cands: list[tuple[float, int, int, Suggestion]] = []
    order = {nid: i for i, nid in enumerate(model.tables)}
    tables = [focus] if focus else list(model.tables)
    seen_stack: set[str] = set()
    for nid in tables:
        t = model.table(nid)
        if t is None:
            continue
        st = model.stack_of(nid)
        if st is not None and not focus:
            if st.id in seen_stack:
                continue                              # the stack's first table speaks for all of them
            seen_stack.add(st.id)
        for spec in _candidates(model, t):
            try:
                p = plan(model, spec)
            except PlanError:
                continue
            recipe = spec["recipe"]
            s = Suggestion(spec=spec, title=p.title, recipe=recipe, score=_score(model, t, spec), why=p.why, view=p.view)
            cands.append((-s.score, order.get(nid, 0), RECIPES.index(recipe), s))
    cands.sort(key=lambda c: c[:3])
    picked: list[Suggestion] = []
    kinds: set[str] = set()
    for *_k, s in cands:                                  # the best of each kind first
        if s.recipe not in kinds:
            picked.append(s); kinds.add(s.recipe)
    for *_k, s in cands:                                  # then the rest, best first
        if s not in picked:
            picked.append(s)
    picked.sort(key=lambda s: (-s.score, RECIPES.index(s.recipe)))
    return _dedupe(picked)[:limit]


def _dedupe(items: list[Suggestion]) -> list[Suggestion]:
    out, seen = [], set()
    for s in items:
        key = json.dumps(s.spec, sort_keys=True)
        if key not in seen:
            seen.add(key); out.append(s)
    return out


def _condition_columns(model: DataModel, t: Table) -> list[list]:
    """Number columns that are one quantity under different conditions, side by side: named alike but for one word
    (Height before, Height after; Trial 1, Trial 2) or for moments (Before, After), holding the same kind of number,
    in a small table without dates."""
    if t.time or (t.rows or t.sampled or 0) > EXPERIMENT_ROWS:
        return []
    ms = [c for c in t.measures if not c.derived]
    if 2 <= len(ms) <= 6 and not t.categories and any(c.role == ID for c in t.columns) and \
            same_kind(model, [[t.node, c.name] for c in ms]):
        return [[t.node, c.name] for c in ms]           # Replicate | Control | Fertilised: every number is a group
    for size in range(min(len(ms), 6), 1, -1):
        for i in range(len(ms) - size + 1):
            run = ms[i:i + size]
            words = [name_words(c.name) for c in run]
            alike = all(len(w) == len(words[0]) and len(w) >= 2 for w in words) and \
                sum(len({w[k] for w in words}) > 1 for k in range(len(words[0]))) == 1
            if (alike or _paired_names([c.name for c in run])) and same_kind(model, [[t.node, c.name] for c in run]):
                return [[t.node, c.name] for c in run]
    return []


def experiment(t: Table, model: DataModel | None = None) -> bool:
    """A small table of measurements in groups with no dates (treated and control plots, before and after): a
    study, whose first question is whether the groups differ. The groups may be a column, or the files (or sheets)
    of tables with the same columns (group_a.csv, group_b.csv)."""
    stack = model.stack_of(t.node) if model is not None else None
    rows = sum((model.tables[n].rows or model.tables[n].sampled or 0) for n in stack.tables) if stack else (t.rows or t.sampled or 0)
    return not t.time and rows <= EXPERIMENT_ROWS and bool(t.measures) and (bool(t.categories) or stack is not None)


def breaks_of(t: Table) -> list[tuple[Column, dict]]:
    """The rows that break a calculated column's rule (a gap of 1.26 where outer area minus inner area is 54.1)."""
    return [(c, b) for c in t.columns for b in (c.derived or {}).get("breaks", [])]


def _score(model: DataModel, t: Table, spec: dict) -> float:
    r = spec["recipe"]
    s = float(WEIGHT[r])
    if r == "groups" and (experiment(t, model) or spec.get("columns")):
        s += 35                                           # the question a study was made to answer
    if experiment(t, model) and r in ("explain", "breakdown"):
        s -= 30                                           # its groups' totals and shares are rarely the question
    if r == "quality" and breaks_of(t):
        s += 70                                           # a value that breaks its own column's rule: say so early
    if t.shape == LOOKUP and r in ("breakdown", "distribution", "top"):
        s -= 40                                           # counting customers per region is rarely the question
    if r == "relationship":
        s += 20 * abs(spec.get("_r", 0.5))
    if r == "gaps" and t.shape != SERIES:
        s -= 25
    if r == "trend" and t.shape == SERIES:
        s += 3
    return round(s, 2)


def _candidates(model: DataModel, t: Table) -> list[dict]:
    out: list[dict] = []
    measures = ordered_measures(t)
    m0 = [t.node, measures[0].name] if measures else None
    stat = default_stat(model, t.node, m0)
    st = model.stack_of(t.node)
    # compare two series of the same quantity
    for r in model.relations:
        if r.kind == "align" and t.node in r.tables:
            m = next((x for x in measures if x.name in r.shared), None)
            if m is not None:
                out.append({"recipe": "compare", "table": r.tables[0], "other": r.tables[1], "measure": [r.tables[0], m.name]})
    if t.time:
        spec = {"recipe": "trend", "table": t.node, "measures": [m0] if m0 else [], "stat": stat if m0 else "count"}
        if st is not None:
            spec["together"] = True
        out.append(spec)
    groups = groupables(model, t.node)
    small = [g for g in groups if 1 < _distinct(model, g) <= GROUP_MAX]
    big = [g for g in groups if _distinct(model, g) > GROUP_MAX]
    if t.shape == SERIES:
        small = [g for g in small if not g[0].startswith("stack:")]     # one line per table already shows that
    others = [x for x in model.tables.values() if x.node != t.node and x.shape != LOOKUP]
    if t.shape == LOOKUP and others:
        small, big = [], []                                              # asked of the tables that link to it
    cond = _condition_columns(model, t)
    if cond:
        out.append({"recipe": "groups", "table": t.node, "columns": cond})
    own_small = [g for g in small if g[0] == t.node and _col(model, g) is not None and
                 (_col(model, g).distinct or 0) * 2 <= (t.rows or t.sampled or 0)]
    if own_small and measures:
        out.append({"recipe": "groups", "table": t.node, "by": own_small[0]})
    elif st is not None and measures and experiment(t, model):
        out.append({"recipe": "groups", "table": t.node, "by": [st.id, "source"]})     # a file (or sheet) per group
    if small and (t.shape != SERIES or st is not None):
        g = sorted(small, key=lambda g: (0 if g[0] == t.node else 1, groups.index(g)))[0]
        out.append({"recipe": "breakdown", "table": t.node, "measure": m0, "stat": stat if m0 else "count", "by": g,
                    **({"together": True} if st is not None else {})})
    if big and m0:
        out.append({"recipe": "top", "table": t.node, "measure": m0, "stat": "sum" if stat == "sum" else "mean", "by": big[0], "n": 10})
    if t.pairs:
        p = t.pairs[0]
        if abs(p["r"]) >= 0.3:
            out.append({"recipe": "relationship", "table": t.node, "x": [t.node, p["x"]], "y": [t.node, p["y"]], "_r": p["r"]})
    if t.time and t.shape == SERIES:                      # gaps mean something in a log, not in a list of orders
        out.append({"recipe": "gaps", "table": t.node})
    if t.shape == SERIES and m0:
        out.append({"recipe": "outliers", "table": t.node, "measure": m0})
    if m0 and not t.time and not small and t.shape != LOOKUP:
        out.append({"recipe": "single", "table": t.node, "measure": m0, "stat": stat})
    if m0 and (t.rows or t.sampled) >= 20 and not (t.shape == LOOKUP and others):
        out.append({"recipe": "distribution", "table": t.node, "measure": m0})
    if reachable(model, t.node) and t.shape != LOOKUP:
        out.append({"recipe": "linked", "table": t.node})
    if st is not None:
        out.append({"recipe": "stacked", "table": t.node})
    rows = t.rows or t.sampled or 0
    if t.time and rows >= 4:
        every = _compare_every(t)
        if (t.span_seconds or 0) >= 2 * parse_bucket(every)[1]:
            out.append({"recipe": "change", "table": t.node, "measure": m0, "stat": (stat if m0 else "count"),
                        "every": every})
    if small:
        out.append({"recipe": "explain", "table": t.node, "by": small[0], "measure": m0, "stat": (stat if m0 else "count")})
    if len(t.columns) >= 2 and rows >= 20:
        out.append({"recipe": "drivers", "table": t.node})
    if t.time and m0 and rows >= 6:
        out.append({"recipe": "forecast", "table": t.node, "measure": m0})
    if t.geo:                                       # the table holds points: it can be drawn on a map
        out.append({"recipe": "map", "table": t.node})
        if t.categories:
            out.append({"recipe": "map", "table": t.node, "color_by": t.categories[0].name})
        if rows >= 4:                               # a handful of points still clusters into a few cells
            out.append({"recipe": "density", "table": t.node})
    for r in model.relations:                        # another table of places: nearest-place match
        if r.kind == "near" and t.node == r.tables[0]:
            out.append({"recipe": "nearest", "table": t.node, "other": r.tables[1]})
        if r.kind == "containment" and t.node == r.tables[0]:
            out.append({"recipe": "place", "table": t.node, "other": r.tables[1]})
    out.append({"recipe": "quality", "table": t.node})
    out.append({"recipe": "describe", "table": t.node})
    return out


def _distinct(model: DataModel, ref: list) -> int:
    if ref[0].startswith("stack:"):
        rel = model.relation(ref[0])
        return len(rel.tables) if rel else 0
    c = _col(model, ref)
    return c.distinct if c is not None else 0


# =================================================================== planning
class _Builder:
    """Collects plan steps while keeping track of what every column is called at the current step."""

    def __init__(self, model: DataModel, spec: dict) -> None:
        self.m, self.spec = model, spec
        self.steps: list[PlanStep] = []
        self.assumptions: list[dict[str, Any]] = []
        self.names: dict[tuple[str, str], str] = {}
        self.cols: list[str] = []
        self.current = ""
        self.avoid = set(spec.get("avoid") or [])

    # -- steps
    def use(self, node: str) -> str:
        key = f"node:{node}"
        if key not in {s.key for s in self.steps}:
            self.steps.append(PlanStep(key, "@", _tlabel(self.m, node), {"node": node}))
        return key

    def table(self, node: str) -> str:
        """A table ready to answer from: without its empty rows or the total row at the bottom, and with values that
        differ only in capitals or spaces spelled one way — each said, each one click to undo."""
        key = self.use(node)
        t = self.m.table(node)
        if t is None:
            return key
        keys = {s.key for s in self.steps}
        if self.spec.get("fix_breaks"):
            fixes = [{"row": b["row"] + 1, "column": c.name, "value": _tidy_number(b["expected"]), "was": b["value"],
                      "note": f"{c.name} is {c.derived['words']}"} for c, b in breaks_of(t) if b.get("row") is not None]
            if fixes:
                k = f"fix:{node}"
                if k not in keys:
                    self.add(k, "fix_values", f"{t.title} with calculated values put right", {"fixes": fixes}, {"in": [key]})
                key = k
        if t.blank_rows and not self.spec.get("keep_blank_rows"):
            k = f"blank:{node}"
            if k not in keys:
                self.add(k, "fix_missing", f"{t.title} without empty rows", {"method": "drop_all"}, {"in": [key]})
                self.assume(f"blank:{node}", f"Left out {t.blank_rows:,} empty row{'s' if t.blank_rows != 1 else ''} of {t.title}",
                            [{"label": "Keep them", "set": {"keep_blank_rows": True}}])
            key = k
        if t.total_row and not self.spec.get("keep_total_row"):
            k = f"total:{node}"
            if k not in keys:
                c, v = t.total_row["column"], t.total_row["value"]
                if t.total_row.get("blank"):          # named by nothing but its sum: blank labels, that exact total
                    rules = [{"column": t.total_row["blank"], "op": "empty"}, {"column": c, "op": "eq", "value": v}]
                    what = f"total row (its {c} is {v:,.10g}, the sum of the rows above)"
                else:
                    rules = [{"column": c, "op": "eq", "value": v, "case_sensitive": True}]
                    what = f"“{v}” row"
                self.add(k, "keep_rows", f"{t.title} without its total row",
                         {"mode": "remove", "conditions": {"match": "all", "rules": rules}}, {"in": [key]})
                self.assume(f"total:{node}", f"Left out the {what} at the bottom of {t.title}, since it adds up the others",
                            [{"label": "Keep it", "set": {"keep_total_row": True}}])
            key = k
        if t.wide:
            k = f"long:{node}"
            if k not in keys:
                w = t.wide
                self.add(k, "unpivot", f"{t.title}: {w['columns'][0]}–{w['columns'][-1]} as rows",
                         {"columns": list(w["columns"]), "name_column": "month", "value_column": "value",
                          "year": str(w["year"] or "")}, {"in": [key]})
                self.assume(f"long:{node}", f"Turned the month columns of {t.title} ({w['columns'][0]} … {w['columns'][-1]}) into rows, "
                            "one per item and month, with the numbers in “value”"
                            + (f", dated in {w['year']}" if w["year"] else ""))
            key = k
        if not self.spec.get("keep_spellings"):
            fixes = [(c.name, c.spellings) for c in t.columns if c.spellings and _safe_spellings(c.spellings)]
            if fixes:
                k = f"spell:{node}"
                if k not in keys:
                    self.add(k, "calculate", f"{t.title} with one spelling per value",
                             {"formulas": [{"name": c, "expr": _spelling_formula(c, sp)} for c, sp in fixes]}, {"in": [key]})
                    groups: dict[str, list[str]] = {}
                    for _, sp in fixes:
                        for variant, target in sorted(sp.items()):
                            groups.setdefault(target, []).append(variant)
                    shown = "; ".join(f"{', '.join(repr(v) for v in vs[:3])} as {target!r}" for target, vs in list(groups.items())[:3])
                    if len(groups) > 3:
                        shown += f"; and {len(groups) - 3} more"
                    self.assume(f"spell:{node}", f"Treated values that differ only in capitals or spaces as one ({shown})",
                                [{"label": "Keep every spelling apart", "set": {"keep_spellings": True}}])
                key = k
        return key

    def add(self, key: str, type_: str, title: str, params: dict, inputs: dict[str, list[str]]) -> str:
        self.steps.append(PlanStep(key, type_, title, params, inputs))
        return key

    def breaks(self, node: str, columns: list[str] | None = None) -> None:
        """Say which rows break their calculated column's rule, and offer to use the calculated values."""
        t = self.m.table(node)
        found = [(c, b) for c, b in breaks_of(t) if columns is None or c.name in columns] if t else []
        if not found or any(a["id"] == "breaks" for a in self.assumptions):
            return
        c, b = found[0]
        where = ", ".join(f"{k} {v}" for k, v in (b.get("where") or {}).items() if v is not None)
        more = f" (and {len(found) - 1} more)" if len(found) > 1 else ""
        if self.spec.get("fix_breaks"):
            self.assume("breaks", f"Used the calculated {c.name} ({c.derived['words']}): {_tidy_number(b['expected']):g} instead of the typed "
                                  f"{b['value']:g}{' for ' + where if where else ''}{more}",
                        [{"label": "Keep the typed values", "set": {"fix_breaks": False}}])
        else:
            self.assume("breaks", f"{c.name} is {c.derived['words']} on {c.derived['holds']} of {c.derived['rows']} rows. "
                                  f"{'For ' + where + ' it' if where else 'One row'} says {b['value']:g}, where the rule gives "
                                  f"{_tidy_number(b['expected']):g}{more}. Kept as typed",
                        [{"label": f"Use {_tidy_number(b['expected']):g}" if len(found) == 1 else "Use the calculated values",
                          "set": {"fix_breaks": True}}])

    def assume(self, id_: str, text: str, choices: list[dict] | None = None) -> None:
        self.assumptions.append({"id": id_, "text": text, "choices": choices or []})

    # -- the table the question is about
    def base(self, together: bool | None = None) -> None:
        spec, m = self.spec, self.m
        table = spec["table"]
        t = m.table(table)
        if t is None:
            raise PlanError("That table is not in the project any more")
        st = m.stack_of(table)
        use_stack = st is not None and (spec.get("together", True) if together is None else together)
        if use_stack:
            label_col = _free_name("source", [c.name for c in t.columns])
            members = [self.table(node) for node in st.tables]
            self.current = self.add("stack", "stack", f"All {len(st.tables)} tables together",
                                    {"label_column": label_col, "labels": list(st.labels)},
                                    {"tables": members})
            self.cols = [c.name for c in t.columns] + [label_col]
            for node in st.tables:
                for c in m.table(node).columns:
                    self.names[(node, c.name)] = c.name
            self.names[(st.id, "source")] = label_col
            self.assume("together", f"Put {', '.join(st.labels)} together. {st.why}",
                        [{"label": f"Only {t.title}", "set": {"together": False}}])
        else:
            self.current = self.table(table)
            self.cols = [c.name for c in t.columns]
            for c in t.columns:
                self.names[(table, c.name)] = c.name
            if st is not None:
                self.assume("together", f"Only {t.title}, not the other tables with the same columns",
                            [{"label": f"Put all {len(st.tables)} together", "set": {"together": True}}])

    def members(self) -> list[str]:
        st = self.m.stack_of(self.spec["table"])
        return list(st.tables) if st is not None and (self.spec.get("together", True)) else [self.spec["table"]]

    # -- columns from other tables
    def need(self, *refs: list | None) -> None:
        """Bring in the tables these columns live in, following links from the base table."""
        for ref in refs:
            if not ref or tuple(ref) in self.names:
                continue
            if ref[0].startswith("stack:"):
                raise PlanError("'Which table' only exists when the tables are put together")
            goal = ref[0]
            chain = _path(self.m, self.members(), goal, self.avoid)
            if chain is None:
                raise PlanError(f"{_tlabel(self.m, goal)} is not linked to {_tlabel(self.m, self.spec['table'])}, "
                                f"so its {ref[1]} cannot be used here")
            for r in chain:
                if (r.tables[1], r.right_on) in self.names:
                    continue
                self._join(r)

    def _join(self, r: Relation) -> None:
        m = self.m
        right = m.table(r.tables[1])
        left_key = self.names.get((r.tables[0], r.left_on))
        if left_key is None:
            raise PlanError(f"{r.left_on} is not available to link {right.title}")
        suffix = "_" + (norm(right.title)[:20] or "2")
        rkey = self.table(right.node)
        self.current = self.add(f"link:{r.id}", "combine", f"Add {right.title} details",
                                {"method": "match", "on": [left_key], "right_on": [r.right_on], "how": "left", "suffix": suffix},
                                {"left": [self.current], "right": [rkey]})
        for c in right.columns:
            if c.name == r.right_on:
                self.names[(right.node, c.name)] = left_key           # the key is kept once, under the first table's name
                continue
            name = c.name + suffix if c.name in self.cols else c.name
            self.cols.append(name)
            self.names[(right.node, c.name)] = name
        others = [x for x in m.relations if x.kind == "link" and x.id != r.id and set(x.tables) == set(r.tables)
                  and x.cardinality != "many-to-many"]
        choices = [{"label": f"Link on {x.left_on} ↔ {x.right_on} instead", "set": {"avoid": sorted(self.avoid | {r.id})}}
                   for x in others[:3]]
        self.assume(f"link:{r.id}", f"Linked {_tlabel(m, r.tables[0])} to {right.title} on {r.left_on} ↔ {r.right_on}. {r.why}", choices)

    def name(self, ref: list | None) -> str:
        if not ref:
            return ""
        n = self.names.get(tuple(ref))
        if n is None:
            raise PlanError(f"{ref[1]} is not available here")
        return n

    def filters(self) -> None:
        rules, formulas = [], []
        ops = {"gt": ">", "lt": "<", "ge": ">=", "le": "<=", "eq": "=", "ne": "!="}
        for f in self.spec.get("filters") or []:
            self.need(f["column"], f.get("column2"))
            if f.get("column2"):                       # one column against another: stock level below reorder level
                formulas.append(f"[{self.name(f['column'])}] {ops[f['op']]} [{self.name(f['column2'])}]")
                continue
            if f["op"] == "outside":                   # not between 10 and 20
                c = self.name(f["column"])
                formulas.append(f"([{c}] < {f['value']} OR [{c}] > {f['value2']})")
                continue
            rule = {"column": self.name(f["column"]), "op": f["op"], "value": f.get("value", "")}
            if f.get("value2") not in (None, ""):
                rule["value2"] = f["value2"]
            rules.append(rule)
        text = "Keep " + filter_text(self.m, self.spec.get("filters") or [])
        if rules:
            self.current = self.add("filter", "keep_rows", text,
                                    {"mode": "keep", "conditions": {"match": "all", "rules": rules}}, {"in": [self.current]})
        if formulas:
            self.current = self.add("compare_columns", "keep_rows", text, {"mode": "keep", "formula": " AND ".join(formulas)},
                                    {"in": [self.current]})


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


def plan(model: DataModel, spec: dict) -> Plan:
    """The steps that answer ``spec``. Raises PlanError with a plain reason when it cannot be answered."""
    spec = {k: v for k, v in spec.items() if not k.startswith("_")}
    recipe = spec.get("recipe")
    if recipe not in RECIPES:
        raise PlanError(f"Unknown kind of answer {recipe!r}")
    if not spec.get("table") or model.table(spec["table"]) is None:
        raise PlanError("Choose a table to answer from")
    b = _Builder(model, copy.deepcopy(spec))
    terminal, view, title, why = PLANNERS[recipe](b)
    b.spec.pop("title", None)
    said = [f for f in spec.get("filters") or [] if not (recipe == "groups" and f["column"] == spec.get("by"))]
    if said and recipe not in ("rows", "single"):
        title += " where " + filter_text(model, said)       # (the groups compared are already in the title)
    st = model.stack_of(spec["table"])
    if st is not None and recipe not in ("compare", "stacked", "gaps", "describe") and (
            not spec.get("together", True)):
        title += f" in {model.tables[spec['table']].title}"      # one of several tables with the same columns
    title = (spec.get("title") or "").strip() or title
    last = next((st for st in b.steps if st.key == terminal), None)
    if last is not None and last.type != "@":
        last.title = title                        # the answer's own step, and a chart's heading, say the whole question
        if last.type == "chart":
            last.params["title"] = title
    return Plan(steps=b.steps, terminal=terminal, title=title, config=spec, view=view,
                assumptions=b.assumptions, why=why, chips=chips(model, spec))


# -------------------------------------------------------------- one planner per recipe
def _plan_trend(b: _Builder):
    m, spec = b.m, b.spec
    t = m.table(spec["table"])
    time = spec.get("time") or ([t.node, t.time] if t.time else None)
    if not time:
        raise PlanError(f"{t.title} has no date or time column")
    measures = [r for r in (spec.get("measures") or []) if r]
    by = spec.get("by")
    stat = spec.get("stat") or ("count" if not measures else default_stat(m, t.node, measures[0]))
    every = spec.get("every") or auto_every(m, t, spec.get("filters"))
    per = EVERY_WORDS.get(every, every)
    if not spec.get("every"):
        span = _asked_span(t, spec.get("filters")) or t.span_seconds
        b.assume("every", f"One point per {per}, so the whole {'period asked about' if _asked_span(t, spec.get('filters')) else 'span'} "
                          f"({_span_text(span)}) fits on one chart",
                 [{"label": f"Per {EVERY_WORDS.get(e, e)}", "set": {"every": e}} for e in EVERY_CHOICES if e != every][:4])
    st = m.stack_of(t.node)
    together = st is not None and spec.get("together", True)
    own = lambda r: r[0] in (st.tables if st else [t.node])            # noqa: E731 - a column of the tables themselves
    ys = [_bucket_name(r[1], stat) for r in measures] or ["rows"]
    if together and not by and all(own(r) for r in measures) and all(
            own(f["column"]) and not f.get("column2") and f["op"] != "outside" for f in spec.get("filters") or []):
        # bucket each table first (small), then stack the buckets with a label: one line per table
        label_col = _free_name("source", [c.name for c in t.columns])
        keys = []
        for node in st.tables:
            cur = b.table(node)
            rules = [_rule(f, f["column"][1]) for f in spec.get("filters") or []]
            if rules:
                cur = b.add(f"filter:{node}", "keep_rows", f"Keep {filter_text(m, spec['filters'])}",
                            {"mode": "keep", "conditions": {"match": "all", "rules": rules}}, {"in": [cur]})
            keys.append(b.add(f"buckets:{node}", "time_buckets", f"{_tlabel(m, node)} per {per}",
                              {"every": every, "time_column": time[1], **_bucket_stats(measures, stat, time[1])}, {"in": [cur]}))
        b.add("stack", "stack", f"All {len(st.tables)} together", {"label_column": label_col, "labels": list(st.labels)},
              {"tables": keys})
        b.assume("together", f"One line for each of {', '.join(st.labels)}. {st.why}",
                 [{"label": f"Only {t.title}", "set": {"together": False}}])
        split = len(ys) > 1                              # several quantities: one panel per table, a line each
        chart = b.add("chart", "chart", "", {"kind": "line", "x": time[1], "series": [{"column": y} for y in ys],
                                             **({"split_by": label_col} if split else {"color_by": label_col}),
                                             "y_label": _y_label(m, measures[0] if measures else None, stat)}, {"in": ["stack"]})
    else:
        b.base()
        b.need(*measures, by)
        b.filters()
        tname = b.name(time) if tuple(time) in b.names else time[1]
        g = b.name(by) if by else None
        params = {"every": every, "time_column": tname, **_bucket_stats([[None, b.name(r)] for r in measures], stat, tname)}
        if g:
            params["by"] = [g]
        b.current = b.add("buckets", "time_buckets", f"Per {per}" + (f" and {group_label(m, by, spec['table'], spec.get('by_words'))}" if by else ""),
                          params, {"in": [b.current]})
        ys = [_bucket_name(b.name(r), stat) for r in measures] or ["rows"]
        extra = {"color_by": g} if g and len(ys) == 1 else ({"split_by": g} if g else {})
        chart = b.add("chart", "chart", "", {"kind": "line", "x": tname, "series": [{"column": y} for y in ys], **extra,
                                             "y_label": _y_label(m, measures[0] if measures else None, stat)}, {"in": [b.current]})
    what = ", ".join(label(m, r) for r in measures) if measures else "rows"
    title = f"{stat_title(stat, what)} per {per}" if stat != "count" else f"Rows per {per}"
    if by:
        title += f", by {group_label(m, by, spec['table'], spec.get('by_words'))}"
    b.steps[-1].title = title
    b.steps[-1].params["title"] = title
    return chart, "chart", title, f"{t.title} has {time[1]}" + (f" and {what}" if measures else "")


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
    from .timeutil import format_seconds
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


def _plan_breakdown(b: _Builder, top: int | None = None):
    m, spec = b.m, b.spec
    t = m.table(spec["table"])
    by, measure, part = spec.get("by"), spec.get("measure"), spec.get("by_part")
    if not by and not part:
        raise PlanError("Choose what to group by")
    stat = spec.get("stat") or ("count" if not measure else default_stat(m, t.node, measure))
    if not measure:
        stat = "count"
    b.base()
    b.need(by, measure)
    b.filters()
    if part:                                          # hour of the day, day of the week: a part of the time
        if not t.time:
            raise PlanError(f"{t.title} has no date or time column to take the {PART_WORDS[part]} from")
        g = _free_name(PART_WORDS[part], b.cols)
        b.current = b.add("part", "calculate", f"{PART_WORDS[part].capitalize()} of each row",
                          {"formulas": [{"name": g, "expr": _part_formula(part, b.name([t.node, t.time]))}]}, {"in": [b.current]})
        b.cols.append(g)
    else:
        g = b.name(by)
    if stat == "count":
        value = "rows"
        params = {"by": [g], "columns": [], "aggregations": [{"column": g, "stats": ["rows"], "alias": "rows"}]}
    else:
        value = b.name(measure)
        params = {"by": [g], "columns": [value], "default_stats": [stat]}
    gl = PART_WORDS[part] if part else group_label(m, by, spec["table"], spec.get("by_words"))
    b.current = b.add("groups", "group_summary", f"{STAT_WORDS.get(stat, stat)} by {gl}", params, {"in": [b.current]})
    if spec.get("share"):                             # each group's part of the whole, in per cent
        share = _free_name("share (%)", [g, value])
        b.current = b.add("share", "calculate", "Share of the whole",
                          {"formulas": [{"name": share, "expr": f"[{value}] / SUM([{value}]) * 100"}]}, {"in": [b.current]})
        value_shown = share
    else:
        value_shown = value
    bottom = bool(spec.get("bottom")) and bool(top)
    if part and not top:                              # hours and weekdays read best in their own order
        b.current = b.add("order", "sort", f"In {gl} order", {"columns": [g], "descending": False}, {"in": [b.current]})
    else:
        b.current = b.add("order", "sort", "Smallest first" if bottom else "Largest first", {"columns": [value], "descending": not bottom},
                          {"in": [b.current]})
    what = "rows" if stat == "count" else label(m, measure)
    if top:
        b.current = b.add("top", "take_sample", f"{'Bottom' if bottom else 'Top'} {top}", {"mode": "first", "rows": int(top)}, {"in": [b.current]})
        title = (f"{'Bottom' if bottom else 'Top'} {top} {plural(gl) if int(top) != 1 else gl} by " +
                 ("number of rows" if stat == "count" else f"{STAT_WORDS.get(stat, stat).lower()} {what}"))
    else:
        title = f"{stat_title(stat, what)} by {gl}" if stat != "count" else f"Rows by {gl}"
    if spec.get("share"):
        title = f"Share of {'rows' if stat == 'count' else what} by {gl} (%)"
    chart = b.add("chart", "chart", title, {"kind": "bar", "category": g, "value": value_shown,
                                            # one row per group already: the bar's statistic only names it right
                                            "stat": stat if stat in ("mean", "min", "max", "median") and not spec.get("share") else "sum",
                                            "title": title, "y_label": "%" if spec.get("share") else _y_label(m, measure, stat)}, {"in": [b.current]})
    if not spec.get("stat") and measure:
        b.assume("stat", ("Averaged the values" if stat == "mean" else "Added the amounts up") + f" for each {gl}",
                 [{"label": "Add them up" if stat == "mean" else "Average them", "set": {"stat": "sum" if stat == "mean" else "mean"}}])
    return chart, "chart", title, f"{gl} splits {t.title} into groups"


PART_WORDS = {"hour": "hour of the day", "weekday": "day of the week", "month": "month of the year", "day": "day of the month"}


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


def _plan_top(b: _Builder):
    n = top_n(b.spec)
    if b.spec.get("every") and not b.spec.get("by"):
        return _plan_top_times(b, n)
    return _plan_breakdown(b, top=n)


def _plan_top_times(b: _Builder, n: int):
    """The days (hours, months) with the largest total or average: each one added up first, then ranked, so
    "the highest sales day" is the day whose sales add up to most, not the single biggest sale."""
    m, spec = b.m, b.spec
    t = m.table(spec["table"])
    if not t.time:
        raise PlanError(f"{t.title} has no date or time column")
    measure = spec.get("measure")
    stat = spec.get("stat") or ("count" if not measure else default_stat(m, t.node, measure))
    if not measure:
        stat = "count"
    every, bottom = spec["every"], bool(spec.get("bottom"))
    per = EVERY_WORDS.get(every, every)
    b.base()
    b.need(measure)
    b.filters()
    tname = b.name([t.node, t.time])
    value = "rows" if stat == "count" else b.name(measure)
    b.current = b.add("buckets", "time_buckets", f"{STAT_WORDS.get(stat, stat)} per {per}",
                      {"every": every, "time_column": tname, **_bucket_stats([[None, value]] if measure else [], stat, tname)},
                      {"in": [b.current]})
    b.current = b.add("order", "sort", "Smallest first" if bottom else "Largest first", {"columns": [value], "descending": not bottom},
                      {"in": [b.current]})
    what = "number of rows" if stat == "count" else stat_title(stat, label(m, measure))
    what = what[0].lower() + what[1:]
    if spec.get("superlative") and n == 1:
        title = f"{spec['superlative'].capitalize()} {spec.get('noun') or per} (by {what})"
    else:
        title = f"{'Bottom' if bottom else 'Top'} {n} {per}s by {what}"
    key = b.add("top", "take_sample", title, {"mode": "first", "rows": n}, {"in": [b.current]})
    if not spec.get("stat") and measure:
        b.assume("stat", ("Averaged the values" if stat == "mean" else "Added the amounts up") + f" for each {per}",
                 [{"label": "Add them up" if stat == "mean" else "Average them", "set": {"stat": "sum" if stat == "mean" else "mean"}}])
    return key, "table", title, f"{t.title} has {t.time}"


def _plan_toprows(b: _Builder):
    """The rows with the largest (or smallest) values: the biggest orders, the best-paid employees."""
    m, spec = b.m, b.spec
    t = m.table(spec["table"])
    measure = spec.get("measure")
    if not measure:
        raise PlanError("Say which number to rank the rows by, for example “biggest orders by amount”")
    n = top_n(spec)
    bottom = bool(spec.get("bottom"))
    b.base()
    b.need(measure)
    b.filters()
    col = b.name(measure)
    b.current = b.add("order", "sort", "Smallest first" if bottom else "Largest first", {"columns": [col], "descending": not bottom},
                      {"in": [b.current]})
    if spec.get("superlative") and n == 1:
        title = f"{spec['superlative'].capitalize()} {spec.get('noun') or 'row'} (by {label(m, measure)})"
    else:
        title = f"{'Smallest' if bottom else 'Biggest'} {n} {spec.get('noun') or t.title} by {label(m, measure)}"
    key = b.add("top", "take_sample", title, {"mode": "first", "rows": n}, {"in": [b.current]})
    return key, "table", title, f"{t.title} ranked by {label(m, measure)}"


def _plan_single(b: _Builder):
    m, spec = b.m, b.spec
    measure = spec.get("measure")
    stat = spec.get("stat") or default_stat(m, spec["table"], measure)
    b.base()
    b.need(measure)
    b.filters()
    if stat == "count" or not measure:
        params = {"by": [], "columns": [], "aggregations": [{"column": b.cols[0], "stats": ["rows"], "alias": "rows"}]}
        title = "Number of rows"
    else:
        params = {"by": [], "columns": [b.name(measure)], "default_stats": [stat]}
        title = stat_title(stat, label(m, measure))
    if spec.get("filters"):
        title += f" where {filter_text(m, spec['filters'])}"
    key = b.add("number", "group_summary", title, params, {"in": [b.current]})
    return key, "table", title, "one number for the whole table"


def _plan_compare(b: _Builder):
    m, spec = b.m, b.spec
    a, other, measure = spec["table"], spec.get("other"), spec.get("measure")
    rel = next((r for r in m.relations if r.kind == "align" and set(r.tables) == {a, other}), None) if other else None
    if rel is None:
        raise PlanError("These two tables do not record the same thing over time")
    if not measure or measure[1] not in rel.shared:
        raise PlanError("Choose a quantity both tables record")
    ta, tb = m.table(a), m.table(other)
    col = measure[1]
    ka, kb = b.table(a), b.table(other)
    tol = spec.get("tolerance") or rel.tolerance
    b.add("pair", "combine", f"Pair {ta.title} with {tb.title}",
          {"method": "nearest_time", "left_time": ta.time, "right_time": tb.time, "direction": "nearest",
           "tolerance": tol, "suffix": "_2"}, {"left": [ka], "right": [kb]})
    b.assume("pairing", f"Paired each reading of {ta.title} with the nearest reading of {tb.title}"
                        + (f" no more than {tol} away" if tol else ""),
             [{"label": "Allow twice as far apart", "set": {"tolerance": _double(tol)}}] if tol else [])
    col_b = rel.pairs.get(col, col)                   # the same quantity may have another name in the second log
    y = f"{col_b}_2" if ta.column(col_b) is not None else col_b    # right-hand names that clash get the suffix
    if spec.get("fit", True):
        b.add("fit", "fit_curve", f"{tb.title} against {ta.title}", {"x": col, "y": y, "kind": "linear"}, {"in": ["pair"]})
        diff = f"{y}_residual"
        b.assume("fit", f"Treated {tb.title} as a straight-line function of {ta.title} (offset and scale). What's left over is the difference",
                 [{"label": "Show the plain difference instead", "set": {"fit": False}}])
    else:
        diff = _free_name("difference", [c.name for c in ta.columns] + [y])
        b.add("fit", "calculate", f"{tb.title} minus {ta.title}", {"formulas": [{"name": diff, "expr": f"[{y}] - [{col}]"}]},
              {"in": ["pair"]})
        b.assume("fit", f"The difference is {tb.title} minus {ta.title}, reading by reading",
                 [{"label": "Fit one to the other first (for sensors with an offset and scale)", "set": {"fit": True}}])
    span = min(x for x in (ta.span_seconds, tb.span_seconds) if x) if (ta.span_seconds or tb.span_seconds) else None
    every = spec.get("every") or bucket_for(span)
    b.add("buckets", "time_buckets", f"Per {EVERY_WORDS.get(every, every)}",
          {"every": every, "time_column": ta.time, "columns": [diff], "default_stats": ["mean"]}, {"in": ["fit"]})
    title = f"{label(m, measure)}: {tb.title} minus {ta.title}"
    chart = b.add("chart", "chart", title, {"kind": "line", "x": ta.time, "series": [{"column": diff, "label": f"{tb.title} minus {'fitted ' if spec.get('fit', True) else ''}{ta.title}"}],
                                            "title": title, "y_label": _y_label(m, measure, "mean")}, {"in": ["buckets"]})
    return chart, "chart", title, f"{ta.title} and {tb.title} both record {label(m, measure)}"


def _double(tol: str) -> str:
    import re
    mm = re.match(r"^(\d+)([a-z]+)$", tol or "")
    return f"{int(mm.group(1)) * 2}{mm.group(2)}" if mm else tol


def _plan_gaps(b: _Builder):
    m, spec = b.m, b.spec
    t = m.table(spec["table"])
    if not t.time:
        raise PlanError(f"{t.title} has no date or time column")
    b.base(together=False)
    key = b.add("gaps", "find_gaps", f"Gaps in {t.title}", {"time_column": t.time}, {"in": [b.current]})
    return key, "table", f"Gaps in {t.title}", f"{t.title} has readings over time"


def _plan_outliers(b: _Builder):
    m, spec = b.m, b.spec
    t = m.table(spec["table"])
    measure = spec.get("measure")
    if not measure and experiment(t, m):
        return _plan_study_outliers(b)
    if not measure:
        raise PlanError("Choose a number to check")
    series = t.shape == SERIES
    st = m.stack_of(t.node)
    flag = _free_name("unusual", [c.name for c in t.columns])
    params = {"columns": [measure[1]], "method": "rolling" if series else "iqr", "action": "flag", "flag_column": flag}
    if series:
        params.update({"window": 51, "threshold": 5.0})
    keep = {"mode": "keep", "conditions": {"match": "all", "rules": [{"column": flag, "op": "true"}]}}
    title = f"Unusual {label(m, measure)}"
    if st is not None and spec.get("together", True) and measure[0] in st.tables:
        # each table checked against its own readings (a spike is local to its log), then the finds put together
        found = []
        for node in st.tables:
            fk = b.add(f"flag:{node}", "remove_outliers", f"Mark unusual {label(m, measure)} in {_tlabel(m, node)}", params,
                       {"in": [b.table(node)]})
            found.append(b.add(f"unusual:{node}", "keep_rows", f"Unusual in {_tlabel(m, node)}", keep, {"in": [fk]}))
        label_col = _free_name("source", [c.name for c in t.columns])
        key = b.add("stack", "stack", title, {"label_column": label_col, "labels": list(st.labels)}, {"tables": found})
        b.assume("together", f"Checked each of {', '.join(st.labels)} against its own readings, then listed them together",
                 [{"label": f"Only {t.title}", "set": {"together": False}}])
    else:
        b.base(together=False)
        b.need(measure)
        params["columns"] = [b.name(measure)]
        flag = _free_name("unusual", b.cols); params["flag_column"] = flag
        keep = {"mode": "keep", "conditions": {"match": "all", "rules": [{"column": flag, "op": "true"}]}}
        b.add("flag", "remove_outliers", f"Mark unusual {label(m, measure)}", params, {"in": [b.current]})
        key = b.add("unusual", "keep_rows", title, keep, {"in": ["flag"]})
    b.assume("outliers", ("A reading is unusual when it is more than 5 typical spreads from the readings around it"
                          if series else "A value is unusual when it lies far outside the middle half of the values"))
    return key, "table", title, f"{label(m, measure)} is recorded steadily"


def row_noun(t: Table) -> str:
    """What a table's rows are, from a column that numbers them: a sample number names each sample."""
    for c in t.by_role(ID):
        w = name_words(c.name)
        if len(w) > 1 and w[0] in ("n", "no", "nr", "num", "number"):
            return plural(" ".join(w[1:]))
    return ""


def _plan_study_outliers(b: _Builder):
    """Unusual rows of a study: every number, each group against its own values (a control sample against the
    other control samples), far outside the middle half of them."""
    m, spec = b.m, b.spec
    t = m.table(spec["table"])
    measures = [c.name for c in ordered_measures(t)]
    by = spec.get("by") or next((g for g in groupables(m, t.node) if g[0] == t.node and 1 < _distinct(m, g) <= GROUP_MAX), None)
    b.base(together=False)
    flag = _free_name("unusual", b.cols)
    params = {"columns": measures, "method": "iqr", "action": "flag", "flag_column": flag}
    if by:
        params["by"] = [b.name(by)]
    b.add("flag", "remove_outliers", f"Mark unusual values in {t.title}", params, {"in": [b.current]})
    noun = spec.get("noun") or row_noun(t) or "rows"
    title = f"Unusual {noun}"
    key = b.add("unusual", "keep_rows", title, {"mode": "keep", "conditions": {"match": "all", "rules": [{"column": flag, "op": "true"}]}},
                {"in": ["flag"]})
    b.assume("outliers", "A value is unusual when it lies far outside the middle half of the values"
             + (f" of its own {group_label(m, by, t.node)}" if by else "") + f", in any of {len(measures)} numbers")
    return key, "table", title, "values far from the rest of their group"


def _plan_relationship(b: _Builder):
    m, spec = b.m, b.spec
    x, y = spec.get("x"), spec.get("y")
    if not x or not y:
        raise PlanError("Choose two numbers")
    b.base()
    b.need(x, y)
    b.filters()
    xn, yn = b.name(x), b.name(y)
    kind = spec.get("fit") or "linear"
    b.add("fit", "fit_curve", f"{label(m, y)} from {label(m, x)}", {"x": xn, "y": yn, "kind": kind}, {"in": [b.current]})
    title = f"{label(m, y)} against {label(m, x)}"
    chart = b.add("chart", "chart", title, {"kind": "scatter", "x": xn, "series": [{"column": yn}], "fit": kind, "title": title,
                                            "y_label": _y_label(m, y, "mean")}, {"in": ["fit"]})
    return chart, "chart", title, f"{label(m, x)} and {label(m, y)} move together"


def _plan_distribution(b: _Builder):
    m, spec = b.m, b.spec
    measure = spec.get("measure")
    if not measure:
        raise PlanError("Choose a number")
    b.base()
    b.need(measure)
    b.filters()
    title = f"Spread of {label(m, measure)}"
    t = m.table(measure[0])
    n = (t.rows or t.sampled or 0) if t is not None else 0
    bins = 60 if not n else max(5, min(60, round(math.sqrt(n))))          # a few values fill a few bins, not sixty
    chart = b.add("chart", "chart", title, {"kind": "histogram", "column": b.name(measure), "bins": bins, "title": title,
                                            "series": [], "mean_line": False}, {"in": [b.current]})
    return chart, "chart", title, "how the values are spread"


def _plan_linked(b: _Builder):
    m, spec = b.m, b.spec
    t = m.table(spec["table"])
    reach = reachable(m, t.node, b.avoid)
    if not reach:
        raise PlanError(f"{t.title} is not linked to another table")
    b.base()
    for node in reach:
        u = m.table(node)
        key = next((c for c in u.columns if c.role != "blank"), None)
        if key is not None:
            b.need([node, key.name])
    title = f"{t.title} with {', '.join(_tlabel(m, n) for n in reach)}"
    b.steps[-1].title = title
    return b.current, "table", title, f"{t.title} links to {len(reach)} other table{'s' if len(reach) > 1 else ''}"


def _plan_stacked(b: _Builder):
    m, spec = b.m, b.spec
    st = m.stack_of(spec["table"])
    if st is None:
        raise PlanError("No other table has the same columns")
    b.base(together=True)
    title = f"{', '.join(st.labels)} in one table"
    b.steps[-1].title = title
    return b.current, "table", title, st.why


def _plan_rows(b: _Builder):
    m, spec = b.m, b.spec
    t = m.table(spec["table"])
    if not spec.get("filters"):
        raise PlanError("Say which rows to keep, for example “where qty above 2”")
    b.base()
    b.filters()
    title = f"{t.title} where {filter_text(m, spec['filters'])}"
    b.steps[-1].title = title
    return b.current, "table", title, "the rows that match"


def _plan_describe(b: _Builder):
    t = b.m.table(b.spec["table"])
    b.base(together=False)
    title = f"What is in {t.title}"
    key = b.add("describe", "summarize", title, {}, {"in": [b.current]})
    return key, "table", title, "one row per column: count, blanks, average, range"


# -------------------------------------------------------------- the new questions
def _plan_change(b: _Builder):
    """This period compared with the one before: what rose, what fell, and by how much."""
    m, spec = b.m, b.spec
    t = m.table(spec["table"])
    time = spec.get("time") or ([t.node, t.time] if t.time else None)
    if not time:
        raise PlanError(f"{t.title} has no date or time column, so nothing can be compared by period")
    every = spec.get("every") or _compare_every(t)
    measure = spec.get("measure")
    stat = spec.get("stat") or (default_stat(m, t.node, measure) if measure else "count")
    if not measure:
        stat = "count"
    if stat == "std":
        raise PlanError("A spread can't be compared period by period here. Ask for the average instead")
    b.base()
    b.need(measure, spec.get("by"))
    b.filters()
    tname = b.name(time) if tuple(time) in b.names else time[1]
    by = spec.get("by")
    params: dict[str, Any] = {"time_column": tname, "every": every, "stat": stat}
    if measure:
        params["measure"] = b.name(measure)
    if by:
        params["by"] = [b.name(by)]
    what = label(m, measure) if measure else "rows"
    title = f"{stat_title(stat, what) if measure else 'Rows'}: latest {EVERY_WORDS.get(every, every)} vs the one before"
    if by:
        title += f", by {group_label(m, by, spec['table'], spec.get('by_words'))}"
    key = b.add("change", "compare_periods", title, params, {"in": [b.current]})
    if not spec.get("every"):
        others = [e for e in ("1d", "1w", "1mo", "1y") if e != every][:3]
        b.assume("every", f"Compared the latest {EVERY_WORDS.get(every, every)} with the one before it",
                 [{"label": f"Per {EVERY_WORDS.get(e, e)}", "set": {"every": e}} for e in others])
    return key, "table", title, f"{t.title} has {time[1]}"


def _plan_explain(b: _Builder):
    """What each group contributes to a total (or to a change): the biggest contributor and its share."""
    m, spec = b.m, b.spec
    t = m.table(spec["table"])
    by = spec.get("by")
    if by is None:
        gs = groupables(m, t.node)
        if not gs:
            raise PlanError(f"{t.title} has nothing to group by, so there is nothing to break the number down into")
        by = gs[0]
    measure = spec.get("measure")
    stat = spec.get("stat") or (default_stat(m, t.node, measure) if measure else "count")
    if not measure:
        stat = "count"
    if stat == "std":
        raise PlanError("A spread can't be split into the parts that make it up. Ask for the average or the total instead")
    b.base()
    b.need(measure, by)
    b.filters()
    params: dict[str, Any] = {"by": [b.name(by)], "stat": stat}
    if measure:
        params["measure"] = b.name(measure)
    time = spec.get("time") or ([t.node, t.time] if t.time else None)
    if time and spec.get("every"):
        params["time_column"] = b.name(time) if tuple(time) in b.names else time[1]
        params["every"] = spec["every"]
    gl = group_label(m, by, spec["table"], spec.get("by_words"))
    what = label(m, measure) if measure else "rows"
    title = f"What drove the change in {what.lower()}: {gl}" if spec.get("every") else f"What drives {what.lower()}: {gl}"
    key = b.add("explain", "contribution", title, params, {"in": [b.current]})
    if not spec.get("stat") and measure:
        b.assume("stat", ("Averaged the values" if stat == "mean" else "Added the amounts up") + f" for each {gl}",
                 [{"label": "Add them up" if stat == "mean" else "Average them", "set": {"stat": "sum" if stat == "mean" else "mean"}}])
    return key, "table", title, f"{gl} splits {t.title} into parts"


def _plan_drivers(b: _Builder):
    """How strongly each column relates to the others (or to one chosen column)."""
    m, spec = b.m, b.spec
    t = m.table(spec["table"])
    b.base(together=False)
    target = spec.get("target")
    params: dict[str, Any] = {}
    if target:
        params["target"] = b.name(target)
    title = f"What relates to {label(m, target)}" if target else f"How the columns of {t.title} move together"
    key = b.add("drivers", "associations", title, params, {"in": [b.current]})
    return key, "table", title, "how the columns move together"


def _plan_forecast(b: _Builder):
    """Where a value is heading, from its trend (and its usual day/week/month pattern)."""
    m, spec = b.m, b.spec
    t = m.table(spec["table"])
    time = spec.get("time") or ([t.node, t.time] if t.time else None)
    if not time:
        raise PlanError(f"{t.title} has no date or time column, so nothing can be projected forward")
    measure = spec.get("measure") or ([t.node, ordered_measures(t)[0].name] if ordered_measures(t) else None)
    if not measure:
        raise PlanError(f"{t.title} has no number to project")
    b.base()
    b.need(measure)
    b.filters()
    tname = b.name(time) if tuple(time) in b.names else time[1]
    col = b.name(measure)
    tc = t.column(t.time) if t.time else None
    if t.shape != SERIES or (tc is not None and not tc.unique) or m.stack_of(t.node) is not None:
        # things that happened (orders, visits), or several readings at once: each period is added up (or averaged)
        # first, and those are projected; a single order's value is not where the business is heading
        every = spec.get("every") or auto_every(m, t, spec.get("filters"))
        stat = default_stat(m, t.node, measure)
        b.current = b.add("buckets", "time_buckets", f"{stat_title(stat, label(m, measure))} per {EVERY_WORDS.get(every, every)}",
                          {"every": every, "time_column": tname, "columns": [col], "default_stats": [stat]}, {"in": [b.current]})
        b.assume("every", f"{'Added up' if stat == 'sum' else 'Averaged'} {label(m, measure)} per {EVERY_WORDS.get(every, every)} "
                          "first, then projected that forward",
                 [{"label": f"Per {EVERY_WORDS.get(e, e)}", "set": {"every": e}} for e in ("1d", "1w", "1mo") if e != every])
    params: dict[str, Any] = {"time_column": tname, "column": col, "method": spec.get("method") or "linear",
                              "horizon": int(spec.get("horizon") or 10)}
    if spec.get("every"):
        params["every"] = spec["every"]
    if spec.get("cycle"):
        params["cycle"] = spec["cycle"]
    if spec.get("threshold") not in (None, ""):
        params["threshold"] = spec["threshold"]
    title = f"Where {label(m, measure)} is heading"
    key = b.add("forecast", "forecast", title, params, {"in": [b.current]})
    return key, "table", title, "a projection from the trend, with a band that says how sure it is"


QUANTITY_NAMES = (READING_WORDS | AMOUNT_WORDS | {"length", "width", "height", "weight", "mass", "area", "volume", "depth",
                                                   "diameter", "radius", "age", "size", "cost", "price"}) - {"total", "value", "reading"}
PAIRED_WORDS = {"before", "after", "pre", "post", "baseline", "follow", "followup", "week", "day", "month", "visit",
                "time", "t0", "t1", "t2", "start", "end", "initial", "final", "first", "second", "trial", "run"}


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


def _plan_group_columns(b: _Builder):
    """Numbers kept in a column per group (Control | Treated, Before | After): turned into rows first, then compared
    as groups; paired when each row holds one thing measured in each (Before and After)."""
    m, spec = b.m, b.spec
    t = m.table(spec["table"])
    refs = [r for r in spec["columns"] if r]
    if len(refs) < 2:
        raise PlanError("Name at least two columns to compare")
    b.base(together=False)
    b.need(*refs)
    b.filters()
    from .units import header_parts
    names = [b.name(r) for r in refs]
    cores = [header_parts(n)[0] for n in names]
    units = {header_parts(n)[2] for n in names} - {""}
    if len(set(cores)) == len(cores) and cores != names and not set(cores) & (set(b.cols) - set(names)):
        # 'Before (cm)', 'After (cm)' are the groups Before and After, of a number in cm
        b.current = b.add("names", "choose_columns", "Groups named without their unit",
                          {"mode": "drop", "columns": [], "rename": dict(zip(names, cores))}, {"in": [b.current]})
        b.cols = [dict(zip(names, cores)).get(c, c) for c in b.cols]
        names = cores
    common = [w for w in name_words(cores[0]) if all(w in name_words(n) for n in cores[1:])]
    value = _free_name((" ".join(common) or "value") + (f" ({units.pop()})" if len(units) == 1 else ""), b.cols)
    group = _free_name("group", b.cols + [value])
    ids = [c for c in t.columns if c.role == ID and (t.node, c.name) in b.names]
    paired = spec.get("paired")
    if paired is None:
        paired = len(refs) == 2 and _paired_names(names)
    pair = b.name([t.node, ids[0].name]) if ids else ""
    if paired and not pair:
        pair = _free_name("row", b.cols + [value, group])
        b.current = b.add("rownum", "calculate", "Number each row", {"formulas": [{"name": pair, "expr": "ROW()"}]}, {"in": [b.current]})
    b.current = b.add("long", "unpivot", f"{', '.join(names)} as rows", {"columns": names, "name_column": group, "value_column": value},
                      {"in": [b.current]})
    params: dict[str, Any] = {"by": group, "columns": [value], "test": spec.get("test") or "auto"}
    if paired:
        params["pair_by"] = pair
    elif ids:
        params["label"] = pair
    who = ", ".join(names[:-1]) + " and " + names[-1]
    title = f"{who} compared"
    key = b.add("groups", "compare_groups", title, params, {"in": [b.current]})
    b.add("chart", "chart", title, {"kind": "bar", "category": group, "value": value, "stat": "mean", "error": "se", "title": title},
          {"in": [b.current]})
    if paired:
        b.assume("paired", f"Each row is one thing measured {len(refs)} ways, so the difference is tested within each row "
                           f"(a paired test{', matched on ' + pair if ids else ''})",
                 [{"label": "The columns are separate groups (not paired)", "set": {"paired": False}}])
    elif len(refs) == 2:
        b.assume("paired", "The columns were treated as separate groups of things",
                 [{"label": "Each row is one thing measured twice (a paired test)", "set": {"paired": True}}])
    return key, "table", title, f"{who} hold the same kind of number"


def _plan_groups(b: _Builder):
    """Do the groups differ? Each number compared between the groups: statistics, a test, how big, in a sentence."""
    m, spec = b.m, b.spec
    t = m.table(spec["table"])
    if spec.get("columns"):
        return _plan_group_columns(b)
    by = spec.get("by")
    if not by:
        own = [g for g in groupables(m, t.node) if g[0] == t.node and 1 < _distinct(m, g) <= GROUP_MAX]
        if not own:
            raise PlanError(f"{t.title} has no column that splits its rows into a few groups")
        by = own[0]
    else:
        gc = _col(m, by)
        if gc is not None and gc.role in ("measure", "time"):
            raise PlanError(f"“{gc.label or gc.name}” is a measured number, not a group. Compare named groups "
                            f"(for example “compare A and B”), or group by a category column")
    if _distinct(m, by) > GROUP_MAX:
        raise PlanError(f"{by[1]} has {_distinct(m, by)} values. Compare a few groups at a time (name them, as in "
                        f"“compare A and B”)")
    measures = [r for r in (spec.get("measures") or []) if r] or [[t.node, c.name] for c in ordered_measures(t)]
    if not measures:
        raise PlanError(f"{t.title} has no numbers to compare")
    b.base()
    b.need(by, *measures)
    b.filters()
    g = b.name(by)
    test = spec.get("test") or "auto"
    if test not in ("auto", "welch", "student", "rank", "none"):
        raise PlanError(f"test must be one of auto, welch, student, rank or none (got {test!r})")
    params: dict[str, Any] = {"by": g, "columns": [b.name(r) for r in measures], "test": test}
    ids = [c for c in t.columns if c.role == ID and (t.node, c.name) in b.names]
    if ids:
        params["label"] = b.name([t.node, ids[0].name])
    gl = group_label(m, by, spec["table"], spec.get("by_words"))
    vals = next((f["value"] for f in spec.get("filters") or [] if f["column"] == by and f["op"] == "in"), None)
    col = _col(m, by) if not by[0].startswith("stack:") else None
    rel = m.relation(by[0]) if by[0].startswith("stack:") else None
    names = [str(v) for v in (vals or (col.values if col is not None else rel.labels if rel is not None else []) or [])]
    who = (", ".join(names[:-1]) + " and " + names[-1]) if 2 <= len(names) <= 3 else f"each {gl}"
    if len(measures) == 1:
        title = f"{who} compared on {label(m, measures[0])}"
    else:
        title = f"{who} compared" if 2 <= len(names) <= 3 else f"{t.title} compared by {gl}"
    key = b.add("groups", "compare_groups", title, params, {"in": [b.current]})
    if len(measures) == 1:
        b.add("chart", "chart", title, {"kind": "bar", "category": g, "value": b.name(measures[0]), "stat": "mean",
                                        "error": "se", "title": title, "y_label": _y_label(m, measures[0], "mean")},
              {"in": [b.current]})
    if not spec.get("test"):
        b.assume("test", "Tested the difference with Welch's t-test (Welch's ANOVA for more than two groups), which does "
                         "not assume the groups vary alike; a rank test checks it when a small sample is not bell-shaped",
                 [{"label": "Student's t-test (the groups vary alike)", "set": {"test": "student"}},
                  {"label": "A rank test (Mann–Whitney)", "set": {"test": "rank"}}])
    for node in b.members():
        b.breaks(node, [r[1] for r in measures])
    return key, "table", title, f"{gl} splits {t.title} into groups"


def _plan_quality(b: _Builder):
    """A check of the table itself: blanks, duplicates, values that are really numbers stored as text."""
    t = b.m.table(b.spec["table"])
    b.base(together=False)
    title = f"Check {t.title}"
    key = b.add("quality", "check_data", title, {}, {"in": [b.current]})
    b.breaks(t.node)
    why = "blanks, duplicates and misread values"
    if breaks_of(t):
        c, _ = breaks_of(t)[0]
        why = f"{len(breaks_of(t))} value{'s' if len(breaks_of(t)) != 1 else ''} break{'s' if len(breaks_of(t)) == 1 else ''} " \
              f"the rule of {c.name} ({c.derived['words']})"
    return key, "table", title, why


# =================================================================== maps, grids and nearest places
CELL_CHOICES = ("0.01", "0.02", "0.05", "0.1", "0.25", "0.5", "1", "5", "10")
NEAR_CHOICES = ("1km", "5km", "10km", "25km", "50km", "100km")


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


def _plan_map(b: _Builder):
    m, spec = b.m, b.spec
    t = m.table(spec["table"])
    if not t.geo:
        raise PlanError(f"{t.title} has no latitude and longitude columns to map")
    b.base(together=False)
    params = {"lat": b.name([t.node, t.geo["lat"]]), "lon": b.name([t.node, t.geo["lon"]])}
    color = spec.get("color_by")
    if color:
        params["color_by"] = b.name([t.node, color])
    size = spec.get("size_by")
    if size:
        params["size_by"] = b.name(size)
    title = f"Map of {t.title}" + (f" by {label(m, [t.node, color])}" if color else "")
    key = b.add("map", "map", title, params, {"in": [b.current]})
    return key, "map", title, f"{t.title} has latitude and longitude columns"


def _plan_density(b: _Builder):
    m, spec = b.m, b.spec
    t = m.table(spec["table"])
    if not t.geo:
        raise PlanError(f"{t.title} has no latitude and longitude columns")
    b.base(together=False)
    lat = b.name([t.node, t.geo["lat"]])
    lon = b.name([t.node, t.geo["lon"]])
    size = str(spec.get("size") or _default_cell(t))
    measure = spec.get("measure")
    params: dict[str, Any] = {"lat": lat, "lon": lon, "size": size, "count_column": "points"}
    if measure:
        params["columns"] = [b.name(measure)]
        params["default_stats"] = [spec.get("stat") or "mean"]
    grid = b.add("grid", "points_grid", f"Points per {size}° cell", params, {"in": [b.current]})
    where = {
        "lat": "cell_lat", "lon": "cell_lon", "cell_size": size, "color_by": "points",
        "title": f"Points per {size}° cell",
    }
    key = b.add("map", "map", f"Density of {t.title}", where, {"in": [grid]})
    if not spec.get("size"):
        b.assume("size", f"Counted points in {size}° cells, so the map reads as a density of {t.title}",
                 [{"label": f"cells of {s}", "set": {"size": s}} for s in CELL_CHOICES if s != size][:4])
    return key, "map", f"Density of {t.title}", "how the points cluster in space"


def _plan_nearest(b: _Builder):
    m, spec = b.m, b.spec
    t = m.table(spec["table"])
    other = m.table(spec.get("other") or "")
    if other is None:
        raise PlanError("Choose the table of places to match against")
    rel = next((r for r in m.relations if r.kind == "near" and set(r.tables) == {t.node, other.node}), None)
    if rel is None:
        raise PlanError(f"{other.title} has no latitude and longitude columns to match against")
    left, right = m.table(rel.tables[0]), m.table(rel.tables[1])
    b.spec["table"] = left.node
    b.base(together=False)
    dist = str(spec.get("distance") or NEAR_DEFAULT)
    dist_col = _free_name("distance", [c.name for c in right.columns] + [c.name for c in left.columns])
    params = {"method": "nearest_feature", "near_how": "left",
              "left_lat": rel.geo["left_lat"], "left_lon": rel.geo["left_lon"],
              "right_lat": rel.geo["right_lat"], "right_lon": rel.geo["right_lon"],
              "max_distance": dist, "units": "km", "distance_column": dist_col}
    key = b.add("nearest", "combine", f"Nearest {right.title} to each {left.title}", params,
                {"left": [b.current], "right": [b.table(right.node)]})
    b.names[(left.node, dist_col)] = dist_col
    b.assume("distance", f"Looked for the nearest place within {dist}",
             [{"label": f"within {d}", "set": {"distance": d}} for d in NEAR_CHOICES if d != dist][:4])
    title = f"Nearest {right.title} to each {left.title}"
    return key, "table", title, rel.why


def _plan_place(b: _Builder):
    """Which region (or other polygon place) each point falls inside."""
    m, spec = b.m, b.spec
    t = m.table(spec["table"])
    other = m.table(spec.get("other") or "")
    if other is None:
        raise PlanError("Choose the table of regions to match against")
    rel = next((r for r in m.relations if r.kind == "containment" and set(r.tables) == {t.node, other.node}), None)
    if rel is None:
        raise PlanError(f"{other.title} has no polygon column to match points against")
    points = m.table(rel.tables[0])
    regions = m.table(rel.tables[1])
    b.spec["table"] = points.node
    b.base(together=False)
    geom = rel.geo.get("right_geometry") or "geometry"
    label = _free_name("region", [c.name for c in points.columns] + [c.name for c in regions.columns])
    params = {"method": "within", "near_how": "left", "left_lat": rel.geo["left_lat"], "left_lon": rel.geo["left_lon"],
              "right_geometry": geom, "place_column": label}
    key = b.add("place", "combine", f"Which {regions.title} each {points.title} is inside", params,
                {"left": [b.current], "right": [b.table(regions.node)]})
    b.names[(points.node, label)] = label
    return key, "table", f"Which {regions.title} each {points.title} is inside", rel.why


PLANNERS = {"compare": _plan_compare, "groups": _plan_groups, "trend": _plan_trend, "breakdown": _plan_breakdown, "top": _plan_top, "toprows": _plan_toprows,
            "relationship": _plan_relationship, "gaps": _plan_gaps, "outliers": _plan_outliers,
            "single": _plan_single, "distribution": _plan_distribution, "linked": _plan_linked, "stacked": _plan_stacked,
            "rows": _plan_rows, "describe": _plan_describe, "change": _plan_change, "explain": _plan_explain,
            "drivers": _plan_drivers, "forecast": _plan_forecast, "quality": _plan_quality,
            "map": _plan_map, "density": _plan_density, "nearest": _plan_nearest, "place": _plan_place}


# =================================================================== chips
def chips(model: DataModel, spec: dict) -> list[dict[str, Any]]:
    """The choices behind an answer, as chips: each has a ``key`` in the spec, a ``text`` and ``choices``
    ([{label, value}]). Choosing one sets ``spec[key] = value``; filters are removed by index."""
    out: list[dict[str, Any]] = []
    r = spec.get("recipe")
    t = model.table(spec.get("table", ""))
    if t is None:
        return out
    measures = [[t.node, c.name] for c in ordered_measures(t)]
    reach = reachable(model, t.node)
    for node in reach:
        u = model.table(node)
        measures += [[node, c.name] for c in ordered_measures(u)]
    cols = lambda refs: [{"label": _ref_label(model, ref, t.node), "value": ref} for ref in refs]  # noqa: E731
    if r in ("trend", "breakdown", "top", "single"):
        first = spec.get("measure") or (spec.get("measures") or [None])[0]
        stat = spec.get("stat") or (default_stat(model, t.node, first) if first else "count")
        out.append({"key": "stat", "text": STAT_WORDS.get(stat, stat), "value": stat,
                    "choices": [{"label": STAT_WORDS[s], "value": s} for s in STAT_CHOICES]})
    if r == "trend":
        ms = spec.get("measures") or []
        out.append({"key": "measures", "text": ", ".join(label(model, x) for x in ms) or "rows", "value": ms,
                    "choices": [{"label": _ref_label(model, ref, t.node), "value": [ref]} for ref in measures]})
        every = spec.get("every") or auto_every(model, t, spec.get("filters"))
        out.append({"key": "every", "text": f"per {EVERY_WORDS.get(every, every)}", "value": every,
                    "choices": [{"label": f"per {EVERY_WORDS.get(e, e)}", "value": e} for e in EVERY_CHOICES]})
    if r in ("breakdown", "top", "single", "outliers", "distribution", "toprows"):
        measure = spec.get("measure")
        choices = cols(measures)
        if r in ("breakdown", "top", "single"):
            choices = [{"label": "rows", "value": None}] + choices
        out.append({"key": "measure", "text": label(model, measure) if measure else "rows", "value": measure, "choices": choices})
    if r == "top" and spec.get("every") and not spec.get("by"):
        every = spec["every"]
        out.append({"key": "every", "text": f"per {EVERY_WORDS.get(every, every)}", "value": every,
                    "choices": [{"label": f"per {EVERY_WORDS.get(e, e)}", "value": e} for e in EVERY_CHOICES]})
    elif r in ("breakdown", "top"):
        by = spec.get("by")
        out.append({"key": "by", "text": f"by {_group_ref_label(model, by, t.node)}" if by else "by …", "value": by,
                    "choices": [{"label": f"by {_group_ref_label(model, g, t.node)}", "value": g} for g in groupables(model, t.node)]})
    if r in ("top", "toprows"):
        n = top_n(spec)
        out.append({"key": "n", "text": f"top {n}", "value": n, "choices": [{"label": f"top {k}", "value": k} for k in TOP_CHOICES]})
    if r == "relationship":
        for key in ("x", "y"):
            out.append({"key": key, "text": ("across: " if key == "x" else "up: ") + label(model, spec.get(key)), "value": spec.get(key),
                        "choices": cols(measures)})
    if r == "compare":
        rel = next((x for x in model.relations if x.kind == "align" and set(x.tables) == {spec["table"], spec.get("other")}), None)
        if rel is not None:
            out.append({"key": "measure", "text": label(model, spec.get("measure")), "value": spec.get("measure"),
                        "choices": [{"label": label(model, [t.node, c]), "value": [t.node, c]} for c in rel.shared]})
    if r in ("change", "explain"):
        measure = spec.get("measure")
        out.append({"key": "measure", "text": label(model, measure) if measure else "rows", "value": measure,
                    "choices": [{"label": "rows", "value": None}] + cols(measures)})
        stat = spec.get("stat") or (default_stat(model, t.node, measure) if measure else "count")
        out.append({"key": "stat", "text": STAT_WORDS.get(stat, stat), "value": stat,
                    "choices": [{"label": STAT_WORDS[s], "value": s} for s in STAT_CHOICES if s != "std"]})
    if r == "change":
        every = spec.get("every") or _compare_every(t)
        out.append({"key": "every", "text": f"per {EVERY_WORDS.get(every, every)}", "value": every,
                    "choices": [{"label": f"per {EVERY_WORDS.get(e, e)}", "value": e} for e in EVERY_CHOICES if not e.endswith("q")]})
    if r == "explain":
        by = spec.get("by")
        out.append({"key": "by", "text": f"by {_group_ref_label(model, by, t.node)}" if by else "by …", "value": by,
                    "choices": [{"label": f"by {_group_ref_label(model, g, t.node)}", "value": g} for g in groupables(model, t.node)]})
    if r == "map":
        cats = [[t.node, c.name] for c in t.categories]
        out.append({"key": "color_by", "text": (f"coloured by {label(model, [t.node, spec['color_by']])}"
                    if spec.get("color_by") else "not coloured"), "value": spec.get("color_by"),
                    "choices": [{"label": "not coloured", "value": None}] +
                               [{"label": f"by {_ref_label(model, c, t.node)}", "value": c[1]} for c in cats]})
    if r == "density":
        size = spec.get("size") or _default_cell(t)
        out.append({"key": "size", "text": f"cells of {size}°", "value": spec.get("size") or size,
                    "choices": [{"label": f"cells of {s}°", "value": s} for s in CELL_CHOICES]})
    if r == "nearest":
        dist = spec.get("distance") or NEAR_DEFAULT
        out.append({"key": "distance", "text": f"within {dist}", "value": spec.get("distance") or dist,
                    "choices": [{"label": f"within {d}", "value": d} for d in NEAR_CHOICES]})
    if r == "groups" and spec.get("columns"):
        test = spec.get("test") or "auto"
        out.append({"key": "test", "text": dict(TEST_CHOICES).get(test, test), "value": test,
                    "choices": [{"label": lab, "value": v} for v, lab in TEST_CHOICES]})
    elif r == "groups":
        by = spec.get("by")
        own = [g for g in groupables(model, t.node) if 1 < _distinct(model, g) <= GROUP_MAX]
        out.append({"key": "by", "text": f"by {_group_ref_label(model, by, t.node)}" if by else "by …", "value": by,
                    "choices": [{"label": f"by {_group_ref_label(model, g, t.node)}", "value": g} for g in own]})
        ms = spec.get("measures") or []
        out.append({"key": "measures", "text": ", ".join(label(model, x) for x in ms) if ms else "every number", "value": ms or None,
                    "choices": [{"label": "every number", "value": None}] + [{"label": _ref_label(model, ref, t.node), "value": [ref]}
                                                                             for ref in measures]})
        test = spec.get("test") or "auto"
        out.append({"key": "test", "text": dict(TEST_CHOICES).get(test, test), "value": test,
                    "choices": [{"label": lab, "value": v} for v, lab in TEST_CHOICES]})
    if r == "drivers":
        out.append({"key": "target", "text": f"related to {label(model, spec.get('target'))}" if spec.get("target") else "every pair",
                    "value": spec.get("target"), "choices": [{"label": "every pair", "value": None}] + cols(measures)})
    if r == "forecast":
        measure = spec.get("measure")
        out.append({"key": "measure", "text": label(model, measure) if measure else "?", "value": measure, "choices": cols(measures)})
        horizon = int(spec.get("horizon") or 10)
        out.append({"key": "horizon", "text": f"{horizon} ahead", "value": horizon,
                    "choices": [{"label": f"{k} ahead", "value": k} for k in (3, 5, 10, 20, 50)]})
        method = spec.get("method") or "linear"
        out.append({"key": "method", "text": ("Seasonal pattern" if method == "seasonal" else "Straight trend"), "value": method,
                    "choices": [{"label": "Straight trend", "value": "linear"}, {"label": "Seasonal pattern", "value": "seasonal"}]})
    for i, f in enumerate(spec.get("filters") or []):
        out.append({"key": f"filters.{i}", "text": filter_text(model, [f]), "value": f, "choices": [{"label": "Remove this filter", "value": None}]})
    return out


def _group_ref_label(model: DataModel, ref: list | None, base: str) -> str:
    """'customers' for a customer's name, else the column (and its table when it is not the base table)."""
    g = group_label(model, ref)
    return g if ref and model.table(ref[0]) is not None and g == model.table(ref[0]).title else _ref_label(model, ref, base)


def _ref_label(model: DataModel, ref: list | None, base: str) -> str:
    if not ref:
        return ""
    if ref[0].startswith("stack:"):
        return "which table"
    text = label(model, ref)
    if ref[0] != base:
        text += f" ({_tlabel(model, ref[0])})"
    return text


def apply_choice(spec: dict, key: str, value: Any) -> dict:
    """A copy of ``spec`` with one chip or assumption choice applied."""
    out = copy.deepcopy(spec)
    if key.startswith("filters."):
        i = int(key.split(".", 1)[1])
        fl = list(out.get("filters") or [])
        if value is None and 0 <= i < len(fl):
            fl.pop(i)
        elif 0 <= i < len(fl):
            fl[i] = value
        out["filters"] = fl
        if not fl:
            out.pop("filters")
        return out
    if key == "set" and isinstance(value, dict):
        out.update(value)
        return out
    if value is None:
        out.pop(key, None)
    else:
        out[key] = value
    return out
