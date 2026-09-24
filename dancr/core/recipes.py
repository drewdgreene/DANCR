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
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from .planner import Plan, PlanStep
from .understand import (DataModel, Table, Column, Relation, MEASURE, CATEGORY, ID, TEXT, TIME_ROLE, SERIES, LOOKUP,
                         EVENTS, bucket_for, norm)

RULES_VERSION = 1           # bump when a change to these rules would build a different plan from the same spec

STAT_WORDS = {"sum": "Total", "mean": "Average", "count": "Number of rows", "max": "Highest", "min": "Lowest",
              "median": "Median", "std": "Spread of"}
STAT_CHOICES = ["sum", "mean", "median", "min", "max", "count"]
EVERY_CHOICES = ["1m", "15m", "1h", "1d", "1w", "1mo", "1q", "1y"]
EVERY_WORDS = {"1s": "second", "10s": "10 seconds", "30s": "30 seconds", "1m": "minute", "5m": "5 minutes",
               "15m": "15 minutes", "30m": "30 minutes", "1h": "hour", "6h": "6 hours", "1d": "day", "1w": "week",
               "1mo": "month", "1q": "quarter", "1y": "year", "1ms": "millisecond", "10ms": "10 ms", "50ms": "50 ms",
               "100ms": "100 ms", "500ms": "half second", "5s": "5 seconds"}
TOP_CHOICES = [5, 10, 20, 50]

# the recipes, in the order that breaks ties
RECIPES = ["compare", "trend", "breakdown", "top", "toprows", "relationship", "gaps", "outliers", "single", "distribution",
           "linked", "stacked", "rows", "describe"]
RECIPE_LABELS = {"compare": "Compare two series", "trend": "Change over time", "breakdown": "Totals by group",
                 "top": "Top items", "relationship": "How two numbers relate", "gaps": "Gaps in the data",
                 "outliers": "Unusual readings", "single": "One number", "distribution": "Spread of values",
                 "linked": "Tables linked together", "stacked": "Tables stacked together", "rows": "Matching rows",
                 "toprows": "Biggest rows",
                 "describe": "Describe the table"}
WEIGHT = {"compare": 100, "trend": 95, "breakdown": 90, "top": 75, "relationship": 60, "gaps": 65, "outliers": 55,
          "single": 30, "distribution": 45, "linked": 50, "stacked": 60, "rows": 25, "toprows": 40, "describe": 20}
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
    for node in _reachable(model, table):
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


def _reachable(model: DataModel, table: str, avoid: set[str] | None = None) -> list[str]:
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


def _ordered_measures(t: Table) -> list[Column]:
    """The numbers most worth answering about first: amounts (sales, visits, quantity), then numbers with a unit
    (what a logger measures), then the rest, each in column order."""
    from .understand import _words

    def rank(c: Column) -> tuple:
        words = set(_words(c.name)) | set(_words(c.label or ""))
        money = (c.unit or "").strip().lower() in CURRENCY_UNITS or words & MONEY_WORDS
        return (0 if money else 1 if words & AMOUNT_WORDS else (2 if c.unit else 3), t.columns.index(c))
    return sorted(t.measures, key=rank)


AMOUNT_WORDS = {"sales", "sale", "revenue", "amount", "amounts", "cost", "costs", "spend", "spent", "profit", "income",
                "qty", "quantity", "quantities", "units", "unit", "count", "counts", "total", "sum", "volume", "orders",
                "items", "visits", "hours", "minutes", "calls", "tickets", "turnover", "paid", "payment", "payments",
                "sold", "bookings", "downloads", "clicks", "views", "impressions", "rainfall", "precipitation", "energy"}
READING_WORDS = {"salary", "salaries", "wage", "wages", "pay", "battery", "bounce", "duration", "time", "temperature", "temp", "pressure", "humidity", "speed", "velocity", "level", "depth", "height",
                 "voltage", "current", "rate", "ratio", "percent", "pct", "percentage", "price", "score", "age", "ph",
                 "conductivity", "salinity", "concentration", "density", "flow", "rating", "latitude", "longitude",
                 "lat", "lon", "lng", "altitude", "elevation", "weight", "mass", "size", "length", "width", "psi",
                 "psia", "bar", "tension", "load", "signal", "strength", "frequency", "value", "reading", "average",
                 "mean", "median", "index", "margin", "utilisation", "utilization", "occupancy", "efficiency"}
MONEY_WORDS = {"amount", "sales", "revenue", "turnover", "income", "cost", "costs", "spend", "profit", "paid",
               "payment", "payments", "debit", "credit", "balance", "value", "total"}
CURRENCY_UNITS = {"$", "€", "£", "¥", "usd", "eur", "gbp", "jpy", "chf", "aud", "cad", "nok", "sek", "dkk", "k$", "m$"}


def default_stat(model: DataModel, table: str, measure: list | None = None) -> str:
    """Amounts are added up (the total sales per month); readings are averaged (the mean pressure per hour).
    Decided by what the number is — its name and unit — and, when that says nothing, by its table: a logger's
    numbers are readings, a lookup's numbers describe its rows (a product's unit cost), anything else adds up."""
    from .understand import _words
    col = _col(model, measure) if measure and not measure[0].startswith("stack:") else None
    if col is not None:
        words = set(_words(col.name)) | set(_words(col.label or ""))
        unit = (col.unit or "").strip().lower()
        if words & AMOUNT_WORDS and not words & {"price", "rate", "average", "mean", "ratio", "percent", "pct"}:
            return "sum"
        if unit in CURRENCY_UNITS:
            return "sum"
        if words & READING_WORDS or unit:
            return "mean"
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


def _score(model: DataModel, t: Table, spec: dict) -> float:
    r = spec["recipe"]
    s = float(WEIGHT[r])
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
    measures = _ordered_measures(t)
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
    if _reachable(model, t.node) and t.shape != LOOKUP:
        out.append({"recipe": "linked", "table": t.node})
    if st is not None:
        out.append({"recipe": "stacked", "table": t.node})
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
                self.assume(f"total:{node}", f"Left out the {what} at the bottom of {t.title}: it adds up the others",
                            [{"label": "Keep it", "set": {"keep_total_row": True}}])
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
            self.assume("together", f"Put {', '.join(st.labels)} together; each row keeps which one it came from ({st.why})",
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
        self.assume(f"link:{r.id}", f"Linked {_tlabel(m, r.tables[0])} to {right.title} on {r.left_on} ↔ {r.right_on}: {r.why}", choices)

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
             "between": "between", "contains": "contains", "in": "is one of"}
    words.update({"year": "in", "month": "in"})
    parts = []
    for f in filters:
        if f.get("text"):                              # as the person typed it: "in March", "after 2024-06-01"
            parts.append(f"{label(model, f['column'])} {f['text']}")
            continue
        v = f.get("value", "")
        if f["op"] == "between":
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
    terminal, view, title, why = globals()[f"_plan_{recipe}"](b)
    b.spec.pop("title", None)
    if spec.get("filters") and recipe not in ("rows", "single"):
        title += " where " + filter_text(model, spec["filters"])
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
    every = spec.get("every") or auto_every(m, t)
    per = EVERY_WORDS.get(every, every)
    if not spec.get("every"):
        b.assume("every", f"One point per {per}, so the whole span ({_span_text(t.span_seconds)}) fits on one chart",
                 [{"label": f"Per {EVERY_WORDS.get(e, e)}", "set": {"every": e}} for e in EVERY_CHOICES if e != every][:4])
    st = m.stack_of(t.node)
    together = st is not None and spec.get("together", True)
    own = lambda r: r[0] in (st.tables if st else [t.node])            # noqa: E731 - a column of the tables themselves
    ys = [_bucket_name(r[1], stat) for r in measures] or ["rows"]
    if together and not by and all(own(r) for r in measures) and all(own(f["column"]) for f in spec.get("filters") or []):
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
        b.assume("together", f"One line for each of {', '.join(st.labels)} ({st.why})",
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
        b.current = b.add("buckets", "time_buckets", f"Per {per}" + (f" and {group_label(m, by)}" if by else ""),
                          params, {"in": [b.current]})
        ys = [_bucket_name(b.name(r), stat) for r in measures] or ["rows"]
        extra = {"color_by": g} if g and len(ys) == 1 else ({"split_by": g} if g else {})
        chart = b.add("chart", "chart", "", {"kind": "line", "x": tname, "series": [{"column": y} for y in ys], **extra,
                                             "y_label": _y_label(m, measures[0] if measures else None, stat)}, {"in": [b.current]})
    what = ", ".join(label(m, r) for r in measures) if measures else "rows"
    title = f"{stat_title(stat, what)} per {per}" if stat != "count" else f"Rows per {per}"
    if by:
        title += f", by {group_label(m, by)}"
    b.steps[-1].title = title
    b.steps[-1].params["title"] = title
    return chart, "chart", title, f"{t.title} has {time[1]}" + (f" and {what}" if measures else "")


def _rule(f: dict, column: str) -> dict:
    rule = {"column": column, "op": f["op"], "value": f.get("value", "")}
    if f.get("value2") not in (None, ""):
        rule["value2"] = f["value2"]
    return rule


def auto_every(model: DataModel, t: Table) -> str:
    """Readings: a few hundred points across the span. Events (sales, visits): a few dozen totals, since each
    point is a sum people read one by one."""
    tc = t.column(t.time) if t.time else None
    return bucket_for(t.span_seconds, tc.cadence if tc else None, target=400 if t.shape == SERIES else 60)


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
    return f"{name} ({unit})" if unit and f"({unit})" not in name and f"[{unit}]" not in name else name


def _span_text(secs: float | None) -> str:
    from .timeutil import format_seconds
    return format_seconds(secs) if secs else "unknown span"


def group_label(model: DataModel, ref: list | None) -> str:
    """A group as people say it: 'customers' for a customer's name, else the column's label."""
    if not ref:
        return ""
    if ref[0].startswith("stack:"):
        rel = model.relation(ref[0])
        from .understand import _words
        common = [w for w in _words(model.tables[rel.tables[0]].title)
                  if all(w in _words(model.tables[n].title) for n in rel.tables)] if rel else []
        return common[0] if common else "table"          # device_1, device_2: "by device"
    t = model.table(ref[0])
    c = _col(model, ref)
    if t is not None and t.shape == LOOKUP and c is not None and c.role in (ID, TEXT) and c.kind == "text":
        return t.title
    return label(model, ref)


def _plan_breakdown(b: _Builder, top: int | None = None):
    m, spec = b.m, b.spec
    t = m.table(spec["table"])
    by, measure = spec.get("by"), spec.get("measure")
    if not by:
        raise PlanError("Choose what to group by")
    stat = spec.get("stat") or ("count" if not measure else default_stat(m, t.node, measure))
    if not measure:
        stat = "count"
    b.base()
    b.need(by, measure)
    b.filters()
    g = b.name(by)
    if stat == "count":
        value = "rows"
        params = {"by": [g], "columns": [], "aggregations": [{"column": g, "stats": ["rows"], "alias": "rows"}]}
    else:
        value = b.name(measure)
        params = {"by": [g], "columns": [value], "default_stats": [stat]}
    gl = group_label(m, by)
    b.current = b.add("groups", "group_summary", f"{STAT_WORDS.get(stat, stat)} by {gl}", params, {"in": [b.current]})
    bottom = bool(spec.get("bottom")) and bool(top)
    b.current = b.add("order", "sort", "Smallest first" if bottom else "Largest first", {"columns": [value], "descending": not bottom},
                      {"in": [b.current]})
    what = "rows" if stat == "count" else label(m, measure)
    if top:
        b.current = b.add("top", "take_sample", f"{'Bottom' if bottom else 'Top'} {top}", {"mode": "first", "rows": int(top)}, {"in": [b.current]})
        title = (f"{'Bottom' if bottom else 'Top'} {top} {gl} by " +
                 ("number of rows" if stat == "count" else f"{STAT_WORDS.get(stat, stat).lower()} {what}"))
    else:
        title = f"{stat_title(stat, what)} by {gl}" if stat != "count" else f"Rows by {gl}"
    chart = b.add("chart", "chart", title, {"kind": "bar", "category": g, "value": value, "stat": "sum" if stat != "mean" else "mean",
                                            "title": title, "y_label": _y_label(m, measure, stat)}, {"in": [b.current]})
    if not spec.get("stat") and measure:
        b.assume("stat", ("Averaged the values" if stat == "mean" else "Added the amounts up") + f" for each {gl}",
                 [{"label": "Add them up" if stat == "mean" else "Average them", "set": {"stat": "sum" if stat == "mean" else "mean"}}])
    return chart, "chart", title, f"{gl} splits {t.title} into groups"


def _plan_top(b: _Builder):
    n = int(b.spec.get("n") or 10)
    return _plan_breakdown(b, top=n)


def _plan_toprows(b: _Builder):
    """The rows with the largest (or smallest) values: the biggest orders, the best-paid employees."""
    m, spec = b.m, b.spec
    t = m.table(spec["table"])
    measure = spec.get("measure")
    if not measure:
        raise PlanError("Say which number to rank the rows by, for example “biggest orders by amount”")
    n = int(spec.get("n") or 10)
    bottom = bool(spec.get("bottom"))
    b.base()
    b.need(measure)
    b.filters()
    col = b.name(measure)
    b.current = b.add("order", "sort", "Smallest first" if bottom else "Largest first", {"columns": [col], "descending": not bottom},
                      {"in": [b.current]})
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
    b.add("fit", "fit_curve", f"{tb.title} against {ta.title}", {"x": col, "y": y, "kind": "linear"}, {"in": ["pair"]})
    b.assume("fit", f"{tb.title} is a straight-line function of {ta.title} (offset and scale); what is left over is the difference")
    span = min(x for x in (ta.span_seconds, tb.span_seconds) if x) if (ta.span_seconds or tb.span_seconds) else None
    every = spec.get("every") or bucket_for(span)
    b.add("buckets", "time_buckets", f"Per {EVERY_WORDS.get(every, every)}",
          {"every": every, "time_column": ta.time, "columns": [f"{y}_residual"], "default_stats": ["mean"]}, {"in": ["fit"]})
    title = f"{label(m, measure)}: {tb.title} minus {ta.title}"
    chart = b.add("chart", "chart", title, {"kind": "line", "x": ta.time, "series": [{"column": f"{y}_residual", "label": f"{tb.title} minus fitted {ta.title}"}],
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
    chart = b.add("chart", "chart", title, {"kind": "histogram", "column": b.name(measure), "bins": 60, "title": title,
                                            "series": [], "mean_line": False}, {"in": [b.current]})
    return chart, "chart", title, "how the values are spread"


def _plan_linked(b: _Builder):
    m, spec = b.m, b.spec
    t = m.table(spec["table"])
    reach = _reachable(m, t.node, b.avoid)
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


# =================================================================== chips
def chips(model: DataModel, spec: dict) -> list[dict[str, Any]]:
    """The choices behind an answer, as chips: each has a ``key`` in the spec, a ``text`` and ``choices``
    ([{label, value}]). Choosing one sets ``spec[key] = value``; filters are removed by index."""
    out: list[dict[str, Any]] = []
    r = spec.get("recipe")
    t = model.table(spec.get("table", ""))
    if t is None:
        return out
    measures = [[t.node, c.name] for c in _ordered_measures(t)]
    reach = _reachable(model, t.node)
    for node in reach:
        u = model.table(node)
        measures += [[node, c.name] for c in _ordered_measures(u)]
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
        every = spec.get("every") or auto_every(model, t)
        out.append({"key": "every", "text": f"per {EVERY_WORDS.get(every, every)}", "value": every,
                    "choices": [{"label": f"per {EVERY_WORDS.get(e, e)}", "value": e} for e in EVERY_CHOICES]})
    if r in ("breakdown", "top", "single", "outliers", "distribution", "toprows"):
        measure = spec.get("measure")
        choices = cols(measures)
        if r in ("breakdown", "top", "single"):
            choices = [{"label": "rows", "value": None}] + choices
        out.append({"key": "measure", "text": label(model, measure) if measure else "rows", "value": measure, "choices": choices})
    if r in ("breakdown", "top"):
        by = spec.get("by")
        out.append({"key": "by", "text": f"by {_group_ref_label(model, by, t.node)}" if by else "by …", "value": by,
                    "choices": [{"label": f"by {_group_ref_label(model, g, t.node)}", "value": g} for g in groupables(model, t.node)]})
    if r in ("top", "toprows"):
        n = int(spec.get("n") or 10)
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
