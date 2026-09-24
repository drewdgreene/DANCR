"""Read a short question typed in plain words and turn it into a spec, deterministically.

Not AI: a fixed grammar over the words of *this* project — its column names and labels, table names,
the values of its categories, statistics ("total", "average"), time words ("per hour", "monthly") and
comparisons ("above 5", "between 1 and 2"). Everything understood becomes a chip the person can change,
and a word that is not understood is reported rather than guessed, so the same words on the same data
always build the same answer::

    average pressure per hour for MJ03F     total qty by region        top 10 customers by qty
    compare MJ03E and MJ03F                 gaps in probe_MJ03E        orders where qty above 2

Pure core, no Qt.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Any

from .recipes import PlanError, plan, groupables, RECIPES
from .understand import DataModel, MEASURE, CATEGORY, TIME_ROLE, ID, TEXT, FLAG, BLANK, CONSTANT, norm, _words

STATS = {"total": "sum", "totals": "sum", "sum": "sum", "sums": "sum", "add up": "sum", "added up": "sum",
         "average": "mean", "averages": "mean", "avg": "mean", "mean": "mean", "typical": "mean",
         "count": "count", "number of": "count", "how many": "count", "count of": "count",
         "max": "max", "maximum": "max", "highest": "max", "peak": "max", "largest": "max", "biggest": "max",
         "min": "min", "minimum": "min", "lowest": "min", "smallest": "min", "median": "median"}
TIME_UNITS = {"second": "s", "seconds": "s", "sec": "s", "secs": "s", "s": "s", "minute": "m", "minutes": "m",
              "min": "m", "mins": "m", "hour": "h", "hours": "h", "hr": "h", "hrs": "h", "h": "h", "day": "d",
              "days": "d", "d": "d", "week": "w", "weeks": "w", "w": "w", "month": "mo", "months": "mo",
              "quarter": "q", "quarters": "q", "year": "y", "years": "y"}
ADVERBS = {"hourly": "1h", "daily": "1d", "weekly": "1w", "monthly": "1mo", "quarterly": "1q", "yearly": "1y",
           "annually": "1y", "annual": "1y", "minutely": "1m"}
RECIPE_WORDS = {"over time": "trend", "trend": "trend", "trends": "trend", "timeline": "trend", "change": "trend",
                "gaps": "gaps", "gap": "gaps", "missing data": "gaps", "dropouts": "gaps", "coverage": "gaps",
                "outliers": "outliers", "outlier": "outliers", "spikes": "outliers", "spike": "outliers",
                "unusual": "outliers", "anomalies": "outliers", "odd": "outliers",
                "distribution": "distribution", "histogram": "distribution", "spread": "distribution",
                "describe": "describe", "summary": "describe", "summarise": "describe", "summarize": "describe",
                "overview": "describe", "what is in": "describe",
                "relationship": "relationship", "correlation": "relationship", "depend on": "relationship",
                "depends on": "relationship", "relate": "relationship",
                "compare": "compare", "comparison": "compare", "versus": "compare", "vs": "compare",
                "against": "compare", "difference between": "compare",
                "together": "stacked", "stacked": "stacked", "stack": "stacked", "combined": "stacked",
                "linked": "linked", "details": "linked", "joined": "linked"}
BY_WORDS = {"by", "per", "for each", "each", "grouped by", "split by", "broken down by", "across", "by each"}
OPS = {"above": "gt", "over": "gt", "more than": "gt", "greater than": "gt", ">": "gt", "exceeds": "gt",
       "exceeding": "gt", "higher than": "gt", "below": "lt", "under": "lt", "less than": "lt", "<": "lt",
       "lower than": "lt", "at least": "ge", ">=": "ge", "at most": "le", "<=": "le", "between": "between",
       "is": "eq", "=": "eq", "equals": "eq", "equal to": "eq", "is not": "ne", "not": "ne", "!=": "ne",
       "except": "ne", "excluding": "ne", "without": "ne"}
STOP = {"show", "me", "the", "a", "an", "of", "what", "whats", "what's", "are", "was", "were", "how", "does",
        "do", "did", "i", "want", "see", "give", "find", "please", "chart", "plot", "graph", "list", "my",
        "for", "in", "where", "when", "with", "to", "on", "from", "data", "and", "all", "it", "its", "that",
        "which", "who", "there", "any", "can", "you", "get", "tell", "about", "has", "have", "had", "only",
        "rows", "row", "values", "readings", "reading", "records", "time", "at", "?", "this", "these", "then",
        "table", "tables", "file", "files", "their", "them", "look", "into", "during", "within", "thing", "things"}
TOP_WORDS = {"top": False, "best": False, "bottom": True, "worst": True}


@dataclass
class Meaning:
    kind: str                  # stat every recipe col table value by op num top unit stop and
    value: Any = None
    refs: list[list] = field(default_factory=list)      # col: every column the words could mean


@dataclass
class Asked:
    """What a question was read as. ``spec`` is None when it could not be read; ``message`` says why."""
    text: str
    spec: dict[str, Any] | None = None
    title: str = ""
    message: str = ""
    unknown: list[str] = field(default_factory=list)
    hints: list[str] = field(default_factory=list)          # "did you mean" phrases
    ambiguous: list[dict[str, Any]] = field(default_factory=list)   # {"text", "chose", "choices": [{label, set}]}
    chips: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.spec is not None

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "ok": self.ok, "spec": self.spec, "title": self.title, "message": self.message,
                "unknown": self.unknown, "hints": self.hints, "ambiguous": self.ambiguous, "chips": self.chips}


# =================================================================== vocabulary
def _tokens(text: str) -> list[str]:
    return re.findall(r">=|<=|!=|[<>=]|-?\d+(?:[.,]\d+)*|[\w'%/]+(?:[.\-][\w]+)*", text.lower())


def vocabulary(model: DataModel) -> dict[tuple[str, ...], list[Meaning]]:
    """Every phrase this project understands, to its meanings (fixed words first, then the project's own)."""
    voc: dict[tuple[str, ...], list[Meaning]] = {}

    def add(phrase: str, m: Meaning) -> None:
        key = tuple(_tokens(phrase))
        if key:
            voc.setdefault(key, []).append(m)

    for w, s in STATS.items():
        add(w, Meaning("stat", s))
    for w, e in ADVERBS.items():
        add(w, Meaning("every", e))
    for w, r in RECIPE_WORDS.items():
        add(w, Meaning("recipe", r))
    for w in BY_WORDS:
        add(w, Meaning("by"))
    for w, o in OPS.items():
        add(w, Meaning("op", o))
    for w, u in TIME_UNITS.items():
        add(w, Meaning("unit", u))
    for w, rev in TOP_WORDS.items():
        add(w, Meaning("top", rev))
    add("and", Meaning("and"))
    table_words: set[str] = set()
    for t in model.tables.values():
        for n in (t.title.lower(), t.title.lower().replace("_", " ")):
            table_words |= {n, n[:-1] if n.endswith("s") else n}
    for node, t in model.tables.items():
        names = {t.title, t.title.replace("_", " "), norm(t.title)}
        names |= {n[:-1] for n in names if len(n) > 3 and n.endswith("s")}        # "products" is also "product"
        for phrase in names:
            add(phrase, Meaning("table", node))
        for c in t.columns:
            if c.role in (BLANK, CONSTANT):
                continue
            ref = [node, c.name]
            words = _words(c.name)
            if c.role == ID and len(words) > 1 and words[-1] in ("id", "key", "code", "no", "number", "ref"):
                stem = " ".join(words[:-1])
                if stem not in table_words:                       # "by patient" is by patient_id
                    _add_ref(voc, tuple(_tokens(stem)), ref)
            for phrase in _column_phrases(c.name, c.label):
                _add_ref(voc, tuple(_tokens(phrase)), ref)
                for n in names:                                  # "product name", "customers region"
                    _add_ref(voc, tuple(_tokens(f"{n} {phrase}")), ref)
            if c.role == CATEGORY:
                for v in c.values:
                    if v is not None and str(v).strip():
                        voc.setdefault(tuple(_tokens(str(v))), []).append(Meaning("value", str(v), [ref]))
    for r in model.relations:
        if r.kind == "stack":
            for node, lab in zip(r.tables, r.labels):
                add(lab, Meaning("table", node))
    return voc


def _add_ref(voc, key, ref) -> None:
    if not key:
        return
    for m in voc.setdefault(key, []):
        if m.kind == "col":
            if ref not in m.refs:
                m.refs.append(ref)
            return
    voc[key].append(Meaning("col", refs=[ref]))


def _column_phrases(name: str, label: str) -> set[str]:
    """pressure_psia -> 'pressure_psia', 'pressure psia', 'pressure'; 'Pressure (bar)' -> 'pressure'."""
    out = {name, name.replace("_", " "), label, re.sub(r"\s*[\(\[].*?[\)\]]\s*$", "", label)}
    words = _words(name)
    if words:
        out.add(" ".join(words))
        if len(words) > 1 and words[-1] not in ("id", "key", "code", "no"):
            out.add(words[0])                         # pressure_psia is 'pressure'
    return {o for o in out if o and o.strip()}


# =================================================================== reading
def ask(model: DataModel, text: str) -> Asked:
    """Read ``text`` as a question about the tables in ``model``."""
    out = Asked(text=text)
    toks = _tokens(text)
    if not toks:
        out.message = "Type a question, for example “average pressure per hour” or “total sales by region”"
        return out
    voc = vocabulary(model)
    maxlen = max((len(k) for k in voc), default=1)
    items: list[tuple[str, Meaning]] = []           # (words, meaning)
    i = 0
    while i < len(toks):
        hit = None
        for n in range(min(maxlen, len(toks) - i), 0, -1):
            key = tuple(toks[i:i + n])
            if key in voc:
                hit = (key, voc[key]); break
        if hit is not None:
            key, ms = hit
            items.append((" ".join(key), _pick(ms)))
            i += len(key); continue
        tok = toks[i]
        num = _number(tok)
        if num is not None:
            items.append((tok, Meaning("num", num)))
        elif tok in STOP:
            items.append((tok, Meaning("stop")))
        else:
            out.unknown.append(tok)
        i += 1
    if out.unknown:
        phrases = [" ".join(k) for k in voc]
        out.hints = sorted({h for u in out.unknown for h in difflib.get_close_matches(u, phrases, n=3, cutoff=0.6)})
        out.message = ("I don't know " + ", ".join(f"“{u}”" for u in out.unknown)
                       + (". Did you mean " + ", ".join(f"“{h}”" for h in out.hints) + "?" if out.hints else
                          ". Use the names of your columns, tables or their values."))
        return out
    try:
        out.spec = _assemble(model, items, out)
    except PlanError as e:
        out.spec, out.message = None, str(e)
        return out
    try:
        p = plan(model, out.spec)
    except PlanError as e:
        out.spec, out.message = None, str(e)
        return out
    out.title, out.chips = p.title, p.chips
    return out


def _pick(ms: list[Meaning]) -> Meaning:
    """When a phrase means several things, a column or value of the project beats a fixed word."""
    order = {"col": 0, "value": 1, "table": 2}
    return sorted(ms, key=lambda m: order.get(m.kind, 3))[0]


def _number(tok: str) -> float | int | None:
    from .dtypes import typed_value
    if not re.fullmatch(r"-?\d+(?:[.,]\d+)*", tok):
        return None
    try:
        return typed_value(tok)
    except Exception:  # noqa: BLE001
        return None


def _assemble(model: DataModel, items: list[tuple[str, Meaning]], out: Asked) -> dict[str, Any]:
    kinds = [m.kind for _, m in items]
    stat = next((m.value for _, m in items if m.kind == "stat"), None)
    recipes = [m.value for _, m in items if m.kind == "recipe"]
    tables = [m.value for _, m in items if m.kind == "table"]
    every = next((m.value for _, m in items if m.kind == "every"), None)
    top, bottom = None, False
    filters: list[dict[str, Any]] = []
    by = None
    cols: list[tuple[str, list[list]]] = []          # (words, candidate refs) in order, for measures
    i = 0
    while i < len(items):
        words, m = items[i]
        nxt = items[i + 1][1] if i + 1 < len(items) else None
        if m.kind == "top":
            n = items[i + 1][1].value if nxt is not None and nxt.kind == "num" else 10
            top, bottom = int(n), bool(m.value)
            i += 2 if nxt is not None and nxt.kind == "num" else 1
            continue
        if m.kind in ("by", "stop") and nxt is not None and nxt.kind == "unit":
            every = f"1{nxt.value}"; i += 2; continue
        if m.kind in ("by", "stop") and nxt is not None and nxt.kind == "num" and i + 2 < len(items) and items[i + 2][1].kind == "unit":
            every = f"{int(nxt.value)}{items[i + 2][1].value}"; i += 3; continue
        if m.kind == "by" and nxt is not None and nxt.kind == "col":
            by = (items[i + 1][0], nxt.refs); i += 2; continue
        if m.kind == "col":
            # a column followed by a comparison is a filter
            if nxt is not None and nxt.kind == "op":
                f, used = _filter(items, i)
                if f is not None:
                    filters.append(f); i += used; continue
            cols.append((words, m.refs)); i += 1; continue
        if m.kind == "value":
            negate = i > 0 and items[i - 1][1].kind == "op" and items[i - 1][1].value == "ne"
            filters.append({"column": m.refs[0], "op": "ne" if negate else "eq", "value": m.value, "_words": words})
            i += 1; continue
        if m.kind == "unit":
            every = f"1{m.value}"; i += 1; continue
        i += 1
    # which table: the one named, else the one most of the columns live in
    base = _base_table(model, tables, cols, by, filters, out)
    resolve = lambda w, refs: _resolve(model, w, refs, base, out)   # noqa: E731
    measures = [resolve(w, r) for w, r in cols]
    measures = [r for r in measures if r is not None]
    by_ref = resolve(*by) if by else None
    for f in filters:
        f.pop("_words", None)
    time_ref = next((r for r in measures if _role(model, r) == TIME_ROLE), None)
    if time_ref is not None:
        measures.remove(time_ref)
    if by_ref is not None and _role(model, by_ref) == TIME_ROLE:
        time_ref, by_ref = by_ref, None
        every = every or None
    if by_ref is not None and _role(model, by_ref) == MEASURE:
        measures.insert(0, by_ref); by_ref = None       # "customers by qty": qty is what is ranked, not a group
    numbers = [r for r in measures if _role(model, r) == MEASURE]
    groups = [r for r in measures if _role(model, r) in (CATEGORY, ID, TEXT, FLAG)]
    if by_ref is None and groups and (numbers or stat == "count" or top):
        by_ref = groups[0]                              # "customers by qty": the group named first
    named_lookups = [t for t in tables if t != base and model.tables[t].shape == "lookup"]
    if by_ref is None and named_lookups and (numbers or stat or top):
        by_ref = _name_column(model, named_lookups[0], base)   # "top 3 customers by qty": customers by their name
    tables = [t for t in tables if t not in named_lookups]
    spec: dict[str, Any] = {"table": base}
    stack = model.stack_of(base)
    if stack is not None:
        spec["together"] = not any(t in stack.tables for t in tables) or len({t for t in tables if t in stack.tables}) > 1
    if filters:
        spec["filters"] = filters
    recipe = recipes[0] if recipes else None
    if recipe == "compare" and len({tuple(r) for r in numbers}) >= 2 and len(set(tables)) < 2:
        recipe = "relationship"                        # "temperature against pressure": two numbers, not two tables
    if recipe == "compare":
        pair = [t for t in tables]
        if stack is not None and len(pair) < 2:
            pair = list(stack.tables[:2])
        if len(pair) >= 2:
            a, b = pair[0], pair[1]
            rel = next((r for r in model.relations if r.kind == "align" and set(r.tables) == {a, b}), None)
            if rel is None:
                raise PlanError(f"{model.tables[a].title} and {model.tables[b].title} do not record the same thing over time")
            if rel.tables[0] != a:
                a, b = rel.tables
            m = next((r[1] for r in numbers if r[1] in rel.shared), rel.shared[0])
            spec.update({"recipe": "compare", "table": a, "other": b, "measure": [a, m]})
            spec.pop("together", None)
            return spec
        recipe = "relationship" if len(numbers) >= 2 else None
        if recipe is None:
            raise PlanError("Compare what? Name two tables (“compare A and B”) or two numbers (“y against x”)")
    if recipe == "relationship" or (recipe is None and len(numbers) == 2 and any(w in ("against", "vs", "versus") for w, _ in items)):
        if len(numbers) < 2:
            raise PlanError("Name the two numbers, for example “temperature against pressure”")
        spec.update({"recipe": "relationship", "x": numbers[1], "y": numbers[0]})
        spec["together"] = False if stack is not None else spec.get("together")
        return _tidy(spec)
    if recipe in ("gaps", "describe", "stacked", "linked"):
        spec["recipe"] = recipe
        return _tidy(spec)
    if recipe in ("outliers", "distribution"):
        if not numbers:
            raise PlanError(f"Name the number to look at, for example “{recipe} in {_first_measure(model, base)}”")
        spec.update({"recipe": recipe, "measure": numbers[0]})
        return _tidy(spec)
    if top is not None:
        if by_ref is None:
            raise PlanError(f"Top {top} what? Say what to rank, for example “top {top} customers by sales”")
        spec.update({"recipe": "top", "n": top, "by": by_ref, "bottom": bottom or None, "measure": numbers[0] if numbers else None,
                     "stat": stat or ("count" if not numbers else "sum")})
        return _tidy(spec)
    if by_ref is not None:
        spec.update({"recipe": "breakdown", "by": by_ref, "measure": numbers[0] if numbers else None})
        if stat:
            spec["stat"] = stat
        return _tidy(spec)
    t = model.tables[base]
    if recipe == "trend" or every or time_ref is not None or (numbers and t.time and not stat) or (stat == "count" and t.time and not numbers):
        if not t.time and time_ref is None:
            raise PlanError(f"{t.title} has no date or time column, so it cannot be shown over time")
        spec.update({"recipe": "trend", "measures": numbers[:3]})
        if every:
            spec["every"] = every
        if stat:
            spec["stat"] = stat
        if time_ref is not None:
            spec["time"] = time_ref
        return _tidy(spec)
    if numbers or stat:
        spec.update({"recipe": "single", "measure": numbers[0] if numbers else None, "stat": stat or "sum"})
        return _tidy(spec)
    if filters:
        spec["recipe"] = "rows"
        return _tidy(spec)
    if tables:
        spec["recipe"] = "describe"
        return _tidy(spec)
    raise PlanError("Say what you would like to know: a number to add up or average, a group, or a time step")


def _name_column(model: DataModel, table: str, base: str) -> list | None:
    """The column that names a lookup table's rows (a customer's name), else its key."""
    g = [r for r in groupables(model, base) if r[0] == table]
    t = model.tables[table]
    texts = [r for r in g if _role(model, r) in (ID, TEXT)]
    if texts:
        return texts[0]
    if g:
        return g[0]
    key = next((c for c in t.columns if c.role == ID), None)
    return [table, key.name] if key is not None else None


def _tidy(spec: dict) -> dict:
    return {k: v for k, v in spec.items() if v is not None and v != []}


def _filter(items: list[tuple[str, Meaning]], i: int) -> tuple[dict | None, int]:
    """col op num [and num] -> a filter rule, and how many items it used."""
    col = items[i][1]
    op = items[i + 1][1].value
    j = i + 2
    if j < len(items) and items[j][1].kind == "op" and op == "eq" and items[j][1].value == "ne":
        op, j = "ne", j + 1                              # "is not"
    if j >= len(items):
        return None, 1
    val = items[j][1]
    if op == "between":
        if val.kind == "num" and j + 2 < len(items) and items[j + 1][1].kind == "and" and items[j + 2][1].kind == "num":
            return {"column": col.refs[0], "op": "between", "value": val.value, "value2": items[j + 2][1].value}, j + 3 - i
        return None, 1
    if val.kind == "num":
        return {"column": col.refs[0], "op": op, "value": val.value}, j + 1 - i
    if val.kind == "value" and op in ("eq", "ne"):
        return {"column": val.refs[0], "op": op, "value": val.value}, j + 1 - i
    return None, 1


def _role(model: DataModel, ref: list) -> str:
    if ref[0].startswith("stack:"):
        return CATEGORY
    t = model.table(ref[0])
    c = t.column(ref[1]) if t else None
    return c.role if c is not None else ""


def _first_measure(model: DataModel, base: str) -> str:
    t = model.tables[base]
    return t.measures[0].name if t.measures else "a column"


def _base_table(model: DataModel, tables: list[str], cols, by, filters, out: Asked) -> str:
    """The table whose rows the question is about: one from which every table it mentions can be reached by
    links (orders reach customers and products, not the other way), preferring a table it names, then one
    holding the numbers asked about, then the biggest, then project order."""
    from .recipes import _reachable
    order = list(model.tables)
    mentioned: list[str] = []
    measure_tables: list[str] = []
    for _, refs in cols:
        own = [r[0] for r in refs]
        mentioned += own
        measure_tables += [r[0] for r in refs if _role(model, r) == MEASURE]
    if by:
        mentioned += [r[0] for r in by[1]]
    mentioned += [f["column"][0] for f in filters]
    mentioned = [m for m in mentioned if not m.startswith("stack:")]

    def covers(t: str) -> bool:
        reach = set(_reachable(model, t)) | {t} | set(_stack_members(model, t))
        # a column phrase can mean columns of several tables: one of them reachable is enough
        groups = [[r[0] for r in refs] for _, refs in cols] + ([[r[0] for r in by[1]]] if by else [])
        groups += [[f["column"][0]] for f in filters]
        return all(any(x in reach or x.startswith("stack:") for x in g) for g in groups)

    candidates = [t for t in order if covers(t)]
    if not candidates:
        if tables:
            return tables[0]
        involved = set(mentioned)
        mm = next((r for r in model.relations if r.kind == "link" and r.cardinality == "many-to-many" and set(r.tables) <= involved), None)
        if mm is not None:
            a, b = (model.tables[x].title for x in mm.tables)
            raise PlanError(f"{a} and {b} share {mm.left_on}, but it repeats in both, so putting them side by side would "
                            f"repeat rows and every total would be wrong. Ask about one of them, or remove the duplicates first.")
        raise PlanError("Those columns are in tables that are not linked to each other")
    named = [t for t in tables if t in candidates]
    if named:
        return named[0]
    def rank(t: str):
        tb = model.tables[t]
        return (0 if t in measure_tables else 1, 0 if t in mentioned else 1,
                -(tb.rows or tb.sampled or 0), order.index(t))
    return sorted(candidates, key=rank)[0]


def _stack_members(model: DataModel, t: str) -> list[str]:
    st = model.stack_of(t)
    return list(st.tables) if st is not None else []


def _resolve(model: DataModel, words: str, refs: list[list], base: str, out: Asked) -> list | None:
    """The column a phrase means: the one in the table the question is about, else one reachable from it
    by links; the same column in stacked tables counts as one; otherwise the first, noted as a choice."""
    if len(refs) == 1:
        return refs[0]
    stack = model.stack_of(base)
    members = set(stack.tables) if stack is not None else {base}
    own = [r for r in refs if r[0] == base]
    if own:
        return own[0]
    if all(r[0] in members for r in refs):
        return next((r for r in refs if r[0] == base), refs[0])
    from .recipes import _reachable
    reach = _reachable(model, base)
    reachable = [r for r in refs if r[0] in reach]
    chosen = (reachable or refs)[0]
    others = [r for r in refs if r != chosen]
    if others:
        out.ambiguous.append({"text": words, "chose": chosen,
                              "choices": [{"label": f"{r[1]} ({model.tables[r[0]].title})", "value": r} for r in others]})
    return chosen
