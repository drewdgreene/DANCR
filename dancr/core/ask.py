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
from datetime import datetime, timedelta
import re
from dataclasses import dataclass, field
from typing import Any

from .recipes import PlanError, plan, groupables, RECIPES
from .understand import DataModel, MEASURE, CATEGORY, TIME_ROLE, ID, TEXT, FLAG, BLANK, CONSTANT, STR, norm, _words

STATS = {"total": "sum", "totals": "sum", "sum": "sum", "sums": "sum", "add up": "sum", "added up": "sum",
         "average": "mean", "averages": "mean", "avg": "mean", "mean": "mean", "typical": "mean",
         "count": "count", "number of": "count", "how many": "count", "count of": "count",
         "max": "max", "maximum": "max", "highest": "max", "peak": "max", "largest": "max", "biggest": "max",
         "min": "min", "minimum": "min", "lowest": "min", "smallest": "min", "median": "median",
         "busiest": "count", "quietest": "count"}
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
       "except": "ne", "excluding": "ne", "without": "ne", "after": "gt", "before": "lt", "since": "ge", "until": "le",
       "from": "ge", "later than": "gt", "earlier than": "lt"}
STOP = {"show", "me", "the", "a", "an", "of", "what", "whats", "what's", "are", "was", "were", "how", "does",
        "do", "did", "i", "want", "see", "give", "find", "please", "chart", "plot", "graph", "list", "my",
        "for", "in", "where", "when", "with", "to", "on", "from", "data", "and", "all", "it", "its", "that",
        "which", "who", "there", "any", "can", "you", "get", "tell", "about", "has", "have", "had", "only",
        "rows", "row", "values", "readings", "reading", "records", "time", "at", "?", "this", "these", "then",
        "table", "tables", "file", "files", "their", "them", "look", "into", "during", "within", "thing", "things",
        "most", "least", "gets", "got", "goes", "went", "been", "be", "being", "biggest", "largest",
        "our", "we", "us", "each", "every", "some", "much", "many", "there's", "theres", "was", "per",
        "respondents", "people", "users", "entries", "responses", "cases", "answers", "answered", "said", "say"}
MONTH_WORDS = {name: i for i, name in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                                 "september", "october", "november", "december"], start=1)}
MONTH_WORDS.update({k[:3]: v for k, v in list(MONTH_WORDS.items())})
MONTH_WORDS["sept"] = 9
BIG_WORDS = {"biggest": False, "largest": False, "highest": False, "smallest": True, "lowest": True}
# superlatives that also say which number: "hottest day" is the largest temperature
ADJECTIVES = {"hottest": (False, {"temperature", "temp", "tmax", "tavg", "heat"}),
              "warmest": (False, {"temperature", "temp", "tmax", "tavg"}),
              "coldest": (True, {"temperature", "temp", "tmin", "tavg"}),
              "wettest": (False, {"rain", "rainfall", "precipitation", "precip"}),
              "windiest": (False, {"wind", "gust", "windspeed"}),
              "fastest": (False, {"speed", "velocity"}), "slowest": (True, {"speed", "velocity"}),
              "most expensive": (False, {"price", "cost", "amount"}), "cheapest": (True, {"price", "cost", "amount"})}
BIG_WORDS.update({w: big for w, (big, _) in ADJECTIVES.items()})
TOP_WORDS = {"top": False, "best": False, "bottom": True, "worst": True}


@dataclass
class Meaning:
    kind: str                  # stat every recipe col table value by op num top unit stop and
    value: Any = None
    refs: list[list] = field(default_factory=list)      # col: every column the words could mean
    alias: bool = False                                 # col: only by a shortened name ("patient" for patient_id)


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
_DATE = r"\d{4}-\d{1,2}-\d{1,2}(?:[ t]\d{1,2}:\d{2}(?::\d{2})?)?|\d{1,2}/\d{1,2}/\d{2,4}"


def _tokens(text: str) -> list[str]:
    return re.findall(_DATE + r"|>=|<=|!=|[<>=]|-?\d+(?:[.,]\d+)*|[\w'%/]+(?:[.\-][\w]+)*", text.lower())


def vocabulary(model: DataModel) -> dict[tuple[str, ...], list[Meaning]]:
    """Every phrase this project understands, to its meanings (fixed words first, then the project's own)."""
    voc: dict[tuple[str, ...], list[Meaning]] = {}

    def add(phrase: str, m: Meaning) -> None:
        key = tuple(_tokens(phrase))
        if key:
            voc.setdefault(key, []).append(m)

    for w, s in STATS.items():
        add(w, Meaning("stat", s))
    for w, (low, _) in ADJECTIVES.items():
        add(w, Meaning("stat", "min" if low else "max"))
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
    for w, n in MONTH_WORDS.items():
        add(w, Meaning("month", n))
    for w, part in PARTS.items():
        add(w, Meaning("part", part))
    for w in SHARE_WORDS:
        add(w, Meaning("share"))
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
            if c.role == BLANK:
                continue
            ref = [node, c.name]
            words = _words(c.name)
            st = model.stack_of(node)
            if c.role == CONSTANT and st is not None and len(words) > 1 and words[-1] in ("id", "key", "code", "no", "number", "ref"):
                for form in _plural_forms(" ".join(words[:-1])):     # one vehicle per file: "by vehicle" is by file
                    _add_ref(voc, tuple(_tokens(form)), [st.id, "source"], alias=True)
            if c.role == ID and len(words) > 1 and words[-1] in ("id", "key", "code", "no", "number", "ref"):
                stem = " ".join(words[:-1])                     # "by patient" is by patient_id
                for form in _plural_forms(stem):
                    _add_ref(voc, tuple(_tokens(form)), ref, alias=True)
            for phrase, short in _column_phrases(c.name, c.label):
                for form in _plural_forms(phrase):              # "customers" for a column called customer
                    _add_ref(voc, tuple(_tokens(form)), ref, alias=short)
                for n in names:                                  # "product name", "customers region"
                    _add_ref(voc, tuple(_tokens(f"{n} {phrase}")), ref)
            if c.role in (MEASURE, TIME_ROLE):                  # "revenue" for Amount, "pays" for salary, "started" for start_date
                for syn in sorted(_synonyms(words + _words(c.label or ""))):
                    _add_ref(voc, tuple(_tokens(syn)), ref, alias=True)
            if c.kind == STR and c.values:
                for v in c.values:
                    # a value spelled as a number is read as that number ("per 15 minutes"), never as a value
                    if v is not None and str(v).strip() and _number(str(v).strip()) is None:
                        _add_value(voc, tuple(_tokens(str(v))), str(v), ref)
    for r in model.relations:
        if r.kind == "stack":
            for node, lab in zip(r.tables, r.labels):
                add(lab, Meaning("table", node))
            # "per device", "by file": which of the stacked tables a row came from
            common = set.intersection(*[set(_words(model.tables[n].title)) for n in r.tables]) if r.tables else set()
            for w in sorted(common | {"table", "file", "source", "log"}):
                for form in _plural_forms(w):
                    _add_ref(voc, tuple(_tokens(form)), [r.id, "source"], alias=True)
    return voc


SYNONYMS = [
    {"amount", "sales", "sale", "revenue", "revenues", "turnover", "income", "takings", "value", "total"},
    {"cost", "costs", "spend", "spending", "spent", "expense", "expenses"},
    {"qty", "quantity", "quantities", "units", "count"},
    {"salary", "salaries", "pay", "wage", "wages", "earnings", "compensation"},
    {"price", "prices", "rate"},
    {"visits", "views", "sessions", "hits", "traffic"},
    {"start", "started", "hired", "joined", "hire"},
    {"stock", "inventory", "on hand"},
]


def _synonyms(words: list[str]) -> set[str]:
    out: set[str] = set()
    for w in words:
        for group in SYNONYMS:
            if w in group:
                out |= group - {w}
    return out


def _stem_word(tok: str) -> list[str]:
    """Everyday endings taken off a word that is not known as it is: pays -> pay, started -> start, earns -> earn."""
    out = []
    for end, back in (("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", ""), ("d", "")):
        if tok.endswith(end) and len(tok) > len(end) + 2:
            out.append(tok[: len(tok) - len(end)] + back)
    return out


def _plural_forms(phrase: str) -> list[str]:
    p = phrase.strip()
    out = [p]
    if len(p) > 3 and p.endswith("s") and not p.endswith("ss"):
        out.append(p[:-1])
    elif len(p) > 2:
        out.append(p + "s")
    return out


def _add_value(voc, key, value: str, ref) -> None:
    if not key:
        return
    for m in voc.setdefault(key, []):
        if m.kind == "value" and m.value == value:
            if ref not in m.refs:
                m.refs.append(ref)
            return
    voc[key].append(Meaning("value", value, [ref]))


def _add_ref(voc, key, ref, alias: bool = False) -> None:
    if not key:
        return
    for m in voc.setdefault(key, []):
        if m.kind == "col":
            if ref not in m.refs:
                m.refs.append(ref)
            m.alias = m.alias and alias                   # a real name wins over a shortened one
            return
    voc[key].append(Meaning("col", refs=[ref], alias=alias))


def _column_phrases(name: str, label: str) -> list[tuple[str, bool]]:
    """(phrase, shortened?): pressure_psia -> 'pressure_psia', 'pressure psia', and 'pressure' (shortened);
    'Pressure (bar)' -> 'pressure (bar)', 'pressure'."""
    full = {name, name.replace("_", " "), label, re.sub(r"\s*[\(\[].*?[\)\]]\s*$", "", label)}
    full |= {re.sub(r"^(avg|average|mean|total|sum of|number of|no of|count of)[\s_.:]+", "", f, flags=re.IGNORECASE) for f in list(full)}
    words = _words(name)
    if words:
        full.add(" ".join(words))
    out = [(p, False) for p in sorted(x for x in full if x and x.strip())]
    if len(words) > 1 and words[-1] not in ("id", "key", "code", "no"):
        out.append((words[0], True))                  # pressure_psia is 'pressure', customer_name is 'customer'
    return out


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

    def match(j: int):
        for n in range(min(maxlen, len(toks) - j), 0, -1):
            key = tuple(toks[j:j + n])
            if key in voc:
                return key, voc[key]
        return None

    def kinds_at(j: int) -> set[str]:
        if j >= len(toks):
            return set()
        h = match(j)
        if h is not None:
            return {m.kind for m in h[1]}
        return {"num"} if _number(toks[j]) is not None else ({"stop"} if toks[j] in STOP else {"?"})

    i = 0
    while i < len(toks):
        hit = match(i)
        if hit is not None:
            key, ms = hit
            prev = next((m.kind for _, m in reversed(items) if m.kind != "stop"), None)
            prev_word = items[-1][0] if items else None
            chosen = _pick(ms, " ".join(key), prev, kinds_at(i + len(key)), prev_word)
            items.append((" ".join(key), chosen))
            i += len(key); continue
        tok = toks[i]
        num = _number(tok)
        if re.fullmatch(_DATE, tok):
            items.append((tok, Meaning("date", _iso_date(tok))))
        elif num is not None:
            items.append((tok, Meaning("num", num)))
        elif tok in STOP:
            items.append((tok, Meaning("stop")))
        else:
            stemmed = next((voc[(w,)] for w in _stem_word(tok) if (w,) in voc), None)
            if stemmed is not None:
                prev = next((m.kind for _, m in reversed(items) if m.kind != "stop"), None)
                items.append((tok, _pick(stemmed, tok, prev, kinds_at(i + 1), items[-1][0] if items else None)))
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


FIXED = {"stat", "every", "recipe", "by", "op", "unit", "top", "and", "month", "part", "share"}
PARTS = {"hour of day": "hour", "hour of the day": "hour", "time of day": "hour", "time of the day": "hour",
         "day of week": "weekday", "day of the week": "weekday", "weekday": "weekday", "weekdays": "weekday",
         "month of year": "month", "month of the year": "month", "day of month": "day", "day of the month": "day"}
SHARE_WORDS = ["share", "share of", "percentage", "percentage of", "percent", "percent of", "proportion", "proportion of",
               "what share", "% of", "fraction"]


def _iso_date(tok: str) -> str:
    """'2024-3-1' -> '2024-03-01'; '03/01/2024' stays as typed (the filter reads it month first, like every typed date)."""
    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})(.*)", tok)
    return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}{m.group(4)}" if m else tok


def _pick(ms: list[Meaning], words: str, prev: str | None, nxt: set[str], prev_word: str | None = None) -> Meaning:
    """What a phrase means here, when it could mean several things. Deterministic, from its neighbours:

    - right after a comparison ("region is Total", "status is on") it is the project's value or column;
    - a fixed word followed by a column, value, table or number is the fixed word ("total amount", "top 10");
    - otherwise the project's own word wins (a column called "total" when nothing follows);
    - a value that is also an everyday word ("on", "for") is that word unless a comparison came just before;
    - after "by"/"per", a table (customers) beats the id column named after it, and a column beats a value.
    """
    fixed = [m for m in ms if m.kind in FIXED]
    proj = [m for m in ms if m.kind not in FIXED]
    if prev == "op":
        if proj:
            return sorted(proj, key=lambda m: {"value": 0, "col": 1, "table": 2}.get(m.kind, 3))[0]
    elif fixed and proj and fixed[0].kind == "unit" and prev in ("by", None, "stat") :
        return fixed[0]                                   # "per day" is a time step, even with a column called day
    elif fixed and proj:
        if nxt & {"col", "value", "table", "num"} or fixed[0].kind in ("by", "op", "and"):
            return fixed[0]
    elif fixed:
        return fixed[0]
    if not proj:
        return fixed[0]
    if all(m.kind == "value" for m in proj) and words in STOP:
        if prev_word in ("for", "in", "at", "from", "where", "only", "of"):
            return proj[0]                                # "for IT", "in On": the value after a word that points at one
        return Meaning("stop")
    tables = [m for m in proj if m.kind == "table"]
    real = [m for m in proj if m.kind == "col" and not m.alias]
    if prev == "by" and real:
        return real[0]                                    # "by product": the column called product, if there is one
    if tables and all(m.alias for m in proj if m.kind == "col"):
        return tables[0]                                  # "customers" is the table, not customer_id shortened
    if prev in ("by", "top"):
        order = {"table": 0, "col": 1, "value": 2}
    else:
        order = {"col": 0, "value": 1, "table": 2}
    return sorted(proj, key=lambda m: order.get(m.kind, 3))[0]


def _number(tok: str) -> float | int | None:
    from .dtypes import typed_value
    if not re.fullmatch(r"-?\d+(?:[.,]\d+)*", tok):
        return None
    try:
        return typed_value(tok)
    except Exception:  # noqa: BLE001
        return None


def _assemble(model: DataModel, items: list[tuple[str, Meaning]], out: Asked) -> dict[str, Any]:
    """Words -> spec. Every word that carries meaning must end up in the spec: one that could not be used makes the
    question fail with that word named, rather than build something that quietly ignores it."""
    used: set[int] = set()
    stat = None
    recipes: list[str] = []
    tables: list[str] = []
    every = None
    top, bottom, big = None, False, None
    by_table, by_table_words = None, ""
    filters: list[dict[str, Any]] = []
    time_filters: list[dict[str, Any]] = []          # dates, months and years: resolved once the table is known
    by = None
    cols: list[tuple[int, str, list[list]]] = []     # (item index, words, candidate refs) for measures
    rows_named: list[int] = []                       # words that name the rows (orders, items) rather than a column
    rows_tables: list[str] = []
    noun_unit = None
    part, share = None, False
    i = 0
    while i < len(items):
        words, m = items[i]
        nxt = items[i + 1][1] if i + 1 < len(items) else None
        if m.kind in ("stop", "and"):
            used.add(i); i += 1; continue
        if m.kind == "stat":
            stat = stat or m.value; used.add(i)
            if words in BIG_WORDS and big is None:
                big = BIG_WORDS[words]
            i += 1; continue
        if m.kind == "recipe":
            recipes.append(m.value); used.add(i); i += 1; continue
        if m.kind == "part":
            part = m.value; used.add(i); i += 1; continue
        if m.kind == "share":
            share = True; used.add(i); i += 1; continue
        if m.kind == "table":
            tables.append(m.value); used.add(i); i += 1; continue
        if m.kind == "every":
            every = m.value; used.add(i); i += 1; continue
        if m.kind == "top":
            n = nxt.value if nxt is not None and nxt.kind == "num" else 10
            top, bottom = int(n), bool(m.value)
            used |= {i, i + 1} if nxt is not None and nxt.kind == "num" else {i}
            i += 2 if nxt is not None and nxt.kind == "num" else 1
            continue
        if m.kind == "by" and nxt is not None and nxt.kind == "unit":
            every = _one_step(every, f"1{nxt.value}"); used |= {i, i + 1}; i += 2; continue
        if m.kind == "by" and nxt is not None and nxt.kind == "num" and i + 2 < len(items) and items[i + 2][1].kind == "unit":
            every = f"{int(nxt.value)}{items[i + 2][1].value}"; used |= {i, i + 1, i + 2}; i += 3; continue
        if m.kind == "by" and nxt is not None and nxt.kind == "col":
            by = (items[i + 1][0], nxt.refs); used |= {i, i + 1}; i += 2; continue
        if m.kind == "by" and nxt is not None and nxt.kind == "table":
            by_table, by_table_words = nxt.value, items[i + 1][0]; used |= {i, i + 1}; i += 2; continue
        if m.kind == "by" and nxt is not None and nxt.kind == "month":
            every = "1mo"; used |= {i, i + 1}; i += 2; continue     # "by month"
        if m.kind == "by":
            used.add(i); i += 1; continue
        if m.kind == "unit" and i > 0 and items[i - 1][0] == "busiest" and m.value in ("h", "d"):
            part = "hour" if m.value == "h" else "weekday"      # "busiest hour": rows by hour of the day
            used.add(i); i += 1; continue
        if m.kind == "unit":
            if big is not None and not every and i + 1 >= len(items):
                used.add(i); noun_unit = words; i += 1; continue    # "hottest day": the day itself, not a step
            every = _one_step(every, f"1{m.value}"); used.add(i); i += 1; continue
        if m.kind in ("month", "date") or (m.kind == "num" and _is_year(m.value) and _time_context(items, i)):
            f, n_used = _time_filter(items, i)
            time_filters.append(f); used |= set(range(i, i + n_used)); i += n_used; continue
        if m.kind == "op" and nxt is not None and nxt.kind in ("date", "month") or (
                m.kind == "op" and nxt is not None and nxt.kind == "num" and _is_year(nxt.value) and m.value in ("gt", "lt", "ge", "le")):
            f, n_used = _time_filter(items, i + 1, op=m.value)
            time_filters.append(f); used |= set(range(i, i + 1 + n_used)); i += 1 + n_used; continue
        if m.kind == "col":
            if nxt is not None and nxt.kind == "op":
                j = i + 1
                if j + 1 < len(items) and items[j + 1][1].kind in ("date", "month"):
                    cols.append((i, words, m.refs)); used.add(i)       # "sales after 2024-01-01": the date is for the time column
                    i += 1; continue
                f, n_used = _filter(items, i)
                if f is not None:
                    filters.append(f); used |= set(range(i, i + n_used)); i += n_used; continue
            if m.alias and any(_role(model, r) == ID for r in m.refs) and not (nxt is not None and nxt.kind == "op"):
                rows_named.append(i); rows_tables.extend(r[0] for r in m.refs if _role(model, r) == ID)
                i += 1; continue                             # "orders", "items", "employees": the rows themselves
            if len(m.refs) == 1 and _names_its_table(model, words, m.refs[0][0]):
                tables.append(m.refs[0][0])                  # "actual jan": Jan of the Actual sheet only
            cols.append((i, words, m.refs)); i += 1; continue
        if m.kind == "value":
            negate = i > 0 and items[i - 1][1].kind == "op" and items[i - 1][1].value == "ne"
            if negate:
                used.add(i - 1)
            filters.append({"column": m.refs[0], "op": "ne" if negate else "eq", "value": m.value})
            if len(m.refs) > 1:
                out.ambiguous.append({"text": words, "chose": m.refs[0],
                                      "choices": [{"label": f"{m.value!r} in {r[1]}", "value": r} for r in m.refs[1:]]})
            used.add(i); i += 1; continue
        i += 1
    lookups_named = [t for t in tables if model.tables[t].shape == "lookup"]
    if rows_tables and lookups_named and rows_tables[0] not in tables:
        tables = [rows_tables[0]] + tables           # "which supplier has the most items": items per supplier
    base = _base_table(model, tables, [(w, r) for _, w, r in cols], by, filters, out)
    t0 = model.tables[base]
    resolve = lambda w, refs: _resolve(model, w, refs, base, out)   # noqa: E731
    measures = []
    for idx, w, r in cols:
        ref = resolve(w, r)
        if ref is not None:
            measures.append((idx, ref))
    by_ref = resolve(*by) if by else None
    time_ref = next((r for _, r in measures if _role(model, r) == TIME_ROLE), None)
    measures = [(k, r) for k, r in measures if r != time_ref]
    if time_ref is not None:
        used |= {idx for idx, w, r in cols if resolve(w, r) == time_ref}
    if by is not None and by_ref is not None and _role(model, by_ref) == MEASURE and not _groups_named(model, cols, tables, resolve):
        pass                                       # "subtotal per table" with a numbered table: the number is the group
    elif by_ref is not None and _role(model, by_ref) == MEASURE:
        measures.insert(0, (-1, by_ref)); by_ref = None
    if by_ref is not None and _role(model, by_ref) == TIME_ROLE:
        time_ref, by_ref = by_ref, None
    numbers = [(k, r) for k, r in measures if _role(model, r) == MEASURE or (_role(model, r) == CONSTANT and _is_number(model, r))]
    groups = [(k, r) for k, r in measures if _role(model, r) in (CATEGORY, ID, TEXT, FLAG)]
    adjective = next((w for w, m in items if m.kind == "stat" and w in ADJECTIVES), None)
    if adjective and not numbers:
        hint = ADJECTIVES[adjective][1]
        match = next((c for c in _measures_of(model, base) if set(_words(c.name)) & hint), None)
        if match is None:
            raise PlanError(f"“{adjective}” needs a {sorted(hint)[0]} column, and {t0.title} has none")
        numbers = [(-1, [base, match.name])]
    texts = [(k, r) for k, r in groups if _col_kind(model, r) == "text"]
    if stat in ("sum", "mean", "median", "min", "max") and not numbers and texts and by_ref is not None:
        g = texts[0][1]
        raise PlanError(f"{g[1]} holds text (for example {_example(model, g)}), so it cannot be "
                        f"{'added up' if stat == 'sum' else 'averaged' if stat == 'mean' else 'ranked as a number'}")
    if by_ref is None and groups and (numbers or stat == "count" or top):
        by_ref = groups[0][1]; used.add(groups[0][0])
    if stat in ("sum", "mean", "median") and not numbers and not adjective:
        some = ", ".join(c.name for c in _measures_of(model, base)[:3])
        raise PlanError(f"{'Total' if stat == 'sum' else 'Average'} of which number? For example "
                        + (f"“{'total' if stat == 'sum' else 'average'} {_measures_of(model, base)[0].name} …” ({some})" if some
                           else "name a column of numbers"))
    if by_table is not None and by_ref is None:
        by_ref = _group_for_table(model, base, by_table)
        if by_ref is None and by_table == base:
            by_ref = _own_key(model, base, by_table_words)
        if by_ref is None:
            if by_table == base:
                raise PlanError(f"{t0.title} is the table being asked about; say which of its columns to group by")
            raise PlanError(f"{model.tables[by_table].title} is not linked to {t0.title}, so its rows cannot be counted per it")
    if by_ref is not None and by_ref[0] == base and _unique_id(model, by_ref) and not top:
        by_ref = None                              # "average tip per ticket": one row per ticket, so the plain average
    named_lookups = [t for t in tables if t != base and model.tables[t].shape == "lookup"]
    if by_ref is None and named_lookups and (numbers or stat or top or rows_named) and not (top is not None and base in tables and not named_lookups):
        by_ref = _name_column(model, named_lookups[0], base)
    others = [t for t in tables if t not in named_lookups]
    # dates, months and years apply to the time column named, else the table's own
    tcol = time_ref if time_ref is not None else ([base, t0.time] if t0.time else None)
    for f in time_filters:
        if tcol is None:
            raise PlanError(f"{t0.title} has no date or time column to take “{f['_words']}” from")
        f.pop("_words", None)
        filters.append({**f, "column": tcol})
    for f in filters:
        _check_filter(model, f)
    spec: dict[str, Any] = {"table": base}
    if by is not None and by_ref is not None and by_ref[0].startswith("stack:"):
        spec["by_words"] = by[0]                       # "by vehicle": said the way it was asked
    stack = model.stack_of(base)
    if stack is not None:
        spec["together"] = not any(t in stack.tables for t in others) or len({t for t in others if t in stack.tables}) > 1
    if filters:
        spec["filters"] = filters
    nums = [r for _, r in numbers]
    recipe = recipes[0] if recipes else None
    if noun_unit and big is not None:
        rows_named = rows_named or [-1]
        if adjective or big is not None:
            top = top or 1                         # "hottest day", "biggest month": that one
            spec["superlative"] = adjective or next((w for w, m in items if w in BIG_WORDS), "")
    if rows_named:
        used |= {k for k in rows_named if k >= 0}
        spec["_rows_named"] = True
        spec["noun"] = items[rows_named[0]][0] if rows_named[0] >= 0 else noun_unit
        if top is not None and by_ref is not None and by_ref[0] == base and _role(model, by_ref) == ID:
            by_ref = None                          # "top 5 orders by amount": the orders themselves, not their ids
    if part:
        if by_ref is not None or every:
            raise PlanError(f"Group by one thing at a time: the {part_word(part)}, or a column")
        spec.update({"recipe": "breakdown", "by_part": part, "measure": nums[0] if nums else None, "share": share or None})
        if stat and stat not in ("max", "min") or (stat and nums):
            spec["stat"] = stat
        return _finish(spec, cols, numbers, groups, items, used, resolve)
    if share:
        if by_ref is None and spec.get("filters") and spec["filters"][-1]["op"] == "eq" and not spec["filters"][-1].get("text"):
            by_ref = spec["filters"].pop()["column"]      # "what share agree": the share of every answer to that question
            if not spec["filters"]:
                spec.pop("filters")
        if by_ref is None:
            raise PlanError("A share of what, by what? For example “share of sales by region”")
        spec.update({"recipe": "breakdown", "by": by_ref, "measure": nums[0] if nums else None, "share": True})
        if stat:
            spec["stat"] = stat
        return _finish(spec, cols, numbers, groups, items, used, resolve)
    spec = _choose_recipe(model, spec, recipe, nums, by_ref, stat, every, top, bottom, big, time_ref, others, tables,
                          stack, base, items)
    return _finish(spec, cols, numbers, groups, items, used, resolve)


def part_word(part: str) -> str:
    from .recipes import PART_WORDS
    return PART_WORDS[part]


def _finish(spec, cols, numbers, groups, items, used, resolve) -> dict[str, Any]:
    """Every word must have been used: one that was not is named, rather than silently left out."""
    in_spec = _refs_in(spec)
    for idx, w, r in cols:
        ref = resolve(w, r)
        if ref is not None and tuple(ref) in in_spec:
            used.add(idx)
    for k, r in numbers + groups:
        if tuple(r) in in_spec and k >= 0:
            used.add(k)
    left = [items[k][0] for k in range(len(items)) if k not in used and items[k][1].kind not in ("op",)]
    left += [items[k][0] for k in range(len(items)) if k not in used and items[k][1].kind == "op"]
    if left:
        raise PlanError("I could not fit " + ", ".join(f"“{w}”" for w in dict.fromkeys(left)) + " into the question. "
                        "Try naming one number, a group (“by region”) and a time step (“per month”)")
    return spec


def _choose_recipe(model, spec, recipe, numbers, by_ref, stat, every, top, bottom, big, time_ref, others, tables, stack,
                   base, items) -> dict[str, Any]:
    t0 = model.tables[base]
    if recipe == "compare" and len({tuple(r) for r in numbers}) >= 2 and len(set(others)) < 2:
        recipe = "relationship"
    if recipe == "compare":
        pair = [t for t in others]
        if stack is not None and len(pair) < 2:
            pair = list(stack.tables[:2])
        if len(pair) >= 2:
            a, b = pair[0], pair[1]
            rel = next((r for r in model.relations if r.kind == "align" and set(r.tables) == {a, b}), None)
            st = model.stack_of(a)
            if rel is None and st is not None and b in st.tables:
                # the same table twice (Budget and Actual, this year and last): side by side, table against table
                measure = numbers[0] if numbers else _first_measure_ref(model, a)
                if measure is None:
                    raise PlanError("Name the number to compare")
                spec.update({"table": a, "together": True, "recipe": "breakdown", "by": [st.id, "source"],
                             "measure": measure, "stat": stat or default_stat_for(model, a, measure)})
                return _tidy(spec)
            if rel is None:
                raise PlanError(f"{model.tables[a].title} and {model.tables[b].title} do not record the same thing over time")
            if rel.tables[0] != a:
                a, b = rel.tables
            col = next((r[1] for r in numbers if r[1] in rel.shared), rel.shared[0])
            spec.update({"recipe": "compare", "table": a, "other": b, "measure": [a, col]})
            spec.pop("together", None)
            return spec
        raise PlanError("Compare what? Name two tables (“compare A and B”) or two numbers (“y against x”)")
    if recipe == "relationship" or (recipe is None and len(numbers) == 2 and any(w in ("against", "vs", "versus") for w, _ in items)):
        if len(numbers) < 2:
            raise PlanError("Name the two numbers, for example “temperature against pressure”")
        spec.update({"recipe": "relationship", "x": numbers[1], "y": numbers[0]})
        return _tidy(spec)
    if recipe in ("gaps", "describe", "stacked", "linked"):
        spec["recipe"] = recipe
        return _tidy(spec)
    if recipe in ("outliers", "distribution"):
        if not numbers:
            raise PlanError(f"Name the number to look at, for example “{recipe} in {_first_measure(model, base)}”")
        spec.update({"recipe": recipe, "measure": numbers[0]})
        return _tidy(spec)
    rows_named = spec.pop("_rows_named", False)
    ranking_rows = (top is not None and by_ref is None) or (top is not None and (base in tables or rows_named)) or \
                   (big is not None and by_ref is None and (base in tables or rows_named) and not every)
    if ranking_rows:                                  # "top 5 orders by amount", "the biggest orders"
        measure = numbers[0] if numbers else _first_measure_ref(model, base)
        if measure is None:
            raise PlanError(f"{t0.title} has no number to rank its rows by")
        spec.update({"recipe": "toprows", "n": top or 10, "measure": measure, "bottom": (bottom if top is not None else big) or None})
        if not rows_named:
            spec.pop("noun", None)
        return _tidy(spec)
    if top is not None:
        spec.update({"recipe": "top", "n": top, "by": by_ref, "bottom": bottom or None, "measure": numbers[0] if numbers else None,
                     "stat": stat or ("count" if not numbers else "sum")})
        return _tidy(spec)
    trendy = every or recipe == "trend" or time_ref is not None
    if by_ref is not None and trendy and (t0.time or time_ref):
        spec.update({"recipe": "trend", "measures": numbers[:3], "by": by_ref})
        _time_bits(spec, every, stat, time_ref)
        return _tidy(spec)
    if by_ref is not None:
        spec.update({"recipe": "breakdown", "by": by_ref, "measure": numbers[0] if numbers else None})
        if stat:
            spec["stat"] = stat
        return _tidy(spec)
    if trendy or (numbers and t0.time and not stat):
        if not t0.time and time_ref is None:
            raise PlanError(f"{t0.title} has no date or time column, so it cannot be shown over time")
        spec.update({"recipe": "trend", "measures": numbers[:3]})
        _time_bits(spec, every, stat, time_ref)
        return _tidy(spec)
    if numbers or stat:                               # "how many employees", "average salary"
        spec.update({"recipe": "single", "measure": numbers[0] if numbers else None, "stat": stat or "sum"})
        return _tidy(spec)
    if spec.get("filters"):
        spec["recipe"] = "rows"
        return _tidy(spec)
    if tables:
        spec["recipe"] = "describe"
        return _tidy(spec)
    raise PlanError("Say what you would like to know: a number to add up or average, a group, or a time step")


def _time_bits(spec, every, stat, time_ref) -> None:
    if every:
        spec["every"] = every
    if stat:
        spec["stat"] = stat
    if time_ref is not None:
        spec["time"] = time_ref


def _refs_in(spec: dict) -> set[tuple]:
    out = set()
    for k in ("measure", "by", "x", "y", "time"):
        if spec.get(k):
            out.add(tuple(spec[k]))
    for r in spec.get("measures") or []:
        out.add(tuple(r))
    for f in spec.get("filters") or []:
        out.add(tuple(f["column"]))
    return out


def _one_step(current: str | None, new: str) -> str:
    if current and current != new:
        from .recipes import EVERY_WORDS
        a, b = EVERY_WORDS.get(current, current), EVERY_WORDS.get(new, new)
        raise PlanError(f"The question names two time steps, “{a}” and “{b}”. Ask per {a} or per {b}; "
                        f"“each {a} of the {b}” (across all {b}s together) cannot be built yet")
    return new


def _names_its_table(model: DataModel, words: str, table: str) -> bool:
    t = model.tables.get(table)
    if t is None or model.stack_of(table) is None:
        return False
    names = {t.title.lower(), t.title.lower().replace("_", " "), norm(t.title)}
    return any(words.startswith(n + " ") for n in names)


def _measures_of(model: DataModel, table: str):
    from .recipes import _ordered_measures
    return _ordered_measures(model.tables[table])


def _unique_id(model: DataModel, ref: list) -> bool:
    """An id with one row each and too many to read as groups (ticket numbers), unlike a short list of names."""
    t = model.table(ref[0]) if not ref[0].startswith("stack:") else None
    c = t.column(ref[1]) if t else None
    return c is not None and c.role == ID and c.unique and c.distinct > 50


def _groups_named(model, cols, tables, resolve) -> bool:
    """Whether the question names a group elsewhere (so a number after 'by' is what is ranked, not a group)."""
    if any(model.tables[t].shape == "lookup" for t in tables):
        return True
    return any(_role(model, resolve(w, r)) in (CATEGORY, ID, TEXT) for _, w, r in cols if resolve(w, r) is not None)


def _col_kind(model: DataModel, ref: list) -> str:
    t = model.table(ref[0]) if not ref[0].startswith("stack:") else None
    c = t.column(ref[1]) if t else None
    return c.kind if c is not None else ""


def _is_year(v: Any) -> bool:
    return isinstance(v, int) and 1900 <= v <= 2100


def _time_context(items, i: int) -> bool:
    """A year is a year when it follows 'in', 'for', 'during' or a month, not when it is a quantity ('above 2000')."""
    prev = items[i - 1] if i > 0 else None
    if prev is None:
        return False
    return (prev[1].kind == "stop" and prev[0] in ("in", "for", "during", "of")) or prev[1].kind == "month"


def _time_filter(items, i: int, op: str | None = None) -> tuple[dict[str, Any], int]:
    """A date, a month, a year, or 'March 2024', as a filter on the time column (the column is set later)."""
    words, m = items[i]
    if m.kind == "month":
        nxt = items[i + 1][1] if i + 1 < len(items) else None
        if nxt is not None and nxt.kind == "num" and _is_year(nxt.value):
            y, mo = int(nxt.value), int(m.value)
            last = (datetime(y + (mo == 12), mo % 12 + 1, 1) - timedelta(days=1)).day
            return {"op": "between", "value": f"{y}-{mo:02d}-01", "value2": f"{y}-{mo:02d}-{last:02d} 23:59:59.999999",
                    "_words": f"{words} {items[i + 1][0]}", "text": f"in {words.title()} {items[i + 1][0]}"}, 2
        return {"op": "month", "value": int(m.value), "_words": words, "text": f"in {words.title()}"}, 1
    if m.kind == "num":
        y = int(m.value)
        if op in ("gt", "lt", "ge", "le"):
            edge = f"{y}-12-31 23:59:59.999999" if op in ("gt", "le") else f"{y}-01-01"
            return {"op": op, "value": edge, "_words": words, "text": f"{_OP_WORD[op]} {y}"}, 1
        return {"op": "year", "value": y, "_words": words, "text": f"in {y}"}, 1
    d = str(m.value)
    if op == "between":
        if i + 2 < len(items) and items[i + 1][1].kind == "and" and items[i + 2][1].kind == "date":
            d2 = str(items[i + 2][1].value)
            if len(d2) <= 10:
                d2 += " 23:59:59.999999"
            return {"op": "between", "value": d, "value2": d2, "_words": f"{words} and {items[i + 2][0]}",
                    "text": f"from {words} to {items[i + 2][0]}"}, 3
        return {"op": "ge", "value": d, "_words": words, "text": f"from {words}"}, 1
    if op:
        if len(d) <= 10 and op in ("gt", "le"):
            d = d + " 23:59:59.999999"                 # "after 2024-03-01" means after that day
        return {"op": op, "value": d, "_words": words, "text": f"{_OP_WORD[op]} {words}"}, 1
    if len(d) <= 10:                                   # "on 2024-03-01": that whole day
        return {"op": "between", "value": d, "value2": d + " 23:59:59.999999", "_words": words, "text": f"on {words}"}, 1
    return {"op": "eq", "value": d, "_words": words, "text": f"at {words}"}, 1


_OP_WORD = {"gt": "after", "lt": "before", "ge": "from", "le": "up to", "eq": "on", "ne": "not on"}


def _check_filter(model: DataModel, f: dict) -> None:
    ref = f["column"]
    if ref[0].startswith("stack:"):
        return
    t = model.table(ref[0])
    c = t.column(ref[1]) if t else None
    if c is None:
        return
    if f["op"] in ("gt", "lt", "ge", "le", "between") and c.kind == "text":
        raise PlanError(f"{c.name} holds text (for example {_example(model, ref)}), so it cannot be compared as a number; "
                        f"check the file's {c.name} column (a few notes among the numbers are read as blanks)")


def _example(model: DataModel, ref: list) -> str:
    t = model.table(ref[0])
    c = t.column(ref[1]) if t else None
    vals = [v for v in (c.values if c is not None and c.values else []) if v is not None]
    return f"“{vals[0]}”" if vals else "words"


def default_stat_for(model: DataModel, base: str, measure: list) -> str:
    from .recipes import default_stat
    return default_stat(model, base, measure)


def _first_measure_ref(model: DataModel, base: str) -> list | None:
    from .recipes import _ordered_measures
    ms = _ordered_measures(model.tables[base])
    return [base, ms[0].name] if ms else None


def _name_column(model: DataModel, table: str, base: str) -> list | None:
    """The column that names a lookup table's rows (a customer's name), else its key."""
    g = [r for r in groupables(model, base) if r[0] == table]
    t = model.tables[table]
    texts = [r for r in g if _role(model, r) in (ID, TEXT)]
    named = [r for r in texts if set(_words(r[1])) & {"name", "title", "label", "description", "product", "customer"}]
    if named or texts:
        return (named or texts)[0]
    if g:
        return g[0]
    key = next((c for c in t.columns if c.role == ID), None)
    return [table, key.name] if key is not None else None


def _is_number(model: DataModel, ref: list) -> bool:
    t = model.table(ref[0])
    c = t.column(ref[1]) if t else None
    return c is not None and c.kind == "number"


def _group_for_table(model: DataModel, base: str, table: str) -> list | None:
    """'per customer' when customers is a table: its name column if it is a lookup the base links to, else the
    base's own key column that links to it."""
    if table == base:
        return None
    t = model.tables[table]
    if t.shape == "lookup":
        g = _name_column(model, table, base)
        if g is not None and (g[0] == base or g[0] in _reach(model, base)):
            return g
    link = next((r for r in model.relations if r.kind == "link" and r.tables == [base, table]), None)
    return [base, link.left_on] if link is not None else None


def _own_key(model: DataModel, table: str, words: str) -> list | None:
    """The id column of ``table`` whose name starts with the words (sensor -> sensor_id), if there is one."""
    want = norm(words)
    want_s = want[:-1] if want.endswith("s") else want
    for c in model.tables[table].columns:
        w = _words(c.name)
        if c.role in (ID, CATEGORY) and len(w) > 1 and norm("".join(w[:-1])) in (want, want_s):
            return [table, c.name]
    return None


def _reach(model: DataModel, base: str) -> list[str]:
    from .recipes import _reachable
    return _reachable(model, base)


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
        # the column named wins ("where billing country is France"), when that value can be in it
        return {"column": col.refs[0], "op": op, "value": val.value}, j + 1 - i
    if val.kind == "col" and op in ("gt", "lt", "ge", "le", "eq", "ne"):
        return {"column": col.refs[0], "op": op, "column2": val.refs[0]}, j + 1 - i   # stock below reorder level
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
    from .recipes import _reachable as _r
    for t in tables:
        if t not in candidates and not any(t in _r(model, c) for c in candidates):
            raise PlanError(f"{model.tables[t].title} is not linked to the table that holds "
                            f"{', '.join(dict.fromkeys(r[1] for _, rs in cols for r in rs)) or 'those columns'}, "
                            "so it cannot be counted that way")
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
