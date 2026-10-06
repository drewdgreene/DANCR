"""Read a short question typed in plain words and turn it into a spec, deterministically.

Not AI: a fixed grammar over the words of *this* project — its column names and labels, table names,
the values of its categories, statistics ("total", "average"), time words ("per hour", "monthly") and
comparisons ("above 5", "between 1 and 2"). Everything understood becomes a chip the person can change,
and a word that is not understood is reported rather than guessed, so the same words on the same data
always build the same answer::

    average pressure per hour for MJ03F     total qty by region        top 10 customers by qty
    compare MJ03E and MJ03F                 gaps in probe_MJ03E        orders where qty above 2
    orders in North or South                sales since March          average price last month

A question is read in three stages: its words are matched to phrases (``_read``), the phrases are joined into
clauses — a comparison with its value, several values of one column, a date or a range of dates (``_clauses``) —
and the clauses are put together into a spec (``_assemble``). Dates like "last month" or "since March" are taken
relative to the latest date in the data, not today's date, so the same question on the same data always keeps
the same rows; the filter says the dates it chose.

Pure core, no Qt.
"""
from __future__ import annotations

import difflib
from datetime import datetime, timedelta
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..recipes import (PlanError, plan, groupables, reachable, ordered_measures, default_stat, EVERY_WORDS, PART_WORDS,
                      experiment, same_kind, GROUP_MAX)
from ..understand import (DataModel, MEASURE, CATEGORY, TIME_ROLE, ID, TEXT, FLAG, BLANK, CONSTANT, STR, norm, name_words,
                         LOOKUP)
from ..units import header_parts, QUANTITY_WORDS

if TYPE_CHECKING:
    from ..bank import Bank


STATS = {"total": "sum", "totals": "sum", "sum": "sum", "sums": "sum", "add up": "sum", "added up": "sum",
         "average": "mean", "averages": "mean", "avg": "mean", "mean": "mean", "typical": "mean",
         "count": "count", "number of": "count", "how many": "count", "count of": "count",
         "max": "max", "maximum": "max", "highest": "max", "peak": "max", "largest": "max", "biggest": "max",
         "min": "min", "minimum": "min", "lowest": "min", "smallest": "min", "median": "median",
         "busiest": "count", "quietest": "count",
         "standard deviation": "std", "standard deviations": "std", "std dev": "std", "stdev": "std", "std": "std",
         "sd": "std", "variability": "std", "variation": "std"}
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
                "depends on": "relationship", "relate": "relationship", "relationship between": "relationship",
                "correlation between": "relationship", "correlate": "relationship", "correlated": "relationship",
                # do the groups differ: a test, a size, a sentence
                "difference": "groups", "differences": "groups", "different": "groups", "differ": "groups",
                "differs": "groups", "significant": "groups", "significantly": "groups", "significance": "groups",
                "t test": "groups", "t-test": "groups", "ttest": "groups", "anova": "groups", "welch": "groups",
                "mann whitney": "groups", "mann-whitney": "groups", "rank test": "groups", "wilcoxon": "groups",
                "higher": "higher", "lower": "higher", "greater": "higher", "larger": "higher", "bigger": "higher",
                "smaller": "higher", "heavier": "higher", "lighter": "higher", "longer": "higher", "shorter": "higher",
                "thicker": "higher", "thinner": "higher", "compare groups": "groups",
                "compare": "compare", "comparison": "compare", "versus": "compare", "vs": "compare",
                "against": "compare", "difference between": "compare", "differences between": "compare",
                "together": "stacked", "stacked": "stacked", "stack": "stacked", "combined": "stacked",
                "linked": "linked", "details": "linked", "joined": "linked",
                # what changed / why / what relates / is the data good / where it is heading
                "what changed": "change", "changed": "change", "up or down": "change", "increase": "change",
                "increased": "change", "increases": "change", "decrease": "change", "decreased": "change",
                "decreases": "change", "rose": "change", "rise": "change", "risen": "change", "fell": "change",
                "fall": "change", "fallen": "change", "drop": "change", "dropped": "change", "drops": "change",
                "grew": "change", "grow": "change", "grown": "change", "declined": "change", "decline": "change",
                "gained": "change", "gain": "change",
                "why": "explain", "explain": "explain", "drives": "explain", "driving": "explain", "drive": "explain",
                "caused": "explain", "cause": "explain", "what caused": "explain", "contribute": "explain",
                "contributes": "explain", "contribution": "explain", "reason": "explain", "because": "explain",
                "relates to": "drivers", "relate to": "drivers", "related to": "drivers", "related": "drivers",
                "relates": "drivers", "association": "drivers", "associations": "drivers", "affects": "drivers",
                "affect": "drivers", "influences": "drivers", "influence": "drivers",
                "data quality": "quality", "quality": "quality", "check the data": "quality", "check data": "quality",
                "problems": "quality", "issues": "quality", "clean": "quality", "missing values": "quality",
                 "forecast": "forecast", "projected": "forecast", "projection": "forecast", "future": "forecast",
                 "heading": "forecast", "when will": "forecast",
                 # places: draw the points, count them per cell, or match each row to its nearest place
                 "map": "map", "mapped": "map", "map of": "map", "on a map": "map", "geographic": "map",
                 "geographical": "map", "geographically": "map", "location": "map", "locations": "map",
                 "on the map": "map",
                 "density": "density", "hotspot": "density", "hotspots": "density", "hot spot": "density",
                 "hot spots": "density", "cluster": "density", "clusters": "density", "clustered": "density",
                 "per cell": "density", "grid": "density", "concentration": "density",
                 "nearest": "nearest", "closest": "nearest", "nearest place": "nearest", "closest place": "nearest",
                 "how far": "nearest", "distance to": "nearest", "distance from": "nearest", "distance between": "nearest",
                 "proximity": "nearest", "nearby": "nearest", "near to": "nearest",
                 "inside": "place", "inside which": "place", "falls inside": "place"}
BY_WORDS = {"by", "per", "for each", "each", "grouped by", "split by", "broken down by", "across", "by each"}
OPS = {"above": "gt", "over": "gt", "more than": "gt", "greater than": "gt", ">": "gt", "exceeds": "gt",
       "exceeding": "gt", "higher than": "gt", "below": "lt", "under": "lt", "less than": "lt", "<": "lt",
       "lower than": "lt", "at least": "ge", ">=": "ge", "at most": "le", "<=": "le", "between": "between",
       "=": "eq", "equals": "eq", "equal to": "eq", "is not": "ne", "isn't": "ne", "are not": "ne", "aren't": "ne",
       "not": "ne", "!=": "ne", "except": "ne", "excluding": "ne", "without": "ne", "other than": "ne",
       "after": "gt", "before": "lt", "since": "ge", "until": "le", "till": "le", "up to": "le", "through": "le",
       "from": "ge", "later than": "gt", "earlier than": "lt"}
COPULA = {"is", "are", "was", "were"}     # "price is above 30", "region is North"; on its own, nothing
RELATIVE = {"last": "last", "previous": "last", "past": "past", "this": "this", "current": "this"}
DAY_WORDS = {"today": ("this", "d"), "yesterday": ("last", "d")}
STOP = {"show", "me", "the", "a", "an", "of", "what", "whats", "what's", "how", "does",
        "do", "did", "i", "want", "see", "give", "find", "please", "chart", "plot", "graph", "list", "my",
        "for", "in", "where", "when", "with", "to", "on", "data", "all", "it", "its", "that",
        "which", "who", "there", "any", "can", "you", "get", "tell", "about", "has", "have", "had", "only",
        "rows", "row", "values", "readings", "reading", "records", "time", "at", "?", "these", "then",
        "table", "tables", "file", "files", "their", "them", "look", "into", "during", "thing", "things",
        "most", "least", "gets", "got", "goes", "went", "been", "be", "being",
        "our", "we", "us", "every", "some", "much", "many", "there's", "theres",
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
    kind: str                  # stat every recipe col table value by op is num top unit rel stop and or month date part share when
    value: Any = None
    refs: list[list] = field(default_factory=list)      # col: every column the words could mean; value: its columns
    aliases: list[list] = field(default_factory=list)   # col: the refs named only by a shortened name ("patient" for patient_id)
    neg: bool = False                                   # value: "not North", "except North"

    @property
    def alias(self) -> bool:
        """Every column these words can mean is named by them only in short."""
        return bool(self.refs) and all(r in self.aliases for r in self.refs)


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
    corrected: list[dict[str, Any]] = field(default_factory=list)   # {"from", "to"}: spellings repaired from the typed words
    chips: list[dict[str, Any]] = field(default_factory=list)
    source: str = "grammar"                                 # "grammar" | "matched" (the question bank fallback)
    matched: str = ""                                       # the canonical phrasing a matched question fitted

    @property
    def ok(self) -> bool:
        return self.spec is not None

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "ok": self.ok, "spec": self.spec, "title": self.title, "message": self.message,
                "unknown": self.unknown, "hints": self.hints, "ambiguous": self.ambiguous, "chips": self.chips,
                "corrected": self.corrected, "source": self.source, "matched": self.matched}


# =================================================================== vocabulary
_DATE = (r"\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:[ t]\d{1,2}:\d{2}(?::\d{2})?)?|\d{1,2}/\d{1,2}/\d{2,4}"
         r"|\d{4}-\d{1,2}(?![\d\w])")                 # 2024-03-01, 2024/03/01, 01/03/2024, and 2024-03 (a month)
# a comma between digits is a thousands separator only before exactly three digits ("1,234", "12,345.5");
# otherwise it separates a list ("customer_id is 1,2" is 1 or 2). A decimal comma is not read: write 30.5
_NUMBER = r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\d,])|-?\d+(?:\.\d+)?|-?\.\d+"      # .1 is a tenth


_THIS_VS_LAST = re.compile(r"\b(?:this|current)\s+(day|week|month|quarter|year)\s+(?:vs\.?|versus|against|compared (?:to|with))"
                           r"\s+(?:last|previous|the one before)(?:\s+\1)?\b", re.IGNORECASE)


def _this_vs_last(text: str) -> str:
    """"sales this month vs last" is a change between two periods: "sales what changed per month"."""
    return _THIS_VS_LAST.sub(lambda m: f"what changed per {m.group(1).lower()}", text)


JOINED = re.compile(r"^(top|bottom|last|past|first|next|best|worst)(\d+)$")


def _split_joined(toks: list[str]) -> list[str]:
    """"top5", "last7" as they are meant: a word and its number."""
    out: list[str] = []
    for t in toks:
        m = JOINED.match(t)
        out += [m.group(1), m.group(2)] if m else [t]
    return out


def _tokens(text: str) -> list[str]:
    return re.findall(_DATE + r"|>=|<=|!=|[<>=]|" + _NUMBER + r"|[\w'%/]+(?:[.\-][\w]+)*", text.lower())


_VOCABS: dict[int, tuple[DataModel, bool, dict]] = {}


def vocabulary(model: DataModel) -> dict[tuple[str, ...], list[Meaning]]:
    """Every phrase this project understands, to its meanings (fixed words first, then the project's own). Built
    once per data model (a model is replaced, not changed, when the tables change), so each question typed does
    not build it again."""
    hit = _VOCABS.get(id(model))
    if hit is not None and hit[0] is model and hit[1] == model.deep:
        return hit[2]
    voc = _vocabulary(model)
    if len(_VOCABS) > 8:
        _VOCABS.clear()
    # the model is kept with it, so its id is not reused; deepen() completes a model in place, so a vocabulary read
    # from the sample model is not used for the deep one
    _VOCABS[id(model)] = (model, model.deep, voc)
    return voc


def _vocabulary(model: DataModel) -> dict[tuple[str, ...], list[Meaning]]:
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
    for w in COPULA:
        add(w, Meaning("is"))
    for w, r in RELATIVE.items():
        add(w, Meaning("rel", r))
    for w, (r, u) in DAY_WORDS.items():
        add(w, Meaning("rel", (r, u)))
    add("and", Meaning("and"))
    add("or", Meaning("or"))
    for node, t in model.tables.items():
        names = {t.title, t.title.replace("_", " "), norm(t.title)}
        names |= {n[:-1] for n in names if len(n) > 3 and n.endswith("s")}        # "products" is also "product"
        for phrase in names:
            add(phrase, Meaning("table", node))
        for c in t.columns:
            if c.role == BLANK:
                continue
            ref = [node, c.name]
            words = name_words(c.name)
            st = model.stack_of(node)
            if c.role == CONSTANT and st is not None and len(words) > 1 and words[-1] in ("id", "key", "code", "no", "number", "ref"):
                for form in _plural_forms(" ".join(words[:-1])):     # one vehicle per file: "by vehicle" is by file
                    _add_ref(voc, tuple(_tokens(form)), [st.id, "source"], alias=True)
            if c.role == ID and len(words) > 1 and words[-1] in ("id", "key", "code", "no", "number", "ref"):
                stem = " ".join(words[:-1])                     # "by patient" is by patient_id
                for form in _plural_forms(stem):
                    _add_ref(voc, tuple(_tokens(form)), ref, alias=True)
            if c.role == ID and len(words) > 1 and words[0] in ("n", "no", "nr", "num", "number"):
                for form in _plural_forms(" ".join(words[1:])):   # "N - Sample" numbers the samples: "samples" are the rows
                    _add_ref(voc, tuple(_tokens(form)), ref, alias=True)
            for phrase, short in _column_phrases(c.name, c.label):
                for form in _plural_forms(phrase):              # "customers" for a column called customer
                    _add_ref(voc, tuple(_tokens(form)), ref, alias=short)
                for n in names:                                  # "product name", "customers region"
                    _add_ref(voc, tuple(_tokens(f"{n} {phrase}")), ref)
            if c.role in (MEASURE, TIME_ROLE):                  # "revenue" for Amount, "pays" for salary, "started" for start_date
                for syn in sorted(_synonyms(words + name_words(c.label or ""))):
                    _add_ref(voc, tuple(_tokens(syn)), ref, alias=True)
            if c.abbrev:                                        # "IA" for 'inner area (IA) cm2'
                _add_ref(voc, tuple(_tokens(c.abbrev)), ref)
            if c.role == MEASURE and c.quantity:                # "mass" for 'm (g)', "mass per area" for g/m2
                q = c.quantity
                forms = [q] + [w for w in QUANTITY_WORDS.get(q, [])]
                if " per " in q:
                    a, b = q.split(" per ", 1)
                    forms += [f"{x} per {y}" for x in QUANTITY_WORDS.get(a, [a]) for y in QUANTITY_WORDS.get(b, [b])]
                for form in sorted(set(forms)):
                    _add_ref(voc, tuple(_tokens(form)), ref, alias=True)
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
            common = set.intersection(*[set(name_words(model.tables[n].title)) for n in r.tables]) if r.tables else set()
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
    # measurement nouns that help find a column when the header spells the quantity out
    {"yield", "yields", "production", "output"},
    {"height", "tall"},
    {"maturity", "maturing", "matures"},
    {"disease", "diseases", "infection"},
    {"lodging"},
    {"temperature", "temp"},
    {"pressure"},
    {"humidity", "rh"},
    {"salinity", "salt"},
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
    if len(p) > 3 and p.endswith("ves"):
        out.append(p[:-3] + "f")                          # shelves -> shelf
    if len(p) > 3 and p.endswith("s") and not p.endswith("ss"):
        out.append(p[:-1])
    elif len(p) > 2:
        out.append(p + "s")
        if p.endswith("f"):
            out.append(p[:-1] + "ves")                    # shelf -> shelves
        elif p.endswith("fe"):
            out.append(p[:-2] + "ves")                    # knife -> knives
        elif p.endswith("y") and p[-2:-1] not in "aeiou":
            out.append(p[:-1] + "ies")
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
                if alias:
                    m.aliases.append(ref)
            elif not alias and ref in m.aliases:
                m.aliases.remove(ref)                     # a real name wins over a shortened one
            return
    voc[key].append(Meaning("col", refs=[ref], aliases=[ref] if alias else []))


def _column_phrases(name: str, label: str) -> list[tuple[str, bool]]:
    """(phrase, shortened?): pressure_psia -> 'pressure_psia', 'pressure psia', and 'pressure' (shortened);
    'Pressure (bar)' -> 'pressure (bar)', 'pressure'; 'inner area (IA) cm2' -> 'inner area' too."""
    full = {name, name.replace("_", " "), label, re.sub(r"\s*[\(\[].*?[\)\]]\s*$", "", label),
            header_parts(name)[0], header_parts(label)[0]}
    full |= {re.sub(r"^(avg|average|mean|total|sum of|number of|no of|count of)[\s_.:]+", "", f, flags=re.IGNORECASE) for f in list(full)}
    full |= {re.sub(r"\s*[x×*]\s*[\d,.]+$", "", f) for f in list(full)}          # 'mass/area x 10000' is also 'mass/area'

    words = name_words(name)
    if words:
        full.add(" ".join(words))
    out = [(p, False) for p in sorted(x for x in full if x and x.strip())]
    if len(words) > 1 and words[-1] not in ("id", "key", "code", "no"):
        out.append((words[0], True))                  # pressure_psia is 'pressure', customer_name is 'customer'
        # the modifier + head noun ('grain yield', 'plant height'), so a question that says both words lands on
        # one column instead of naming it twice, plus the head noun on its own ('yield', 'height')
        tail = list(words[1:])
        while tail and (len(tail[-1]) < 3 or not tail[-1].isalnum() or tail[-1] in _FILLER_WORDS):
            tail.pop()
        if tail:
            out.append((" ".join(words[:1 + len(tail)]), True))
            head = tail[-1]
            if head.isalpha() and len(head) >= 3 and head not in _FILLER_WORDS:
                out.append((head, True))
    return out


# words that must never stand alone as a column name (grammar words, statistic words, units spelled short)
_FILLER_WORDS = {
    "per", "and", "to", "of", "for", "the", "a", "an", "in", "on", "at", "by", "vs", "with",
    "avg", "average", "mean", "median", "max", "min", "sum", "total", "count", "std", "sd", "se",
    "pct", "percent", "num", "number", "no", "id", "key", "code", "new", "old",
    "day", "days", "hour", "hours", "minute", "minutes", "sec", "second", "seconds",
    "cm", "mm", "km", "kg", "mg", "ml", "ha", "bar", "psi", "kpa", "mph", "ph",
}




def _number(tok: str) -> float | int | None:
    from ..dtypes import typed_value
    if re.fullmatch(r"-?\.\d+", tok):
        tok = tok.replace(".", "0.", 1)               # .1 is 0.1, -.5 is -0.5
    if not re.fullmatch(r"-?\d+(?:[.,]\d+)*", tok):
        return None
    try:
        return typed_value(tok)
    except Exception:  # noqa: BLE001
        return None


PARTS = {"hour of day": "hour", "hour of the day": "hour", "time of day": "hour", "time of the day": "hour",
         "day of week": "weekday", "day of the week": "weekday", "weekday": "weekday", "weekdays": "weekday",
         "month of year": "month", "month of the year": "month", "day of month": "day", "day of the month": "day"}
SHARE_WORDS = ["share", "share of", "percentage", "percentage of", "percent", "percent of", "proportion", "proportion of",
               "what share", "% of", "fraction"]
