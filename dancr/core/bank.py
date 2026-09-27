"""The question bank: every question DANCR can answer for a project, kept as data and indexed by its words.

The grammar in :mod:`dancr.core.ask` reads the questions people actually type, precisely: every word must be
understood and fitted into the answer, or the question is refused. This module is the recall layer *behind*
the grammar: it enumerates the answers the recipes can build for the project's tables, keeps each one's
canonical phrasing and spec, and matches a typed question against them by its words. It is a fallback only —
the grammar stays authoritative, and a question the grammar can read is never sent here.

There is no model and no randomness. The bank is built from the data model, so the same model always gives the
same questions, and the same question always gives the same matches and the same specs. A match must clear
:data:`MATCH_THRESHOLD` before it is used, so an unrelated or gibberish question is still refused.

Pure core, no Qt.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

from . import recipes
from .ask import _tokens, STOP
from .understand import DataModel, name_words

MATCH_THRESHOLD = 0.8       # a match must cover the question's words and most of the phrasing before it is used
MAX_VARIANTS = 6            # surface phrasings kept per canonical question
MIN_QUERY_TOKENS = 2        # distinct content words a typed question needs before the bank will answer it
K1 = 1.5                    # BM25 term saturation
B = 0.75                    # BM25 length normalisation

# words that carry no answer of their own, so they neither help nor hurt a match (ask.STOP plus the joiners)
BANK_STOP = STOP | {"and", "or"}

# the leading word of a title that a person may leave out ("total quantity by region" for "Total quantity by region")
LEAD_WORDS = {"total", "average", "mean", "median", "highest", "lowest", "spread", "number", "rows",
              "top", "bottom", "biggest", "smallest"}

# a time step said as an adverb ("per day" is "daily")
UNIT_ADVERB = {"second": "secondly", "seconds": "secondly", "minute": "minutely", "minutes": "minutely",
               "hour": "hourly", "hours": "hourly", "day": "daily", "days": "daily", "week": "weekly",
               "weeks": "weekly", "month": "monthly", "months": "monthly", "quarter": "quarterly",
               "quarters": "quarterly", "year": "yearly", "years": "yearly"}


@dataclass(frozen=True)
class Question:
    """One answer DANCR can build: the spec, the canonical phrasing of it, and where it comes from."""
    spec: dict[str, Any]
    canonical: str
    recipe: str
    table: str
    key: str


@dataclass(frozen=True)
class Match:
    """A typed question matched to a bank question, with how well it fits (0..1)."""
    question: Question
    score: float


def _content(text: str) -> list[str]:
    """The words of a question that carry meaning: its tokens without the everyday words."""
    return [t for t in _tokens(text) if t not in BANK_STOP]


def _key(table: str, recipe: str, canonical: str, spec: dict) -> str:
    """A stable id for a question, the same on every build."""
    seed = json.dumps({"t": table, "r": recipe, "c": canonical, "s": spec}, sort_keys=True, default=str)
    return hashlib.sha1(seed.encode()).hexdigest()[:12]


def _expand(tokens: tuple[str, ...]) -> tuple[str, ...]:
    """A token split the way people write it: pressure_bar -> pressure bar; orders's -> orders."""
    out: list[str] = []
    for t in tokens:
        t = t.replace("_", " ")
        if t.endswith("'s"):
            t = t[:-2]
        out += [w for w in t.split() if w]
    return tuple(out)


def _adverbial(tokens: tuple[str, ...]) -> tuple[str, ...] | None:
    """'total quantity per day' is also 'total quantity daily'."""
    for i in range(len(tokens) - 1):
        if tokens[i] in ("per", "by") and tokens[i + 1] in UNIT_ADVERB:
            return tokens[:i] + (UNIT_ADVERB[tokens[i + 1]],) + tokens[i + 2:]
    return None


def _root(tok: str) -> str:
    """A word reduced to compare without plural or possessive endings: customers and customer's are customer."""
    t = tok.replace("_", " ")
    t = t.split()[-1] if " " in t else t
    if t.endswith("'s"):
        t = t[:-2]
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        t = t[:-1]
    return t


def _required_groups(model: DataModel, spec: dict) -> tuple[frozenset[str], ...]:
    """The columns a question is *about* (the number and the groups it names). A typed question must mention
    each of them, or an omitted number would match the phrasing that happens to share the other words."""
    refs: list[Any] = []
    for k in ("measure", "by", "x", "y", "target"):
        v = spec.get(k)
        if isinstance(v, list) and len(v) == 2:
            refs.append(v)
    refs += list(spec.get("measures") or [])
    groups: list[frozenset[str]] = []
    for ref in refs:
        if not ref or ref[0].startswith("stack:"):
            continue
        t = model.table(ref[0])
        c = t.column(ref[1]) if t else None
        words = set(name_words(c.name)) | set(name_words(c.label or "")) if c is not None else set()
        g = frozenset(_root(w) for w in words if w)
        if g and g not in groups:
            groups.append(g)
    return tuple(groups)


def _mentions(raw: set[str], group: frozenset[str]) -> bool:
    return any(_root(t) in group for t in raw)


def _project_words(model: DataModel, spec: dict, table: str) -> tuple[str, ...]:
    """The project's own words for this question: the columns it names and the table it is about."""
    refs: list[Any] = []
    for k in ("measure", "by", "x", "y", "time", "target"):
        v = spec.get(k)
        if isinstance(v, list) and len(v) == 2:
            refs.append(v)
    refs += list(spec.get("measures") or [])
    refs += [f.get("column") for f in (spec.get("filters") or [])]
    words: list[str] = []
    for ref in refs:
        if not ref or ref[0].startswith("stack:"):
            continue
        t = model.table(ref[0])
        c = t.column(ref[1]) if t else None
        if c is None:
            continue
        words += name_words(c.name) + name_words(c.label or "")
    t = model.table(table)
    if t is not None:
        words += name_words(t.title)
    return tuple(w for w in dict.fromkeys(words) if w and w not in BANK_STOP)


def _variants(model: DataModel, q: Question) -> tuple[tuple[str, ...], ...]:
    """The surface phrasings of a canonical question, most faithful first, at most MAX_VARIANTS."""
    base = tuple(_content(q.canonical))
    cands: list[tuple[str, ...]] = [base]
    if base and base[0] in LEAD_WORDS:
        cands.append(base[1:])
    if "by" in base:
        cands.append(tuple("per" if t == "by" else t for t in base))
    elif "per" in base:
        cands.append(tuple("by" if t == "per" else t for t in base))
    cands.append(_expand(base))
    adv = _adverbial(base)
    if adv is not None:
        cands.append(adv)
    extra = _project_words(model, q.spec, q.table)
    if extra:
        cands.append(base + extra)
    out: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for c in cands:
        if not c or c in seen:
            continue
        seen.add(c)
        out.append(c)
        if len(out) >= MAX_VARIANTS:
            break
    return tuple(out)


class Bank:
    """The questions a project can answer, indexed by their words, for a deterministic recall fallback."""

    def __init__(self, questions: list[Question], variants: list[tuple[tuple[str, ...], ...]],
                 required: list[tuple[frozenset[str], ...]] | None = None,
                 table_order: dict[str, int] | None = None) -> None:
        self._questions = list(questions)
        self._table_order = dict(table_order or {})
        self._variants: dict[str, tuple[tuple[str, ...], ...]] = \
            {q.key: v for q, v in zip(self._questions, variants)}
        self._required: dict[str, tuple[frozenset[str], ...]] = \
            {q.key: r for q, r in zip(self._questions, required or [()] * len(self._questions))}
        self._postings: dict[str, list[int]] = {}
        df: Counter[str] = Counter()
        for i, q in enumerate(self._questions):
            for t in self._tokens_of(q.key):
                self._postings.setdefault(t, []).append(i)
                df[t] += 1
        n = len(self._questions)
        lengths = [len(v) for vs in self._variants.values() for v in vs]
        self._avgdl = (sum(lengths) / len(lengths)) if lengths else 1.0
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self._idf_unseen = math.log(1 + (n + 0.5) / 0.5)

    def _tokens_of(self, key: str) -> set[str]:
        return {t for v in self._variants.get(key, ()) for t in v}

    # ---------------------------------------------------------------- building
    @classmethod
    def from_model(cls, model: DataModel, limit_tables: int | None = None) -> "Bank":
        """Every answer the recipes can build for the model's tables, indexed. A candidate that cannot be
        planned is left out, so nothing unbuildable is ever matched. ``limit_tables`` keeps the first N tables."""
        nodes = list(model.tables)
        if limit_tables is not None:
            nodes = nodes[:max(0, int(limit_tables))]
        table_order = {n: i for i, n in enumerate(model.tables)}
        questions: list[Question] = []
        variants: list[tuple[tuple[str, ...], ...]] = []
        required: list[tuple[frozenset[str], ...]] = []
        seen: set[str] = set()
        for node in nodes:
            t = model.table(node)
            if t is None:
                continue
            for cand in recipes._candidates(model, t):
                try:
                    p = recipes.plan(model, cand)
                except recipes.PlanError:
                    continue
                canonical = " ".join(str(p.title).split())
                if not canonical:
                    continue
                spec = {k: v for k, v in p.config.items() if not k.startswith("_")}
                recipe = str(spec.get("recipe") or cand.get("recipe") or "")
                key = _key(node, recipe, canonical, spec)
                if key in seen:
                    continue
                seen.add(key)
                q = Question(spec=spec, canonical=canonical, recipe=recipe, table=node, key=key)
                questions.append(q)
                variants.append(_variants(model, q))
                required.append(_required_groups(model, spec))
        return cls(questions, variants, required, table_order)

    # ---------------------------------------------------------------- reading
    def entries(self) -> list[Question]:
        """Every question in the bank, in the stable order it was built."""
        return list(self._questions)

    def match(self, text: str, limit: int = 5) -> list[Match]:
        """The bank questions closest to a typed question, best first. Empty when the words say too little."""
        quniq = tuple(dict.fromkeys(_content(text)))
        if len(quniq) < MIN_QUERY_TOKENS:
            return []
        raw = set(_tokens(text))
        cand: set[int] = set()
        for t in quniq:
            cand.update(self._postings.get(t, ()))
        if not cand:
            return []
        qnorm = tuple(_content(text))
        out: list[Match] = []
        for i in cand:
            q = self._questions[i]
            if not all(_mentions(raw, g) for g in self._required.get(q.key, ())):
                continue
            best = 0.0
            for vtoks in self._variants[q.key]:
                s = self._score(quniq, vtoks)
                if s > best:
                    best = s
            if qnorm == tuple(_content(q.canonical)) or text.strip().lower() == q.canonical.strip().lower():
                best = 1.0
            out.append(Match(q, round(best, 6)))
        out.sort(key=lambda m: (-m.score, self._recipe_rank(m.question.recipe),
                                self._table_order.get(m.question.table, 0), m.question.canonical, m.question.key))
        return out[:max(0, limit)]

    def complete(self, prefix: str, limit: int = 8) -> list[str]:
        """Canonical phrasings that start with ``prefix``, in a stable order, no repeats."""
        p = " ".join(prefix.split()).lower()
        out: list[str] = []
        seen: set[str] = set()
        for q in sorted(self._questions, key=lambda q: (self._recipe_rank(q.recipe),
                                                        self._table_order.get(q.table, 0), q.canonical)):
            if q.canonical.lower().startswith(p) and q.canonical not in seen:
                seen.add(q.canonical)
                out.append(q.canonical)
                if len(out) >= limit:
                    break
        return out

    # ---------------------------------------------------------------- scoring
    def _recipe_rank(self, recipe: str) -> int:
        return recipes.RECIPES.index(recipe) if recipe in recipes.RECIPES else len(recipes.RECIPES)

    def _idf_of(self, token: str) -> float:
        return self._idf.get(token, self._idf_unseen)

    def _score(self, quniq: tuple[str, ...], vtoks: tuple[str, ...]) -> float:
        """How well one phrasing answers the question: its BM25 score over the words it covers, out of the
        score the question's own words would get. 0 when nothing matches, 1 when the phrasing is the question."""
        if not vtoks:
            return 0.0
        doc = Counter(vtoks)
        dl = len(vtoks)
        matched = [t for t in quniq if t in doc]
        if not matched:
            return 0.0
        raw = sum(self._idf_of(t) * (doc[t] * (K1 + 1)) / (doc[t] + K1 * (1 - B + B * dl / self._avgdl))
                  for t in matched)
        dq = Counter(quniq)
        ideal = sum(self._idf_of(t) * (dq[t] * (K1 + 1)) / (dq[t] + K1 * (1 - B + B * len(quniq) / self._avgdl))
                    for t in quniq)
        return raw / ideal if ideal > 0 else 0.0
