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
import threading
from collections import OrderedDict
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
from .vocab import (  # noqa: E402,F401 - re-exported so `dancr.core.ask` is unchanged
    ADJECTIVES,
    ADVERBS,
    Asked,
    BIG_WORDS,
    BY_WORDS,
    COPULA,
    DAY_WORDS,
    JOINED,
    MONTH_WORDS,
    Meaning,
    OPS,
    PARTS,
    RECIPE_WORDS,
    RELATIVE,
    SHARE_WORDS,
    STATS,
    STOP,
    SYNONYMS,
    TIME_UNITS,
    TOP_WORDS,
    _DATE,
    _FILLER_WORDS,
    _NUMBER,
    _THIS_VS_LAST,
    _add_ref,
    _add_value,
    _column_phrases,
    _number,
    _plural_forms,
    _split_joined,
    _stem_word,
    _synonyms,
    _this_vs_last,
    _tokens,
    _vocabulary,
    vocabulary,
)
# =================================================================== reading
def ask(model: DataModel, text: str, bank: "Bank | None" = None, aliases: dict[str, str] | None = None) -> Asked:
    """Read ``text`` as a question about the tables in ``model``.

    The grammar is tried first and is authoritative: a question it can read is returned exactly as before.
    Only when it cannot read the question is the question bank tried as a fallback (``bank``, or one built
    from the model), and only when a match clears its confidence threshold; otherwise the same message and
    "did you mean" hints as before are returned."""
    out = Asked(text=text)
    toks = _split_joined(_tokens(_this_vs_last(text)))
    if not toks:
        out.message = "Type a question, for example “average pressure per hour” or “total sales by region”"
        return out
    voc = vocabulary(model)
    items = _read(toks, voc, out)
    if out.unknown:
        # one repair attempt before refusing: a misspelled word matched to the project's own words
        from ..lexicon import correct as _correct
        fixed, fixes = _correct(toks, voc, aliases)
        if fixes:
            trial = Asked(text=text)
            trial_items = _read(fixed, voc, trial)
            if not trial.unknown:
                out.corrected = [{"from": a, "to": b} for a, b in fixes]
                out.unknown, out.hints, out.message = [], [], ""
                items = trial_items
    if out.unknown:
        phrases = [" ".join(k) for k in voc]
        # the project's names that hold the word ('inner area' for 'area') before names that merely look like it
        holding = {u: sorted(" ".join(k) for k in voc if len(k) > 1 and u in k and any(m.kind in ("col", "table", "value") for m in voc[k]))[:3]
                   for u in out.unknown}
        out.hints = sorted({h for u in out.unknown for h in (holding[u] or difflib.get_close_matches(u, phrases, n=3, cutoff=0.6))})
        out.message = ("I don't know " + ", ".join(f"“{u}”" for u in out.unknown)
                       + (". Did you mean " + ", ".join(f"“{h}”" for h in out.hints) + "?" if out.hints else
                          ". Use the names of your columns, tables or their values."))
        return _matched(model, out, bank)
    try:
        out.spec = _assemble(model, _clauses(model, items), out)
        p = plan(model, out.spec)
    except PlanError as e:
        out.spec, out.message = None, str(e)
        return _matched(model, out, bank)
    out.title, out.chips = p.title, p.chips
    return out


def _matched(model: DataModel, out: Asked, bank: "Bank | None") -> Asked:
    """The question bank as a fallback: if the best match clears its threshold and still plans, answer from
    it; otherwise leave ``out`` exactly as the grammar left it (a refusal with hints), so nothing is guessed."""
    if "is named twice" in out.message:
        return out                                    # a column named twice is refused, not matched to something close
    try:
        from ..bank import MATCH_THRESHOLD
        b = bank if bank is not None else _bank_of(model)
        top = b.match(out.text, limit=1)
        if not top or top[0].score < MATCH_THRESHOLD:
            return out
        p = plan(model, top[0].question.spec)
    except Exception:  # noqa: BLE001 - a fallback must never break a question the grammar already refused
        return out
    out.spec, out.title, out.chips = p.config, p.title, p.chips
    out.source, out.matched = "matched", top[0].question.canonical
    out.unknown, out.hints, out.message, out.ambiguous = [], [], "", []
    return out


_BANKS: "OrderedDict[int, tuple[DataModel, bool, Any]]" = OrderedDict()
_BANKS_LOCK = threading.Lock()
_BANKS_MAX = 8


def _bank_of(model: DataModel):
    """The question bank of a data model, built once (it plans every answer the recipes can build). A small LRU,
    so a caller keeping many live models does not rebuild the whole bank on every question."""
    from ..bank import Bank
    key = id(model)
    with _BANKS_LOCK:
        hit = _BANKS.get(key)
        if hit is not None and hit[0] is model and hit[1] == model.deep:
            _BANKS.move_to_end(key)
            return hit[2]
    bank = Bank.from_model(model)                   # built outside the lock: it is slow, and it is idempotent
    with _BANKS_LOCK:
        _BANKS[key] = (model, model.deep, bank)
        _BANKS.move_to_end(key)
        while len(_BANKS) > _BANKS_MAX:
            _BANKS.popitem(last=False)
    return bank


Item = tuple[str, Meaning]                      # the words, and what they were read as


def _read(toks: list[str], voc: dict, out: Asked) -> list[Item]:
    """Words -> phrases: the longest phrase the project knows at each point, its meaning chosen by context
    (``_pick``); numbers, dates and everyday words; anything else is noted in ``out.unknown``."""
    maxlen = max((len(k) for k in voc), default=1)
    items: list[Item] = []

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

    def prev_kind() -> str | None:
        return next((m.kind for _, m in reversed(items) if m.kind != "stop"), None)

    i = 0
    while i < len(toks):
        hit = match(i)
        if hit is not None:
            key, ms = hit
            words = " ".join(key)
            items.append((words, _pick(ms, words, prev_kind(), kinds_at(i + len(key)), items[-1][0] if items else None)))
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
                items.append((tok, _pick(stemmed, tok, prev_kind(), kinds_at(i + 1), items[-1][0] if items else None)))
            else:
                out.unknown.append(tok)
        i += 1
    return items


FIXED = {"stat", "every", "recipe", "by", "op", "is", "unit", "top", "and", "or", "month", "part", "share", "rel"}


def _iso_date(tok: str) -> str:
    """'2024-3-1' and '2024/3/1' -> '2024-03-01'; '2024-3' -> '2024-03' (a month); '25/12/2024' and '12/25/2024'
    -> '2024-12-25', as only one reading is a date. '01/02/2024' reads either way and stays as typed (refused
    when the question is put together)."""
    m = re.fullmatch(r"(\d{4})[-/](\d{1,2})(?:[-/](\d{1,2}))?(.*)", tok)
    if m:
        # the question is lowercased before it is read, so an ISO 'T' between date and time arrives as a
        # lowercase 't'; the date/time parser wants a space, so normalise it rather than pass it through
        rest = re.sub(r"^t", " ", m.group(4), count=1)
        return f"{m.group(1)}-{int(m.group(2)):02d}" + (f"-{int(m.group(3)):02d}{rest}" if m.group(3) else "")
    a, b, y = (int(x) for x in tok.split("/"))
    y += 2000 if y < 100 else 0
    if a > 12 >= b:
        a, b = b, a                                  # day first
    elif not (b > 12 >= a or a == b):
        return tok
    return f"{y:04d}-{a:02d}-{b:02d}"


def _pick(ms: list[Meaning], words: str, prev: str | None, nxt: set[str], prev_word: str | None = None) -> Meaning:
    """What a phrase means here, when it could mean several things. Deterministic, from its neighbours:

    - right after a comparison ("region is Total", "status is on") it is the project's value or column;
    - a fixed word followed by a column, value, table or number is the fixed word ("total amount", "top 10");
    - a time word after "per", "last" or "this" is the time word, even with a column called day;
    - otherwise the project's own word wins (a column called "total" when nothing follows);
    - a value that is also an everyday word ("on", "for") is that word unless a comparison came just before;
    - after "by"/"per", a table (customers) beats the id column named after it, and a column beats a value.
    """
    fixed = [m for m in ms if m.kind in FIXED]
    proj = [m for m in ms if m.kind not in FIXED]
    if prev in ("recipe", "and", "col") and fixed and fixed[0].kind == "op" and \
            any(m.kind == "col" and not m.alias for m in proj) and not nxt & {"num", "month", "date", "rel", "unit", "value"}:
        return next(m for m in proj if m.kind == "col" and not m.alias)   # "compare before and after": the columns
    if prev in ("op", "is"):
        if proj:
            return sorted(proj, key=lambda m: {"value": 0, "col": 1, "table": 2}.get(m.kind, 3))[0]
    elif fixed and proj and fixed[0].kind == "unit" and prev in ("by", None, "stat", "rel") and not (
            len(words) == 1 and prev in (None, "stat") and any(m.kind == "col" and not m.alias for m in proj)):
        return fixed[0]                                   # "per day" is a time step, even with a column called day;
        # but "average D" is the column D, not a day
    elif fixed and proj:
        if nxt & {"col", "value", "table", "num"} or fixed[0].kind in ("by", "op", "is", "and", "or") or (
                fixed[0].kind == "rel" and "unit" in nxt):
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
    if tables and prev_word in ("how many", "number of", "count of"):
        return tables[0]                                  # "how many products": the products, not orders per product
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




# =================================================================== clauses
TIME_OPS = {"after", "before", "since", "until", "till", "through", "from", "later than", "earlier than"}
NOT_OP = {"gt": "le", "lt": "ge", "ge": "lt", "le": "gt", "between": "outside"}   # "not above 30" is at most 30
RANGE_TO = {"to", "until", "till", "through"}      # "from March to May"; "and" only after "between" or "from"


def _clauses(model: DataModel, items: list[Item]) -> list[Item]:
    """Join phrases that belong together into one clause each, so the question can be put together in any order:

    - a date, month, year or "last 7 days", with its comparison and the other end of a range ("from March to May",
      "between 2024-01-01 and 2024-02-01", "since March") -> one ``when``;
    - "is", "are": part of the comparison that follows ("price is above 30"), else "is" itself ("region is North"),
      and nothing at all when no comparison follows ("what is the average price");
    - "not", "except" in front of a comparison turns it round ("not above 30"), and in front of values marks them;
    - values of one column joined by "and", "or" or nothing ("North and South", "Leeds vs York") -> one value
      holding them all: rows with any of them.
    """
    items = _join_times(model, items)
    items = _join_comparisons(items)
    return _join_values(items)


def _is_year(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and 1900 <= v <= 2100


def _prev_real(items: list[Item], i: int) -> Item | None:
    return next((items[k] for k in range(i - 1, -1, -1) if items[k][1].kind != "stop"), None)


def _point(items: list[Item], i: int, year_ok: bool) -> tuple[dict | None, int]:
    """A point in time at ``i``: a date, "March", "March 2024", a year (when ``year_ok``), "last month",
    "the last 7 days", "today". Returns it and how many items it used."""
    if i >= len(items):
        return None, 0
    words, m = items[i]
    nxt = items[i + 1][1] if i + 1 < len(items) else None
    if m.kind == "date":
        if "/" not in str(m.value) and not _real_date(str(m.value)):   # 2024-02-30, 2024-13
            raise PlanError(f"“{words}” is not a date. Write it as year-month-day, for example 2024-03-01")
        if re.fullmatch(r"\d{4}-\d{2}", str(m.value)):                # 2024-03: March 2024
            y, mo = (int(x) for x in str(m.value).split("-"))
            return {"month": mo, "year": y, "text": words}, 1
        return {"date": str(m.value), "text": words}, 1
    if m.kind == "num" and _is_day(m.value) and nxt is not None and nxt.kind == "month":     # 1 Feb, 1 Feb 2024
        return _day_point(items, i, nxt.value, m.value)
    if m.kind == "month" and _is_day(nxt.value if nxt is not None and nxt.kind == "num" else None):   # Feb 1, Feb 1 2024
        return _day_point(items, i, m.value, nxt.value)
    if m.kind == "month":
        if nxt is not None and nxt.kind == "num" and _is_year(nxt.value):
            return {"month": int(m.value), "year": int(nxt.value), "text": f"{words.title()} {items[i + 1][0]}"}, 2
        return {"month": int(m.value), "text": words.title()}, 1
    if m.kind == "num" and year_ok and _is_year(m.value):
        return {"year": int(m.value), "text": words}, 1
    if m.kind == "rel":
        if isinstance(m.value, tuple):                               # today, yesterday
            return {"rel": m.value[0], "n": None, "unit": m.value[1], "text": words}, 1
        if nxt is not None and nxt.kind == "unit" and nxt.value != "s":
            return {"rel": m.value, "n": None, "unit": nxt.value, "text": f"{words} {items[i + 1][0]}"}, 2
        if (m.value in ("last", "past") and nxt is not None and nxt.kind == "num" and i + 2 < len(items)
                and items[i + 2][1].kind == "unit"):
            return {"rel": "past", "n": nxt.value, "unit": items[i + 2][1].value,
                    "text": f"{words} {items[i + 1][0]} {items[i + 2][0]}"}, 3
    return None, 0


def _real_date(d: str) -> bool:
    """Whether '2024-03-01' (or the month '2024-03', or '2024-03-01 09:30') is a day that exists."""
    try:
        datetime.fromisoformat(d.replace("t", " ") if len(d) > 7 else d + "-01")
        return True
    except ValueError:
        return False


def _is_day(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= 31


def _day_point(items: list[Item], i: int, month: int, day: int) -> tuple[dict, int]:
    """A day of a month starting at ``i`` ("1 Feb", "Feb 1"), with its year when one follows ("1 Feb 2024")."""
    words = [items[i][0], items[i + 1][0]]
    p = {"month": int(month), "day": int(day)}
    if i + 2 < len(items) and items[i + 2][1].kind == "num" and _is_year(items[i + 2][1].value):
        p["year"] = int(items[i + 2][1].value)
        words.append(items[i + 2][0])
    p["text"] = " ".join(words).title()
    return p, len(words)


def _join_times(model: DataModel, items: list[Item]) -> list[Item]:
    out: list[Item] = []
    i = 0
    while i < len(items):
        words, m = items[i]
        before = _prev_real(items, i)
        op = m.value if m.kind == "op" and m.value in ("gt", "lt", "ge", "le", "between") else None
        start = i + 1 if op else i
        # a year is a year after a comparison with no number column in front of it ("orders after 2023", not
        # "price above 2000"), after "in", "for", "during", or as the start of a range
        # "since 2024", "before 2025", "from 2023 to 2024": words about time take a year as a year, whatever column
        # comes before them; "above 2000" after a number column is the number 2000
        nxt_val = items[start][1].value if start < len(items) and items[start][1].kind == "num" else None
        timely = words in TIME_OPS and any(t.time for t in model.tables.values()) and not (
            before is not None and before[1].kind == "col" and _all_numbers(model, before[1].refs)
            and _within(model, before[1].refs, nxt_val))        # "price from 2000 to 3000": prices, not years
        year_ok = timely or (op is not None and not (before is not None and before[1].kind == "col" and not before[1].alias
                                                     and _all_numbers(model, before[1].refs))) or \
                  (op is None and i > 0 and items[i - 1][1].kind == "stop" and items[i - 1][0] in ("in", "for", "during", "of"))
        a, n = _point(items, start, year_ok)
        if a is None:
            if m.kind == "rel" and m.value == "this":
                out.append((words, Meaning("stop")))                 # "this table"
            elif m.kind == "op" and m.value == "ge" and words == "from" and not (
                    i + 1 < len(items) and items[i + 1][1].kind == "num"):
                out.append((words, Meaning("stop")))                 # "sales from customers in North", not "from 10 to 20"
            else:
                out.append(items[i])
            i += 1; continue
        j = start + n
        b = None
        sep = items[j] if j < len(items) else None
        # the other end of a range: "from March to May", "between 2024-01-01 and 2024-02-01", "from March and
        # April" (both months), "March until May"
        if sep is not None and op in (None, "ge", "between") and (
                sep[0] in RANGE_TO or (sep[0] == "and" and (op == "between" or words == "from"))
                or (sep[1].kind == "op" and sep[1].value == "le")):
            b, n2 = _point(items, j + 1, True)
            if b is not None:
                j += 1 + n2
        when = {"op": op, "a": a, "b": b}
        out.append((" ".join(w for w, _ in items[i:j]), Meaning("when", when)))
        i = j
    return out


def _within(model: DataModel, refs: list[list], v: Any) -> bool:
    """Whether a number lies within the values a number column holds."""
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return False
    for r in refs:
        t = model.table(r[0]) if not r[0].startswith("stack:") else None
        c = t.column(r[1]) if t else None
        try:
            if c is not None and c.minimum is not None and float(c.minimum) <= v <= float(c.maximum):
                return True
        except (TypeError, ValueError):
            continue
    return False


def _all_numbers(model: DataModel, refs: list[list]) -> bool:
    return all(_col_kind(model, r) == "number" for r in refs)


def _join_comparisons(items: list[Item]) -> list[Item]:
    out: list[Item] = []
    i = 0
    while i < len(items):
        words, m = items[i]
        nxt = items[i + 1] if i + 1 < len(items) else None
        if m.kind == "is":
            if nxt is not None and nxt[1].kind == "op":           # "is above", "is not", "is between"
                items[i + 1] = (f"{words} {nxt[0]}", nxt[1])
                i += 1; continue
            before = _prev_real(items, i)
            if before is not None and before[1].kind == "col" and nxt is not None and nxt[1].kind in ("value", "num", "col"):
                items[i] = (words, Meaning("op", "eq"))           # "region is North", "day is 3"
                continue
            out.append((words, Meaning("stop")))                  # "what is the average price"
            i += 1; continue
        if m.kind == "op" and m.value == "between" and nxt is not None and nxt[1].kind == "value":
            out.append((words, Meaning("stop")))                  # "between North and South": the values, not a range
            i += 1; continue
        if m.kind == "op" and m.value == "ne" and nxt is not None and nxt[1].kind == "op" and nxt[1].value in NOT_OP:
            items[i + 1] = (f"{words} {nxt[0]}", Meaning("op", NOT_OP[nxt[1].value]))
            i += 1; continue
        if m.kind == "op" and m.value in ("eq", "ne"):
            j = i + 1                                              # "not in North", "except for North"
            while j < len(items) and items[j][1].kind == "stop" and items[j][0] in ("in", "for", "from", "at", "on"):
                j += 1
            if j < len(items) and items[j][1].kind == "value":
                v = items[j][1]
                items[j] = (" ".join(w for w, _ in items[i:j + 1]),
                            Meaning("value", v.value, v.refs, neg=(m.value == "ne") != v.neg))
                i = j; continue
        out.append(items[i])
        i += 1
    return out


def _join_values(items: list[Item]) -> list[Item]:
    """Values of one column side by side become one: "North and South", "Leeds, York or Hull", "Leeds vs York"
    (the "vs" is kept, it asks for a comparison)."""
    out: list[Item] = []
    i = 0
    while i < len(items):
        words, m = items[i]
        if m.kind != "value":
            out.append(items[i]); i += 1; continue
        vals = [m.value] if not isinstance(m.value, list) else list(m.value)
        refs = list(m.refs)
        all_words, kept = [words], []
        j = i + 1
        while j < len(items):
            k = j
            seps = []
            while k < len(items) and (items[k][1].kind in ("and", "or") or
                                      items[k][1].kind == "recipe" and items[k][1].value == "compare"):
                seps.append(items[k]); k += 1
            if k >= len(items) or items[k][1].kind != "value" or items[k][1].neg or len(seps) > 1:
                break
            common = [r for r in refs if r in items[k][1].refs]
            if not common:
                break
            refs = common
            vals.append(items[k][1].value)
            all_words += [w for w, _ in seps] + [items[k][0]]
            kept += [s for s in seps if s[1].kind == "recipe"]
            j = k + 1
        out.append((" ".join(all_words), Meaning("value", vals if len(vals) > 1 else vals[0], refs, neg=m.neg)))
        out += kept
        i = j
    return out


# =================================================================== putting a question together
@dataclass
class _Parts:
    """What the clauses of a question said, before it is known which table it is about."""
    stat: str | None = None                  # said outright: total, average
    sup: str | None = None                   # a superlative: biggest, hottest
    sup_stat: str | None = None
    big: bool | None = None                  # the superlative's direction: True for the smallest
    recipes: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)
    every: str | None = None
    top: int | None = None
    bottom: bool = False
    by: tuple[int, str, Meaning] | None = None
    by_table: str | None = None
    by_table_words: str = ""
    filters: list[dict[str, Any]] = field(default_factory=list)
    whens: list[dict[str, Any]] = field(default_factory=list)
    cols: list[tuple[int, str, Meaning]] = field(default_factory=list)   # (item index, words, meaning): numbers or groups
    rows_named: list[int] = field(default_factory=list)                 # words naming the rows (orders, items)
    rows_tables: list[str] = field(default_factory=list)
    noun_unit: str | None = None
    part: str | None = None
    share: bool = False
    values: list[dict[str, Any]] = field(default_factory=list)          # filters of several values ("Leeds and York")
    ranked_by: bool = False                  # "top 3 days by sales": the number after "by" is ranked


def _assemble(model: DataModel, items: list[Item], out: Asked) -> dict[str, Any]:
    """Clauses -> spec. Every word that carries meaning must end up in the spec: one that could not be used makes the
    question fail with that word named, rather than build something that quietly ignores it."""
    used: set[int] = set()
    q = _Parts()

    def words_at(k: int) -> str:
        return items[k][0] if k < len(items) else ""

    def sup_later(k: int) -> bool:                  # "which day had the highest sales": the day, not a step
        return any(x.kind == "stat" and w in BIG_WORDS for w, x in items[k + 1:])

    i = 0
    while i < len(items):
        words, m = items[i]
        nxt = items[i + 1][1] if i + 1 < len(items) else None
        if m.kind in ("stop", "and"):
            used.add(i); i += 1; continue
        if m.kind == "stat":
            if words in BIG_WORDS:
                q.sup, q.sup_stat = q.sup or words, q.sup_stat or m.value
                q.big = BIG_WORDS[words] if q.big is None else q.big
            else:
                q.stat = q.stat or m.value
            used.add(i); i += 1; continue
        if m.kind == "recipe":
            q.recipes.append(m.value); used.add(i); i += 1; continue
        if m.kind == "part":
            q.part = m.value; used.add(i); i += 1; continue
        if m.kind == "share":
            q.share = True; used.add(i); i += 1; continue
        if m.kind == "table":
            q.tables.append(m.value); used.add(i); i += 1; continue
        if m.kind == "every":
            q.every = m.value; used.add(i); i += 1; continue
        if m.kind == "when":
            q.whens.append({**m.value, "words": words}); used.add(i); i += 1; continue
        if m.kind == "top":
            # "top 5", "top customers" (ten), "best day" (one day)
            n = nxt.value if nxt is not None and nxt.kind == "num" else 1 if nxt is not None and nxt.kind == "unit" and \
                not words_at(i + 1).endswith("s") else 10
            if isinstance(n, bool) or n != int(n) or n < 1:
                raise PlanError(f"“{words} {words_at(i + 1)}” is not a number of rows. Say a whole number of 1 or more, "
                                f"as in “{words} 5”")
            q.top, q.bottom = int(n), bool(m.value)
            used |= {i, i + 1} if nxt is not None and nxt.kind == "num" else {i}
            i += 2 if nxt is not None and nxt.kind == "num" else 1
            continue
        if m.kind == "by" and nxt is not None:
            if nxt.kind == "unit":
                q.every = _one_step(q.every, f"1{nxt.value}"); used |= {i, i + 1}; i += 2; continue
            if nxt.kind == "num" and i + 2 < len(items) and items[i + 2][1].kind == "unit":
                if float(nxt.value) != int(nxt.value) or int(nxt.value) <= 0:
                    # a fractional or zero step would be truncated to a wrong bucket (or 0, which can't run)
                    raise PlanError(f"A step has to be a whole number of {items[i + 2][0]}, not {nxt.value:g}")
                q.every = f"{int(nxt.value)}{items[i + 2][1].value}"; used |= {i, i + 1, i + 2}; i += 3; continue
            if nxt.kind == "col":
                if q.by is not None:
                    raise PlanError(f"Group by one thing at a time, either {q.by[1]} or {items[i + 1][0]}. "
                                    "Two groups at once (a cross-tab) can't be built yet")
                q.by = (i + 1, items[i + 1][0], nxt); used |= {i, i + 1}; i += 2; continue
            if nxt.kind == "table":
                q.by_table, q.by_table_words = nxt.value, items[i + 1][0]; used |= {i, i + 1}; i += 2; continue
            if nxt.kind == "month":
                q.every = "1mo"; used |= {i, i + 1}; i += 2; continue     # "by month"
        if m.kind == "by":
            used.add(i); i += 1; continue
        if m.kind == "unit" and i > 0 and items[i - 1][0] == "busiest" and m.value in ("h", "d"):
            q.part = "hour" if m.value == "h" else "weekday"      # "busiest hour": rows by hour of the day
            used.add(i); i += 1; continue
        if m.kind == "unit":
            rest = {x.kind for _, x in items[i + 1:]} - {"stop", "and", "table", "value", "when"}
            # "hottest day", "biggest month", also "biggest month by quantity": the period itself is ranked, not
            # stepped over — a bare unit after the superlative. A unit after "per" has already set every, so
            # "highest quantity per month" (a monthly maximum) is not caught here.
            if (q.big is not None or sup_later(i)) and not q.every and not rest - {"stat", "col", "by"}:
                used.add(i); q.noun_unit = words; i += 1; continue    # "hottest day (in site A)": the day itself, not a step
            q.every = _one_step(q.every, f"1{m.value}"); used.add(i); i += 1; continue
        if m.kind == "col":
            if nxt is not None and nxt.kind == "op":
                f, n_used = _filter(model, items, i)
                if f is not None:
                    q.filters += f; used |= set(range(i, i + n_used)); i += n_used; continue
            if nxt is not None and nxt.kind == "value" and _holds(model, m, nxt):
                ref = next((r for r in m.refs if r in nxt.refs), m.refs[0])     # "region North", "region is North"
                _add_values(q, ref, nxt)
                used |= {i, i + 1}; i += 2; continue
            if m.alias and any(_role(model, r) == ID for r in m.refs) and not (nxt is not None and nxt.kind == "op"):
                q.rows_named.append(i); q.rows_tables.extend(r[0] for r in m.refs if _role(model, r) == ID)
                i += 1; continue                             # "orders", "items", "employees": the rows themselves
            if len(m.refs) == 1 and _names_its_table(model, words, m.refs[0][0]):
                q.tables.append(m.refs[0][0])                # "actual jan": Jan of the Actual sheet only
            if words in STATS and m.alias:
                q.stat = q.stat or STATS[words]              # "total by product": the total of the value it names
            q.cols.append((i, words, m)); i += 1; continue
        if m.kind == "value":
            _add_values(q, m.refs[0], m)
            if len(m.refs) > 1:
                shown = m.value if not isinstance(m.value, list) else " or ".join(map(str, m.value))
                out.ambiguous.append({"text": words, "chose": m.refs[0],
                                      "choices": [{"label": f"{shown!r} in {r[1]}", "value": r} for r in m.refs[1:]]})
            used.add(i); i += 1; continue
        i += 1
    if any(m.kind == "or" for k, (_, m) in enumerate(items) if k not in used):
        raise PlanError("“or” only joins values of one column, as in “North or South”. "
                        "Ask about each condition on its own")
    return _build(model, items, q, used, out)


def _add_values(q: _Parts, ref: list, m: Meaning) -> None:
    """A value or several of one column as filters: "North" is one; "North and South" is either; "not North or
    South" is neither."""
    vals = m.value if isinstance(m.value, list) else [m.value]
    if len(vals) == 1:
        q.filters.append({"column": ref, "op": "ne" if m.neg else "eq", "value": vals[0]})
    elif m.neg:
        q.filters += [{"column": ref, "op": "ne", "value": v} for v in vals]
    else:
        f = {"column": ref, "op": "in", "value": list(vals)}
        q.filters.append(f)
        q.values.append(f)


def _holds(model: DataModel, col: Meaning, value: Meaning) -> bool:
    """Whether a value just after a column is that column's ("billing country is France"): it is one of the
    column's values, or the column holds text (so a value seen only elsewhere can still be in it). A number column
    never does: in "total quantity except North", North is a region."""
    return any(r in value.refs for r in col.refs) or any(_col_kind(model, r) == "text" for r in col.refs)


def _build(model: DataModel, items: list[Item], q: _Parts, used: set[int], out: Asked) -> dict[str, Any]:
    named = ([q.by] if q.by else []) + q.cols
    for k, (_, w, m) in enumerate(named):              # "by region region": the second one has nothing to say
        if any(m2.refs == m.refs for _, _, m2 in named[:k]):
            raise PlanError(f"“{w}” is named twice. Name each column once")
    tables = list(q.tables)
    lookups_named = [t for t in tables if model.tables[t].shape == LOOKUP]
    if q.rows_tables and lookups_named and q.rows_tables[0] not in tables:
        tables = [q.rows_tables[0]] + tables           # "which supplier has the most items": items per supplier
    by = (q.by[1], q.by[2].refs) if q.by else None
    # "nearest clinic to each village": two tables joined by their coordinates, not by a key. The near
    # relation picks the points table (the one with more rows); every word that names either table is used.
    refs = [(w, m.refs) for _, w, m in q.cols]
    if q.by is not None:
        refs.append((q.by[1], q.by[2].refs))
    if "nearest" in q.recipes:
        near = _near_relation(model, tables, refs)
        if near is not None:
            for idx, _, _ in q.cols:
                used.add(idx)
            return {"table": near.tables[0], "other": near.tables[1], "recipe": "nearest"}
    if "place" in q.recipes:
        cont = _containment_relation(model, tables, refs)
        if cont is not None:
            for idx, _, _ in q.cols:
                used.add(idx)
            return {"table": cont.tables[0], "other": cont.tables[1], "recipe": "place"}
    base = _base_table(model, tables, [(w, m.refs) for _, w, m in q.cols], by, q.filters)
    t0 = model.tables[base]
    if t0.shape == LOOKUP and ((q.whens and q.top is not None and not t0.time) or
                               ("compare" in q.recipes and q.values and base not in q.tables)):
        # "top 3 customers in March" ranks customers by their orders in March; "compare North and South" (regions
        # of the customers table) compares the orders of those customers, not how many customers each has
        base = _fact_of(model, base, timed=bool(q.whens)) or base
        t0 = model.tables[base]
    resolve = lambda w, m: _resolve(model, w, m, base, out)   # noqa: E731
    measures = []
    for idx, w, m in q.cols:
        ref = resolve(w, m)
        if ref is not None:
            measures.append((idx, ref, m))
    by_ref = _readable(model, resolve(q.by[1], q.by[2]), q.by[2], base) if q.by else None
    time_ref = next((r for _, r, _ in measures if _role(model, r) == TIME_ROLE), None)
    if time_ref is not None:
        used |= {idx for idx, r, _ in measures if r == time_ref}
    measures = [(k, r, m) for k, r, m in measures if r != time_ref]
    if by_ref is not None and _role(model, by_ref) == MEASURE:
        # "top 3 days by sales", "biggest orders by price", "compare North and South by price": the number ranked or
        # compared; but in "subtotal per table" with a numbered table, the number is the group
        ranked = (q.top is not None or q.sup is not None or ("compare" in q.recipes and bool(q.values))) and \
            not any(_role(model, r) == MEASURE for _, r, _ in measures)
        if ranked or _groups_named(model, measures, tables):
            measures.insert(0, (q.by[0], by_ref, q.by[2])); by_ref = None
            q.ranked_by = ranked
    if by_ref is not None and _role(model, by_ref) == TIME_ROLE:
        time_ref, by_ref = by_ref, None
    stat = q.stat or q.sup_stat
    # a table that holds points is named by its table word ("the villages"); on a map that names the points,
    # so a column word like it is dropped and marked used rather than refused as a stray word
    if t0.geo is not None and (set(q.recipes) & {"map", "density", "nearest"}) and measures:
        keep = []
        for k, r, m in measures:
            if r[0] == base and _named_by([(items[k][0], m.refs)], r[1]):
                used.add(k)
            else:
                keep.append((k, r, m))
        measures = keep
    numbers = [(k, r) for k, r, _ in measures if _role(model, r) == MEASURE or (_role(model, r) == CONSTANT and _is_number(model, r))]
    groups = [(k, _readable(model, r, m, base)) for k, r, m in measures if _role(model, r) in (CATEGORY, ID, TEXT, FLAG)]
    adjective = next((w for w, m in items if m.kind == "stat" and w in ADJECTIVES), None)
    if adjective and not numbers:
        hint = ADJECTIVES[adjective][1]
        match = next((c for c in ordered_measures(t0) if set(name_words(c.name)) & hint), None)
        if match is None:
            raise PlanError(f"“{adjective}” needs a {sorted(hint)[0]} column, and {t0.title} has none")
        numbers = [(-1, [base, match.name])]
    if not numbers and (stat in ("sum", "mean", "median", "min", "max") or q.top) and not adjective:
        # "total quantity above 3": the number compared is also the one to add up
        f = next((f for f in q.filters if f["op"] in ("gt", "lt", "ge", "le", "between", "outside") and _role(model, f["column"]) == MEASURE), None)
        if f is not None:
            numbers = [(-1, f["column"])]
    texts = [(k, r) for k, r in groups if _col_kind(model, r) == "text"]
    if stat in ("sum", "mean", "median", "min", "max") and not numbers and texts and by_ref is not None:
        g = texts[0][1]
        raise PlanError(f"{g[1]} holds text (for example {_example(model, g)}), so it cannot be "
                        f"{'added up' if stat == 'sum' else 'averaged' if stat == 'mean' else 'ranked as a number'}")
    if by_ref is None and groups and (numbers or stat == "count" or q.top):
        by_ref = groups[0][1]; used.add(groups[0][0])
    if stat in ("sum", "mean", "median") and not numbers and not adjective:
        some = ", ".join(c.name for c in ordered_measures(t0)[:3])
        raise PlanError(f"{'Total' if stat == 'sum' else 'Average'} of which number? For example "
                        + (f"“{'total' if stat == 'sum' else 'average'} {ordered_measures(t0)[0].name} …” ({some})" if some
                           else "name a column of numbers"))
    if q.by_table is not None and by_ref is None:
        by_ref = _group_for_table(model, base, q.by_table)
        if by_ref is None and q.by_table == base:
            by_ref = _own_key(model, base, q.by_table_words)
        if by_ref is None:
            if q.by_table == base:
                raise PlanError(f"{t0.title} is the table you're asking about. Say which of its columns to group by")
            raise PlanError(f"{model.tables[q.by_table].title} isn't linked to {t0.title}, so {t0.title} can't be counted per {model.tables[q.by_table].title}")
    if by_ref is not None and by_ref[0] == base and _unique_id(model, by_ref) and not q.top:
        # one row per ticket: "average tip per ticket" is the plain average (and the highest, the lowest), but a
        # total or a count per ticket is each row on its own, so that is refused rather than answered for all rows
        if stat not in ("mean", "median", "min", "max") or q.share:
            word = q.by[1] if q.by else q.by_table_words if q.by_table else by_ref[1]
            ranked = numbers[0][1][1] if numbers else _first_measure(model, base)
            raise PlanError(f"{by_ref[1]} is different on every row of {t0.title}, so “{word}” leaves each row on its own. "
                            f"Ask without it, or for the rows themselves (“top 10 {t0.title} by {ranked}”)")
        by_ref = None
    named_lookups = [t for t in tables if t != base and model.tables[t].shape == LOOKUP]
    if by_ref is None and named_lookups and (numbers or stat or q.top or q.rows_named) and not (q.top is not None and base in tables and not named_lookups):
        by_ref = _name_column(model, named_lookups[0], base)
    others = [t for t in tables if t not in named_lookups]
    # dates, months and years apply to the time column named, else the table's own
    tcol = time_ref if time_ref is not None else ([base, t0.time] if t0.time else None)
    spec: dict[str, Any] = {"table": base}
    # "lowest temperature in logger_site_B" is one number; "biggest orders by price", "cheapest orders" and "orders
    # with the highest price" are rows (the number is not named on its own, or the rows are named before it)
    sup_at = next((k for k, (w, m) in enumerate(items) if m.kind == "stat" and w in BIG_WORDS), None)
    if sup_at is not None and not q.ranked_by and any(k >= 0 for k, _ in numbers) and not any(
            m.kind == "table" and m.value == base and k < sup_at for k, (_, m) in enumerate(items)):
        spec["_one_value"] = True
    if q.by is not None and by_ref is not None and by_ref[0].startswith("stack:"):
        spec["by_words"] = q.by[1]                     # "by vehicle": said the way it was asked
    stack = model.stack_of(base)
    if stack is not None:
        spec["together"] = not any(t in stack.tables for t in others) or len({t for t in others if t in stack.tables}) > 1
    filters = list(q.filters)
    plain = [w for w in q.whens if w["op"] is None and w["b"] is None]
    if len(plain) > 1:
        raise PlanError(f"No row is in both {plain[0]['a']['text']} and {plain[1]['a']['text']}. Ask for a range "
                        f"(“from {plain[0]['a']['text']} to {plain[1]['a']['text']}”) or one at a time")
    for w in q.whens:
        if tcol is None:
            raise PlanError(f"{t0.title} has no date or time column to take “{w['words']}” from")
        # "last month" is the month before the latest date of every table asked about
        members = stack.tables if stack is not None and spec["together"] and tcol[0] == base else [tcol[0]]
        filters.append({**_when_filter(model, w, tcol, members), "column": tcol})
    _check_dates_meet(filters[len(filters) - len(q.whens):])
    for f in filters:
        _check_filter(model, f)
    if filters:
        spec["filters"] = filters
    nums = [r for _, r in numbers]
    top = q.top
    if q.noun_unit and q.big is not None:
        top = 1 if top is None else top            # "hottest day", "biggest month": that one
        spec["superlative"] = q.sup or ""
        spec["noun"] = q.noun_unit
        every = _unit_step(q.noun_unit)
        if every and _several_per(model, t0, tcol, every):
            # many rows a day: the day whose rows add up (or average) to most, not the single biggest row
            spec.update({"recipe": "top", "every": every, "n": top, "bottom": q.big or None,
                         "measure": nums[0] if nums else _first_measure_ref(model, base), "stat": q.stat})
            return _finish(_tidy(spec), q, numbers, groups, items, used, resolve)
        q.rows_named = q.rows_named or [-1]
    if q.rows_named:
        used |= {k for k in q.rows_named if k >= 0}
        spec["_rows_named"] = True
        spec.setdefault("noun", items[q.rows_named[0]][0] if q.rows_named[0] >= 0 else q.noun_unit)
        if top is not None and by_ref is not None and by_ref[0] == base and _role(model, by_ref) == ID:
            by_ref = None                          # "top 5 orders by amount": the orders themselves, not their ids
    if q.part:
        if by_ref is not None or q.every:
            raise PlanError(f"Group by one thing at a time, either the {PART_WORDS[q.part]} or a column")
        spec.update({"recipe": "breakdown", "by_part": q.part, "measure": nums[0] if nums else None, "share": q.share or None})
        if stat and stat not in ("max", "min") or (stat and nums):
            spec["stat"] = stat
        return _finish(spec, q, numbers, groups, items, used, resolve)
    if q.share:
        if by_ref is None and spec.get("filters") and spec["filters"][-1]["op"] == "eq" and not spec["filters"][-1].get("text"):
            by_ref = spec["filters"].pop()["column"]      # "what share agree": the share of every answer to that question
            if not spec["filters"]:
                spec.pop("filters")
        if by_ref is None:
            raise PlanError("A share of what, by what? For example “share of sales by region”")
        spec.update({"recipe": "breakdown", "by": by_ref, "measure": nums[0] if nums else None, "share": True})
        if stat:
            spec["stat"] = stat
        return _finish(spec, q, numbers, groups, items, used, resolve)
    recipe = next((r for r in q.recipes if r == "groups"), q.recipes[0] if q.recipes else None)
    if recipe == "higher":
        # "is mass higher in treated samples": in a study, a test of the groups; "is price higher in North" in a table of
        # orders: price by region, every region, so North can be read against the others
        if experiment(t0, model):
            recipe = "groups"
        else:
            named = [f for f in spec.get("filters") or [] if f["op"] in ("eq", "in") and not f.get("text")
                     and _role(model, f["column"]) in (CATEGORY, ID, TEXT, FLAG)]
            if named and by_ref is None:
                by_ref = named[-1]["column"]
                spec["filters"] = [f for f in spec["filters"] if f is not named[-1]]
                if not spec["filters"]:
                    spec.pop("filters")
                if not stat and nums:
                    stat = default_stat(model, base, nums[0])
                    if stat == "sum":
                        stat = "mean"                          # higher per order, not in total: an average
            recipe = None
    if recipe == "compare" and q.values and len(set(others)) < 2 and by_ref is None:
        if experiment(t0, model):
            recipe = "groups"                              # "compare treated and control": two groups of a study, tested
        else:
            by_ref, recipe = q.values[-1]["column"], None  # "compare Leeds and York": sales by store, those two stores
            nums = nums or [r for r in [_first_measure_ref(model, base)] if r]
            for f in spec.get("filters") or []:
                if f is q.values[-1] and len(f["value"]) < 2:
                    spec["filters"].remove(f)
    if q.sup and not q.stat:
        spec["_sup_only"] = True                      # "highest revenue region": a superlative, no statistic of its own
    spec = _choose_recipe(model, spec, recipe, nums, by_ref, stat, q.every, top, q.bottom, q.big, time_ref, others, tables,
                          stack, base, items)
    return _finish(spec, q, numbers, groups, items, used, resolve)


def _finish(spec, q: _Parts, numbers, groups, items, used, resolve) -> dict[str, Any]:
    """Every word must have been used: one that was not is named, rather than silently left out."""
    in_spec = _refs_in(spec)
    for idx, w, m in q.cols:
        ref = resolve(w, m)
        if ref is not None and tuple(ref) in in_spec:
            used.add(idx)
    for k, r in numbers + groups:
        if tuple(r) in in_spec and k >= 0:
            used.add(k)
    extra = [items[k][0] for k, _ in groups if k >= 0 and k not in used]
    if extra and spec.get("by"):
        raise PlanError(f"Group by one thing at a time, either {spec['by'][1]} or {extra[0]}. "
                        "Two groups at once (a cross-tab) can't be built yet")
    left = [items[k][0] for k in range(len(items)) if k not in used and items[k][1].kind not in ("op",)]
    left += [items[k][0] for k in range(len(items)) if k not in used and items[k][1].kind == "op"]
    if left:
        raise PlanError("I could not fit " + ", ".join(f"“{w}”" for w in dict.fromkeys(left)) + " into the question. "
                        "Try naming one number, a group (“by region”) and a time step (“per month”)")
    return {k: v for k, v in spec.items() if not k.startswith("_")}     # notes between the stages stay out of the spec


def _choose_recipe(model, spec, recipe, numbers, by_ref, stat, every, top, bottom, big, time_ref, others, tables, stack,
                   base, items) -> dict[str, Any]:
    t0 = model.tables[base]
    if recipe in ("compare", "groups") and len({tuple(r) for r in numbers}) >= 2 and len(set(others)) < 2 and \
            by_ref is None and not t0.time and same_kind(model, numbers) and \
            not any(w in ("against", "vs", "versus") for w, _ in items):
        # "compare control and treated": two columns of the same kind of number are two groups, not x against y
        spec.update({"recipe": "groups", "columns": [list(r) for r in dict.fromkeys(tuple(r) for r in numbers)],
                     "test": next((t for w, t in TEST_WORDS.items() if w in " ".join(w for w, _ in items)), None)})
        return _tidy(spec)
    if recipe == "groups":
        return _groups_spec(model, spec, numbers, by_ref, base, items)
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
            if rel is None and st is not None and b in st.tables and experiment(model.tables[a], model):
                # a study kept a file (or sheet) per group: the groups compared on every number, or the ones named
                spec.update({"table": a, "together": True, "recipe": "groups", "by": [st.id, "source"],
                             "measures": [list(r) for r in numbers] or None})
                return _tidy(spec)
            if rel is None and st is not None and b in st.tables:
                # the same table twice (Budget and Actual, this year and last): side by side, table against table
                measure = numbers[0] if numbers else _first_measure_ref(model, a)
                if measure is None:
                    raise PlanError("Name the number to compare")
                spec.update({"table": a, "together": True, "recipe": "breakdown", "by": [st.id, "source"],
                             "measure": measure, "stat": stat or default_stat(model, a, measure)})
                return _tidy(spec)
            if rel is None:
                raise PlanError(f"{model.tables[a].title} and {model.tables[b].title} do not record the same thing over time")
            if rel.tables[0] != a:
                a, b = rel.tables
            col = next((r[1] for r in numbers if r[1] in rel.shared), rel.shared[0])
            spec.update({"recipe": "compare", "table": a, "other": b, "measure": [a, col]})
            spec.pop("together", None)
            return spec
        raise PlanError("Compare what? Name two tables (“compare A and B”), two values (“compare North and South”) "
                        "or two numbers (“y against x”)")
    if recipe == "relationship" or (recipe is None and len(numbers) == 2 and any(w in ("against", "vs", "versus") for w, _ in items)):
        if len(numbers) < 2:
            raise PlanError("Name the two numbers, for example “temperature against pressure”")
        spec.update({"recipe": "relationship", "x": numbers[1], "y": numbers[0]})
        return _tidy(spec)
    if recipe in ("gaps", "describe", "stacked", "linked"):
        spec["recipe"] = recipe
        return _tidy(spec)
    if recipe == "map":
        spec["recipe"] = "map"
        if numbers:
            spec["color_by"] = numbers[0][1][1]
        elif by_ref is not None and by_ref[0] == spec.get("table"):
            spec["color_by"] = by_ref[1]            # "map the villages by status": colour by that column
        return _tidy(spec)
    if recipe == "density":
        spec["recipe"] = "density"
        return _tidy(spec)
    if recipe == "nearest":
        rel = next((r for r in model.relations if r.kind == "near" and base in r.tables), None)
        if rel is None:
            raise PlanError(f"{t0.title} has no second table of places to match against, or no coordinates")
        spec.update({"recipe": "nearest", "table": rel.tables[0], "other": rel.tables[1]})
        return _tidy(spec)
    if recipe == "place":
        rel = next((r for r in model.relations if r.kind == "containment" and base in r.tables), None)
        if rel is None:
            raise PlanError(f"{t0.title} has no table of regions to match its points against")
        spec.update({"recipe": "place", "table": rel.tables[0], "other": rel.tables[1]})
        return _tidy(spec)
    if recipe == "quality":
        spec["recipe"] = "quality"
        return _tidy(spec)
    if recipe == "change":
        spec.update({"recipe": "change", "measure": numbers[0] if numbers else None,
                     "stat": stat, "every": every, "by": by_ref})
        return _tidy(spec)
    if recipe == "explain" and experiment(t0, model) and numbers and not every:
        recipe = "drivers"                                 # in a study, what goes with a number, not what adds up to it
    if recipe == "explain":
        spec.update({"recipe": "explain", "measure": numbers[0] if numbers else None,
                     "stat": stat, "every": every, "by": by_ref})
        return _tidy(spec)
    if recipe == "drivers":
        spec.update({"recipe": "drivers", "target": numbers[0] if numbers else None})
        return _tidy(spec)
    if recipe == "forecast":
        spec.update({"recipe": "forecast", "measure": numbers[0] if numbers else _first_measure_ref(model, base),
                     "every": every, "method": None, "horizon": None})
        return _tidy(spec)
    if recipe == "outliers" and not numbers and experiment(t0, model):
        spec.update({"recipe": "outliers"})                  # "unusual samples": every number, each group on its own
        spec.pop("superlative", None)
        return _tidy(spec)
    if recipe in ("outliers", "distribution"):
        if not numbers:
            raise PlanError(f"Name the number to look at, for example “{recipe} in {_first_measure(model, base)}”")
        spec.update({"recipe": recipe, "measure": numbers[0]})
        return _tidy(spec)
    rows_named = spec.pop("_rows_named", False)
    one_value = spec.pop("_one_value", False)
    if top is not None and every and by_ref is None and not rows_named:
        # "top 3 days by sales", "best week": each day added up (or averaged) first, then the top days
        measure = numbers[0] if numbers else _first_measure_ref(model, base)
        word = next((w for w, m in items if m.kind == "top"), "")
        spec.update({"recipe": "top", "every": every, "n": top, "bottom": bottom or None, "measure": measure,
                     "stat": stat if stat not in ("max", "min") else None,
                     "superlative": word if top == 1 else None, "noun": _unit_noun(every) if top == 1 else None})
        return _tidy(spec)
    ranking_rows = (top is not None and by_ref is None) or (top is not None and (base in tables or rows_named)) or \
                   (big is not None and by_ref is None and (rows_named or base in tables and not one_value) and not every)
    if ranking_rows:                                  # "top 5 orders by amount", "the biggest orders"
        measure = numbers[0] if numbers else _first_measure_ref(model, base)
        if measure is None:
            raise PlanError(f"{t0.title} has no number to rank its rows by")
        noun = str(spec.get("noun") or "")
        one = top is None and big is not None and rows_named and noun and (not noun.endswith("s") or noun.endswith("ss"))
        spec.update({"recipe": "toprows", "n": (1 if one else 10) if top is None else top, "measure": measure,
                     "bottom": (bottom if top is not None else big) or None})
        if one:
            spec["superlative"] = next((w for w, m in items if m.kind == "stat" and w in BIG_WORDS), None)
        if not rows_named:
            spec.pop("noun", None)
        return _tidy(spec)
    spec.pop("noun", None)
    spec.pop("superlative", None)
    if top is not None:
        spec.update({"recipe": "top", "n": top, "by": by_ref, "bottom": bottom or None, "measure": numbers[0] if numbers else None,
                     "stat": stat or ("count" if not numbers else default_stat(model, base, numbers[0]))})
        return _tidy(spec)
    trendy = every or recipe == "trend" or time_ref is not None
    if by_ref is not None and trendy and (t0.time or time_ref):
        spec.update({"recipe": "trend", "measures": numbers[:3], "by": by_ref})
        _time_bits(spec, every, stat, time_ref)
        return _tidy(spec)
    if by_ref is not None and big is not None and spec.pop("_sup_only", False) and numbers and not every:
        # "which region had the lowest revenue": the region whose revenue adds up (or averages) to least, not the
        # smallest single order in each region
        spec.update({"recipe": "top", "n": 1, "by": by_ref, "bottom": big or None, "measure": numbers[0],
                     "stat": default_stat(model, base, numbers[0])})
        spec.pop("superlative", None); spec.pop("noun", None)
        return _tidy(spec)
    spec.pop("_sup_only", None)
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
        spec.update({"recipe": "single", "measure": numbers[0] if numbers else None,
                     "stat": stat or ("count" if not numbers else default_stat(model, base, numbers[0]))})
        return _tidy(spec)
    if spec.get("filters"):
        spec["recipe"] = "rows"
        return _tidy(spec)
    if tables:
        spec["recipe"] = "describe"
        return _tidy(spec)
    raise PlanError("Say what you would like to know: a number to add up or average, a group, or a time step")


TEST_WORDS = {"t test": "welch", "t-test": "welch", "ttest": "welch", "welch": "welch", "mann whitney": "rank",
              "mann-whitney": "rank", "rank test": "rank", "wilcoxon": "rank"}


def _groups_spec(model: DataModel, spec: dict, numbers: list, by_ref: list | None, base: str, items) -> dict[str, Any]:
    """"compare treated and control", "is mass higher in treated samples", "t test inner area": the groups compared, number by
    number. The groups are the column the values named belong to (else the one named after "by", else the table's
    first few-valued column); a filter is kept only when it picks two or more groups out of more."""
    filters = spec.get("filters") or []
    named = [f for f in filters if f["op"] in ("in", "eq") and not f.get("text") and _role(model, f["column"]) in (CATEGORY, ID, TEXT, FLAG)]
    # the groups are the column two or more values are named from ("Control and Drug A"); a value named alone
    # ("for F") limits the rows; with single values only, the first named is the one compared ("higher in Drug A")
    several = [f for f in named if f["op"] == "in"]
    by = by_ref or (several[-1]["column"] if several else named[0]["column"] if named else None)
    if by is None:
        own = [g for g in groupables(model, base) if g[0] == base and _col_distinct(model, g) and 1 < _col_distinct(model, g) <= GROUP_MAX]
        if not own:
            raise PlanError(f"{model.tables[base].title} has no column that splits its rows into groups. Name one (“by …”)")
        by = own[0]
    col = model.table(by[0]).column(by[1]) if not by[0].startswith("stack:") and model.table(by[0]) else None
    every = {str(v) for v in (col.values if col is not None else [])}
    for f in [f for f in named if f["column"] == by]:
        vals = f["value"] if isinstance(f["value"], list) else [f["value"]]
        if len(vals) < 2 or (every and {str(v) for v in vals} >= every):
            filters.remove(f)                              # one group against the rest, or every group: no filter
    if filters:
        spec["filters"] = filters
    else:
        spec.pop("filters", None)
    words = " ".join(w for w, _ in items)
    test = next((t for w, t in TEST_WORDS.items() if w in words), None)
    spec.update({"recipe": "groups", "by": by, "measures": numbers or None, "test": test})
    for k in ("noun", "superlative"):
        spec.pop(k, None)
    return _tidy(spec)


def _col_distinct(model: DataModel, ref: list) -> int:
    if ref[0].startswith("stack:"):
        rel = model.relation(ref[0])
        return len(rel.tables) if rel else 0
    t = model.table(ref[0])
    c = t.column(ref[1]) if t else None
    return c.distinct if c is not None else 0


def _time_bits(spec, every, stat, time_ref) -> None:
    if every:
        spec["every"] = every
    if stat:
        spec["stat"] = stat
    if time_ref is not None:
        spec["time"] = time_ref


def _refs_in(spec: dict) -> set[tuple]:
    out = set()
    for k in ("measure", "by", "x", "y", "time", "target"):
        if spec.get(k):
            out.add(tuple(spec[k]))
    for r in (spec.get("measures") or []) + (spec.get("columns") or []):
        out.add(tuple(r))
    for f in spec.get("filters") or []:
        out.add(tuple(f["column"]))
    return out


def _one_step(current: str | None, new: str) -> str:
    if current and current != new:
        a, b = EVERY_WORDS.get(current, current), EVERY_WORDS.get(new, new)
        raise PlanError(f"The question names two time steps, “{a}” and “{b}”. Ask per {a} or per {b}. "
                        f"Asking for “each {a} of the {b}” across all {b}s together isn't possible yet")
    return new


UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 7 * 86400, "mo": 28 * 86400, "q": 90 * 86400, "y": 365 * 86400}


def _unit_noun(every: str) -> str:
    """The noun for a step like “1h” or “15m” (a two-digit step is not the unit on its own, so the number is
    split off rather than assumed to be a leading “1”)."""
    m = re.fullmatch(r"(\d+)([a-z]+)", every or "")
    unit = m.group(2) if m else every
    return {"s": "second", "m": "minute", "h": "hour", "d": "day", "w": "week", "mo": "month", "q": "quarter",
            "y": "year"}.get(unit, every)


def _unit_step(word: str) -> str | None:
    u = TIME_UNITS.get(word)
    return f"1{u}" if u else None


def _several_per(model: DataModel, t, tcol: list | None, every: str) -> bool:
    """Whether a table has more than one row in some of these time steps (readings every second, several stores a
    day), so that one step has to be added up before it can be ranked."""
    if tcol is None or tcol[0].startswith("stack:"):
        return False
    tt = model.table(tcol[0])
    c = tt.column(tcol[1]) if tt else None
    if c is None:
        return False
    if model.stack_of(tcol[0]) is not None or not c.unique:
        return True
    return c.cadence is not None and c.cadence < 0.999 * UNIT_SECONDS[every[1:]]


def _names_its_table(model: DataModel, words: str, table: str) -> bool:
    t = model.tables.get(table)
    if t is None or model.stack_of(table) is None:
        return False
    names = {t.title.lower(), t.title.lower().replace("_", " "), norm(t.title)}
    return any(words.startswith(n + " ") for n in names)


def _unique_id(model: DataModel, ref: list) -> bool:
    """An id with one row each and too many to read as groups (ticket numbers), unlike a short list of names."""
    t = model.table(ref[0]) if not ref[0].startswith("stack:") else None
    c = t.column(ref[1]) if t else None
    return c is not None and c.role == ID and c.unique and c.distinct > 50


def _groups_named(model, measures, tables) -> bool:
    """Whether the question names a group elsewhere (so a number after 'by' is what is ranked, not a group)."""
    if any(model.tables[t].shape == LOOKUP for t in tables):
        return True
    return any(_role(model, r) in (CATEGORY, ID, TEXT) for _, r, _ in measures)


def _col_kind(model: DataModel, ref: list) -> str:
    t = model.table(ref[0]) if not ref[0].startswith("stack:") else None
    c = t.column(ref[1]) if t else None
    return c.kind if c is not None else ""


# =================================================================== dates
_OP_WORD = {"gt": "after", "lt": "before", "ge": "from", "le": "up to", "between": "from"}
_DAY_END = " 23:59:59.999999"


def _when_filter(model: DataModel, w: dict, tcol: list, members: list[str]) -> dict[str, Any]:
    """A date clause as a filter rule on ``tcol``: "in March" (any year), "since March" (the latest March in the
    data), "from 2024-02-01 to 2024-03-01", "last month" (the month before the data's latest). The rule says in
    its text the dates it chose."""
    op, a, b = w["op"], w["a"], w["b"]
    spans = [x for x in (_extent(model, [n, tcol[1]]) for n in members) if x is not None]
    latest = (min(x[0] for x in spans), max(x[1] for x in spans)) if spans else None
    if b is not None:
        hi = _span_of(b, latest, w)
        if "month" in a and "year" not in a:         # "from November to February": the November before
            y, mo = int(hi[0][:4]), int(hi[0][5:7])
            a = {**a, "year": y if a["month"] <= mo else y - 1}
        lo = _span_of(a, latest, w)
        if lo[0] > hi[1]:
            raise PlanError(f"{lo[2]} comes after {hi[2]}. Put the earlier one first (“from {hi[2]} to {lo[2]}”)")
        return {"op": "between", "value": lo[0], "value2": hi[1], "text": f"from {lo[2]} to {hi[2]}"}
    if op is None and a.get("rel") == "past":
        lo, _, text = _span_of(a, latest, w)     # nothing is later than the latest date, so "after" is enough
        return {"op": "gt", "value": lo, "text": text}
    if op is None:
        if "month" in a and "year" not in a and "day" not in a:
            return {"op": "month", "value": a["month"], "text": f"in {a['text']}"}
        if "year" in a and "month" not in a:
            return {"op": "year", "value": a["year"], "text": f"in {a['year']}"}
        lo, hi, text = _span_of(a, latest, w)
        if "date" in a and lo == hi:
            return {"op": "eq", "value": lo, "text": f"at {text}"}
        return {"op": "between", "value": lo, "value2": hi,
                "text": ("on " if "date" in a or "day" in a else "in " if "rel" not in a else "") + text}
    lo, hi, text = _span_of(a, latest, w)
    if op == "between":
        op = "ge"                                    # "between March" with no other end: from March on
    value = hi if op in ("gt", "le") else lo         # "after March" is after its last moment, "before" its first
    return {"op": op, "value": value, "text": f"{_OP_WORD[op]} {text}"}


def _date_span(f: dict) -> tuple[str, str, bool] | None:
    """(from, to, an end left out?) of a date rule, as text that sorts like the dates; None for a month of any year."""
    op, v = f["op"], str(f.get("value"))
    return {"gt": (v, "~", True), "ge": (v, "~", False), "lt": ("", v, True), "le": ("", v, False),
            "between": (v, str(f.get("value2")), False), "eq": (v, v, False),
            "year": (f"{v}-01-01", f"{v}-12-31{_DAY_END}", False)}.get(op)


def _check_dates_meet(rules: list[dict]) -> None:
    """Two dates that no row can be in at once ("before February and after March") are refused, not emptied."""
    for i, f in enumerate(rules):
        for g in rules[i + 1:]:
            a, b = _date_span(f), _date_span(g)
            if a is None or b is None or f["column"] != g["column"]:
                continue
            lo, hi = max(a[0], b[0]), min(a[1], b[1])
            if lo > hi or (lo == hi and (a[2] or b[2])):
                raise PlanError(f"No row is both {f['text']} and {g['text']}. Ask for one at a time")


def _extent(model: DataModel, tcol: list) -> tuple[datetime, datetime] | None:
    """The earliest and latest dates of a time column: what "last month" and a month without its year are taken from."""
    t = model.table(tcol[0])
    c = t.column(tcol[1]) if t else None
    if c is None:
        return None
    lo, hi = (t.start, t.end) if tcol[1] == t.time else (c.minimum, c.maximum)
    lo, hi = _as_datetime(lo), _as_datetime(hi)
    return (lo, hi) if lo is not None and hi is not None else None


def _as_datetime(v: Any) -> datetime | None:
    from datetime import date
    if isinstance(v, datetime):
        return v.replace(tzinfo=None)
    if isinstance(v, date):
        return datetime(v.year, v.month, v.day)
    return None


def _month_year(month: int, extent: tuple[datetime, datetime]) -> int:
    """The year of a month named without one: the latest such month that starts at or before the data's latest
    date ("since December" on data from January to April 2024 is since December 2023)."""
    last = extent[1]
    return last.year if month <= last.month else last.year - 1


def _span_of(p: dict, extent: tuple[datetime, datetime] | None, w: dict) -> tuple[str, str, str]:
    """(first moment, last moment, how it is said) of a point in time. ``extent`` is the data's first and last date."""
    if "date" in p:
        d = p["date"]
        if "/" in d:
            x, z, y = (int(v) for v in d.split("/"))
            y += 2000 if y < 100 else 0
            raise PlanError(f"“{d}” could be day/month or month/day. Write it as year-month-day, for example "
                            f"{y}-{z:02d}-{x:02d} or {y}-{x:02d}-{z:02d}")
        return (d, d + _DAY_END, d) if len(d) <= 10 else (d, d, d)
    if "year" in p and "month" not in p:
        y = p["year"]
        return f"{y}-01-01", f"{y}-12-31{_DAY_END}", str(y)
    if extent is None:
        raise PlanError(f"There are no dates to take “{w['words']}” from")
    latest = extent[1]
    if "day" in p:
        y = p.get("year") or _month_year(p["month"], extent)
        try:
            d = datetime(y, p["month"], p["day"])
        except ValueError:
            raise PlanError(f"{p['text']} {y} is not a date") from None
        return _fmt(d), _fmt(d + timedelta(days=1) - timedelta(microseconds=1), True), f"{d:%Y-%m-%d}"
    if "month" in p:
        mo = p["month"]
        y = p.get("year") or _month_year(mo, extent)
        start = datetime(y, mo, 1)
        end = _shift(start, "mo", 1) - timedelta(microseconds=1)
        return _fmt(start), _fmt(end, True), f"{start:%B} {y}"
    unit, n = p["unit"], p["n"]
    if n is not None and n <= 0:
        raise PlanError(f"“{p['text']}” is no time at all. Ask for one or more")
    try:
        if p["rel"] == "past":                       # "the last 7 days": that long up to the latest date
            start = _shift(latest, unit, -(n or 1))
            return _fmt(start), _fmt(latest, True), f"in the {p['text']} (after {_fmt(start)})"
        start = _floor(latest, unit)                 # "this month", "last week": calendar periods
        if p["rel"] == "last":
            start = _shift(start, unit, -1)
        end = _shift(start, unit, 1) - timedelta(microseconds=1)
        return _fmt(start), _fmt(end, True), f"{p['text']} ({_period_text(start, unit)})"
    except (OverflowError, ValueError):
        # a typed span can run off the end of the calendar (year 1 or 9999); refuse it in plain words
        raise PlanError(f"“{p['text']}” reaches past the calendar. Ask for a shorter span") from None


def _floor(t: datetime, unit: str) -> datetime:
    if unit == "m":
        return t.replace(second=0, microsecond=0)
    if unit == "h":
        return t.replace(minute=0, second=0, microsecond=0)
    day = t.replace(hour=0, minute=0, second=0, microsecond=0)
    if unit == "d":
        return day
    if unit == "w":
        return day - timedelta(days=day.weekday())
    if unit == "mo":
        return day.replace(day=1)
    if unit == "q":
        return day.replace(month=(day.month - 1) // 3 * 3 + 1, day=1)
    return day.replace(month=1, day=1)


def _shift(t: datetime, unit: str, n: int) -> datetime:
    if unit in ("mo", "q", "y"):
        months = n * {"mo": 1, "q": 3, "y": 12}[unit]
        k = t.year * 12 + t.month - 1 + months
        y, mo = divmod(k, 12)
        last = (datetime(y + (mo == 11), (mo + 1) % 12 + 1, 1) - timedelta(days=1)).day
        return t.replace(year=y, month=mo + 1, day=min(t.day, last))
    return t + timedelta(seconds=n * UNIT_SECONDS[unit])


def _fmt(t: datetime, end: bool = False) -> str:
    if end:
        return f"{t:%Y-%m-%d %H:%M:%S.%f}"
    return f"{t:%Y-%m-%d}" if t == t.replace(hour=0, minute=0, second=0, microsecond=0) else f"{t:%Y-%m-%d %H:%M:%S}"


def _period_text(start: datetime, unit: str) -> str:
    if unit == "y":
        return f"{start.year}"
    if unit == "q":
        return f"Q{(start.month - 1) // 3 + 1} {start.year}"
    if unit == "mo":
        return f"{start:%B %Y}"
    if unit == "w":
        return f"week from {start:%Y-%m-%d}"
    if unit == "d":
        return f"{start:%Y-%m-%d}"
    return f"from {start:%Y-%m-%d %H:%M}"


def _check_filter(model: DataModel, f: dict) -> None:
    ref = f["column"]
    if ref[0].startswith("stack:"):
        return
    t = model.table(ref[0])
    c = t.column(ref[1]) if t else None
    if c is None:
        return
    if c.kind == "date/time" and f["op"] in ("gt", "lt", "ge", "le", "between", "outside", "eq", "ne") and not f.get("text") \
            and not f.get("column2") and any(isinstance(f.get(k), (int, float)) and not isinstance(f.get(k), bool)
                                             for k in ("value", "value2")):
        raise PlanError(f"{c.name} holds dates, so it can't be compared with a plain number. Give a date "
                        f"(“{c.name} after 2024-03-01”), a month or a year")
    if f["op"] in ("gt", "lt", "ge", "le", "between", "outside") and c.kind == "text":
        raise PlanError(f"{c.name} holds text (for example {_example(model, ref)}), so it can't be compared as a number. "
                        f"Check the {c.name} column in the file (a few notes among numbers are read as blanks)")


def _example(model: DataModel, ref: list) -> str:
    t = model.table(ref[0])
    c = t.column(ref[1]) if t else None
    vals = [v for v in (c.values if c is not None and c.values else []) if v is not None]
    return f"“{vals[0]}”" if vals else "words"


def _first_measure_ref(model: DataModel, base: str) -> list | None:
    ms = ordered_measures(model.tables[base])
    return [base, ms[0].name] if ms else None


def _name_column(model: DataModel, table: str, base: str) -> list | None:
    """The column that names a lookup table's rows (a customer's name), else its key."""
    g = [r for r in groupables(model, base) if r[0] == table]
    t = model.tables[table]
    # a name shared by two customers would put their rows together: only a name every row has its own of
    texts = [r for r in g if _role(model, r) in (ID, TEXT) and (t.column(r[1]) is None or t.column(r[1]).unique)]
    named = [r for r in texts if set(name_words(r[1])) & {"name", "title", "label", "description", "product", "customer"}]
    if named or texts:
        return (named or texts)[0]
    if any(_role(model, r) in (ID, TEXT) for r in g):
        key = next((c for c in t.columns if c.role == ID and c.unique), None)
        if key is not None:
            return [table, key.name]
    if g:
        return g[0]
    key = next((c for c in t.columns if c.role == ID), None)
    return [table, key.name] if key is not None else None


def _readable(model: DataModel, ref: list | None, m: Meaning, base: str) -> list | None:
    """A group named in short by an id ("per product" for product_id) is shown by the name of what it links to
    (the product's name), when the id is a key into a lookup table that has one."""
    if ref is None or ref not in m.aliases or _role(model, ref) != ID:
        return ref
    link = next((r for r in model.relations if r.kind == "link" and r.tables[0] == ref[0] and r.left_on == ref[1]
                 and r.cardinality != "many-to-many"), None)
    if link is None or model.tables[link.tables[1]].shape != LOOKUP:
        return ref
    g = _name_column(model, link.tables[1], base)
    return g if g is not None and g[0] == link.tables[1] and _col_kind(model, g) == "text" else ref


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
    if t.shape == LOOKUP:
        g = _name_column(model, table, base)
        if g is not None and (g[0] == base or g[0] in reachable(model, base)):
            return g
    link = next((r for r in model.relations if r.kind == "link" and r.tables == [base, table]), None)
    return [base, link.left_on] if link is not None else None


def _own_key(model: DataModel, table: str, words: str) -> list | None:
    """The id column of ``table`` whose name starts with the words (sensor -> sensor_id), if there is one."""
    want = norm(words)
    want_s = want[:-1] if want.endswith("s") else want
    for c in model.tables[table].columns:
        w = name_words(c.name)
        if c.role in (ID, CATEGORY) and len(w) > 1 and norm("".join(w[:-1])) in (want, want_s):
            return [table, c.name]
    return None


def _tidy(spec: dict) -> dict:
    return {k: v for k, v in spec.items() if v is not None and v != []}


def _filter(model: DataModel, items: list[Item], i: int) -> tuple[list[dict] | None, int]:
    """col op num [and num] -> filter rules, and how many items they used: "price above 30", "price between 10 and
    20", "price from 10 to 20", "price not between 10 and 20", "customer_id is 1, 2 or 3" (any of them). A value that is
    not the column's is left to its own column ("quantity except North")."""
    col = items[i][1]
    op_words, op = items[i + 1][0], items[i + 1][1].value
    j = i + 2
    if j >= len(items):
        return None, 1
    val = items[j][1]
    ref = col.refs[0]
    two = val.kind == "num" and j + 2 < len(items) and items[j + 2][1].kind == "num"
    if op in ("between", "outside"):
        if two and items[j + 1][1].kind == "and":
            _check_range(items, j)
            return [{"column": ref, "op": op, "value": val.value, "value2": items[j + 2][1].value}], j + 3 - i
        return None, 1
    if op == "ge" and op_words == "from" and two and items[j + 1][0] in RANGE_TO | {"and"}:
        _check_range(items, j)
        return [{"column": ref, "op": "between", "value": val.value, "value2": items[j + 2][1].value}], j + 3 - i
    if val.kind == "num":
        vals, k = [val.value], j + 1
        while op in ("eq", "ne"):                         # "is 1, 2 or 3": the commas are gone, "and"/"or" may be there
            k2 = k + 1 if k < len(items) and items[k][1].kind in ("and", "or") else k
            if k2 >= len(items) or items[k2][1].kind != "num":
                break
            vals.append(items[k2][1].value); k = k2 + 1
        if len(vals) == 1:
            return [{"column": ref, "op": op, "value": val.value}], j + 1 - i
        if op == "eq":
            return [{"column": ref, "op": "in", "value": vals}], k - i
        return [{"column": ref, "op": "ne", "value": v} for v in vals], k - i
    if val.kind == "col" and op in ("gt", "lt", "ge", "le", "eq", "ne"):
        return [{"column": ref, "op": op, "column2": val.refs[0]}], j + 1 - i   # stock below reorder level
    return None, 1


def _check_range(items: list[Item], j: int) -> None:
    """A range of numbers written backwards ("between 20 and 10") is refused, as a backwards range of dates is,
    rather than answered with nothing."""
    (lo, a), (hi, b) = items[j], items[j + 2]
    if a.value > b.value:
        raise PlanError(f"{lo} is more than {hi}. Put the smaller one first (“{hi} {items[j + 1][0]} {lo}”)")


def _role(model: DataModel, ref: list) -> str:
    if ref[0].startswith("stack:"):
        return CATEGORY
    t = model.table(ref[0])
    c = t.column(ref[1]) if t else None
    return c.role if c is not None else ""


def _first_measure(model: DataModel, base: str) -> str:
    t = model.tables[base]
    return t.measures[0].name if t.measures else "a column"


def _base_table(model: DataModel, tables: list[str], cols, by, filters) -> str:
    """The table whose rows the question is about: one from which every table it mentions can be reached by
    links (orders reach customers and products, not the other way), preferring a table it names, then one
    holding the numbers asked about, then the biggest, then project order."""
    order = list(model.tables)
    mentioned: list[str] = []
    measure_tables: list[str] = []
    for _, refs in cols:
        mentioned += [r[0] for r in refs]
        measure_tables += [r[0] for r in refs if _role(model, r) == MEASURE]
    if by:
        mentioned += [r[0] for r in by[1]]
    mentioned += [f["column"][0] for f in filters]
    mentioned = [m for m in mentioned if not m.startswith("stack:")]
    # a column phrase can mean columns of several tables: one of them reachable is enough
    groups = [[r[0] for r in refs] for _, refs in cols] + ([[r[0] for r in by[1]]] if by else [])
    groups += [[f["column"][0]] for f in filters]

    def covers(t: str) -> bool:
        reach = set(reachable(model, t)) | {t} | set(_stack_members(model, t))
        return all(any(x in reach or x.startswith("stack:") for x in g) for g in groups)

    candidates = [t for t in order if covers(t)]
    if not candidates:
        if tables:
            return tables[0]
        involved = set(mentioned)
        mm = next((r for r in model.relations if r.kind == "link" and r.cardinality == "many-to-many" and set(r.tables) <= involved), None)
        if mm is not None:
            a, b = (model.tables[x].title for x in mm.tables)
            raise PlanError(f"{a} and {b} share {mm.left_on}, but it repeats in both. Putting them side by side would "
                            f"repeat rows and make the totals wrong. Ask about one of them, or remove the duplicates first.")
        raise PlanError("Those columns are in tables that are not linked to each other")
    for t in tables:
        if t not in candidates and not any(t in reachable(model, c) for c in candidates):
            raise PlanError(f"{model.tables[t].title} is not linked to the table that holds "
                            f"{', '.join(dict.fromkeys(r[1] for _, rs in cols for r in rs)) or 'those columns'}, "
                            "so it cannot be counted that way")
    named = [t for t in tables if t in candidates]
    if named:
        return named[0]

    def rank(t: str):
        tb = model.tables[t]
        return (0 if t in measure_tables else 1, 0 if t in mentioned else 1, -(tb.rows or tb.sampled or 0), order.index(t))
    return sorted(candidates, key=rank)[0]


def _fact_of(model: DataModel, lookup: str, timed: bool) -> str | None:
    """The table whose rows point at a lookup (orders for customers): the biggest one linked to it, with a date
    column when ``timed``."""
    order = list(model.tables)
    found = [t for t in order if t != lookup and model.tables[t].shape != LOOKUP and lookup in reachable(model, t)
             and (model.tables[t].time or not timed)]
    return min(found, key=lambda t: (-(model.tables[t].rows or model.tables[t].sampled or 0), order.index(t)), default=None)


def _stack_members(model: DataModel, t: str) -> list[str]:
    st = model.stack_of(t)
    return list(st.tables) if st is not None else []


def _resolve(model: DataModel, words: str, m: Meaning, base: str, out: Asked) -> list | None:
    """The column a phrase means: one it names in full before one it names in short ("product" is the product
    column before product_id); among those, the one in the table the question is about, else one reachable from
    it by links; the same column in stacked tables counts as one; otherwise the first, noted as a choice."""
    refs = m.refs
    if len(refs) == 1:
        return refs[0]
    stack = model.stack_of(base)
    members = set(stack.tables) if stack is not None else {base}
    reach = set(reachable(model, base)) | members
    full = [r for r in refs if r not in m.aliases and r[0] in reach]
    pool = full or refs
    own = [r for r in pool if r[0] == base]
    if own:
        return own[0]
    if all(r[0] in members for r in pool):
        return pool[0]
    chosen = next((r for r in pool if r[0] in reach), pool[0])
    others = [r for r in refs if r != chosen]
    if others and not any(a["text"] == words and a["chose"] == chosen for a in out.ambiguous):
        out.ambiguous.append({"text": words, "chose": chosen,
                              "choices": [{"label": f"{r[1]} ({model.tables[r[0]].title})", "value": r} for r in others]})
    return chosen


def words_for(model: DataModel, items: list[Item]) -> list[tuple[str, list[list]]]:
    """The (word, column refs) of every column phrase read from the question, for checking which words name a
    table's own rows rather than a column."""
    return [(w, m.refs) for _, (w, m) in enumerate(items) if m.kind in ("col", "value") and m.refs]


def _named_by(items: list[tuple[str, list[list]]], column: str) -> bool:
    """True when some word in the question names this column, so dropping it loses nothing: only reached for a
    mappable table's columns, where a word like 'villages' names the points (village) as much as a column."""
    return any(ref and ref[1] == column for _, refs in items for ref in refs)


def _containment_relation(model: DataModel, tables: list[str], cols) -> "object | None":
    """The containment relation (points inside polygons) connecting the tables a question names, or None."""
    mentioned = {r[0] for _, refs in cols for r in refs if not r[0].startswith("stack:")} | set(tables)
    for rel in model.relations:
        if rel.kind == "containment" and set(rel.tables) <= mentioned:
            return rel
    return None


def _near_relation(model: DataModel, tables: list[str], cols) -> "object | None":
    """The near relation that connects the tables a question names ('nearest clinic to each village'), or None.
    ``cols`` is [(words, refs), …]."""
    mentioned = {r[0] for _, refs in cols for r in refs if not r[0].startswith("stack:")} | set(tables)
    for rel in model.relations:
        if rel.kind == "near" and set(rel.tables) <= mentioned:
            return rel
    return None
