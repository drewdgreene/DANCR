"""Profile a table and suggest how several tables relate, for the guided build wizard.

Pure core, no Qt. Profiling is what lets the wizard ask two or three questions instead of
twenty: it works out the types, the likely keys, the measures and the categories, and it
proposes which files can be linked (a shared key) or stacked (the same shape). Everything
here is best-effort and cheap: it reads a bounded sample, never the whole file, for per-column
facts, and counts rows in one streaming pass.
"""
from __future__ import annotations

import difflib
import re
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

from .expr import _kind_of_dtype, NUM, TIME, STR, BOOL
from .nodes.load import scan_file
from .registry import Ctx

SAMPLE_ROWS = 100_000          # rows read for per-column facts
MAX_VALUES = 5_000             # distinct values kept per column, for link suggestions
LINK_MIN_SCORE = 0.60
STACK_MIN_SIMILARITY = 0.70
KEY_SUFFIXES = ("id", "key", "code", "no", "number", "ref", "sku", "uuid", "guid")


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    kind: str
    nulls: int = 0
    null_pct: float = 0.0
    distinct: int = 0
    samples: list[Any] = field(default_factory=list)
    values: list[Any] = field(default_factory=list)     # distinct values (as text), for link overlap
    value_cut: int | None = None                         # values kept are those whose _vhash <= this; None = all
    minimum: Any = None
    maximum: Any = None
    constant: bool = False
    key_like: bool = False
    looks_like_key: bool = False
    measure: bool = False
    category: bool = False

    @property
    def join_candidate(self) -> bool:
        """A column worth testing as a link key: an id-like or key-like column, or a text column."""
        return self.distinct >= 2 and (self.key_like or self.looks_like_key or self.category or self.kind == STR)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "dtype": self.dtype, "kind": self.kind, "nulls": self.nulls,
                "null_pct": round(self.null_pct, 3), "distinct": self.distinct, "samples": self.samples,
                "constant": self.constant, "key_like": self.key_like, "looks_like_key": self.looks_like_key,
                "measure": self.measure, "category": self.category}


@dataclass
class TableProfile:
    path: str
    name: str
    rows: int
    columns: list[ColumnProfile] = field(default_factory=list)
    sampled: int = 0
    time_column: str | None = None
    key_columns: list[str] = field(default_factory=list)
    numeric_columns: list[str] = field(default_factory=list)
    category_columns: list[str] = field(default_factory=list)
    measures: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    def column(self, name: str) -> ColumnProfile | None:
        return next((c for c in self.columns if c.name == name), None)

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "name": self.name, "rows": self.rows, "sampled": self.sampled,
                "time_column": self.time_column, "key_columns": self.key_columns,
                "numeric_columns": self.numeric_columns, "category_columns": self.category_columns,
                "measures": self.measures, "problems": self.problems,
                "columns": [c.to_dict() for c in self.columns]}


def profile_table(path: str | Path, name: str | None = None) -> TableProfile:
    """Read a sample of one file and describe it. Raises ValueError with a plain message if it cannot be read."""
    p = Path(path).expanduser().resolve()
    ctx = Ctx(p.parent, "profile", "profile", preview=True)
    lf, _messages = scan_file(ctx, {"path": str(p), "parse_dates": True})
    schema = lf.collect_schema()
    rows = int(lf.select(pl.len()).collect(engine="streaming")[0, 0])
    sample = lf.head(SAMPLE_ROWS).collect(engine="streaming")
    sampled = sample.height
    cols: list[ColumnProfile] = []
    for col, dt in schema.items():
        s = sample[col]
        kind = _kind_of_dtype(dt)
        nulls = int(s.null_count())
        distinct = int(s.n_unique())
        looks_key = _looks_like_key(col)
        cp = ColumnProfile(
            name=col, dtype=str(dt), kind=kind, nulls=nulls,
            null_pct=(nulls / sampled) if sampled else 0.0, distinct=distinct,
            samples=_display_values(s), minimum=_extreme(s, "min"), maximum=_extreme(s, "max"),
            constant=(distinct == 1 and sampled > 0),
            key_like=(sampled >= 5 and nulls == 0 and distinct == sampled and (kind == STR or dt.is_integer())),
            looks_like_key=looks_key,
            category=(kind in (STR, BOOL) and 1 < distinct <= 50 and distinct < sampled and nulls / sampled <= 0.5 if sampled else False),
        )
        cp.measure = kind == NUM and distinct > 1 and not cp.key_like and not looks_key
        if cp.join_candidate:
            cp.values, cp.value_cut = _capped_values(s)
        cols.append(cp)
    tp = TableProfile(path=str(p), name=name or p.stem, rows=rows, columns=cols, sampled=sampled)
    tp.time_column = next((c.name for c in cols if c.kind == TIME and c.distinct > 1), None)
    tp.key_columns = [c.name for c in cols if c.key_like]
    tp.numeric_columns = [c.name for c in cols if c.kind == NUM]
    tp.category_columns = [c.name for c in cols if c.category]
    tp.measures = [c.name for c in cols if c.measure]
    tp.problems = _problems(tp)
    return tp


def profile_files(paths: list[str | Path]) -> tuple[list[TableProfile], list[tuple[str, str]]]:
    """Profile each file, skipping (with a reason) any that cannot be read, so one bad file never
    blocks the others."""
    profiles: list[TableProfile] = []
    skipped: list[tuple[str, str]] = []
    for p in paths:
        try:
            profiles.append(profile_table(p))
        except Exception as e:  # noqa: BLE001 - a file we cannot read is reported, not raised
            skipped.append((str(p), str(e).strip() or type(e).__name__))
    return profiles, skipped


# ------------------------------------------------------------------- helpers
def _display_values(s: pl.Series, n: int = 8) -> list[Any]:
    vals = s.drop_nulls().unique(maintain_order=True).head(n).to_list()
    return [_clean(v) for v in vals]


def _vhash(v: str) -> int:
    """A hash that is the same for the same text in every table and every run."""
    return zlib.crc32(v.encode("utf-8", "surrogatepass"))


def _capped_values(s: pl.Series, cap: int = MAX_VALUES) -> tuple[list[str], int | None]:
    """At most ``cap`` distinct values, chosen by hash rather than by position in the file: two tables keep
    the same values out of the ones they share, so their overlap is measured fairly even when one file is
    sorted and the other is not. Returns (values, the largest hash kept, or None when every value is kept)."""
    vals = s.drop_nulls().cast(pl.Utf8).unique().to_list()
    if len(vals) <= cap:
        return vals, None
    kept = sorted(vals, key=_vhash)[:cap]
    return kept, _vhash(kept[-1])


def _comparable(a: "ColumnProfile", b: "ColumnProfile") -> tuple[set[str], set[str]]:
    """The two columns' values restricted to the same hash range, so a value missing from one side was
    really not there rather than cut off by the cap."""
    cuts = [c for c in (a.value_cut, b.value_cut) if c is not None]
    if not cuts:
        return set(a.values), set(b.values)
    cut = min(cuts)
    return {v for v in a.values if _vhash(v) <= cut}, {v for v in b.values if _vhash(v) <= cut}


def _clean(v: Any) -> Any:
    if isinstance(v, float) and v != v:
        return None
    return v


def _extreme(s: pl.Series, how: str) -> Any:
    try:
        v = getattr(s, how)()
    except Exception:  # noqa: BLE001 - some dtypes have no min/max
        return None
    return _clean(v)


def _problems(tp: TableProfile) -> list[str]:
    out: list[str] = []
    blank = [c.name for c in tp.columns if c.null_pct >= 0.5 and tp.sampled]
    if blank:
        out.append(f"{len(blank)} column{'s' if len(blank) != 1 else ''} mostly blank: {', '.join(blank[:4])}")
    const = [c.name for c in tp.columns if c.constant and c.null_pct < 0.5]
    if const:
        out.append(f"never changes: {', '.join(const[:4])}")
    return out


# ------------------------------------------------------------------- relations
@dataclass
class LinkSuggestion:
    left: str          # table path
    left_col: str
    right: str
    right_col: str
    score: float
    match_pct: float   # how many of the smaller side's values are found on the other side

    def to_dict(self) -> dict[str, Any]:
        return {"left": self.left, "left_col": self.left_col, "right": self.right,
                "right_col": self.right_col, "score": self.score, "match_pct": self.match_pct}


@dataclass
class StackSuggestion:
    paths: list[str]
    columns: list[str]
    similarity: float

    def to_dict(self) -> dict[str, Any]:
        return {"paths": self.paths, "columns": self.columns, "similarity": self.similarity}


def suggest_links(profiles: list[TableProfile]) -> list[LinkSuggestion]:
    """Pairs of columns in different tables that look like the same key (name + value overlap)."""
    out: list[LinkSuggestion] = []
    for i, a in enumerate(profiles):
        for b in profiles[i + 1:]:
            for ca in a.columns:
                if not ca.join_candidate or not ca.values:
                    continue
                for cb in b.columns:
                    if not cb.join_candidate or not cb.values:
                        continue
                    name = _name_similarity(ca.name, cb.name)
                    if name < 0.34:
                        continue
                    va, vb = _comparable(ca, cb)
                    overlap = _overlap(va, vb)
                    if overlap < 0.5:
                        continue
                    score = 0.6 * name + 0.4 * overlap
                    if score >= LINK_MIN_SCORE:
                        pct = round(containment_percent(va, vb), 1)
                        out.append(LinkSuggestion(a.path, ca.name, b.path, cb.name, round(score, 3), pct))
    out.sort(key=lambda s: -s.score)
    return out[:12]


def suggest_stacks(profiles: list[TableProfile]) -> list[StackSuggestion]:
    """Groups of tables with the same shape (the same columns), so they can be appended."""
    out: list[StackSuggestion] = []
    used: set[str] = set()
    for i, a in enumerate(profiles):
        if a.path in used:
            continue
        members = [a]
        best = 1.0
        for b in profiles[i + 1:]:
            if b.path in used:
                continue
            sim = _column_similarity(a, b)
            if sim >= STACK_MIN_SIMILARITY:
                members.append(b)
                best = min(best, sim)
        if len(members) >= 2:
            for m in members:
                used.add(m.path)
            out.append(StackSuggestion([m.path for m in members], [c.name for c in members[0].columns], round(best, 3)))
    return out


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _words(name: str) -> list[str]:
    """'CustomerID' -> customer, id; 'order_no' -> order, no; 'Amount paid' -> amount, paid."""
    return [w.lower() for w in re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+", str(name))]


def _looks_like_key(name: str) -> bool:
    """A name whose last word is a key word (customer_id, OrderNo, sku), not one that merely ends in those
    letters (Amount paid, valid, Humid). A bare 'id' or 'sku' counts; a bare 'number' or 'no' does not."""
    w = _words(name)
    if not w or w[-1] not in KEY_SUFFIXES:
        return False
    return len(w) > 1 or w[-1] in ("id", "key", "sku", "uuid", "guid", "code", "ref")


def _stem(name: str) -> str:
    """The name without its key word: customer_id -> customer."""
    w = _words(name)
    return "".join(w[:-1]) if len(w) > 1 and w[-1] in KEY_SUFFIXES else _norm(name)


def _name_similarity(a: str, b: str) -> float:
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    sa, sb = _stem(a), _stem(b)
    if sa != na and sb != nb:          # both are key-suffixed: compare the stems (customer vs product)
        return difflib.SequenceMatcher(None, sa, sb).ratio()
    if sa == sb:
        return 0.95
    if sa in sb or sb in sa:
        return 0.85
    return difflib.SequenceMatcher(None, na, nb).ratio()


def match_percent(a: "ColumnProfile", b: "ColumnProfile") -> float:
    """The 'match %' shown for a link between two profiled columns."""
    return containment_percent(*_comparable(a, b))


def _overlap(a_values: Any, b_values: Any) -> float:
    """How alike two value sets are: half containment (the smaller side found in the larger) and
    half Jaccard, so a tiny set that happens to sit inside a big one does not score as a match."""
    a, b = set(a_values), set(b_values)
    if not a or not b:
        return 0.0
    inter = len(a & b)
    probe = min(len(a), len(b))
    containment = inter / probe
    jaccard = inter / len(a | b)
    return 0.5 * containment + 0.5 * jaccard


def containment_percent(a_values: Any, b_values: Any) -> float:
    """The intuitive 'match %' for a link: how many of the smaller side's distinct values appear on
    the other side (0-100). '100% match' means every value on the smaller side is found in the other."""
    a, b = set(a_values), set(b_values)
    if not a or not b:
        return 0.0
    probe, other = (a, b) if len(a) <= len(b) else (b, a)
    return 100.0 * sum(1 for v in probe if v in other) / len(probe)


def _column_similarity(a: TableProfile, b: TableProfile) -> float:
    ca = {_norm(c.name) for c in a.columns}
    cb = {_norm(c.name) for c in b.columns}
    if not ca or not cb:
        return 0.0
    inter = len(ca & cb)
    union = len(ca | cb)
    return inter / union if union else 0.0
