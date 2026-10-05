"""Understand the tables in a project: what each column means, what each table is, how tables relate.

This is what lets DANCR offer answers instead of asking questions. It reads the *output of steps* (never a
file on the side), so what it describes is exactly what a step downstream will see: the loader's own
settings (sheet, header row, day-first dates) are part of what is read.

Two passes, both deterministic:

- ``understand(...)`` reads a bounded sample of each table (a spread over the whole result when a step has
  already run, else its first rows) and works out column roles, table shapes and candidate relations.
- ``deepen(...)`` then reads every row of each table in one streaming pass (a second only to count keys exactly
  in a source not run yet) and one pass per link, for the facts a sample cannot give: exact row counts and time
  spans, whether a key really is unique, and how many keys really match.
  Anything built from the model (an answer) is always planned on the deep model.

Pure core: no Qt, no execution beyond reading step outputs.
"""
from __future__ import annotations

import difflib
import math
import re
import zlib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable

import polars as pl

from .expr import kind_of_dtype, NUM, TIME, STR, BOOL
from .registry import registry
from .units import header_parts, quantity_of, unit_from_name

SAMPLE_ROWS = 100_000          # rows read per table for column facts
CADENCE_ROWS = 20_000          # consecutive rows read to measure how often rows arrive
MAX_VALUES = 5_000             # distinct values kept per key-like column, for link overlap
CATEGORY_MAX = 50              # a text column with at most this many values (and repeats) is a category
LINK_MIN_SCORE = 0.60
LINKS_PER_PAIR = 3             # alternatives kept per pair of tables (the first is used; the others are offered)
STACK_MIN_SIMILARITY = 0.70
EXACT_KEY_ROWS = 5_000_000     # up to this many rows, key uniqueness is counted exactly; above, estimated
EXACT_MATCH_ROWS = 20_000_000  # up to this many rows, link matches are counted exactly
KEY_SUFFIXES = ("id", "key", "code", "no", "number", "ref", "sku", "uuid", "guid")
# numbers that name something rather than count it: nobody adds up zip codes or phone numbers. The name must be
# about the code itself (zip, zip_code, phone_number, customer_phone, account_no), not merely hold the word:
# account_balance and phone_calls are amounts
CODE_WORDS = {"zip", "zipcode", "postcode", "postal", "phone", "tel", "telephone", "mobile", "fax", "account", "acct",
              "iban", "ssn", "isbn", "ean", "upc", "barcode", "pin"}
CODE_ENDINGS = {"code", "no", "nr", "num", "number"}
CALENDAR_WORDS = ("year", "month", "quarter", "week", "weekday", "day", "hour")

# column roles
TIME_ROLE, ID, CATEGORY, MEASURE, FLAG, TEXT, CONSTANT, BLANK = (
    "time", "id", "category", "measure", "flag", "text", "constant", "blank")
# table shapes
SERIES, LOOKUP, EVENTS, TABLE = "series", "lookup", "events", "table"


@dataclass
class Column:
    name: str
    dtype: str
    kind: str                       # number | text | true/false | date/time
    role: str
    label: str = ""                 # what the person sees (column registry label, else the name)
    unit: str = ""                  # from the column registry, else from the header ("Pressure (bar)")
    null_pct: float = 0.0
    distinct: int = 0               # in the sample (or exactly / estimated over every row after deepen)
    unique: bool = False            # every filled value is different
    unique_exact: bool = False      # ... counted over every row, not just the sample
    values: list[Any] = field(default_factory=list)   # a category's values, most frequent first
    minimum: Any = None
    maximum: Any = None
    cadence: float | None = None    # time: the typical seconds between one row and the next
    regular: bool = False           # time: most steps are that typical step
    spellings: dict[str, str] = field(default_factory=dict)   # category: a variant -> the spelling most rows use
    abbrev: str = ""                # the short name its header gives it: 'IA' in 'inner area (IA) cm2'
    quantity: str = ""              # what its unit measures: mass, area, mass per area …
    derived: dict[str, Any] | None = None   # worked out from other columns: {formula, words, operands, holds, rows, breaks}
    geo: str = ""                   # "lat" / "lon" when the column is a coordinate of a detected pair
    _keys: set[str] = field(default_factory=set, repr=False)       # capped distinct values, for link overlap
    _key_cut: int | None = field(default=None, repr=False)

    @property
    def link_candidate(self) -> bool:
        return self.distinct >= 2 and self.role in (ID, CATEGORY, TEXT) and self.kind in (STR, NUM)

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in asdict(self).items() if not k.startswith("_")}
        return _json(d)


@dataclass
class Table:
    node: str                       # the step whose output this is
    title: str
    source: str | None = None       # the file, for a loader
    rows: int | None = None
    rows_exact: bool = False
    sampled: int = 0
    complete: bool = False          # the sample holds every row
    shape: str = TABLE
    time: str | None = None         # the main time column
    start: Any = None               # first and last time
    end: Any = None
    span_seconds: float | None = None
    columns: list[Column] = field(default_factory=list)
    pairs: list[dict[str, Any]] = field(default_factory=list)   # measure pairs that move together (sample): {x, y, r}
    total_row: dict[str, Any] | None = None   # a last row that adds up the others: {"column", "value"} naming it
    wide: dict[str, Any] | None = None        # months as columns: {"columns": [Jan, Feb …], "year": 2024 or None}
    blank_rows: int = 0             # rows with nothing in them (a blank line in a CSV)
    deep: bool = False              # deepen() has read every row
    geo: dict[str, str] | None = None   # {"lat": name, "lon": name} when the table holds points

    def column(self, name: str) -> Column | None:
        return next((c for c in self.columns if c.name == name), None)

    def by_role(self, *roles: str) -> list[Column]:
        return [c for c in self.columns if c.role in roles]

    @property
    def measures(self) -> list[Column]:
        return self.by_role(MEASURE)

    @property
    def categories(self) -> list[Column]:
        return self.by_role(CATEGORY)

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in asdict(self).items() if k != "columns"}
        d["columns"] = [c.to_dict() for c in self.columns]
        return _json(d)


@dataclass
class Relation:
    """How two or more tables belong together.

    - ``link``: rows of ``tables[0]`` find their row in ``tables[1]`` by a key (``left_on`` → ``right_on``).
      ``cardinality`` says whether the second table has one row per key (``many-to-one`` / ``one-to-one``)
      or several (``many-to-many``: linking would multiply rows, so it is never used without asking).
    - ``stack``: the tables have the same columns and can be appended; ``labels`` name each one.
    - ``align``: two time series of the same quantities that can be lined up by time (``tolerance``).
    """
    id: str
    kind: str
    tables: list[str]
    left_on: str = ""
    right_on: str = ""
    cardinality: str = ""
    match_pct: float = 0.0
    exact: bool = False
    score: float = 0.0
    labels: list[str] = field(default_factory=list)
    shared: list[str] = field(default_factory=list)     # align/stack: the columns both have (align: the first table's names)
    pairs: dict[str, str] = field(default_factory=dict)  # align: first table's column -> the second table's
    tolerance: str = ""
    why: str = ""
    geo: dict[str, str] = field(default_factory=dict)   # near: left_lat/left_lon/right_lat/right_lon

    def to_dict(self) -> dict[str, Any]:
        return _json(asdict(self))


@dataclass
class DataModel:
    tables: dict[str, Table] = field(default_factory=dict)          # node id -> table, in project order
    relations: list[Relation] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)           # node id -> why it could not be read
    deep: bool = False

    def table(self, node: str) -> Table | None:
        return self.tables.get(node)

    def relation(self, rid: str) -> Relation | None:
        return next((r for r in self.relations if r.id == rid), None)

    def stack_of(self, node: str) -> Relation | None:
        return next((r for r in self.relations if r.kind == "stack" and node in r.tables), None)

    def to_dict(self) -> dict[str, Any]:
        return {"tables": [t.to_dict() for t in self.tables.values()],
                "relations": [r.to_dict() for r in self.relations],
                "skipped": dict(self.skipped), "deep": self.deep}


# =================================================================== reading tables
def default_tables(pipe) -> list[str]:
    """The tables a project's answers start from: every step that brings data in (loaders, typed data)."""
    return [nid for nid, n in pipe.nodes.items() if registry.get(n.type).kind == "source"]


def understand(pipe, executor, nodes: Iterable[str] | None = None) -> DataModel:
    """Describe the output of each of ``nodes`` (default: every source step) from a sample, and find how
    they relate. A step that cannot be read is listed under ``skipped`` with a plain reason."""
    from .executor import friendly_error
    model = DataModel()
    for nid in (list(nodes) if nodes is not None else default_tables(pipe)):
        if nid not in pipe.nodes:
            continue
        try:
            model.tables[nid] = _read_table(pipe, executor, nid)
        except Exception as e:  # noqa: BLE001 - an unreadable table is reported, never raised
            model.skipped[nid] = friendly_error(e)
    model.relations = find_relations(model)
    return model


def _read_table(pipe, executor, nid: str) -> Table:
    node = pipe.nodes[nid]
    lf, kind = executor.sample_frame(nid, SAMPLE_ROWS)
    sample = lf.head(SAMPLE_ROWS).collect(engine="streaming")
    # a spread sample (every k-th row of a stored result) is fair for values but not for timing: how often rows
    # arrive is measured on consecutive rows
    run = sample if kind != "spread" else pl.scan_parquet(executor.state(nid).output).head(CADENCE_ROWS).collect()
    st = executor.state(nid)
    rows = st.rows if st.status == "done" and st.rows is not None else None
    nt = registry.get(node.type)
    source = str(node.params.get("path")) if nt.kind == "source" and node.params.get("path") else None
    t = Table(node=nid, title=node.title, source=source, rows=rows if rows is not None else None,
              rows_exact=rows is not None, sampled=sample.height)
    if rows is not None:
        t.complete = rows <= sample.height
    else:
        # a source not run yet gives its first rows, so fewer than asked means that is all of it; any other
        # step not run yet is computed from samples of its inputs, which says nothing about its full size
        t.complete = nt.kind == "source" and sample.height < SAMPLE_ROWS
    if t.complete and t.rows is None:
        t.rows, t.rows_exact = sample.height, True
    t.total_row = _total_row(sample.tail(3), sample) if t.complete else None
    t.blank_rows = int(sample.select(pl.all_horizontal(pl.all().is_null()).sum()).item()) if sample.width else 0
    rows_kept = None if kind == "spread" else _kept_rows(sample, t)     # row numbers of the step's output, in order
    sample = _without_extra_rows(sample, t)
    t.columns = [_describe_column(sample[c], pipe) for c in sample.columns]
    if t.complete:
        for c in t.columns:
            c.unique_exact = True
    _numbering(t, sample)
    _derived(t, sample, rows_kept)
    _settle_time(t, sample, run)
    _as_long(t, node)
    t.shape = _shape_of(t)
    t.pairs = _pairs(sample, t.measures, _definitions(t))
    _settle_geo(t)
    return t


def _settle_geo(t: Table) -> None:
    """Note a table's latitude/longitude pair, if it has one, so answers can offer a map or a nearest-place match.

    A pair needs a name that says latitude (lat, latitude, y) and one that says longitude, and both ranges must
    fit their coordinate; a plain x/y is only accepted when the values leave no doubt. This never changes a
    column's role: a coordinate stays a number, it just also carries where it is."""
    from .geo import lat_lon_pair
    nums = [(c.name, c.minimum, c.maximum) for c in t.columns if c.kind == NUM]
    pair = lat_lon_pair(nums)
    if not pair:
        return
    t.geo = {"lat": pair[0], "lon": pair[1]}
    for c in t.columns:
        if c.name == pair[0]:
            c.geo = "lat"
        elif c.name == pair[1]:
            c.geo = "lon"


def _polygon_column(t: Table) -> str | None:
    """The column of a table that holds polygons as WKT text (a GeoJSON file loads one called 'geometry'), or
    None. Detected by the column's name and the WKT it starts with in a sample."""
    for c in t.columns:
        if c.name == "geometry" or (c.kind == STR and "geom" in norm(c.name)):
            return c.name
    return None


NUMBERING_WORDS = {"n", "no", "nr", "num", "number", "rep", "replicate", "trial", "sample", "plot", "quadrat", "subject",
                   "run", "specimen", "individual", "plant", "tree", "site", "station", "animal", "patient"}


def _numbering(t: Table, sample: pl.DataFrame) -> None:
    """A column that numbers the rows within each group (1 … 15 for one group, 1 … 15 again for the other) names
    the rows; it is not a quantity. Whole numbers 1 … k, each the same number of times, in the first
    column or under a name like 'N', 'Rep' or 'Plot'."""
    for i, c in enumerate(t.columns):
        if c.role != MEASURE or not sample[c.name].dtype.is_integer() or c.distinct < 3:
            continue
        if i != 0 and not set(name_words(c.name)) & NUMBERING_WORDS:
            continue
        vc = sample[c.name].drop_nulls().value_counts()
        vals = sorted(vc[c.name].to_list())
        counts = set(vc["count"].to_list())
        if vals[0] in (0, 1) and vals == list(range(vals[0], vals[0] + len(vals))) and len(counts) == 1 and counts != {1}:
            c.role = ID


def _derived(t: Table, sample: pl.DataFrame, rows: list[int] | None) -> None:
    """Columns worked out from other columns (core.derived), noted on the column. Each row that breaks the rule
    is noted with its row number in the step's output (when the sample holds the rows in order) and the values
    that name it (its group, its number)."""
    from .derived import find
    names = [c.name for c in t.columns if c.role == MEASURE]
    if len(names) < 2:
        return
    try:
        found = find(sample, names)
    except Exception:  # noqa: BLE001 - a search that fails finds nothing
        return
    naming = [c.name for c in t.columns if c.role in (CATEGORY, ID) and c.kind in (STR, NUM)][:2]
    for d in found:
        col = t.column(d.target)
        if col is None:
            continue
        info = d.to_dict()
        for b in info["breaks"]:
            pos = b["row"]
            b["row"] = rows[pos] if rows is not None and pos < len(rows) else None
            b["where"] = {n: _clean(sample[n][pos]) for n in naming}
        col.derived = info


def _definitions(t: Table) -> set[frozenset]:
    """Pairs of columns related by definition: a calculated column and each column it is made from."""
    out = set()
    for c in t.columns:
        for o in (c.derived or {}).get("operands", []):
            out.add(frozenset((c.name, o)))
    return out


TOTAL_WORDS = {"total", "totals", "grand total", "sum", "all", "overall", "subtotal", "total:", "totals:"}


def _total_row(tail: pl.DataFrame, whole: pl.DataFrame | None = None) -> dict[str, Any] | None:
    """A row at the bottom that sums the others, named by a word like 'Total' in one of its text cells. With the
    whole table at hand, one of its numbers must also be the sum of that column's other rows (so a real region
    called 'All' is not taken for a total); without it, only the unmistakable words count."""
    for i in range(tail.height - 1, -1, -1):
        row = tail.row(i, named=True)
        for c, v in row.items():
            if not (isinstance(v, str) and v.strip().lower() in TOTAL_WORDS):
                continue
            if whole is None:
                if v.strip().lower() in ("total", "totals", "grand total"):
                    return {"column": c, "value": v}
                continue
            rest = whole.filter(pl.col(c).is_null() | (pl.col(c).cast(pl.Utf8) != v))
            for n, x in row.items():
                if isinstance(x, (int, float)) and not isinstance(x, bool) and whole[n].dtype.is_numeric():
                    s = rest[n].sum()
                    if s is not None and abs(float(s) - float(x)) <= 0.005 * max(1.0, abs(float(s))):
                        return {"column": c, "value": v}
        if whole is not None and any(v is not None for v in row.values()):
            # no word, but every label blank and a number that is the sum of the rows above: a total row all the same
            labels = [c for c, v in row.items() if not (isinstance(v, (int, float)) and not isinstance(v, bool))]
            nums = [(c, v) for c, v in row.items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
            if labels and nums and all(row[c] is None for c in labels) and whole.height > 3:
                rest = whole.slice(0, whole.height - (tail.height - i))
                for c, v in nums:
                    tot = rest[c].sum()
                    if tot is not None and abs(float(tot) - float(v)) <= 0.005 * max(1.0, abs(float(tot))) and abs(float(v)) > 0:
                        return {"column": c, "value": v, "blank": labels[0]}
            return None
        if any(v is not None for v in row.values()):
            return None
    return None


def _kept_rows(df: pl.DataFrame, t: Table) -> list[int]:
    """The positions in ``df`` of the rows ``_without_extra_rows`` keeps."""
    idx = df.with_columns(pl.Series("__pos", range(df.height)))
    kept = _without_extra_rows(idx.drop("__pos"), t, positions=idx["__pos"])
    return kept


def _without_extra_rows(df: pl.DataFrame, t: Table, positions: pl.Series | None = None):
    if t.total_row is not None:
        c, v = t.total_row["column"], t.total_row["value"]
        if t.total_row.get("blank"):
            keep = ~(pl.col(t.total_row["blank"]).is_null() & (pl.col(c) == v))
        else:
            keep = pl.col(c).is_null() | (pl.col(c).cast(pl.Utf8) != v)
        if positions is not None:                     # the positions follow every filter the rows go through
            positions = positions.filter(df.select(keep.fill_null(False)).to_series())
        df = df.filter(keep)
    if t.blank_rows and df.width:
        keep = ~pl.all_horizontal(pl.all().is_null())
        if positions is not None:
            positions = positions.filter(df.select(keep).to_series())
        df = df.filter(keep)
    return positions.to_list() if positions is not None else df


def _as_long(t: Table, node) -> None:
    """A wide table (a column per month: Jan, Feb … Dec) is described as the long table it stands for — one row per
    item and month, with 'month', 'value' and, when the file names its year, 'date' — so it can be totalled and
    charted by month. Answers put the reshaping step in front of it."""
    from .nodes.basic import month_of
    months = [c for c in t.columns if c.kind == NUM and month_of(c.name)]
    if len(months) < 3 or len({month_of(c.name) for c in months}) < len(months):
        return
    m = re.search(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)", f"{t.title} {t.source or ''}")
    year = int(m.group(1)) if m else None
    names = [c.name for c in months]
    t.wide = {"columns": names, "year": year}
    rest = [c for c in t.columns if c not in months]
    month = Column(name="month", dtype="String", kind=STR, role=CATEGORY, label="month", distinct=len(names), values=list(names))
    value = Column(name="value", dtype="Float64", kind=NUM, role=MEASURE, label="value", distinct=max(c.distinct for c in months))
    extra = [month, value]
    if year:
        import datetime as _dt
        start, end = _dt.datetime(year, min(month_of(n) for n in names), 1), _dt.datetime(year, max(month_of(n) for n in names), 1)
        date = Column(name="date", dtype="Datetime(time_unit='us', time_zone=None)", kind=TIME, role=TIME_ROLE, label="date",
                      distinct=len(names), minimum=start, maximum=end, cadence=30.44 * 86400, regular=True)
        extra.append(date)
        t.time, t.start, t.end, t.span_seconds = "date", start, end, (end - start).total_seconds()
    t.columns = rest + extra
    t.pairs = []


def _pairs(sample: pl.DataFrame, measures: list[Column], skip: set[frozenset] | None = None,
           limit: int = 8) -> list[dict[str, Any]]:
    """How strongly each pair of measures moves together in the sample (Pearson r), strongest first. Only used
    to rank suggestions: an answer built from it computes its fit on every row. Pairs related by definition (a
    calculated column and what it is made from) are left out: that they move together is not news."""
    names = [m.name for m in measures[:limit]]
    out = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if skip and frozenset((a, b)) in skip:
                continue
            try:
                d = sample.select(pl.col(a).cast(pl.Float64), pl.col(b).cast(pl.Float64)).drop_nulls().drop_nans()
                if d.height < 10:
                    continue
                r = d.select(pl.corr(a, b)).item()
            except Exception:  # noqa: BLE001
                continue
            if r is not None and r == r:
                out.append({"x": a, "y": b, "r": round(float(r), 4)})
    out.sort(key=lambda p: (-abs(p["r"]), p["x"], p["y"]))
    return out


def _describe_column(s: pl.Series, pipe) -> Column:
    n = s.len()
    kind = kind_of_dtype(s.dtype)
    filled = s.drop_nulls()
    if s.dtype == pl.Float64 or s.dtype == pl.Float32:
        filled = filled.filter(filled.is_not_nan())
    nulls = n - filled.len()
    distinct = int(filled.n_unique()) if filled.len() else 0
    label = pipe.column_label(s.name) if hasattr(pipe, "column_label") else s.name
    unit = (pipe.column_unit(s.name) if hasattr(pipe, "column_unit") else "") or unit_from_name(s.name)
    c = Column(name=s.name, dtype=str(s.dtype), kind=kind, role=TEXT, label=label or s.name, unit=unit,
               null_pct=(nulls / n) if n else 0.0, distinct=distinct,
               unique=bool(filled.len()) and distinct == filled.len())
    c.abbrev = header_parts(s.name)[1]
    c.quantity = quantity_of(unit) if kind == NUM else ""
    c.minimum, c.maximum = _extreme(filled, "min"), _extreme(filled, "max")
    c.role = _role_of(c, filled, n)
    if c.role == CATEGORY or (kind == STR and c.role in (ID, TEXT) and 1 < c.distinct <= 200 and _short_text(filled)):
        c.values = _category_values(filled)      # names too (departments, stores): they can be asked about by value
        if kind == STR:
            c.spellings = _spellings(filled)
    if c.link_candidate:
        c._keys, c._key_cut = _capped_values(filled)
    return c


def _role_of(c: Column, filled: pl.Series, n: int) -> str:
    if n == 0 or filled.len() == 0:
        return BLANK
    if c.distinct == 1:
        return CONSTANT
    if c.kind == TIME:
        return TIME_ROLE
    if c.kind == BOOL:
        return FLAG
    words = name_words(c.name)
    if c.kind == NUM:
        integer = filled.dtype.is_integer()
        if looks_like_key(c.name) and (integer or c.unique):
            return ID
        if integer and (_code_name(words) or _fixed_width_code(filled, c)):
            return ID                                        # zip, phone, account number: a code, not a quantity
        if integer and words and words[-1] in CALENDAR_WORDS and c.distinct <= 400:
            return CATEGORY                                  # year, month, week: a group, not a quantity
        if integer and c.unique and filled.len() >= 5 and _is_sequence(filled) and (filled.min() or 0) <= 1:
            return ID                                        # 1, 2, 3 … (or 0, 1, 2 …): row numbers
        return MEASURE
    if c.kind == STR:
        if looks_like_key(c.name) or (c.unique and filled.len() >= 5 and _short_text(filled)):
            return ID
        if 1 < c.distinct <= CATEGORY_MAX and c.distinct < filled.len() and c.null_pct <= 0.5:
            return CATEGORY
        return TEXT
    return TEXT


def _code_name(words: list[str]) -> bool:
    """zip, zip_code, postal_code, phone_number, customer_phone, account_no: the last word (before a closing code
    or number word) is a code word."""
    while len(words) > 1 and words[-1] in CODE_ENDINGS:
        words = words[:-1]
    return bool(words) and words[-1] in CODE_WORDS


def _fixed_width_code(s: pl.Series, c: Column) -> bool:
    """Whole numbers of seven or more digits, all the same width, mostly different and numbered from one base, so
    that their first three digits are the same (account 40001234 … 40009876), in a column whose name gives no unit.
    Amounts spread across their width (a population of 1,200,000 to 9,800,000) are not codes."""
    if c.unit or s.len() < 5 or c.distinct < 0.5 * s.len():
        return False
    lo, hi = s.min(), s.max()
    return (lo is not None and hi is not None and lo >= 1_000_000 and len(str(lo)) == len(str(hi))
            and str(lo)[:3] == str(hi)[:3])


def _is_sequence(s: pl.Series) -> bool:
    lo, hi = s.min(), s.max()
    return lo is not None and hi is not None and (hi - lo + 1) <= 2 * s.len()


def _short_text(s: pl.Series) -> bool:
    """Codes and ids are short and have no sentences in them; free text is long."""
    try:
        lens = s.cast(pl.Utf8).str.len_chars()
        return (lens.max() or 0) <= 40 and float(s.cast(pl.Utf8).str.contains(" ").mean() or 0) < 0.5
    except Exception:  # noqa: BLE001
        return False


def _category_values(s: pl.Series) -> list[Any]:
    vc = s.value_counts(sort=False)
    name = vc.columns[0]
    vc = vc.with_columns(pl.col(name).cast(pl.Utf8).alias("_text")).sort(["count", "_text"], descending=[True, False])
    return [_clean(v) for v in vc[name].to_list()]


def _spellings(s: pl.Series) -> dict[str, str]:
    """Values that differ only in capitals or surrounding spaces (North, north, 'North '): each variant maps to
    the spelling most rows use (ties: the first in text order)."""
    vc = s.cast(pl.Utf8).value_counts(sort=False)
    name = vc.columns[0]
    vc = vc.with_columns(pl.col(name).str.strip_chars().str.to_lowercase().alias("_k")).sort(["count", name], descending=[True, False])
    out: dict[str, str] = {}
    for key in vc["_k"].unique(maintain_order=True).to_list():
        variants = vc.filter(pl.col("_k") == key)[name].to_list()
        target = variants[0].strip()                     # the spelling most rows use, without stray spaces
        for v in variants:
            if v != target:
                out[v] = target
    return out


def _settle_time(t: Table, sample: pl.DataFrame, consecutive: pl.DataFrame) -> None:
    """The main time column (the first date/time column that varies) and how regularly rows arrive, measured on
    ``consecutive`` rows."""
    times = t.by_role(TIME_ROLE)
    if not times:
        return
    main = max(times, key=lambda c: (c.distinct, -t.columns.index(c)))     # the finest: Date Time rather than Date
    t.time = main.name
    s = consecutive[main.name].drop_nulls() if main.name in consecutive.columns else sample[main.name].drop_nulls()
    if s.len() >= 3:
        try:
            srt = s.sort()
            if isinstance(s.dtype, pl.Datetime):
                us = srt.dt.epoch("us").cast(pl.Float64)
            else:
                us = srt.cast(pl.Datetime("us")).dt.epoch("us").cast(pl.Float64)
            steps = us.diff().drop_nulls()
            steps = steps.filter(steps > 0)
            if steps.len():
                med = float(steps.median())
                main.cadence = med / 1e6
                main.regular = float(((steps - med).abs() <= 0.1 * med).mean()) >= 0.8
        except Exception:  # noqa: BLE001 - odd time types simply get no cadence
            pass
    t.start, t.end = main.minimum, main.maximum
    t.span_seconds = _span(t.start, t.end)


def _span(a: Any, b: Any) -> float | None:
    try:
        return float((b - a).total_seconds())
    except Exception:  # noqa: BLE001
        return None


def _shape_of(t: Table) -> str:
    """series: readings at a steady rate (a logger); lookup: one row per key (customers, sites);
    events: things that happened, with ids or times (orders, visits); table: anything else."""
    time = t.column(t.time) if t.time else None
    ids = t.by_role(ID)
    unique_ids = [c for c in ids if c.unique]
    small = (t.rows if t.rows is not None else t.sampled) <= 100_000
    if time is not None and t.measures and time.regular and time.unique and not ids:
        return SERIES                                   # one reading per moment, at a steady rate
    if unique_ids and small and time is None and not [c for c in ids if not c.unique]:
        return LOOKUP
    if time is not None or ids:
        return EVENTS
    return TABLE


# =================================================================== relations
def find_relations(model: DataModel) -> list[Relation]:
    tables = list(model.tables.values())
    out: list[Relation] = []
    stacks = _find_stacks(tables)
    out += stacks
    out += _find_aligns(tables)
    out += _find_nears(tables, stacks)
    out += _find_contains(tables, stacks)
    same = [set(r.tables) for r in stacks]
    # tables of one stack are the same kind of table, not lookups of each other
    out += [r for r in _find_links(tables) if not any(set(r.tables) <= g for g in same)]
    return out


def _find_links(tables: list[Table]) -> list[Relation]:
    out: list[Relation] = []
    for i, a in enumerate(tables):
        for b in tables[i + 1:]:
            for ca in a.columns:
                if not ca.link_candidate or not ca._keys:
                    continue
                for cb in b.columns:
                    if not cb.link_candidate or not cb._keys or _kind_family(ca) != _kind_family(cb):
                        continue
                    name = _link_name_score(a, ca, b, cb)
                    if name < 0.34:
                        continue
                    va, vb = _comparable(ca, cb)
                    overlap = _overlap(va, vb)
                    if overlap < 0.5:
                        continue
                    # 1, 2, 3 … on both sides overlap whatever they count, so only the names can say they are one key
                    score = name if _counts_from_one(ca) and _counts_from_one(cb) else 0.6 * name + 0.4 * overlap
                    if score < LINK_MIN_SCORE:
                        continue
                    out.append(_orient_link(a, ca, b, cb, va, vb, round(score, 3)))
    out.sort(key=lambda r: (-r.score, r.id))
    kept: list[Relation] = []
    per_pair: dict[frozenset, int] = {}
    for r in out:                                  # the best few ways to link each pair of tables, not every one
        k = frozenset(r.tables)
        if per_pair.get(k, 0) < LINKS_PER_PAIR:
            per_pair[k] = per_pair.get(k, 0) + 1
            kept.append(r)
    return kept


def _counts_from_one(c: Column) -> bool:
    """Whole numbers starting near 1 with few gaps (row numbers, small ids)."""
    lo, hi = c.minimum, c.maximum
    if c.kind != NUM or not isinstance(lo, int) or not isinstance(hi, int) or isinstance(lo, bool):
        return False
    return lo <= 1 and hi - lo + 1 <= 2 * max(c.distinct, 1)


BARE_KEYS = ("id", "key", "code", "ref", "uuid", "guid", "no", "number")
ROW_IDS = ("id", "uuid", "guid")            # a table's own row number, not a key it shares


def _singular(word: str) -> str:
    w = norm(word)
    for end, rep in (("ies", "y"), ("ses", "s"), ("s", "")):
        if w.endswith(end) and len(w) > len(end) + 2:
            return w[: -len(end)] + rep
    return w


def _link_name_score(a: Table, ca: Column, b: Table, cb: Column) -> float:
    """How much two column names say they are the same key. A bare 'id' is a table's own row id: it matches
    '<that table>_id' in another table (customers.id and orders.customer_id), never another table's bare 'id'."""
    bare_a, bare_b = norm(ca.name) in BARE_KEYS, norm(cb.name) in BARE_KEYS
    if norm(ca.name) in ROW_IDS and norm(cb.name) in ROW_IDS:
        return 0.0 if _singular(a.title) != _singular(b.title) else 1.0
    if bare_b and _stem(ca.name) == _singular(b.title):
        return 0.95
    if bare_a and _stem(cb.name) == _singular(a.title):
        return 0.95
    return name_similarity(ca.name, cb.name)


def _kind_family(c: Column) -> str:
    return "number" if c.kind == NUM else "text"


def _orient_link(a: Table, ca: Column, b: Table, cb: Column, va: set[str], vb: set[str], score: float) -> Relation:
    """Put the side with one row per key second (the lookup), so linking never multiplies rows."""
    ua, ub = ca.unique, cb.unique
    if ub and not ua:
        left, lc, right, rc, lv, rv = a, ca, b, cb, va, vb
    elif ua and not ub:
        left, lc, right, rc, lv, rv = b, cb, a, ca, vb, va
    else:
        # both unique (one-to-one) or neither (many-to-many): the bigger table first
        ra, rb = (a.rows or a.sampled), (b.rows or b.sampled)
        if rb > ra:
            left, lc, right, rc, lv, rv = b, cb, a, ca, vb, va
        else:
            left, lc, right, rc, lv, rv = a, ca, b, cb, va, vb
    card = "many-to-many" if not rc.unique else ("one-to-one" if lc.unique else "many-to-one")
    pct = round(100.0 * len(lv & rv) / len(lv), 1) if lv else 0.0
    rel = Relation(id=f"link:{left.node}.{lc.name}>{right.node}.{rc.name}", kind="link", tables=[left.node, right.node],
                   left_on=lc.name, right_on=rc.name, cardinality=card, match_pct=pct, score=score,
                   exact=lc.unique_exact and rc.unique_exact and left.complete and right.complete)
    rel.why = link_why(rel, left, right)
    return rel


def link_why(rel: Relation, left: Table, right: Table) -> str:
    est = "" if rel.exact else " (from a sample)"
    many = {"many-to-one": f"each {rel.right_on} appears once in {right.title}, so no rows are multiplied",
            "one-to-one": f"each {rel.left_on} appears once in both",
            "many-to-many": f"{rel.right_on} repeats in {right.title}, so linking would repeat rows"}[rel.cardinality]
    return f"{rel.match_pct:g}% of {left.title}'s {rel.left_on} values are in {right.title}{est}. {many[0].upper()}{many[1:]}"


def _find_stacks(tables: list[Table]) -> list[Relation]:
    out: list[Relation] = []
    used: set[str] = set()
    for i, a in enumerate(tables):
        if a.node in used:
            continue
        members, worst = [a], 1.0
        for b in tables[i + 1:]:
            if b.node in used:
                continue
            sim = _column_similarity(a, b)
            if sim >= STACK_MIN_SIMILARITY and _kinds_agree(a, b):
                members.append(b)
                worst = min(worst, sim)
        if len(members) < 2:
            continue
        used.update(m.node for m in members)
        labels = distinct_labels([m.source or m.title for m in members], [m.title for m in members])
        shared = [c.name for c in a.columns if all(m.column(c.name) is not None for m in members)]
        rel = Relation(id="stack:" + "+".join(m.node for m in members), kind="stack", tables=[m.node for m in members],
                       score=round(worst, 3), labels=labels, shared=shared, exact=True,
                       why=f"The {len(members)} tables have the same columns ({', '.join(shared[:4])}"
                           f"{'…' if len(shared) > 4 else ''}). Each row keeps a note of which one it came from")
        out.append(rel)
    return out


def _find_aligns(tables: list[Table]) -> list[Relation]:
    """Two time series measuring the same things: they can be lined up reading by reading."""
    out: list[Relation] = []
    series = [t for t in tables if t.time and t.measures and t.shape == SERIES]    # logs, not orders or budgets
    for i, a in enumerate(series):
        for b in series[i + 1:]:
            pairs = {m.name: m.name for m in a.measures if b.column(m.name) is not None and b.column(m.name).role == MEASURE}
            if not pairs:
                pairs = {m.name: n.name for m in a.measures for n in b.measures if m.unit and m.unit == n.unit}
                pairs = dict(list(pairs.items())[:1])
            if not pairs:
                continue
            shared = list(pairs)
            ca, cb = a.column(a.time), b.column(b.time)
            cad = max(x for x in (ca.cadence, cb.cadence, 0.0) if x is not None)
            overlap = _time_overlap(a, b)
            if overlap is not None and overlap <= 0:
                continue
            tol = duration_text(cad) if cad else ""
            score = 0.7 + 0.3 * (overlap if overlap is not None else 0.5)
            rel = Relation(id=f"align:{a.node}~{b.node}", kind="align", tables=[a.node, b.node], left_on=a.time,
                           right_on=b.time, shared=shared, pairs=pairs, tolerance=tol, score=round(score, 3), exact=False)
            rel.why = (f"Both record {', '.join(shared[:3])} over time. Each reading of {a.title} is paired with the "
                       f"nearest reading of {b.title}" + (f" within {tol}" if tol else ""))
            out.append(rel)
    return out


def _find_contains(tables: list[Table], stacks: list[Relation]) -> list[Relation]:
    """A table of points and a table of polygons (a WKT 'geometry' column): each point can be matched to the
    place it falls inside."""
    out: list[Relation] = []
    stack_pairs = [set(r.tables) for r in stacks]
    points = [t for t in tables if t.geo]
    polys = [t for t in tables if _polygon_column(t) is not None]
    for p in points:
        for poly in polys:
            if p.node == poly.node or any({p.node, poly.node} == sp for sp in stack_pairs):
                continue
            geom = _polygon_column(poly)
            rel = Relation(id=f"within:{p.node}>{poly.node}", kind="containment", tables=[p.node, poly.node],
                           score=0.7, exact=False,
                           left_on=p.geo["lat"], right_on=geom, geo={"left_lat": p.geo["lat"], "left_lon": p.geo["lon"],
                                                                     "right_geometry": geom or "geometry"})
            rel.why = f"Each place of {p.title} can be matched to the {poly.title} region it falls inside"
            out.append(rel)
    return out


def _find_nears(tables: list[Table], stacks: list[Relation]) -> list[Relation]:
    """Two tables that both hold points: every row of one can be matched to its nearest place in the other.
    The table with more rows is the points (the left side); the smaller is the set of places to search."""
    out: list[Relation] = []
    stack_pairs = [set(r.tables) for r in stacks]
    geo_tables = [t for t in tables if t.geo]
    for i, a in enumerate(geo_tables):
        for b in geo_tables[i + 1:]:
            if any({a.node, b.node} == sp for sp in stack_pairs):
                continue
            ra, rb = (a.rows or a.sampled or 0), (b.rows or b.sampled or 0)
            left, right = (a, b) if ra >= rb else (b, a)
            score = round(0.7 + 0.25 * _geo_overlap(left, right), 3)
            rel = Relation(id=f"near:{left.node}>{right.node}", kind="near", tables=[left.node, right.node],
                           left_on=left.geo["lat"], right_on=right.geo["lat"], score=score, exact=False,
                           geo={"left_lat": left.geo["lat"], "left_lon": left.geo["lon"],
                                "right_lat": right.geo["lat"], "right_lon": right.geo["lon"]})
            rel.why = (f"Both have coordinates: each row of {left.title} can be matched to its nearest place in "
                       f"{right.title}")
            out.append(rel)
    out.sort(key=lambda r: (-r.score, r.id))
    return out


def _geo_overlap(a: Table, b: Table) -> float:
    """How much of the smaller table's coordinate box the other covers (0..1): a share of places side."""
    try:
        def box(t: Table) -> tuple:
            la, lo = t.column(t.geo["lat"]), t.column(t.geo["lon"])
            return (float(lo.minimum), float(lo.maximum), float(la.minimum), float(la.maximum))
        ax0, ax1, ay0, ay1 = box(a)
        bx0, bx1, by0, by1 = box(b)
        ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
        iy = max(0.0, min(ay1, by1) - max(ay0, by0))
        smaller = min((ax1 - ax0) * (ay1 - ay0) or 1e-9, (bx1 - bx0) * (by1 - by0) or 1e-9)
        return max(0.0, min(1.0, (ix * iy) / smaller))
    except (TypeError, ValueError):
        return 0.0


def _time_overlap(a: Table, b: Table) -> float | None:
    """The share of the shorter span that both tables cover, or None when the spans are not known."""
    try:
        lo, hi = max(a.start, b.start), min(a.end, b.end)
        shorter = min(a.span_seconds or 0, b.span_seconds or 0)
        if not shorter:
            return None
        return max(0.0, (hi - lo).total_seconds()) / shorter
    except Exception:  # noqa: BLE001 - dates of different kinds or zones
        return None


def _kinds_agree(a: Table, b: Table) -> bool:
    for c in a.columns:
        d = b.column(c.name)
        if d is not None and c.kind != d.kind and BLANK not in (c.role, d.role):
            return False
    return True


# =================================================================== the full pass
def deepen(pipe, executor, model: DataModel, cancel=None) -> DataModel:
    """Read every row of each table to replace sample facts by exact ones: row counts, time spans, key
    uniqueness and link matches. Tables that cannot be read in full keep their sample facts.

    One streaming pass per table; a second only to count keys exactly in a table whose size was not known before
    (a source not run yet); then one pass over the two key columns of each link."""
    frames: dict[str, pl.LazyFrame | None] = {}          # each table is opened once (an Excel file is parsed once)
    for t in model.tables.values():
        if cancel is not None and cancel():
            return model
        try:
            frames[t.node] = lf = full_frame(pipe, executor, t.node)
            if lf is not None:
                st = executor.state(t.node)
                _deepen_table(t, lf, st.rows if st.status == "done" else None)
        except Exception:  # noqa: BLE001 - the sample facts stay; the answer says they are estimates
            frames[t.node] = None
            continue
    # shapes can change once exact uniqueness and spans are known; links are re-pointed before they are counted
    for t in model.tables.values():
        t.shape = _shape_of(t)
    _reorient_links(model)
    for r in model.relations:
        if cancel is not None and cancel():
            return model
        if r.kind == "link":
            try:
                _deepen_link(model, r, frames)
            except Exception:  # noqa: BLE001
                continue
    model.deep = True
    return model


def full_frame(pipe, executor, nid: str) -> pl.LazyFrame | None:
    """Every row of a step's output without running it: its stored result, or for a source step not run yet
    the source read in full with the same settings a run uses. None for other steps not run yet."""
    st = executor.state(nid)
    if st.status == "done" and st.output:
        return pl.scan_parquet(st.output)
    node = pipe.nodes[nid]
    nt = registry.get(node.type)
    if nt.kind != "source":
        return None
    from .registry import NodeResult
    ctx = executor._ctx(nid, preview=False)
    res = nt.apply(ctx, {}, node.params)
    return res.frame if isinstance(res, NodeResult) else res


def _deepen_table(t: Table, lf: pl.LazyFrame, known_rows: int | None) -> None:
    """Counts, time span, blank rows, the last rows (for a total row) and key counts, in one pass. Keys are
    counted exactly in that pass when the table is known to be small enough, else estimated there and counted
    exactly in a second pass once its size is known."""
    schema = lf.collect_schema()
    aggs: list[pl.Expr] = [pl.len().alias("__rows")]
    if t.time and t.time in schema:
        aggs += [pl.col(t.time).min().alias("__tmin"), pl.col(t.time).max().alias("__tmax")]
    keys = [c for c in t.columns if c.name in schema and c.role in (ID, CATEGORY, TEXT)]
    # the columns that looked unique in the sample are the ones worth counting exactly
    exact = [c for c in keys if c.unique and c.role == ID and not t.complete]
    exact_now = known_rows is not None and known_rows <= EXACT_KEY_ROWS
    for c in keys:
        aggs.append(pl.col(c.name).drop_nulls().approx_n_unique().alias(f"__u_{c.name}"))
        aggs.append(pl.col(c.name).drop_nulls().len().alias(f"__n_{c.name}"))
    if exact_now:
        aggs += [pl.col(c.name).drop_nulls().n_unique().alias(f"__x_{c.name}") for c in exact]
    aggs.append(pl.all_horizontal(pl.all().is_null()).sum().alias("__blank"))
    tail = t.total_row is None and not t.complete
    if tail:
        aggs += [pl.col(n).tail(3).implode().alias(f"__t_{n}") for n in schema]
    row = lf.select(aggs).collect(engine="streaming").row(0, named=True)
    t.rows, t.rows_exact, t.deep = int(row["__rows"]), True, True
    t.blank_rows = int(row["__blank"] or 0)
    if tail:
        last = pl.DataFrame({n: row[f"__t_{n}"] for n in schema}, schema=dict(schema))
        t.total_row = _total_row(last)
    t.complete = t.rows <= t.sampled
    if "__tmin" in row:
        t.start, t.end = _clean(row["__tmin"]), _clean(row["__tmax"])
        t.span_seconds = _span(t.start, t.end)
        col = t.column(t.time)
        if col is not None:
            col.minimum, col.maximum = t.start, t.end
    counts = {c.name: row[f"__x_{c.name}"] for c in exact if f"__x_{c.name}" in row}
    if not exact_now and exact and t.rows <= EXACT_KEY_ROWS:
        counts = lf.select([pl.col(c.name).drop_nulls().n_unique().alias(c.name) for c in exact]).collect(engine="streaming").row(0, named=True)
    for c in keys:
        est, filled = int(row[f"__u_{c.name}"]), int(row[f"__n_{c.name}"])
        if not t.complete:
            c.distinct = min(max(c.distinct, est), filled)       # an estimate: never more values than filled cells
        if c.name in counts:
            n = int(counts[c.name])
            c.unique, c.unique_exact, c.distinct = n == filled, True, n
        elif c.unique and not t.complete:
            c.unique = est >= 0.98 * filled           # HyperLogLog is within about 2% (free text, or a very long table)
        else:
            c.unique_exact = True                     # the whole table was read, or a repeat was already seen
        if c.role == CATEGORY and c.distinct > CATEGORY_MAX * 2:
            c.role, c.values = TEXT, []                    # the sample looked like a category; the whole table does not


def _deepen_link(model: DataModel, r: Relation, frames: dict[str, pl.LazyFrame | None]) -> None:
    """The share of the first table's keys found in the second, counted over every row in one pass of each."""
    left, right = model.tables[r.tables[0]], model.tables[r.tables[1]]
    if (left.rows or 0) > EXACT_MATCH_ROWS:
        return
    lf, rf = frames.get(left.node), frames.get(right.node)
    if lf is None or rf is None:
        return
    lk = lf.select(pl.col(r.left_on).cast(pl.Utf8).alias("k")).drop_nulls().unique()
    rk = rf.select(pl.col(r.right_on).cast(pl.Utf8).alias("k")).drop_nulls().unique().with_columns(pl.lit(1).alias("hit"))
    row = lk.join(rk, on="k", how="left").select(pl.len().alias("n"), pl.col("hit").sum().alias("found")) \
        .collect(engine="streaming").row(0, named=True)
    total, found = int(row["n"]), int(row["found"] or 0)
    r.match_pct = round(100.0 * found / total, 1) if total else 0.0
    lc, rc = left.column(r.left_on), right.column(r.right_on)
    r.exact = bool(lc and rc and lc.unique_exact and rc.unique_exact)
    r.why = link_why(r, left, right)


def _reorient_links(model: DataModel) -> None:
    """Point each link at the side exact counts showed to be the lookup; a link turned round has its match share
    measured again from the other side."""
    for r in model.relations:
        if r.kind != "link":
            continue
        a, b = model.tables[r.tables[0]], model.tables[r.tables[1]]
        ca, cb = a.column(r.left_on), b.column(r.right_on)
        if ca is None or cb is None:
            continue
        if ca.unique and not cb.unique:                 # exact counts showed the other side is the lookup
            a, b, ca, cb = b, a, cb, ca
            r.tables, r.left_on, r.right_on = [a.node, b.node], ca.name, cb.name
            r.id = f"link:{a.node}.{ca.name}>{b.node}.{cb.name}"
            va, vb = _comparable(ca, cb)
            r.match_pct = round(100.0 * len(va & vb) / len(va), 1) if va else 0.0
        r.cardinality = "many-to-many" if not cb.unique else ("one-to-one" if ca.unique else "many-to-one")
        r.why = link_why(r, a, b)


# =================================================================== names and values
def name_words(name: str) -> list[str]:
    """'CustomerID' -> customer, id; 'order_no' -> order, no; 'Amount paid' -> amount, paid."""
    return [w.lower() for w in re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+", str(name))]


def norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def looks_like_key(name: str) -> bool:
    """A name whose last word is a key word (customer_id, OrderNo, sku), not one that merely ends in those
    letters (Amount paid, valid, Humid). A bare 'id' or 'sku' counts; a bare 'number' or 'no' does not."""
    w = name_words(name)
    if not w or w[-1] not in KEY_SUFFIXES:
        return False
    return len(w) > 1 or w[-1] in ("id", "key", "sku", "uuid", "guid", "code", "ref")


def _stem(name: str) -> str:
    w = name_words(name)
    return "".join(w[:-1]) if len(w) > 1 and w[-1] in KEY_SUFFIXES else norm(name)


def name_similarity(a: str, b: str) -> float:
    na, nb = norm(a), norm(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    sa, sb = _stem(a), _stem(b)
    if sa != na and sb != nb:          # both are key-suffixed: the stems must be one word (cust_id, customer_id)
        if sa == sb:
            return 1.0
        short, long_ = sorted((sa, sb), key=len)
        return 0.85 if len(short) >= 3 and long_.startswith(short) else 0.0    # stock_id is not store_id
    if sa == sb:
        return 0.95
    if sa in sb or sb in sa:
        return 0.85
    return difflib.SequenceMatcher(None, na, nb).ratio()


def distinct_labels(names: list[str], fallback: list[str]) -> list[str]:
    """Short labels telling apart files with similar names: the words that differ between them.
    probe_MJ03E.csv, probe_MJ03F.csv -> MJ03E, MJ03F. Falls back to the titles when nothing differs."""
    split = [[w for w in re.split(r"[_\s.]+", Path(str(n)).stem) if w] for n in names]
    common = set(split[0]).intersection(*map(set, split[1:])) if split else set()
    out = []
    lead = next((w for w in split[0] if w in common), "") if split else ""
    for words, fb in zip(split, fallback):
        rest = [w for w in words if w not in common]
        label = "_".join(rest) if rest else fb
        if label.isdigit() and lead:
            label = f"{lead} {label}"                  # device_1, device_2 -> "device 1", not just "1"
        out.append(label)
    if len(set(out)) < len(out):
        return list(fallback) if len(set(fallback)) == len(fallback) else [f"{fb} {i + 1}" for i, fb in enumerate(fallback)]
    return out


def _vhash(v: str) -> int:
    return zlib.crc32(v.encode("utf-8", "surrogatepass"))


def _capped_values(s: pl.Series, cap: int = MAX_VALUES) -> tuple[set[str], int | None]:
    """At most ``cap`` distinct values, chosen by hash rather than position, so two tables keep the same
    values out of the ones they share and their overlap is measured fairly."""
    vals = s.cast(pl.Utf8).unique().to_list()
    if len(vals) <= cap:
        return set(vals), None
    kept = sorted(vals, key=_vhash)[:cap]
    return set(kept), _vhash(kept[-1])


def _comparable(a: Column, b: Column) -> tuple[set[str], set[str]]:
    cuts = [c for c in (a._key_cut, b._key_cut) if c is not None]
    if not cuts:
        return set(a._keys), set(b._keys)
    cut = min(cuts)
    return {v for v in a._keys if _vhash(v) <= cut}, {v for v in b._keys if _vhash(v) <= cut}


def _overlap(a: set[str], b: set[str]) -> float:
    """Half containment (the smaller side found in the larger) and half Jaccard, so a tiny set that happens
    to sit inside a big one does not score as a match."""
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return 0.5 * inter / min(len(a), len(b)) + 0.5 * inter / len(a | b)


def _column_similarity(a: Table, b: Table) -> float:
    ca = {norm(c.name) for c in a.columns}
    cb = {norm(c.name) for c in b.columns}
    if not ca or not cb:
        return 0.0
    return len(ca & cb) / len(ca | cb)


def _extreme(s: pl.Series, how: str) -> Any:
    try:
        return _clean(getattr(s, how)())
    except Exception:  # noqa: BLE001 - some dtypes have no min/max
        return None


def _clean(v: Any) -> Any:
    if isinstance(v, float) and (v != v or math.isinf(v)):
        return None
    return v


def _json(d: Any) -> Any:
    """JSON-ready: dates as ISO text, no NaN."""
    import datetime as _dt
    from .dtypes import json_safe

    def walk(v: Any) -> Any:
        if isinstance(v, dict):
            return {k: walk(x) for k, x in v.items()}
        if isinstance(v, (list, tuple, set)):
            return [walk(x) for x in v]
        if isinstance(v, (_dt.datetime, _dt.date, _dt.time, _dt.timedelta)):
            return str(v) if isinstance(v, _dt.timedelta) else v.isoformat()
        return v
    return json_safe(walk(d))


# =================================================================== durations for people
NICE_STEPS = [(1e-3, "1ms"), (1e-2, "10ms"), (0.05, "50ms"), (0.1, "100ms"), (0.5, "500ms"), (1, "1s"), (5, "5s"),
              (10, "10s"), (30, "30s"), (60, "1m"), (300, "5m"), (900, "15m"), (1800, "30m"), (3600, "1h"),
              (6 * 3600, "6h"), (86400, "1d"), (7 * 86400, "1w"), (30.44 * 86400, "1mo"), (91.31 * 86400, "1q"),
              (365.25 * 86400, "1y")]


def duration_text(secs: float) -> str:
    """A tolerance for pairing readings taken ``secs`` apart: that spacing, written the short way."""
    if secs <= 0:
        return ""
    for unit, scale in (("d", 86400), ("h", 3600), ("m", 60), ("s", 1), ("ms", 1e-3)):
        v = secs / scale
        if v >= 1 and abs(v - round(v)) < 1e-9:
            return f"{int(round(v))}{unit}"
    ms = secs * 1000
    if ms >= 1:
        return f"{int(math.ceil(ms))}ms"
    return f"{int(math.ceil(secs * 1e6))}us"


def bucket_for(span_seconds: float | None, cadence: float | None = None, target: int = 400) -> str:
    """The time bucket that turns a span into roughly ``target`` points (minutes for a day, days for months),
    never finer than the rows arrive (``cadence``): hourly rows are not bucketed per minute."""
    if not span_seconds or span_seconds <= 0:
        return "1d"
    ideal = span_seconds / target
    fits = [text for secs, text in NICE_STEPS if secs <= ideal * 2]
    best = fits[-1] if fits else NICE_STEPS[0][1]
    if cadence:
        floor = next((text for secs, text in NICE_STEPS if secs >= cadence * 0.999), NICE_STEPS[-1][1])
        if dict((t, s) for s, t in NICE_STEPS)[best] < dict((t, s) for s, t in NICE_STEPS)[floor]:
            best = floor
    return best
