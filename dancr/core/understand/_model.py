"""The data model: a project's tables, their columns and how they relate (pure types)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any

from ..expr import STR, NUM


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

def _clean(v: Any) -> Any:
    if isinstance(v, float) and (v != v or math.isinf(v)):
        return None
    return v



def _json(d: Any) -> Any:
    """JSON-ready: dates as ISO text, no NaN."""
    import datetime as _dt
    from ..dtypes import json_safe

    def walk(v: Any) -> Any:
        if isinstance(v, dict):
            return {k: walk(x) for k, x in v.items()}
        if isinstance(v, (list, tuple, set)):
            return [walk(x) for x in v]
        if isinstance(v, (_dt.datetime, _dt.date, _dt.time, _dt.timedelta)):
            return str(v) if isinstance(v, _dt.timedelta) else v.isoformat()
        return v
    return json_safe(walk(d))
