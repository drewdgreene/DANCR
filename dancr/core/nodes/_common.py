from __future__ import annotations

from pathlib import Path

from typing import Any

import polars as pl

from ..expr import kind_of_dtype, NUM, TIME

STAT_CHOICES = [
    ("mean", "average"), ("median", "median"), ("min", "minimum"), ("max", "maximum"),
    ("std", "standard deviation"), ("sum", "total"), ("count", "count of filled values"), ("first", "first"), ("last", "last"),
    ("n_unique", "distinct count"), ("rows", "number of rows (blank or not)"),
]
def number_param(params: dict[str, Any], key: str, default: float, what: str, *, whole: bool = False,
                 above: float | None = None, at_least: float | None = None, at_most: float | None = None) -> float:
    """A numeric setting, where only a missing value means the default: an explicit 0 is 0 (``x or default``
    would silently replace it). Out-of-range values are refused with a plain message."""
    v = params.get(key)
    n: float = default if v is None or v == "" else (int(v) if whole else float(v))
    if above is not None and not n > above:
        raise ValueError(f"{what} must be more than {above:g}")
    if at_least is not None and n < at_least:
        raise ValueError(f"{what} must be at least {at_least:g}")
    if at_most is not None and n > at_most:
        raise ValueError(f"{what} must be at most {at_most:g}")
    return n


STAT_HELP = ("Any of: " + ", ".join(v for v, _ in STAT_CHOICES) + ". With one statistic the columns keep their names. "
             "With several, or when choosing column by column, each name gets a suffix")


def check_stats(stats: list[str] | tuple[str, ...]) -> None:
    valid = {v for v, _ in STAT_CHOICES}
    for st in stats:
        if st not in valid:
            raise ValueError(f"Unknown statistic {st!r}. Use: {', '.join(sorted(valid))}")


def stat_expr(col: str, stat: str, order_by: str | None = None) -> pl.Expr:
    """One statistic of a column. ``order_by`` (a time column) decides what first and last mean: the
    earliest and latest reading, not the first and last row in the file."""
    c = pl.col(col)
    if order_by is not None and stat in ("first", "last"):
        c = c.sort_by(order_by)
    if stat in ("mean", "median", "min", "max", "std", "sum"):
        c = c.fill_nan(None)                  # NaN is a blank, as everywhere else: it does not poison or skew a statistic
    if stat == "mean":
        return c.mean()
    if stat == "median":
        return c.median()
    if stat == "min":
        return c.min()
    if stat == "max":
        return c.max()
    if stat == "std":
        return c.std()
    if stat == "sum":
        return c.sum()
    if stat == "count":
        return c.count()
    if stat == "first":
        return c.first()
    if stat == "last":
        return c.last()
    if stat == "n_unique":
        return c.n_unique()
    if stat == "rows":
        return pl.len()                       # every row of the group, whatever its values
    raise ValueError(f"Unknown statistic {stat!r}")


def numeric_columns(schema: dict[str, pl.DataType], exclude: list[str] | None = None) -> list[str]:
    ex = set(exclude or [])
    return [c for c, dt in schema.items() if kind_of_dtype(dt) == NUM and c not in ex]


def temporal_columns(schema: dict[str, pl.DataType]) -> list[str]:
    return [c for c, dt in schema.items() if kind_of_dtype(dt) == TIME]


def build_aggregations(schema: dict[str, pl.DataType], aggregations: list[dict[str, Any]] | None,
                       exclude: list[str], default_stats: tuple[str, ...] = ("mean",),
                       only: list[str] | None = None, order_by: str | None = None) -> list[pl.Expr]:
    """[{column, stats:[...], alias?}] -> agg exprs. Empty -> default stats of the chosen (or every) numeric column.
    With the default statistics, a single statistic keeps the original column names and several add a _stat
    suffix; column-by-column choices are always named column_stat (or alias / alias_stat)."""
    exprs: list[pl.Expr] = []
    check_stats(default_stats)
    if not aggregations:
        for c in (only or numeric_columns(schema, exclude)):
            for st in default_stats:
                name = c if len(default_stats) == 1 else f"{c}_{st}"
                exprs.append(stat_expr(c, st, order_by).alias(name))
        return exprs
    names: set[str] = set()
    for a in aggregations:
        col = a.get("column")
        if col not in schema:
            raise ValueError(f"There is no column called {col!r}")
        stats = a.get("stats") or ["mean"]
        check_stats(stats)
        alias = (a.get("alias") or "").strip()
        for st in stats:
            name = (alias if len(stats) == 1 else f"{alias}_{st}") if alias else f"{col}_{st}"
            if name in names:
                raise ValueError(f"Two of the chosen statistics would both be called {name!r}. Remove one or give it another name")
            names.add(name)
            exprs.append(stat_expr(col, st, order_by).alias(name))
    return exprs


def column_title(ctx: Any, name: str | None) -> str:
    """How a column is shown to a person: its display label, with the unit when it has one."""
    if not name:
        return ""
    meta = (getattr(ctx, "columns", None) or {}).get(name) or {}
    lab = meta.get("label") or name
    unit = meta.get("unit") or ""
    return f"{lab} ({unit})" if unit and unit not in lab else lab


def require_column(schema: dict[str, pl.DataType], name: str | None, what: str, kind: str | None = None) -> str:
    if not name:
        raise ValueError(f"Choose the {what}")
    if name not in schema:
        # case-insensitive fallback
        for c in schema:
            if c.lower() == str(name).lower():
                name = c
                break
        else:
            raise ValueError(f"There is no column called {name!r} for the {what}. Columns: {list(schema)[:15]}")
    if kind and kind_of_dtype(schema[name]) != kind:
        raise ValueError(f"The {what} ({name}) must be a {kind} column, but it is {kind_of_dtype(schema[name])}. "
                         f"Use a 'Fix numbers and dates' step first.")
    return name


def schema_of(lf: pl.LazyFrame) -> dict[str, pl.DataType]:
    return dict(lf.collect_schema())


def first_input(inputs: dict[str, list[pl.LazyFrame]], port: str = "in") -> pl.LazyFrame:
    frames = inputs.get(port) or []
    if not frames:
        raise ValueError("Nothing is connected to this step's input")
    return frames[0]


def private_temp(out: Path) -> Path:
    """A temporary file next to ``out`` that no other writer (another process, another thread) uses, so two
    runs writing the same file never overwrite each other's half-written copy. Keeps the extension."""
    import os
    import uuid
    return out.with_name(f".{out.stem}.{os.getpid()}.{uuid.uuid4().hex[:10]}.tmp{out.suffix}")
