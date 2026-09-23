from __future__ import annotations

from typing import Any

import polars as pl

from ..expr import _kind_of_dtype, NUM, TIME

STAT_CHOICES = [
    ("mean", "average"), ("median", "median"), ("min", "minimum"), ("max", "maximum"),
    ("std", "standard deviation"), ("sum", "total"), ("count", "count"), ("first", "first"), ("last", "last"),
    ("n_unique", "distinct count"),
]
STAT_HELP = "Any of: " + ", ".join(v for v, _ in STAT_CHOICES) + ". One statistic keeps the column names; several add a suffix"


def stat_expr(col: str, stat: str) -> pl.Expr:
    c = pl.col(col)
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
    raise ValueError(f"Unknown statistic {stat!r}")


def numeric_columns(schema: dict[str, pl.DataType], exclude: list[str] | None = None) -> list[str]:
    ex = set(exclude or [])
    return [c for c, dt in schema.items() if _kind_of_dtype(dt) == NUM and c not in ex]


def temporal_columns(schema: dict[str, pl.DataType]) -> list[str]:
    return [c for c, dt in schema.items() if _kind_of_dtype(dt) == TIME]


def build_aggregations(schema: dict[str, pl.DataType], aggregations: list[dict[str, Any]] | None,
                       exclude: list[str], default_stats: tuple[str, ...] = ("mean",),
                       only: list[str] | None = None) -> list[pl.Expr]:
    """[{column, stats:[...], alias?}] -> agg exprs. Empty -> default stats of the chosen (or every) numeric column.
    A single statistic keeps the original column names; several add a _stat suffix."""
    exprs: list[pl.Expr] = []
    valid = {v for v, _ in STAT_CHOICES}
    for st in default_stats:
        if st not in valid:
            raise ValueError(f"Unknown statistic {st!r}. Use: {', '.join(sorted(valid))}")
    if not aggregations:
        for c in (only or numeric_columns(schema, exclude)):
            for st in default_stats:
                name = c if len(default_stats) == 1 else f"{c}_{st}"
                exprs.append(stat_expr(c, st).alias(name))
        return exprs
    for a in aggregations:
        col = a.get("column")
        if col not in schema:
            raise ValueError(f"There is no column called {col!r}")
        stats = a.get("stats") or ["mean"]
        for st in stats:
            if st not in valid:
                raise ValueError(f"Unknown statistic {st!r}. Use: {', '.join(sorted(valid))}")
        alias = (a.get("alias") or "").strip()
        for st in stats:
            name = (alias if len(stats) == 1 else f"{alias}_{st}") if alias else f"{col}_{st}"
            exprs.append(stat_expr(col, st).alias(name))
    return exprs


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
    if kind and _kind_of_dtype(schema[name]) != kind:
        raise ValueError(f"The {what} ({name}) must be a {kind} column, but it is {_kind_of_dtype(schema[name])}. "
                         f"Use a 'Change type' node first.")
    return name


def schema_of(lf: pl.LazyFrame) -> dict[str, pl.DataType]:
    return dict(lf.collect_schema())


def first_input(inputs: dict[str, list[pl.LazyFrame]], port: str = "in") -> pl.LazyFrame:
    frames = inputs.get(port) or []
    if not frames:
        raise ValueError("Nothing is connected to this node's input")
    return frames[0]
