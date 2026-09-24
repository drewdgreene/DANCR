"""Column statistics over a LazyFrame. One column at a time on the streaming engine, so memory stays
flat. Everything is exact: quartiles come from a streaming (out-of-core) sort of the column."""
from __future__ import annotations

from typing import Any

import polars as pl

from ..core.expr import _kind_of_dtype, NUM, TIME, STR, BOOL

SUMMARY_SCHEMA = {
    "column": pl.Utf8, "type": pl.Utf8, "rows": pl.Int64, "missing": pl.Int64,
    "mean": pl.Float64, "std": pl.Float64, "min": pl.Float64, "q25": pl.Float64, "median": pl.Float64,
    "q75": pl.Float64, "max": pl.Float64, "earliest": pl.Utf8, "latest": pl.Utf8, "distinct": pl.Int64,
}
STREAM = "streaming"


def exact_quantiles(lf: pl.LazyFrame, expr: pl.Expr, qs: tuple[float, ...] = (0.25, 0.5, 0.75)) -> list[float | None]:
    """Exact quantiles (nearest-rank) of a numeric expression over every row: one streaming sort, then the ranks."""
    src = lf.select(expr.cast(pl.Float64).alias("v")).filter(pl.col("v").is_not_null() & pl.col("v").is_not_nan())
    n = int(src.select(pl.len()).collect(engine=STREAM)[0, 0])
    if n == 0:
        return [None] * len(qs)
    ranks = [int(round(q * (n - 1))) for q in qs]
    hits = (src.sort("v").with_row_index("i").filter(pl.col("i").is_in(sorted(set(ranks))))
            .collect(engine=STREAM))
    by_rank = dict(zip(hits["i"].to_list(), hits["v"].to_list()))
    return [by_rank.get(r) for r in ranks]


def column_summary(lf: pl.LazyFrame, columns: list[str] | None = None) -> pl.DataFrame:
    schema = dict(lf.collect_schema())
    cols = columns or list(schema)
    for c in cols:
        if c not in schema:
            raise ValueError(f"There is no column called {c!r}")
    total = int(lf.select(pl.len()).collect(engine=STREAM)[0, 0])
    records = []
    for c in cols:
        kind = _kind_of_dtype(schema[c])
        e = pl.col(c)
        rec: dict[str, Any] = {k: None for k in SUMMARY_SCHEMA}
        rec.update({"column": c, "type": kind, "rows": total})
        if kind == NUM:
            f = e.cast(pl.Float64)
            r = lf.select([f.null_count().alias("missing"), f.mean().alias("mean"), f.std().alias("std"),
                           f.min().alias("min"), f.max().alias("max")]).collect(engine=STREAM).row(0, named=True)
            rec.update(r)
            q25, med, q75 = exact_quantiles(lf, f)
            rec.update({"q25": q25, "median": med, "q75": q75})
        elif kind == TIME:
            r = lf.select([e.null_count().alias("missing"), e.min().cast(pl.Utf8).alias("earliest"),
                           e.max().cast(pl.Utf8).alias("latest")]).collect(engine=STREAM).row(0, named=True)
            rec.update(r)
        elif kind == BOOL:
            r = lf.select([e.null_count().alias("missing"), e.cast(pl.Float64).mean().alias("mean")]).collect(engine=STREAM).row(0, named=True)
            rec.update(r)
        else:
            r = lf.select([e.null_count().alias("missing"), e.drop_nulls().n_unique().alias("distinct")]).collect(engine=STREAM).row(0, named=True)
            rec.update(r)
        rec["missing"] = int(rec["missing"] or 0)
        records.append(rec)
    return pl.DataFrame(records, schema=SUMMARY_SCHEMA, strict=False)


def quick_column_info(lf: pl.LazyFrame, column: str, sample_rows: int = 200_000) -> dict[str, Any]:
    """Fast stats on a head sample for header tooltips."""
    schema = dict(lf.collect_schema())
    dt = schema[column]
    kind = _kind_of_dtype(dt)
    s = lf.select(column).head(sample_rows).collect(engine=STREAM)[column]
    info: dict[str, Any] = {"column": column, "dtype": str(dt), "kind": kind, "sampled": len(s), "missing": int(s.null_count())}
    if kind == NUM and len(s) > s.null_count():
        f = s.cast(pl.Float64)
        info.update({"min": float(f.min()), "max": float(f.max()), "mean": float(f.mean())})
    elif kind == TIME and len(s) > s.null_count():
        info.update({"earliest": str(s.min()), "latest": str(s.max())})
    elif kind == STR:
        info["distinct"] = int(s.n_unique())
    return info
