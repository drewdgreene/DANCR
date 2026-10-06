from __future__ import annotations

import math
import re
from typing import Any

import polars as pl

from ...params import Param
from ...registry import NodeType, Ctx, NodeResult, registry
from ...findings import finding, fmt_number, fmt_pct
from ...expr import NUM, TIME, STR, kind_of_dtype
from ...dtypes import resolve_number, temp_name
from ...timeutil import parse_bucket
from .._common import first_input, schema_of, require_column, number_param
from ._shared import (
    STAT_CHOICES,
    ASSOC_ROWS,
    ASSOC_MAX_COLS,
    CATEGORY_CAP,
    _looks_like_key,
    _label,
    _stat,
    _value_expr,
    _stat_label,
    _direction,
    _verb,
    _every,
    _truncate,
    _latest_two,
    _period_label,
    _signed,
)

def _pearson(df: pl.DataFrame, a: str, b: str) -> float | None:
    d = df.select(pl.col(a).cast(pl.Float64), pl.col(b).cast(pl.Float64)).drop_nulls().drop_nans()
    if d.height < 10:
        return None
    try:
        r = d.select(pl.corr(a, b)).item()
    except Exception:  # noqa: BLE001
        return None
    return float(r) if r is not None and r == r else None


def _eta_squared(df: pl.DataFrame, cat: str, num: str) -> float | None:
    d = df.select(pl.col(cat).cast(pl.Utf8).alias("g"), pl.col(num).cast(pl.Float64).alias("y")).drop_nulls().drop_nans()
    if d.height < 10 or d["g"].n_unique() < 2:
        return None
    grand = d["y"].mean()
    g = d.group_by("g").agg([pl.col("y").mean().alias("m"), pl.len().alias("n")])
    ss_between = float((((g["m"] - grand) ** 2) * g["n"]).sum() or 0.0)
    ss_total = float(((d["y"] - grand) ** 2).sum() or 0.0)
    if ss_total <= 0:
        return None
    return max(0.0, min(1.0, ss_between / ss_total))


def _top_categories(df: pl.DataFrame, col: str) -> list[str]:
    vc = df[col].cast(pl.Utf8).value_counts(sort=True)
    name = vc.columns[0]
    return [v for v in vc[name].head(CATEGORY_CAP).to_list() if v is not None]


def _cramers_v(df: pl.DataFrame, a: str, b: str) -> float | None:
    va, vb = _top_categories(df, a), _top_categories(df, b)
    if len(va) < 2 or len(vb) < 2:
        return None
    d = df.select(pl.col(a).cast(pl.Utf8).alias("a"), pl.col(b).cast(pl.Utf8).alias("b")).drop_nulls() \
          .filter(pl.col("a").is_in(va) & pl.col("b").is_in(vb))
    n = d.height
    if n < 10:
        return None
    tab = d.group_by(["a", "b"]).len()
    rt = tab.group_by("a").agg(pl.col("len").sum().alias("rt"))
    ct = tab.group_by("b").agg(pl.col("len").sum().alias("ct"))
    t = tab.join(rt, on="a").join(ct, on="b")
    # chi² = n·(Σ O²/(row·col) − 1): the pairs that never occur count too (their expected share is not 0),
    # and they are exactly the ones a table of the pairs seen leaves out
    chi2 = n * (float((t["len"].cast(pl.Float64) ** 2 / (t["rt"].cast(pl.Float64) * t["ct"])).sum() or 0.0) - 1.0)
    denom = n * (min(rt.height, ct.height) - 1)          # the categories left once blanks are dropped
    return max(0.0, min(1.0, math.sqrt(max(chi2, 0.0) / denom))) if denom > 0 else None


def _associations(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any]) -> NodeResult:
    lf = first_input(inputs)
    schema = schema_of(lf)
    target = (params.get("target") or "").strip()
    target = require_column(schema, target, "target column") if target else ""
    given = [require_column(schema, c, "column") for c in (params.get("columns") or [])]
    pool = given or [c for c in schema if kind_of_dtype(schema[c]) in (NUM, STR)]
    pool = [c for c in pool if c != target]
    measures = [c for c in pool if kind_of_dtype(schema[c]) == NUM]
    others = [c for c in pool if c != target and kind_of_dtype(schema[c]) != NUM]
    chosen = (([target] if target else []) + (measures + others)[:ASSOC_MAX_COLS])
    if len(chosen) < 2:
        raise ValueError("Need at least two columns to look for relationships between")
    df = lf.select(chosen).head(ASSOC_ROWS).collect(engine="streaming")
    rows: list[dict[str, Any]] = []
    # with a target, only how each column relates to it; without one, every pair
    pairs = [(target, b) for b in chosen[1:]] if target else \
        [(a, b) for i, a in enumerate(chosen) for b in chosen[i + 1:]]
    for a, b in pairs:
        ka, kb = kind_of_dtype(schema[a]), kind_of_dtype(schema[b])
        if ka == NUM and kb == NUM:
            value, kind, label = _pearson(df, a, b), "straight line", "r"
        elif ka == NUM or kb == NUM:
            cat, num = (b, a) if ka == NUM else (a, b)
            e2 = _eta_squared(df, cat, num)
            # eta (the root of eta²) is on the same scale as r, so the two rank fairly against each other
            value, kind, label = (math.sqrt(e2) if e2 is not None else None), "group difference", "eta"
        else:
            value, kind, label = _cramers_v(df, a, b), "association", "V"
        if value is None:
            continue
        rows.append({"left": _label(ctx, a), "right": _label(ctx, b), "kind": kind, "measure": label,
                     "strength": round(abs(float(value)), 4), "value": round(float(value), 4)})
    if not rows:
        raise ValueError("No two columns had enough values to compare")
    out = pl.DataFrame(rows).sort("strength", descending=True)
    best = out.row(0, named=True)
    strong = int((out["strength"] >= 0.5).sum())
    movers = {"straight line": "move together", "group difference": "differ between groups",
              "association": "go together"}[best["kind"]]
    direction = "flat"
    if best["kind"] == "straight line":
        direction = _direction(best["value"])
    statement = (f"{best['left']} and {best['right']} {movers} most strongly "
                 f"({best['measure']} = {fmt_number(best['value'], 3)})"
                 + (f"; {strong} of {out.height} pairs are strong" if strong > 1 else ""))
    report = {"pairs": out.height, "strong": strong, "sample_rows": df.height, "target": _label(ctx, target) or None,
              "finding": finding("association", statement, magnitude=best["strength"], direction=direction,
                                 exact=False, sample_rows=df.height)}
    msgs = [f"From the first {df.height:,} rows. The strongest is {best['left']} ↔ {best['right']}"]
    return NodeResult(out.lazy(), report=report, messages=msgs)


registry.register(NodeType(
    key="associations", uses_labels=True, label="What relates to what", category="Analyse & model", icon="∞",
    description="Rank how strongly the columns move together: straight-line relationships between numbers, "
                "differences between groups, and associations between categories.",
    apply=_associations,
    summary=lambda p: f"relating to {p.get('target')}" if p.get("target") else "every pair of columns",
    params=[
        Param("target", "Related to", "column", help="Blank = every pair of columns"),
        Param("columns", "Columns to compare", "columns", default=[], help="Blank = all"),
    ],
))


# =================================================================== check data
