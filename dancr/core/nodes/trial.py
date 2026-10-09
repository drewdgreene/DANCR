"""Trial analysis: a randomised complete block design, fitted as a mixed model.

A breeding trial measures a value on each plot, and the same genotype is grown in several blocks (reps), so the
blocks must be accounted for before genotypes are compared. This fits the standard model

    value = mean + block + genotype + error,   genotype ~ N(0, sigma2_g), error ~ N(0, sigma2_e)

with the **block effects fixed** and the **genotype effects random**, estimating the variance components by
REML — the same estimates a breeder gets from a mixed-model package — and reporting each genotype's **BLUP**
(its shrunken estimate) and the trial's **heritability**.

Deterministic and offline: a small fixed-point iteration over the mixed-model equations, no model, no
dependency beyond numpy. Balanced and unbalanced designs both work; an optional group column fits each trial
(or site-year) separately.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from ..params import Param
from ..registry import NodeType, InputSpec, Ctx, NodeResult, registry
from ..findings import finding, fmt_number, plural
from ._common import first_input, schema_of, require_column

REML_MAX_ITER = 200
REML_TOL = 1e-10
MIN_GROUPS = 2


def _design(labels: list[str]) -> tuple[np.ndarray, list[str]]:
    """A 0/1 incidence matrix, one column per distinct label, in sorted (deterministic) order."""
    levels = sorted(set(labels))
    index = {v: i for i, v in enumerate(levels)}
    z = np.zeros((len(labels), len(levels)))
    for r, v in enumerate(labels):
        z[r, index[v]] = 1.0
    return z, levels


def _reml(y: np.ndarray, X: np.ndarray, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """REML fit of ``y = X b + Z u + e`` with ``u ~ N(0, s2g I)``, ``e ~ N(0, s2e I)``.

    Returns (b, u, s2g, s2e). The fixed-effect design X carries the intercept and the block effects; Z carries
    the genotype effects. The variance components are updated from the mixed-model solution until they settle;
    the iteration is a deterministic fixed point, so the same data always gives the same answer."""
    n, p, q = y.shape[0], X.shape[1], Z.shape[1]
    XtX, XtZ, ZtZ = X.T @ X, X.T @ Z, Z.T @ Z
    ZtX = XtZ.T
    Xty, Zty = X.T @ y, Z.T @ y
    beta0, *_ = np.linalg.lstsq(X, y, rcond=None)
    r0 = y - X @ beta0
    s2e = max(float(r0 @ r0) / max(1, n - p), 1e-12)
    s2g = max(s2e * 0.1, 1e-12)
    beta = beta0
    u = np.zeros(q)
    for _ in range(REML_MAX_ITER):
        lam = s2e / s2g
        C = np.block([[XtX, XtZ], [ZtX, ZtZ + lam * np.eye(q)]])
        Cinv = np.linalg.pinv(C)                       # pinv tolerates a rank-deficient design, deterministically
        sol = Cinv @ np.concatenate([Xty, Zty])
        beta, u = sol[:p], sol[p:]
        r = y - X @ beta - Z @ u
        trace_uu = float(np.trace(Cinv[p:, p:]))
        # at the MME solution Z'r = lambda*u, so the residual update carries that term too (REML, not a plain r'r)
        new_e = max((float(r @ r) + lam * float(u @ u)) / max(1, n - p), 1e-12)
        new_g = max((float(u @ u) + new_e * trace_uu) / q, 1e-12)
        settled = abs(new_e - s2e) <= REML_TOL * max(1.0, s2e) and abs(new_g - s2g) <= REML_TOL * max(1.0, s2g)
        s2e, s2g = new_e, new_g
        if settled:
            break
    return beta, u, s2g, s2e


def _fit(df: pl.DataFrame, value: str, geno: str, block: str | None) -> dict[str, Any]:
    """One trial's fit: per-genotype BLUP and mean, the variance components, and the heritability."""
    df = df.drop_nulls([value, geno] + ([block] if block else []))
    y = df[value].cast(pl.Float64).to_numpy()
    g = [str(v) for v in df[geno].to_list()]
    Z, levels = _design(g)
    if block and df[block].n_unique() >= 2:
        b = [str(v) for v in df[block].to_list()]
        B, _ = _design(b)
        X = np.hstack([np.ones((len(y), 1)), B[:, 1:]])   # intercept + block effects (first block is the reference)
    else:
        X = np.ones((len(y), 1))
    grand = float(np.mean(y)) if len(y) else 0.0
    if len(levels) < MIN_GROUPS or y.shape[0] <= X.shape[1]:
        # too little to separate a genetic variance: report the plain means, no shrinkage
        rows = {lv: {"n": 0, "sum": 0.0} for lv in levels}
        for v, lv in zip(y.tolist(), g):
            rows[lv]["n"] += 1; rows[lv]["sum"] += v
        out = [{"genotype": lv, "n": r["n"], "mean": (r["sum"] / r["n"] if r["n"] else None),
                "blup": (r["sum"] / r["n"] if r["n"] else None)} for lv, r in rows.items()]
        return {"rows": out, "s2g": None, "s2e": None, "h2": None, "grand": grand, "levels": levels}
    _, u, s2g, s2e = _reml(y, X, Z)
    idx = {lv: i for i, lv in enumerate(levels)}
    counts = np.bincount([idx[lv] for lv in g], minlength=len(levels))
    sums = np.zeros(len(levels))
    for v, lv in zip(y.tolist(), g):
        sums[idx[lv]] += v
    n_bar = float(np.mean(counts)) or 1.0
    h2 = s2g / (s2g + s2e / n_bar) if (s2g + s2e) > 0 else 0.0
    out = [{"genotype": lv, "n": int(counts[i]), "mean": float(sums[i] / counts[i]) if counts[i] else None,
            "blup": grand + float(u[i])} for lv, i in sorted(idx.items())]
    return {"rows": out, "s2g": s2g, "s2e": s2e, "h2": h2, "grand": grand, "levels": levels}


def _trial_analysis(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any]) -> NodeResult:
    lf = first_input(inputs)
    schema = schema_of(lf)
    value = require_column(schema, params.get("value"), "value column")
    geno = require_column(schema, params.get("genotype"), "genotype (line) column")
    block = (params.get("block") or "").strip() or None
    if block:
        block = require_column(schema, block, "block (rep) column")
    group = (params.get("group") or "").strip() or None
    if group:
        group = require_column(schema, group, "trial/site column")
    cols = [value, geno] + ([block] if block else []) + ([group] if group else [])
    df = lf.select(cols).collect(engine="streaming")
    if df.height == 0:
        raise ValueError("The trial table is empty")

    out_rows: list[dict[str, Any]] = []
    fits: list[dict[str, Any]] = []
    if group:
        for gval, sub in df.partition_by(group, maintain_order=True, as_dict=True).items():
            gv = gval[0] if isinstance(gval, tuple) else gval
            fit = _fit(sub, value, geno, block)
            fit["group"] = gv
            fits.append(fit)
            for r in fit["rows"]:
                out_rows.append({"group": gv, **r})
    else:
        fit = _fit(df, value, geno, block)
        fit["group"] = None
        fits.append(fit)
        out_rows = list(fit["rows"])

    # rank each group's genotypes by BLUP, best first
    out = pl.DataFrame(out_rows, infer_schema_length=None) if out_rows else pl.DataFrame(
        {"genotype": [], "n": [], "mean": [], "blup": []})
    if "blup" in out.columns:
        rank_expr = pl.col("blup").rank("ordinal", descending=True).cast(pl.Int64).alias("rank")
        if group:
            out = out.with_columns(rank_expr.over("group")).sort(["group", "rank"])
        else:
            out = out.with_columns(rank_expr).sort("rank")

    components = [{"group": f["group"], "genotypes": len(f["levels"]),
                   "variance_genotype": f["s2g"], "variance_residual": f["s2e"], "heritability": f["h2"]} for f in fits]
    h2s = [f["h2"] for f in fits if f["h2"] is not None]
    mean_h2 = float(np.mean(h2s)) if h2s else None
    top = out.row(0, named=True) if out.height else None
    name = _label(ctx, value)
    grand = next((f["grand"] for f in fits if f["group"] == (top or {}).get("group")), fits[0]["grand"] if fits else None)
    if top is not None and top.get("blup") is not None:
        where = f" in {top['group']}" if group and top.get("group") is not None else ""
        said = (f"{top['genotype']} has the highest {name} BLUP{where} "
                f"({fmt_number(top['blup'])} vs a trial mean of {fmt_number(grand)})"
                + (f"; heritability {mean_h2:.2f}" if mean_h2 is not None else ""))
    else:
        said = f"Fitted {plural(len(fits), 'trial')} of {name}"
    report = {"kind": "trial_analysis", "method": "RCBD (block fixed, genotype random, REML)", "value": value,
              "genotypes": len({r["genotype"] for r in out_rows}), "trials": len(fits),
              "components": components, "heritability": mean_h2, "overall_mean": fits[0]["grand"] if fits else None,
              "finding": finding("summary", said, magnitude=(mean_h2 or 0.0), exact=True)}
    return NodeResult(out.lazy(), report=report, messages=[said])


def _label(ctx: Ctx, column: str) -> str:
    return ((ctx.columns or {}).get(column) or {}).get("label") or column


registry.register(NodeType(
    key="trial_analysis", label="Analyse a trial (RCBD)", category="Analyse & model", icon="grid-four",
    description="Compare genotypes in a field trial, accounting for blocks: fits a randomised complete block "
                "design as a mixed model, estimates the genetic and residual variance, and gives each genotype a "
                "BLUP (a shrunken estimate) with the trial's heritability. Add a trial/site column to fit each "
                "trial separately. The output is one row per genotype, best BLUP first.",
    apply=_trial_analysis,
    summary=lambda p: f"BLUP of {p.get('value') or '?'} by {p.get('genotype') or '?'}",
    params=[
        Param("value", "Value to compare", "column", column_group="numeric", required=True,
              help="The measured number, for example grain yield."),
        Param("genotype", "Genotype (line) column", "column", required=True,
              help="The line, hybrid or variety whose performance you want — treated as a random effect."),
        Param("block", "Block (rep) column", "column", default="",
              help="The replicate or block each plot was in. Blocks are accounted for as a fixed effect."),
        Param("group", "Fit each trial separately", "column", default="",
              help="Optional: a trial or site column, so each site-year is its own analysis."),
    ],
))
