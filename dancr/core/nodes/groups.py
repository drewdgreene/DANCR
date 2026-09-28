"""Compare groups: are the numbers different from one group to another, by how much, and how sure is that?

The question behind most small studies — sun leaves against shade leaves, treated plots against control plots,
this machine against that one — and a question big tables ask too (do orders from the web differ from orders in
store?). For each number the step gives each group's count, average, standard deviation, standard error and
median, the difference, a test of whether it is more than chance, and how big it is, then says the clearest
result in a sentence.

It also notices what a person in a hurry would miss, and says it:

- which test fits and why (groups that vary differently: Welch's t-test, not Student's);
- values that are not bell-shaped in a small sample, and whether a rank test agrees;
- values unusually far from the rest of their group, named by the row they are in;
- a number that grows with the size of what is measured (a gap in a leaf's outline grows with the leaf) in groups
  of different sizes: compared relative to size as well, and said when that turns the result round.

Counts, averages and spreads are computed over every row (streaming); the rank test, the checks of shape and the
size check use the first 100,000 rows. Deterministic, no Qt.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import polars as pl

from ..params import Param
from ..registry import NodeType, Ctx, NodeResult, registry
from ..findings import finding, fmt_number
from ..expr import NUM, kind_of_dtype
from ._common import first_input, schema_of, require_column

SAMPLE_ROWS = 100_000       # rows read for the rank test, the checks of shape and the size check
MAX_GROUPS = 12
SIZE_R = 0.6                # a number moves with size when its rank correlation with size inside the groups is this strong
ALPHA = 0.05
TESTS = [("auto", "choose for me"), ("welch", "Welch's t-test / ANOVA (groups may vary differently)"),
         ("student", "Student's t-test / ANOVA (groups vary alike)"), ("rank", "rank test (Mann–Whitney / Kruskal–Wallis)"),
         ("none", "no test")]


def _label(ctx: Ctx, name: str) -> str:
    """A column as a sentence names it: its label if it has one; else its header without the unit and short name,
    with what it measures in front of a cryptic one ('m (g)' is 'mass (m)', 'Leaf area (LA) cm2' is 'Leaf area')."""
    from ..units import header_parts, quantity_of
    meta = (ctx.columns or {}).get(name) or {}
    if meta.get("label") and meta["label"] != name:
        return meta["label"]
    core, _abbrev, unit = header_parts(name)
    q = quantity_of(meta.get("unit") or unit)
    if q and (len(core) <= 3 or "/" in core) and q.lower() not in core.lower():
        return f"{q} ({core})"
    return core


def _unit(ctx: Ctx, name: str) -> str:
    from ..units import unit_from_name
    meta = (ctx.columns or {}).get(name) or {}
    return meta.get("unit") or unit_from_name(name)


def p_text(p: float | None) -> str:
    if p is None or p != p:
        return "p not known"
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.2g}" if p < 0.1 else f"p = {p:.2f}"


def _clear(p: float | None) -> bool:
    """A difference more than chance: p below 5%. A p of 0 (a big sample) is the clearest, not "unknown"."""
    return p is not None and p == p and p < ALPHA


def _summaries(lf: pl.LazyFrame, by: str, cols: list[str]) -> pl.DataFrame:
    aggs: list[pl.Expr] = [pl.len().alias("__rows"), pl.col("__order").min().alias("__first")]
    for i, c in enumerate(cols):
        f = pl.col(c).cast(pl.Float64)
        x = pl.when(f.is_finite()).then(f)            # NaN and infinity are blanks, not values to average
        aggs += [x.count().alias(f"n{i}"), x.mean().alias(f"mean{i}"), x.std().alias(f"sd{i}"),
                 x.median().alias(f"med{i}"), x.min().alias(f"min{i}"), x.max().alias(f"max{i}")]
    # the groups as text, as the sample of rows is: True, dates and codes then match their rows exactly
    return (lf.with_row_index("__order").filter(pl.col(by).is_not_null()).with_columns(pl.col(by).cast(pl.Utf8))
            .group_by(by).agg(aggs).sort("__first").collect(engine="streaming"))


def _t_test(g: list[dict], welch: bool) -> tuple[float, float, float] | None:
    """(t, degrees of freedom, p) for two groups from their counts, averages and spreads."""
    from scipy import stats
    a, b = g
    if a["n"] < 2 or b["n"] < 2:
        return None
    va, vb = a["sd"] ** 2, b["sd"] ** 2
    if welch:
        se2 = va / a["n"] + vb / b["n"]
        if se2 <= 0:
            return None
        t = (a["mean"] - b["mean"]) / math.sqrt(se2)
        df = se2 ** 2 / ((va / a["n"]) ** 2 / (a["n"] - 1) + (vb / b["n"]) ** 2 / (b["n"] - 1))
    else:
        df = a["n"] + b["n"] - 2
        sp2 = ((a["n"] - 1) * va + (b["n"] - 1) * vb) / df
        if sp2 <= 0:
            return None
        t = (a["mean"] - b["mean"]) / math.sqrt(sp2 * (1 / a["n"] + 1 / b["n"]))
    return t, df, float(2 * stats.t.sf(abs(t), df))


def _anova(g: list[dict], welch: bool) -> tuple[float, float, float, float] | None:
    """(F, df1, df2, p) for several groups from their counts, averages and spreads (Welch's ANOVA when the groups
    may vary differently)."""
    from scipy import stats
    g = [x for x in g if x["n"] >= 2]
    k = len(g)
    if k < 2:
        return None
    if welch:
        if any(x["sd"] <= 0 for x in g):
            return None
        w = [x["n"] / x["sd"] ** 2 for x in g]
        sw = sum(w)
        mw = sum(wi * x["mean"] for wi, x in zip(w, g)) / sw
        a = sum(wi * (x["mean"] - mw) ** 2 for wi, x in zip(w, g)) / (k - 1)
        lam = sum((1 - wi / sw) ** 2 / (x["n"] - 1) for wi, x in zip(w, g))
        f = a / (1 + 2 * (k - 2) / (k * k - 1) * lam)
        df1, df2 = k - 1, (k * k - 1) / (3 * lam) if lam > 0 else float("inf")
    else:
        n = sum(x["n"] for x in g)
        grand = sum(x["n"] * x["mean"] for x in g) / n
        ssb = sum(x["n"] * (x["mean"] - grand) ** 2 for x in g)
        ssw = sum((x["n"] - 1) * x["sd"] ** 2 for x in g)
        df1, df2 = k - 1, n - k
        if ssw <= 0 or df2 <= 0:
            return None
        f = (ssb / df1) / (ssw / df2)
    return f, df1, df2, float(stats.f.sf(f, df1, df2))


def _effect(g: list[dict]) -> tuple[float | None, str]:
    """How big the difference is, whatever the sample size: Hedges' g for two groups, eta² for more."""
    if len(g) == 2:
        a, b = g
        df = a["n"] + b["n"] - 2
        if df <= 0:
            return None, "g"
        sp = math.sqrt(((a["n"] - 1) * a["sd"] ** 2 + (b["n"] - 1) * b["sd"] ** 2) / df)
        if sp <= 0:
            return None, "g"
        return (a["mean"] - b["mean"]) / sp * (1 - 3 / (4 * df - 1)), "g"
    n = sum(x["n"] for x in g)
    grand = sum(x["n"] * x["mean"] for x in g) / n
    ssb = sum(x["n"] * (x["mean"] - grand) ** 2 for x in g)
    sst = ssb + sum((x["n"] - 1) * x["sd"] ** 2 for x in g)
    return (ssb / sst if sst > 0 else None), "eta²"


def _size_word(e: float | None, kind: str) -> str:
    if e is None:
        return ""
    e = abs(e)
    if kind in ("g", "dz"):
        return "very large" if e >= 1.2 else "large" if e >= 0.8 else "medium" if e >= 0.5 else "small" if e >= 0.2 else "negligible"
    return "large" if e >= 0.14 else "medium" if e >= 0.06 else "small" if e >= 0.01 else "negligible"


def _rank_test(vals: list[np.ndarray]) -> float | None:
    from scipy import stats
    vals = [v for v in vals if v.size >= 2]
    if len(vals) < 2:
        return None
    try:
        if len(vals) == 2:
            return float(stats.mannwhitneyu(vals[0], vals[1], alternative="two-sided").pvalue)
        return float(stats.kruskal(*vals).pvalue)
    except ValueError:
        return None


def _bell_shaped(v: np.ndarray) -> bool:
    """Whether a small sample could well come from a bell curve (Shapiro–Wilk at 5%). Large samples are taken as
    they are: the tests on averages hold for them either way."""
    from scipy import stats
    if v.size < 3 or v.size > 50 or float(np.ptp(v)) == 0.0:
        return True
    return float(stats.shapiro(v).pvalue) >= ALPHA


def _unusual(v: np.ndarray) -> tuple[float, float]:
    q1, q3 = np.percentile(v, [25, 75])
    return q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)


def _compare(lf: pl.LazyFrame, by: str, cols: list[str], names: dict[str, str], test: str) -> tuple[list[dict], list[str]]:
    """One result per number: the groups' statistics, the test and its verdict."""
    s = _summaries(lf, by, cols)
    if s.height < 2:
        raise ValueError(f"{by} has only one group here, so there is nothing to compare")
    if s.height > MAX_GROUPS:
        raise ValueError(f"{by} has {s.height} groups. Compare at most {MAX_GROUPS} (filter the rows first, or group "
                         f"by something with fewer values)")
    groups = [str(v) for v in s[by].to_list()]
    out = []
    for i, c in enumerate(cols):
        g = [{"group": groups[r], "n": int(s[f"n{i}"][r] or 0), "mean": s[f"mean{i}"][r], "sd": s[f"sd{i}"][r] or 0.0,
              "median": s[f"med{i}"][r], "min": s[f"min{i}"][r], "max": s[f"max{i}"][r]} for r in range(s.height)]
        g = [x for x in g if x["n"] > 0]
        if len(g) < 2:
            continue
        for x in g:
            x["se"] = x["sd"] / math.sqrt(x["n"]) if x["n"] > 0 else None
        out.append({"column": c, "label": names.get(c, c), "groups": g})
    return out, groups


def _paired(res: dict, test: str, sample: pl.DataFrame, by: str, pair_by: str) -> None:
    """Two groups measured on the same things (each leaf before and after, each patient on two drugs): the test is
    on the differences within each pair."""
    from scipy import stats
    g, c = res["groups"], res["column"]
    if len(g) != 2:
        raise ValueError("A paired comparison is of two groups. Choose two (filter the others out first)")
    a, b = (sample.filter(pl.col(by) == x["group"]).select(pl.col(pair_by), pl.col(c).cast(pl.Float64).alias(f"v{i}"))
            .drop_nulls().unique(pair_by, keep="first") for i, x in enumerate(g))
    both = a.join(b, on=pair_by, how="inner")
    d = (both["v0"] - both["v1"]).to_numpy()
    res["pairs"] = int(d.size)
    res["notes"] = [f"{d.size} pairs matched on {pair_by}"]
    res["effect_kind"] = "dz"
    if d.size < 2:
        res["test"], res["statistic"], res["df"], res["p"], res["effect"] = "paired t-test", None, None, None, None
        return
    sd = float(np.std(d, ddof=1))
    res["effect"] = float(np.mean(d)) / sd if sd > 0 else None
    if test == "rank":
        res["test"] = "Wilcoxon signed-rank test"
        try:
            res["p"] = float(stats.wilcoxon(d).pvalue)
        except ValueError:
            res["p"] = None
        res["statistic"] = res["df"] = None
        return
    res["test"] = "paired t-test"
    t = float(np.mean(d)) / (sd / math.sqrt(d.size)) if sd > 0 else None
    res["statistic"], res["df"] = t, d.size - 1
    res["p"] = float(2 * stats.t.sf(abs(t), d.size - 1)) if t is not None else None


def _test_one(res: dict, test: str, sample: pl.DataFrame, by: str, pair_by: str = "") -> None:
    """Fills in the test, its p value, the effect size and the notes for one number."""
    g = res["groups"]
    c = res["column"]
    if pair_by:
        _paired(res, test, sample, by, pair_by)
        res["unusual"] = []
        return
    vals = [v[np.isfinite(v)] for v in (sample.filter(pl.col(by).cast(pl.Utf8) == x["group"])[c].cast(pl.Float64)
                                        .drop_nulls().to_numpy() for x in g)] if c in sample.columns else []
    shaped = all(_bell_shaped(v) for v in vals) if vals else True
    small = min(x["n"] for x in g) < 30
    choice = test
    if test == "auto":
        choice = "welch"
    res["notes"] = []
    stat = df = p = None
    if choice in ("welch", "student"):
        if len(g) == 2:
            r = _t_test(g, welch=choice == "welch")
            if r:
                stat, df, p = r
            res["test"] = "Welch's t-test" if choice == "welch" else "Student's t-test"
        else:
            r = _anova(g, welch=choice == "welch")
            if r:
                stat, df, p = r[0], r[2], r[3]
            res["test"] = "Welch's ANOVA" if choice == "welch" else "one-way ANOVA"
        if test == "auto":
            sds = [x["sd"] for x in g if x["sd"]]
            res["uneven"] = len(sds) >= 2 and max(sds) >= 1.5 * min(sds)
        if vals and small and not shaped:
            rp = _rank_test(vals)
            if rp is not None:
                agree = _clear(rp) == _clear(p)
                res["rank_p"] = rp
                res["notes"].append(f"its values are not bell-shaped in a small sample, and a rank test "
                                    f"{'agrees' if agree else 'does not agree'} ({p_text(rp)})")
    elif choice == "rank":
        p = _rank_test(vals)
        res["test"] = "Mann–Whitney test" if len(g) == 2 else "Kruskal–Wallis test"
    else:
        res["test"] = ""
    res["statistic"], res["df"], res["p"] = stat, df, p
    res["effect"], res["effect_kind"] = _effect(g)
    res["unusual"] = []
    for x, v in zip(g, vals):
        if 4 <= v.size <= 1000:
            lo, hi = _unusual(v)
            res["unusual"] += [(x["group"], float(u)) for u in v if u < lo or u > hi]


def _verdict(res: dict, ctx_unit: str) -> str:
    """The result for one number in words: 'Shade higher (p < 0.001)', 'no clear difference (p = 0.4)'."""
    g, p = res["groups"], res.get("p")
    hi = max(g, key=lambda x: x["mean"])
    lo = min(g, key=lambda x: x["mean"])
    if p is None:
        return f"{hi['group']} highest" if len(g) > 2 else f"{hi['group']} higher"
    if p >= ALPHA:
        return f"no clear difference ({p_text(p)})"
    return (f"{hi['group']} higher ({p_text(p)})" if len(g) == 2 else
            f"differs ({p_text(p)}): {hi['group']} highest, {lo['group']} lowest")


def sentence(res: dict, unit: str) -> str:
    """The clearest way to say one result: 'Shade has 2.6 times the leaf area of Sun (126.6 vs 48.5 cm²; p < 0.001)'."""
    g, p = res["groups"], res.get("p")
    hi = max(g, key=lambda x: x["mean"])
    lo = min(g, key=lambda x: x["mean"])
    what = res["label"]
    u = f" {unit}" if unit else ""
    if hi["mean"] == lo["mean"]:
        return f"{what[:1].upper()}{what[1:]} is the same in every group ({fmt_number(hi['mean'], 4)}{u})"
    nums = f"{fmt_number(hi['mean'], 4)} vs {fmt_number(lo['mean'], 4)}{u}" if len(g) == 2 else \
        f"{hi['group']} {fmt_number(hi['mean'], 4)}, {lo['group']} {fmt_number(lo['mean'], 4)}{u}"
    if p is not None and p >= ALPHA:
        seen = ", ".join(f"{x['group']} {fmt_number(x['mean'], 4)}" for x in g[:4])
        if what.lower() == "value":
            return f"{' and '.join(x['group'] for x in g) if len(g) <= 3 else 'The groups'} do not differ clearly ({seen}{u}; {p_text(p)})"
        return f"{what[:1].upper()}{what[1:]} does not differ clearly between " \
               f"{' and '.join(x['group'] for x in g) if len(g) <= 3 else 'the groups'} ({seen}{u}; {p_text(p)})"
    tail = f" ({nums}; {p_text(p)})" if p is not None else f" ({nums})"
    if what.lower() == "value" and lo["mean"] and lo["mean"] > 0:        # groups kept in columns: the groups are the story
        ratio = hi["mean"] / lo["mean"]
        return (f"{hi['group']} is {ratio:.1f} times {lo['group']}" if ratio >= 2 else
                f"{hi['group']} is {(ratio - 1) * 100:.0f}% higher than {lo['group']}") + tail
    if lo["mean"] and lo["mean"] > 0 and hi["mean"] / lo["mean"] >= 2:
        return f"{hi['group']} has {hi['mean'] / lo['mean']:.1f} times the {what} of {lo['group']}" + tail
    if lo["mean"] and lo["mean"] > 0:
        pct = (hi["mean"] - lo["mean"]) / lo["mean"] * 100
        return f"{hi['group']} has {pct:.0f}% more {what} than {lo['group']}" + tail
    return f"{hi['group']} has the highest {what}" + tail


def _within_r(sample: pl.DataFrame, by: str, a: str, b: str) -> float | None:
    """How closely two numbers go together inside the groups: a rank correlation of each group's own ranks, so two
    groups that differ in both are not taken for a relation, and one mistyped value does not hide one."""
    d = sample.select(pl.col(by).cast(pl.Utf8).alias("g"), pl.col(a).cast(pl.Float64).alias("a"),
                      pl.col(b).cast(pl.Float64).alias("b")).drop_nulls().drop_nans()
    if d.height < 8:
        return None
    d = d.with_columns(((pl.col("a").rank().over("g") - 0.5) / pl.len().over("g")).alias("a"),
                       ((pl.col("b").rank().over("g") - 0.5) / pl.len().over("g")).alias("b"))
    r = d.select(pl.corr("a", "b")).item()
    return float(r) if r is not None and r == r else None


def _size_checks(sample: pl.DataFrame, by: str, results: list[dict], names: dict[str, str], test: str,
                 derivations: list) -> list[dict]:
    """Numbers that grow with the size of what is measured, in groups of different sizes, compared relative to
    size as well; kept when that turns the result round or changes whether it is clear."""
    cols = [r["column"] for r in results]
    ratio_nums = {d.operands[0] for d in derivations if d.op == "ratio"}       # already put relative to something
    ratio_cols = {d.target for d in derivations if d.op == "ratio"}
    pairs_in = [set(d.operands) for d in derivations]
    positive = [c for c in cols if c in sample.columns and c not in ratio_cols and
                bool((sample[c].cast(pl.Float64).drop_nulls() > 0).all()) and sample[c].drop_nulls().len() >= 8]
    if len(positive) < 2:
        return []
    by_col = {r["column"]: r for r in results}
    # the size: the number most bound up with the others, and different between the groups
    def bound(c: str) -> float:
        rs = [abs(_within_r(sample, by, c, o) or 0.0) for o in positive if o != c]
        return sum(rs) / len(rs) if rs else 0.0
    sizes = [c for c in positive if _clear(by_col[c].get("p"))]
    if not sizes:
        return []
    size = max(sizes, key=lambda c: (bound(c), -cols.index(c)))
    out = []
    for r in results:
        y = r["column"]
        if y == size or y in ratio_nums or y in ratio_cols or any({y, size} <= s for s in pairs_in) or y not in sample.columns:
            continue
        rw = _within_r(sample, by, y, size)
        if rw is None or abs(rw) < SIZE_R:
            continue
        name = f"{names.get(y, y)} per {names.get(size, size)}"
        rel = sample.select(pl.col(by).cast(pl.Utf8), (pl.col(y).cast(pl.Float64) / pl.col(size).cast(pl.Float64)).alias(name))
        res, _ = _compare(rel.lazy(), by, [name], {name: name}, test)
        if not res:
            continue
        res = res[0]
        _test_one(res, test, rel, by)
        a, b = r["groups"], res["groups"]
        top_raw = max(a, key=lambda x: x["mean"])["group"]
        top_rel = max(b, key=lambda x: x["mean"])["group"]
        clear_raw = _clear(r.get("p"))
        clear_rel = _clear(res.get("p"))
        if (top_raw != top_rel and clear_rel) or clear_raw != clear_rel:
            res["size"] = size
            res["of"] = y
            res["r"] = rw
            res["turned"] = top_raw != top_rel
            out.append(res)
    return out[:3]


def _apply(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any]) -> NodeResult:
    lf = first_input(inputs)
    schema = schema_of(lf)
    by = require_column(schema, params.get("by"), "column that names the groups")
    label = (params.get("label") or "").strip()
    label = require_column(schema, label, "column that names the rows") if label else ""
    pair_by = (params.get("pair_by") or "").strip()
    pair_by = require_column(schema, pair_by, "column that pairs the rows") if pair_by else ""
    given = [require_column(schema, c, "column to compare") for c in (params.get("columns") or [])]
    cols = given or [c for c, dt in schema.items() if kind_of_dtype(dt) == NUM and c not in (by, label, pair_by)]
    cols = [c for c in cols if c != by]
    if not cols:
        raise ValueError("Choose the numbers to compare between the groups")
    bad = [c for c in cols if kind_of_dtype(schema[c]) != NUM]
    if bad:
        raise ValueError(f"{', '.join(bad)} {'holds' if len(bad) == 1 else 'hold'} text, so {'it' if len(bad) == 1 else 'they'} "
                         f"can't be compared as numbers")
    test = params.get("test") or "auto"
    names = {c: _label(ctx, c) for c in cols}
    results, groups = _compare(lf, by, cols, names, test)
    if not results:
        raise ValueError("No group has any of these numbers filled in")
    keep = list(dict.fromkeys([by] + cols + [x for x in (label, pair_by) if x]))
    sample = lf.select(keep).head(SAMPLE_ROWS).collect(engine="streaming").with_columns(pl.col(by).cast(pl.Utf8))
    for r in results:
        _test_one(r, test, sample, by, pair_by)
    derivs = []
    if params.get("size_check", True) and len(cols) >= 2 and not pair_by:
        from ..derived import find
        try:
            derivs = find(sample, cols)
        except Exception:  # noqa: BLE001 - the size check is a bonus, never a failure
            derivs = []
        extra = _size_checks(sample, by, results, names, test, derivs)
    else:
        extra = []
    rel = (params.get("relative_to") or "").strip()
    if rel:
        rel = require_column(schema, rel, "column to compare relative to")
        for r in results:
            if r["column"] == rel:
                continue
            name = f"{names[r['column']]} per {_label(ctx, rel)}"                       # noqa: E501
            d = lf.select(pl.col(by).cast(pl.Utf8), (pl.col(r["column"]).cast(pl.Float64) / pl.col(rel).cast(pl.Float64)).alias(name))
            res, _ = _compare(d, by, [name], {name: name}, test)
            if res:
                _test_one(res[0], test, d.head(SAMPLE_ROWS).collect(), by)
                if not any(e["label"] == name for e in extra):
                    extra.append(res[0])
    rows = []
    for r in results + extra:
        unit = _unit(ctx, r["column"]) if r in results else ""
        row: dict[str, Any] = {"measure": r["label"], "unit": unit}
        for x in r["groups"]:
            row[f"{x['group']} n"] = x["n"]
            row[f"{x['group']} mean"] = x["mean"]
            row[f"{x['group']} SD"] = x["sd"]
            row[f"{x['group']} SE"] = x["se"]
            row[f"{x['group']} median"] = x["median"]
        hi = max(r["groups"], key=lambda x: x["mean"])
        lo = min(r["groups"], key=lambda x: x["mean"])
        first, second = r["groups"][0], r["groups"][1]
        diff = (first["mean"] - second["mean"]) if len(r["groups"]) == 2 else (hi["mean"] - lo["mean"])
        base = second["mean"] if len(r["groups"]) == 2 else lo["mean"]
        row["difference"] = diff
        row["difference (%)"] = (diff / base * 100) if base else None
        row["test"] = r.get("test", "")
        row["statistic"] = r.get("statistic")
        row["p value"] = r.get("p")
        row["effect size"] = r.get("effect")
        row["effect"] = f"{_size_word(r.get('effect'), r.get('effect_kind', 'g'))} ({r.get('effect_kind', 'g')} = {fmt_number(r.get('effect'))})" \
            if r.get("effect") is not None else ""
        row["result"] = _verdict(r, unit)
        r["sentence"] = sentence(r, unit)
        rows.append(row)
    out = pl.DataFrame(rows, infer_schema_length=None)
    # the clearest result first in the finding: a clear difference with the biggest effect
    clear = [r for r in results if r.get("p") is not None and r["p"] < ALPHA]
    best = max(clear or results, key=lambda r: (abs(r.get("effect") or 0.0), -results.index(r)))
    first_label = f" in {len(groups)} groups" if len(groups) > 2 else ""
    statement = best["sentence"]
    msgs = []
    tests = list(dict.fromkeys(r.get("test") for r in results if r.get("test")))
    why = ""
    if pair_by:
        why = f" ({tests[0] if tests else 'paired'}: each {pair_by} is compared with itself)"
    elif tests and any(r.get("uneven") for r in results):
        why = f" ({tests[0]}, which does not assume the groups vary alike: here they don't)"
    elif tests:
        why = f" ({', '.join(tests)})"
    among = ' and '.join(groups) if len(groups) <= 3 else f'the {len(groups)} groups'
    msgs.append(f"{len(clear)} of {len(results)} number{'s' if len(results) != 1 else ''} differ{'s' if len(results) == 1 else ''} "
                f"clearly between {among}{why}")
    for r in results:
        msgs.append(r["sentence"] + (f". Note: {'; '.join(r['notes'])}" if r.get("notes") else ""))
    for e in extra:
        if e.get("size"):
            of, size = names.get(e["of"], e["of"]), names.get(e["size"], e["size"])
            turned = "the other way round" if e["turned"] else ("clear" if _clear(e.get("p")) else "no longer clear")
            msgs.append(f"{of[:1].upper()}{of[1:]} grows with {size} (rank correlation {e['r']:.2f} within the groups), and the "
                        f"groups differ in {size}. Relative to {size} the result is {turned}: {e['sentence']}")
    named = {}
    if label:
        for r in results:
            for grp, v in r.get("unusual", []):
                hit = sample.filter((pl.col(by) == grp) & (pl.col(r["column"]).cast(pl.Float64) == v))
                if hit.height:
                    named.setdefault(r["label"], []).append(f"{fmt_number(v, 4)} ({label} {hit[label][0]}, {grp})")
    else:
        for r in results:
            for grp, v in r.get("unusual", []):
                named.setdefault(r["label"], []).append(f"{fmt_number(v, 4)} ({grp})")
    for what, items in list(named.items())[:4]:
        msgs.append(f"Unusual for its group — {what}: {', '.join(items[:4])}")
    turned = [e for e in extra if e.get("turned")]
    if turned:
        e = turned[0]
        top = max(e["groups"], key=lambda x: x["mean"])["group"]
        statement += f". Relative to {names.get(e['size'], e['size'])}, {names.get(e['of'], e['of'])} is the other way round: " \
                     f"{top} is higher ({p_text(e.get('p'))})"
    report = {"groups": groups, "compared": len(results), "clear": len(clear), "by": by,
              "results": [{"measure": r["label"], "test": r.get("test"), "p": r.get("p"), "effect": r.get("effect"),
                           "sentence": r["sentence"], "notes": r.get("notes", [])} for r in results],
              "relative": [{"measure": e["label"], "p": e.get("p"), "sentence": e["sentence"], "turned": e.get("turned", False)}
                           for e in extra],
              "finding": finding("groups", statement, magnitude=max(abs(best.get("effect") or 0.0), 1.0 if clear else 0.0),
                                 direction="flat", confidence=1.0 if best.get("p") is not None and best["p"] < ALPHA else 0.5,
                                 exact=True, groups=groups)}
    return NodeResult(out.lazy(), report=report, messages=msgs + ([f"Groups{first_label}: {', '.join(groups)}"] if len(groups) > 2 else []))


registry.register(NodeType(
    key="compare_groups", uses_labels=True, label="Compare groups", category="Analyse & model", icon="⫿",
    description="Are the numbers different from one group to another? Each group's count, average, spread and "
                "standard error, the difference, a test of whether it is more than chance, and how big it is.",
    apply=_apply,
    summary=lambda p: f"by {p.get('by')}" if p.get("by") else "choose the groups",
    params=[
        Param("by", "Groups are in", "column", required=True, help="The column that says which group each row is in"),
        Param("columns", "Numbers to compare", "columns", default=[], column_group="numeric",
              help="Blank = every number column"),
        Param("test", "Test", "choice", default="auto", choices=TESTS,
              help="'Chosen for me' uses Welch's test, which does not assume the groups vary alike, and checks it "
                   "with a rank test when a small sample is not bell-shaped"),
        Param("pair_by", "Pairs are matched by", "column", default="",
              help="When the same things are measured in both groups (each plant before and after), the column that "
                   "says which is which: the test is then on the difference within each pair"),
        Param("label", "Rows are named by", "column", default="", advanced=True,
              help="A column that names each row (leaf number, sample id), used to point at unusual values"),
        Param("size_check", "Check for size effects", "bool", default=True, advanced=True,
              help="When a number grows with the size of what is measured and the groups differ in size, compare it "
                   "relative to size as well and say when that changes the result"),
        Param("relative_to", "Also compare relative to", "column", default="", column_group="numeric", advanced=True,
              help="Divide each number by this one (per area, per person) and compare that too"),
    ],
))
