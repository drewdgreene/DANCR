"""Trial analysis: an RCBD fitted as a mixed model (BLUPs, variance components, heritability)."""
import numpy as np
import polars as pl
import pytest

from conftest import run_one
from dancr.core import Pipeline
from dancr.core.executor import Executor


def _balanced(q: int = 4, b: int = 3, seed: int = 1) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    geno = [f"G{i + 1}" for i in range(q)]
    block = [f"B{j + 1}" for j in range(b)]
    ge = rng.normal(0, 4, q); ge -= ge.mean()
    be = rng.normal(0, 2, b); be -= be.mean()
    rows = [{"line": g, "rep": bl, "yield": round(50 + ge[i] + be[j] + rng.normal(0, 1), 4)}
            for i, g in enumerate(geno) for j, bl in enumerate(block)]
    return pl.DataFrame(rows)


def _project(tmp_path, df):
    f = tmp_path / "trial.parquet"; df.write_parquet(f)
    p = Pipeline("t"); p.path = tmp_path / "t.json"
    p.add_node("load_file", params={"path": str(f)}, id="src")
    p.add_node("trial_analysis", params={"value": "yield", "genotype": "line", "block": "rep"}, id="ta")
    p.connect("src", "ta")
    return p


def test_trial_analysis_matches_the_closed_form_rcbd(tmp_path):
    df = _balanced()
    p = _project(tmp_path, df)
    out = run_one(p, "ta")
    st = Executor(p).state("ta")
    comp = st.report["components"][0]
    grand = df["yield"].mean()
    q, b = df["line"].n_unique(), df["rep"].n_unique()
    geno_means = df.group_by("line").agg(pl.col("yield").mean()).sort("line")
    ss_geno = b * ((geno_means["yield"] - grand) ** 2).sum()
    block_means = df.group_by("rep").agg(pl.col("yield").mean()).sort("rep")
    ss_block = q * ((block_means["yield"] - grand) ** 2).sum()
    ss_err = ((df["yield"] - grand) ** 2).sum() - ss_geno - ss_block
    ms_g, ms_e = ss_geno / (q - 1), ss_err / ((q - 1) * (b - 1))
    # the BLUP is exactly the shrunken deviation for the fitted components (balanced n_bar = b)
    shrink = comp["variance_genotype"] / (comp["variance_genotype"] + comp["variance_residual"] / b)
    got = {r["genotype"]: r["blup"] for r in out.iter_rows(named=True)}
    for r in geno_means.iter_rows(named=True):
        assert got[r["line"]] == pytest.approx(grand + shrink * (r["yield"] - grand), abs=1e-6)
    assert st.report["heritability"] == pytest.approx(shrink, abs=1e-9)
    # and for a balanced design REML reproduces the exact ANOVA estimates
    assert comp["variance_residual"] == pytest.approx(ms_e, abs=1e-6)
    assert comp["variance_genotype"] == pytest.approx((ms_g - ms_e) / b, abs=1e-4)
    assert st.report["heritability"] == pytest.approx(1 - ms_e / ms_g, abs=1e-6)
    assert out["rank"].to_list() == list(range(1, q + 1))        # ranked by BLUP, best first


def test_trial_analysis_fits_each_trial_separately(tmp_path):
    a = _balanced(seed=2).with_columns(pl.lit("T1").alias("trial"))
    b = _balanced(seed=3).with_columns(pl.lit("T2").alias("trial"))
    df = pl.concat([a, b])
    f = tmp_path / "trials.parquet"; df.write_parquet(f)
    p = Pipeline("t"); p.path = tmp_path / "t.json"
    p.add_node("load_file", params={"path": str(f)}, id="src")
    p.add_node("trial_analysis", params={"value": "yield", "genotype": "line", "block": "rep", "group": "trial"}, id="ta")
    p.connect("src", "ta")
    out = run_one(p, "ta")
    st = Executor(p).state("ta")
    assert set(out["group"].to_list()) == {"T1", "T2"}
    # a rank per trial: each trial has its own 1..q
    for t in ("T1", "T2"):
        assert sorted(out.filter(pl.col("group") == t)["rank"].to_list()) == [1, 2, 3, 4]
    assert st.report["trials"] == 2 and len(st.report["components"]) == 2
