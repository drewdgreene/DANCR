"""The maize-trial starter pack: the main project, the stewardship audit and the guarded knowledge base.

The pack is the example that shows DANCR end to end on domain-accurate (fictional) agricultural data, so
these tests pin the parts a newcomer is meant to see: a real statistical comparison, the seeded data
defects found, and the restricted document withheld from search by default.
"""
import polars as pl

from dancr.core import Pipeline
from dancr.core.executor import Executor
from dancr.core.samples import write_example


def _run(path):
    return Executor(Pipeline.load(path)).run()


def test_the_pack_writes_its_projects_and_a_walkthrough(tmp_path):
    write_example("trial", tmp_path)
    for name in ("Maize trial & data stewardship.json", "Stewardship audit.json",
                 "Data cleanup.json", "Knowledge base.json", "Trial analysis.json", "START-HERE.md"):
        assert (tmp_path / name).exists(), name


def test_the_trial_analysis_project_fits_an_rcbd(tmp_path):
    write_example("trial", tmp_path)
    res = _run(tmp_path / "Trial analysis.json")
    assert res["blups"].status == "done", res["blups"].error
    out = pl.read_parquet(res["blups"].output)
    assert {"genotype", "n", "mean", "blup", "rank", "group"} <= set(out.columns)
    assert out.height == 6 * 24                                  # 6 trials x 24 lines
    # a rank within each trial, best BLUP first, and a heritability per trial
    for g in out["group"].unique().to_list():
        assert sorted(out.filter(pl.col("group") == g)["rank"].to_list()) == list(range(1, 25))
    comps = res["blups"].report["components"]
    assert len(comps) == 6 and all(c["heritability"] is not None for c in comps)


def test_the_trial_project_answers_the_hero_question(tmp_path):
    res = _run(write_example("trial", tmp_path))
    hero = [s for s in res.values() if (s.report or {}).get("results")]
    assert hero, "the comparison produced no result"
    r = hero[0].report["results"][0]
    assert r["test"].startswith("Welch") and r["p"] < 0.01           # the right test, chosen on its own
    assert "rainfed" in r["sentence"] and "drought_stress" in r["sentence"] and "p < 0.001" in r["sentence"]


def test_the_stewardship_audit_finds_the_seeded_defects(tmp_path):
    write_example("trial", tmp_path)
    res = _run(tmp_path / "Stewardship audit.json")
    checks = set()
    for s in res.values():
        if s.node_id.startswith("check_contract") and s.output:
            checks |= set(pl.read_parquet(s.output)["check"].to_list())
    assert {"unique", "allowed values", "required", "reference"} <= checks


def test_the_pack_benchmark_reads_its_questions_as_intended(tmp_path):
    # the example's questions are a small benchmark: each must still read as the recipe it is meant to
    from dancr.headless import run_eval
    p = Pipeline.load(write_example("trial", tmp_path))
    cases = [
        {"id": "hero", "question": "t test grain yield rainfed vs drought_stress", "expect_recipe": "groups"},
        {"id": "breakdown", "question": "average grain yield by treatment", "expect_recipe": "breakdown"},
        {"id": "relationship", "question": "relationship between grain yield and plant height", "expect_recipe": "relationship"},
        {"id": "top", "question": "top 5 lines by grain yield", "expect_recipe": "top"},
        {"id": "quality", "question": "check the measurements", "expect_recipe": "quality"},
        {"id": "trend", "question": "grain yield per month", "expect_recipe": "trend"},
    ]
    out = run_eval(p, cases)
    assert out["ok"], out["cases"]


def test_the_knowledge_base_withholds_the_restricted_dossier(tmp_path):
    write_example("trial", tmp_path)
    res = _run(tmp_path / "Knowledge base.json")
    search = res["search"]
    assert search.status == "done", search.error
    assert search.report["hits"] >= 1 and search.report["withheld"] == 1
    # the shareable copy drops the restricted row and keeps the rest
    shareable = pl.read_csv(tmp_path / "ai_knowledge" / "corpus_shareable.csv")
    raw = pl.read_csv(tmp_path / "ai_knowledge" / "corpus.csv")
    assert shareable.height == raw.height - 1
    assert "restricted" not in [str(v).lower() for v in shareable["sensitivity"].to_list()]


def test_the_cleanup_repairs_the_defects_at_the_source(tmp_path):
    write_example("trial", tmp_path)
    res = _run(tmp_path / "Data cleanup.json")
    assert all(s.status == "done" for s in res.values())
    # the contract checks pass once the defects are repaired
    assert pl.read_parquet(res["reg_check"].output).height == 0
    assert pl.read_parquet(res["ev_check"].output).height == 0
    # the clean table drops the blank yields and keeps the corrected one
    clean = pl.read_csv(tmp_path / "trials" / "phenotype_measurements_clean.csv")
    raw = pl.read_csv(tmp_path / "trials" / "phenotype_measurements.csv")
    assert clean.height < raw.height and clean["grain_yield_q_ha"].max() < 200
