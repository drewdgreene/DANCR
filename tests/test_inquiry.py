"""Inquiry steps and their recipes: compare periods, contribution, associations, data checks, forecast.

These are the steps that *tell* a person something, so the tests check both the table they produce and the
finding they attach to it.
"""
from datetime import datetime, timedelta

import polars as pl
import pytest

from dancr.core import Pipeline
from dancr.core.ask import ask
from dancr.core.executor import Executor
from dancr.core.findings import collect, finding, fmt_number, fmt_pct, headline, record
from dancr.core.planner import instantiate
from dancr.core.recipes import plan
from dancr.core.understand import deepen, understand


def _pipe(df: pl.DataFrame, tmp_path, name="in.parquet") -> Pipeline:
    df.write_parquet(tmp_path / name)
    p = Pipeline("t")
    p.path = tmp_path / "p.json"
    p.add_node("load_file", params={"path": name}, id="src")
    return p


def _run(p: Pipeline, type_key: str, params: dict, nid: str = "s"):
    p.add_node(type_key, params=params, id=nid)
    p.connect("src", nid)
    st = Executor(p).run(targets=[nid])[nid]
    assert st.status == "done", st.error
    return st, pl.read_parquet(st.output)


def _model(p: Pipeline):
    ex = Executor(p)
    return deepen(p, ex, understand(p, ex))


def _sales(tmp_path) -> Pipeline:
    """Three regions, four months, growing sales — enough for periods, shares, fits and a trend."""
    rows = []
    for m in range(4):
        for region, base in (("North", 100), ("South", 80), ("East", 60)):
            rows.append({"time": datetime(2024, m + 1, 5 + (len(region) % 3)), "region": region,
                         "sales": base * (m + 1), "qty": (m + 1) * 2})
    return _pipe(pl.DataFrame(rows).with_columns(pl.col("time").cast(pl.Datetime("us"))), tmp_path, "sales.parquet")


# ------------------------------------------------------------------- the nodes
def test_compare_periods_says_what_changed(tmp_path):
    st, out = _run(_sales(tmp_path), "compare_periods",
                   {"time_column": "time", "every": "1mo", "measure": "sales", "stat": "sum", "by": ["region"]})
    assert {"previous", "current", "change", "change_percent"} <= set(out.columns)
    assert st.report["finding"]["kind"] == "change"
    assert "rose" in st.report["finding"]["statement"]
    assert out["current"].sum() > out["previous"].sum()


def test_contribution_ranks_and_shares(tmp_path):
    st, out = _run(_sales(tmp_path), "contribution", {"by": ["region"], "measure": "sales", "stat": "sum"})
    assert out["share_percent"].sum() == pytest.approx(100.0, abs=0.01)
    assert out.row(0, named=True)["region"] == "North"          # largest first
    assert st.report["finding"]["kind"] == "share"


def test_contribution_to_a_change(tmp_path):
    st, out = _run(_sales(tmp_path), "contribution",
                   {"by": ["region"], "measure": "sales", "stat": "sum", "time_column": "time", "every": "1mo"})
    assert "change" in out.columns and "contribution_percent" in out.columns
    assert st.report["finding"]["detail"]["mode"] == "change"


def test_associations_find_the_relationship(tmp_path):
    n = 40
    df = pl.DataFrame({"x": list(range(n)), "y": [2.0 * i for i in range(n)],
                       "g": ["a" if i < 20 else "b" for i in range(n)]})
    st, out = _run(_pipe(df, tmp_path, "assoc.parquet"), "associations", {"target": "x"})
    top = out.row(0, named=True)
    assert {top["left"], top["right"]} == {"x", "y"} and top["strength"] == pytest.approx(1.0, abs=1e-6)
    assert st.report["finding"]["kind"] == "association"


def test_check_data_flags_numbers_stored_as_text(tmp_path):
    df = pl.DataFrame({"id": [1, 2, 3, 4, 5, 6],
                       "amount": ["1,200", "n/a", "", "3,000", "4,500", "5,000"],
                       "region": ["North", None, "North", "South", "South", "South"]})
    st, out = _run(_pipe(df, tmp_path, "dirty.parquet"), "check_data", {})
    assert st.report["finding"]["kind"] == "quality"
    assert any("numbers stored as text" in (r or "") for r in out["issue"].to_list())


def test_check_data_says_when_it_is_clean(tmp_path):
    df = pl.DataFrame({"a": [1, 2, 3, 4, 5, 6], "b": ["x", "y", "z", "x", "y", "z"]})
    st, _ = _run(_pipe(df, tmp_path, "clean.parquet"), "check_data", {})
    assert "looks clean" in st.report["finding"]["statement"]


def test_forecast_projects_within_its_band_and_crosses(tmp_path):
    t0 = datetime(2024, 1, 1)
    df = pl.DataFrame({"time": [t0 + timedelta(days=i) for i in range(30)], "v": [10.0 + i for i in range(30)]})
    st, out = _run(_pipe(df, tmp_path, "lin.parquet"), "forecast",
                   {"time_column": "time", "column": "v", "horizon": 20, "every": "1d", "threshold": 50})
    assert out.height == 20 and out.width == 4
    assert st.report["slope_per_step"] == pytest.approx(1.0, abs=1e-6)
    assert "crosses 50" in st.report["finding"]["statement"]
    low = [c for c in out.columns if "lower" in c][0]
    high = [c for c in out.columns if "upper" in c][0]
    assert (out[low] <= out["v"]).all() and (out["v"] <= out[high]).all()


# ------------------------------------------------------------------- the recipes and questions
@pytest.mark.parametrize("spec", [
    {"recipe": "change", "table": "src", "measure": ["src", "sales"], "stat": "sum", "every": "1mo", "by": ["src", "region"]},
    {"recipe": "explain", "table": "src", "by": ["src", "region"], "measure": ["src", "sales"], "stat": "sum"},
    {"recipe": "drivers", "table": "src", "target": ["src", "sales"]},
    {"recipe": "quality", "table": "src"},
    {"recipe": "forecast", "table": "src", "measure": ["src", "sales"]},
])
def test_new_recipes_plan_and_run(spec, tmp_path):
    p = _sales(tmp_path)
    m = _model(p)
    pl_ = plan(m, spec)
    q = Pipeline.from_dict(p.to_dict(), p.path)
    resolved = instantiate(q, pl_)
    st = Executor(q).run(targets=[resolved[pl_.terminal]])[resolved[pl_.terminal]]
    assert st.status == "done", st.error
    assert (st.report or {}).get("finding", {}).get("statement")


@pytest.mark.parametrize("question,recipe", [
    ("what changed", "change"),
    ("what drives sales", "explain"),
    ("what relates to sales", "drivers"),
    ("check the data", "quality"),
    ("forecast sales", "forecast"),
])
def test_ask_understands_the_new_questions(question, recipe, tmp_path):
    m = _model(_sales(tmp_path))
    a = ask(m, question)
    assert a.ok, (question, a.message, a.unknown)
    assert a.spec["recipe"] == recipe


# ------------------------------------------------------------------- findings themselves
def test_findings_are_ranked_and_spoken():
    quiet = finding("association", "a mild link", magnitude=0.2)
    loud = finding("change", "sales fell 40%", magnitude=40, direction="down")
    records = [record("associations", "A", {"finding": quiet}), record("compare_periods", "C", {"finding": loud}), None]
    found = collect(records)
    assert [f["title"] for f in found] == ["C", "A"]
    assert headline(found) == "sales fell 40%"
    assert record("compare_periods", "C", {}) is None
    assert record("compare_periods", "C", {"finding": loud}, status="failed") is None


def test_number_and_percent_formatting():
    assert fmt_number(1234.5) == "1,234" or fmt_number(1234.5) == "1,230"
    assert fmt_number(None) == "–" and fmt_number(float("nan")) == "–"
    assert fmt_pct(12.0) == "12%" and fmt_pct(0.5) == "0.5%" and fmt_pct(None) == "–"


# ------------------------------------------------------------------- regressions
def _t(m: int) -> datetime:
    return datetime(2024, m, 5)


def test_compare_periods_averages_the_whole_period_not_the_groups(tmp_path):
    """With groups, the headline figure is the whole period's average (15 → 40), not a sum of group averages."""
    df = pl.DataFrame({"time": [_t(1)] * 2 + [_t(2)] * 3, "g": ["a", "b", "a", "b", "c"], "v": [10.0, 20, 30, 40, 50]})
    st, _ = _run(_pipe(df, tmp_path), "compare_periods",
                 {"time_column": "time", "every": "1mo", "measure": "v", "stat": "mean", "by": ["g"]})
    assert st.report["previous_total"] == pytest.approx(15.0) and st.report["current_total"] == pytest.approx(40.0)


def test_contribution_ranks_a_group_without_values_last(tmp_path):
    df = pl.DataFrame({"g": ["a", "a", "b", "b", "c"], "v": [None, None, 5.0, 1.0, 2.0]})
    st, out = _run(_pipe(df, tmp_path), "contribution", {"by": ["g"], "measure": "v", "stat": "mean"})
    assert out["g"].to_list() == ["b", "c", "a"]
    assert st.report["top"] == "b"


def test_contribution_of_a_zero_total_has_no_shares(tmp_path):
    df = pl.DataFrame({"g": ["a", "b"], "v": [0.0, 0.0]})
    st, out = _run(_pipe(df, tmp_path), "contribution", {"by": ["g"], "measure": "v", "stat": "sum"})
    assert out["share_percent"].null_count() == 2 and st.report["top_share"] is None


def test_contribution_to_a_change_ranks_a_missing_change_last(tmp_path):
    df = pl.DataFrame({"time": [_t(1)] * 2 + [_t(2)] * 3, "g": ["a", "b", "a", "b", "c"], "v": [10.0, 20, 30, 40, 50]})
    st, out = _run(_pipe(df, tmp_path), "contribution",
                   {"by": ["g"], "measure": "v", "stat": "mean", "time_column": "time", "every": "1mo"})
    assert out["g"].to_list()[-1] == "c"
    assert "came from c" not in st.report["finding"]["statement"]


def test_associations_with_a_target_only_pair_the_target(tmp_path):
    n = 40
    df = pl.DataFrame({"x": [float(i) for i in range(n)], "y": [float(i % 7) for i in range(n)],
                       "z": [float(i % 7) * 2 for i in range(n)]})
    _, out = _run(_pipe(df, tmp_path), "associations", {"target": "x"})
    assert all("x" in (r["left"], r["right"]) for r in out.to_dicts()) and out.height == 2


def test_cramers_v_of_a_perfect_association_is_one(tmp_path):
    """Pairs that never occur count in the chi-square too; leaving them out halved it (V = 0.71 for a perfect link)."""
    n = 40
    df = pl.DataFrame({"h": ["p" if i % 2 else "q" for i in range(n)],
                       "k": [("u" if i % 2 else "v") if i < 20 else None for i in range(n)]})
    _, out = _run(_pipe(df, tmp_path), "associations", {})
    assert out.row(0, named=True)["strength"] == pytest.approx(1.0)


def test_check_data_flags_repeated_numbers_stored_as_text(tmp_path):
    """Few distinct values (10, 20) with a stray 'n/a' is the usual case of numbers kept as text."""
    df = pl.DataFrame({"amount": ["10", "20", "10", "20", "n/a"] * 20, "code": ["001", "002"] * 50})
    _, out = _run(_pipe(df, tmp_path), "check_data", {})
    issues = dict(zip(out["column"].to_list(), out["issue"].to_list()))
    assert "numbers stored as text" in issues["amount"] and issues["code"] == ""


@pytest.mark.parametrize("every,expected", [
    ("1d", [datetime(2024, 1, 31), datetime(2024, 2, 1), datetime(2024, 2, 2)]),
    ("1w", [datetime(2024, 2, 6), datetime(2024, 2, 13), datetime(2024, 2, 20)]),
    ("1mo", [datetime(2024, 2, 29), datetime(2024, 3, 30), datetime(2024, 4, 30)]),
])
def test_forecast_steps_are_one_step_apart(every, expected, tmp_path):
    """Step i was offset by f'{i}{step}', so '1d' became 11 days, 21 days, 31 days …"""
    t0 = datetime(2024, 1, 1)
    df = pl.DataFrame({"time": [t0 + timedelta(days=i) for i in range(30)], "v": [float(i) for i in range(30)]})
    _, out = _run(_pipe(df, tmp_path), "forecast", {"time_column": "time", "column": "v", "horizon": 3, "every": every})
    assert out["time"].to_list() == expected


def test_forecast_refuses_a_daily_pattern_on_dates_only(tmp_path):
    t0 = datetime(2024, 1, 1)
    df = pl.DataFrame({"day": [(t0 + timedelta(days=i)).date() for i in range(30)], "v": [float(i) for i in range(30)]})
    p = _pipe(df, tmp_path)
    p.add_node("forecast", params={"time_column": "day", "column": "v", "method": "seasonal", "cycle": "hour"}, id="s")
    p.connect("src", "s")
    st = Executor(p).run(targets=["s"])["s"]
    assert st.status == "failed" and "holds dates only" in st.error
