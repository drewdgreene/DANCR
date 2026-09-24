"""Steps and formulas give the answers a person expects (audit 2026-09-24, A2–A10, A17, S1–S25)."""
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from conftest import run_one
from dancr.core import Pipeline, PipelineError
from dancr.core.executor import Executor
from dancr.core.expr import compile_formula
from dancr.core.conditions import rule_mask
from dancr.core.dtypes import datetime_literal, text_to_number_expr, typed_value


def _pipe(df: pl.DataFrame, tmp_path, name="in.parquet") -> Pipeline:
    df.write_parquet(tmp_path / name)
    p = Pipeline("t")
    p.add_node("load_file", "Load", {"path": name}, id="src")
    p.path = tmp_path / "p.json"
    return p


def _f(formula: str, df: pl.DataFrame) -> list:
    e, _, _ = compile_formula(formula, dict(df.schema), {})
    return df.select(e.alias("r"))["r"].to_list()


def _step(p: Pipeline, key: str, params: dict, port: str | None = None, src: str = "src") -> str:
    nid = p.add_node(key, params=params).id
    p.connect(src, nid, port)
    return nid


# ------------------------------------------------------------------ numbers typed as text
@pytest.mark.parametrize("text,value", [("1,200.5", 1200.5), ("1,5", 1.5), ("2,25", 2.25), ("1,500", 1500.0),
                                        ("1.234.567,5", 1234567.5), (" -3 ", -3.0), ("abc", None)])
def test_text_is_read_as_the_number_a_person_means(text, value):
    got = pl.DataFrame({"s": [text]}).select(text_to_number_expr(pl.col("s")))["s"][0]
    assert got == value


def test_inputs_keep_their_value_exactly():
    p = Pipeline("t")
    assert p.set_input("id", "12345678901234567890").value == 12345678901234567890
    assert p.set_input("rate", "1,5").value == 1.5
    assert p.set_input("label", "nan").value == "nan"            # not a number: kept as text
    with pytest.raises(PipelineError):
        p.set_input("bad", float("nan"))
    with pytest.raises(PipelineError):
        p.set_input("bad", {"a": 1})
    assert typed_value("1_000") is None


def test_numeric_settings_are_checked_not_truncated():
    p = Pipeline("t")
    p.add_node("take_sample", id="s")
    with pytest.raises(ValueError, match="whole number"):
        p.set_params("s", rows=2.9)
    with pytest.raises(ValueError):
        p.set_params("s", rows=1e400)
    p.set_params("s", rows="1,000")
    assert p.nodes["s"].params["rows"] == 1000


# ------------------------------------------------------------------ Excel arithmetic
def test_round_is_excel_round():
    df = pl.DataFrame({"x": [2.5, -2.5, 1.005, 1234.5, 0.125]})
    assert _f("ROUND(x)", df) == [3, -3, 1, 1235, 0]
    assert _f("ROUND(x, 2)", df) == [2.5, -2.5, 1.01, 1234.5, 0.13]
    assert _f("ROUND(x, -2)", df) == [0, 0, 0, 1200, 0]


def test_whole_number_conversion_rounds_like_excel(tmp_path):
    p = _pipe(pl.DataFrame({"s": ["2.5", "3.5", "-2.5"]}), tmp_path)
    n = _step(p, "change_type", {"columns": ["s"], "to": "integer"})
    assert run_one(p, n)["s"].to_list() == [3, 4, -3]


def test_percentile_and_quartiles_interpolate():
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    assert _f("PERCENTILE(x, 0.5)", df)[0] == 2.5
    assert _f("PERCENTILE(x, 25)", df)[0] == 1.75
    from dancr.views.stats import exact_quantiles
    assert exact_quantiles(df.lazy(), pl.col("x")) == [1.75, 2.5, 3.25]


def test_iqr_outliers_use_excel_quartiles(tmp_path):
    vals = list(range(1, 10)) + [100]
    p = _pipe(pl.DataFrame({"v": [float(v) for v in vals]}), tmp_path)
    n = _step(p, "remove_outliers", {"columns": ["v"], "method": "iqr", "iqr_factor": 1.5, "action": "remove"})
    q1, q3 = np.quantile(vals, [0.25, 0.75])
    keep = [v for v in vals if q1 - 1.5 * (q3 - q1) <= v <= q3 + 1.5 * (q3 - q1)]
    assert run_one(p, n)["v"].to_list() == keep


# ------------------------------------------------------------------ comparisons and joins in formulas
def test_text_holding_numbers_compares_as_numbers():
    df = pl.DataFrame({"s": ["9", "10", "abc"]})
    assert _f("s > 5", df) == [True, True, None]
    assert _f("s = 10", df) == [False, True, None]


def test_join_with_blanks_and_whole_numbers():
    df = pl.DataFrame({"u": ["a", None], "x": [1.0, 4.5]})
    assert _f('u & " (" & x & ")"', df) == ["a (1)", " (4.5)"]


def test_text_literals_follow_excel():
    df = pl.DataFrame({"x": [1]})
    assert _f(r'"C:\new"', df) == ["C:\\new"]
    assert _f('"say ""hi"""', df) == ['say "hi"']


def test_negative_literals_and_small_integers():
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "i": pl.Series([-128, 1, 2], dtype=pl.Int8)})
    assert _f("LAG(x, -1)", df) == [2.0, 3.0, None]
    assert _f("ABS(i)", df)[0] == 128
    with pytest.raises(ValueError, match="TRUE or FALSE"):
        _f('ROLLING_MEAN(x, 2, "no")', df)


def test_elapsed_counts_from_the_first_row():
    t = [datetime(2024, 1, 1, 0, 10), datetime(2024, 1, 1, 0, 0)]
    assert _f('ELAPSED(t, "min")', pl.DataFrame({"t": t})) == [0.0, -10.0]


# ------------------------------------------------------------------ typed dates
def test_ambiguous_or_impossible_typed_dates_are_refused():
    with pytest.raises(ValueError, match="day/month or month/day"):
        datetime_literal("01/05/2024", pl.Datetime("us"))
    assert pl.select(datetime_literal("13/05/2024", pl.Datetime("us")))[0, 0] == datetime(2024, 5, 13)
    with pytest.raises(ValueError, match="does not exist"):
        datetime_literal("2024-03-31 01:30", pl.Datetime("us", "Europe/London"))
    assert pl.select(datetime_literal("2024-06-01 12:00+02:00", pl.Datetime("us")))[0, 0] == datetime(2024, 6, 1, 12)


def test_is_one_of_splits_numbers_on_commas():
    df = pl.DataFrame({"n": [100.0, 200.0, 100200300.0, 1000.0]})
    assert df.filter(rule_mask(df.schema, {"column": "n", "op": "in", "value": "100,200,300"})).height == 2
    assert df.filter(rule_mask(df.schema, {"column": "n", "op": "in", "value": "1,000; 7"})).height == 1


# ------------------------------------------------------------------ loading
def test_a_forced_date_format_only_touches_date_columns(tmp_path):
    pl.DataFrame({"when": ["05/01/2024", "06/01/2024"], "name": ["alpha", "beta"]}).write_csv(tmp_path / "x.csv")
    p = Pipeline("t"); p.add_node("load_file", params={"path": "x.csv", "date_format": "%d/%m/%Y"}, id="src"); p.path = tmp_path / "p.json"
    df = run_one(p, "src")
    assert df["name"].to_list() == ["alpha", "beta"] and df["when"][0] == datetime(2024, 1, 5)


def test_typed_in_dates_are_read_cell_by_cell(tmp_path):
    p = Pipeline("t")
    p.add_node("enter_data", params={"columns": [{"name": "t", "type": "datetime"}],
                                     "rows": [["2024-06-03 09:00"], ["2024-06-10"], ["junk"]]}, id="e")
    p.path = tmp_path / "p.json"
    st = Executor(p).run()["e"]
    assert pl.read_parquet(st.output)["t"].to_list() == [datetime(2024, 6, 3, 9), datetime(2024, 6, 10), None]
    assert any("'junk'" in m for m in st.messages)


# ------------------------------------------------------------------ time steps
def test_regular_grid_keeps_its_last_tick(tmp_path):
    t0 = datetime(2024, 1, 1)
    p = _pipe(pl.DataFrame({"t": [t0, t0 + timedelta(milliseconds=300)], "v": [1.0, 2.0]}), tmp_path)
    n = _step(p, "regular_grid", {"every": "100ms", "method": "nearest"})
    assert run_one(p, n).height == 4


def test_spans_shorter_than_a_microsecond_are_refused():
    from dancr.core.timeutil import parse_duration
    with pytest.raises(ValueError, match="microsecond"):
        parse_duration("500ns")


def test_summarise_around_samples_one_window_apart(tmp_path):
    base = datetime(2024, 1, 1)
    samples = pl.DataFrame({"at": [base, base + timedelta(hours=2), None]})
    log = pl.DataFrame({"t": [base + timedelta(hours=1)], "v": [5.0]})
    samples.write_parquet(tmp_path / "s.parquet"); log.write_parquet(tmp_path / "l.parquet")
    p = Pipeline("t")
    p.add_node("load_file", params={"path": "s.parquet"}, id="s"); p.add_node("load_file", params={"path": "l.parquet"}, id="l")
    p.add_node("summarise_around", params={"window": "2h", "side": "around", "stats": ["mean", "count"]}, id="a")
    p.connect("s", "a", "samples"); p.connect("l", "a", "log"); p.path = tmp_path / "p.json"
    df = run_one(p, "a")
    assert df.height == 3                                           # the sample without a time is kept
    assert df["v_mean"].to_list()[:2] == [5.0, 5.0]                 # both ends of a closed window see the log row
    assert df["v_count"].to_list()[2] == 0


def test_first_and_last_in_buckets_follow_time(tmp_path):
    base = datetime(2024, 1, 1)
    p = _pipe(pl.DataFrame({"t": [base + timedelta(minutes=30), base], "v": [2.0, 1.0]}), tmp_path)
    n = _step(p, "time_buckets", {"every": "1h", "default_stats": ["first", "last"]})
    df = run_one(p, n)
    assert (df["v_first"][0], df["v_last"][0]) == (1.0, 2.0)


def test_rows_without_a_time_are_reported(tmp_path):
    p = _pipe(pl.DataFrame({"t": [datetime(2024, 1, 1), None], "v": [1.0, 2.0]}), tmp_path)
    n = _step(p, "time_buckets", {"every": "1h"})
    assert any("1 rows with a blank t were left out" in m for m in Executor(p).run()[n].messages)


# ------------------------------------------------------------------ other steps
def test_combine_date_with_datetime_keys(tmp_path):
    pl.DataFrame({"d": [datetime(2024, 1, 1).date(), datetime(2024, 1, 2).date()], "x": [1, 2]}).write_parquet(tmp_path / "l.parquet")
    pl.DataFrame({"d": [datetime(2024, 1, 2)], "y": [9]}).write_parquet(tmp_path / "r.parquet")
    p = Pipeline("t")
    p.add_node("load_file", params={"path": "l.parquet"}, id="l"); p.add_node("load_file", params={"path": "r.parquet"}, id="r")
    p.add_node("combine", params={"method": "match", "on": ["d"], "right_on": ["d"], "how": "left"}, id="j")
    p.connect("l", "j", "left"); p.connect("r", "j", "right"); p.path = tmp_path / "p.json"
    df = run_one(p, "j")
    assert df["y"].to_list() == [None, 9] and df.schema["d"] == pl.Date


def test_combine_says_when_keys_repeat(tmp_path):
    pl.DataFrame({"k": [1, 2]}).write_parquet(tmp_path / "l.parquet")
    pl.DataFrame({"k": [1, 1], "y": [7, 8]}).write_parquet(tmp_path / "r.parquet")
    p = Pipeline("t")
    p.add_node("load_file", params={"path": "l.parquet"}, id="l"); p.add_node("load_file", params={"path": "r.parquet"}, id="r")
    p.add_node("combine", params={"method": "match", "on": ["k"], "how": "left"}, id="j")
    p.connect("l", "j", "left"); p.connect("r", "j", "right"); p.path = tmp_path / "p.json"
    st = Executor(p).run()["j"]
    assert st.rows == 3 and any("share their key" in m for m in st.messages)


def test_categorical_columns_convert(tmp_path):
    p = _pipe(pl.DataFrame({"c": pl.Series(["1,5", "2"], dtype=pl.Categorical)}), tmp_path)
    n = _step(p, "change_type", {"columns": ["c"], "to": "number"})
    assert run_one(p, n)["c"].to_list() == [1.5, 2.0]


def test_falling_saturating_fit(tmp_path):
    x = np.linspace(0.5, 20, 100)
    p = _pipe(pl.DataFrame({"x": x, "y": -5 * x / (3 + x)}), tmp_path)
    n = _step(p, "fit_curve", {"x": "x", "y": "y", "kind": "saturating"})
    st = Executor(p).run()[n]
    assert st.report["r_squared"] == pytest.approx(1.0)
    assert st.report["parameters"] == pytest.approx([-5.0, 3.0], rel=1e-4)


def test_a_fit_worse_than_flat_says_so(tmp_path):
    x = np.linspace(0.5, 20, 100)
    p = _pipe(pl.DataFrame({"x": x, "y": -5 * x / (3 + x) + 10}), tmp_path)
    n = _step(p, "fit_curve", {"x": "x", "y": "y", "kind": "saturating"})
    assert any("worse than a flat line" in m for m in Executor(p).run()[n].messages)


def test_limits_do_not_judge_blanks(tmp_path):
    p = _pipe(pl.DataFrame({"v": [1.0, float("nan"), None, 5.0]}), tmp_path)
    lo = _step(p, "check_limits", {"column": "v", "min": "2"})
    hi = _step(p, "check_limits", {"column": "v", "max": "2"})
    r = Executor(p).run()
    assert (r[lo].report["outside"], r[lo].report["blank"]) == (1, 2)
    assert (r[hi].report["outside"], r[hi].report["blank"]) == (1, 2)


def test_corrections_that_changed_nothing_are_reported(tmp_path):
    p = _pipe(pl.DataFrame({"v": [1, 2]}), tmp_path)
    n = _step(p, "fix_values", {"fixes": [{"row": 5, "column": "v", "value": "9", "was": "1"},
                                          {"row": 1, "column": "gone", "value": "1"},
                                          {"row": 2, "column": "v", "value": "9007199254740993", "was": "7"}]})
    st = Executor(p).run()[n]
    text = " ".join(st.messages)
    assert "past the end" in text and "no longer in the table" in text and "was '7'" in text
    assert pl.read_parquet(st.output)["v"].to_list() == [1, 9007199254740993]


def test_nan_is_a_blank_for_fill_blanks(tmp_path):
    p = _pipe(pl.DataFrame({"v": [1.0, float("nan"), None]}), tmp_path)
    n = _step(p, "fix_missing", {"method": "zero"})
    assert run_one(p, n)["v"].to_list() == [1.0, 0.0, 0.0]


def test_explicit_zero_is_not_the_default(tmp_path):
    p = _pipe(pl.DataFrame({"v": list(range(1000))}), tmp_path)
    n = _step(p, "take_sample", {"mode": "random", "fraction": 0})
    assert run_one(p, n).height == 0


def test_range_ends_can_be_inputs_and_thousands(tmp_path):
    p = _pipe(pl.DataFrame({"v": [500.0, 1500.0, 2500.0]}), tmp_path)
    p.set_input("top", 2000)
    n = _step(p, "remove_outliers", {"columns": ["v"], "method": "range", "min": "1,000", "max": "top", "action": "remove"})
    assert run_one(p, n)["v"].to_list() == [1500.0]


def test_steps_refuse_to_overwrite_columns(tmp_path):
    p = _pipe(pl.DataFrame({"t": [datetime(2024, 1, 1)], "v": [1.0], "v_per_second": [0.0], "v_ok": [True]}), tmp_path)
    r = _step(p, "rate_of_change", {"columns": ["v"]})
    c = _step(p, "check_limits", {"column": "v", "max": "5"})
    res = Executor(p).run()
    assert "already a column" in res[r].error and "already a column" in res[c].error


def test_duplicate_aggregations_are_explained(tmp_path):
    p = _pipe(pl.DataFrame({"t": [datetime(2024, 1, 1)], "v": [1.0]}), tmp_path)
    n = _step(p, "time_buckets", {"every": "1h", "aggregations": [{"column": "v", "stats": ["mean"]}, {"column": "v", "stats": ["mean"]}]})
    assert "would both be called" in Executor(p).run()[n].error
