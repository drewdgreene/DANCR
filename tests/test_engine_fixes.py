"""Regression tests for the 2026-09-23 engine review (docs/history/REVIEW_2026-09-23.md, items E1-E18, Q1-Q3)."""
from datetime import datetime, timedelta, date

import numpy as np
import polars as pl
import pytest

from dancr.core import Pipeline
from dancr.core.executor import Executor
from dancr.core.expr import compile_formula, check_formula
from dancr.core.conditions import rule_mask, ops_for_kind
from dancr.core.dtypes import datetime_literal, temp_name, text_to_bool, number_from_text
from dancr.core.timeutil import parse_duration
from dancr.core import fits
from dancr.views import lod
from conftest import run_one


def pipe_with(tmp_path, df, name="t.parquet"):
    f = tmp_path / name; df.write_parquet(f)
    p = Pipeline(); p.path = tmp_path / "p.json"
    p.add_node("load_file", params={"path": str(f)}, id="src")
    return p


def failed(p, node_id):
    st = Executor(p).run(targets=[node_id])[node_id]
    assert st.status == "failed", "expected a failure"
    return st.error


# ------------------------------------------------------------------ E1 time zones
def test_offset_literal_keeps_its_instant_against_a_zoned_column():
    lit = pl.select(datetime_literal("2024-06-01T12:00:00+02:00", pl.Datetime("us", "Europe/Oslo"))).item()
    assert (lit.hour, lit.utcoffset().total_seconds()) == (12, 7200)
    naive = pl.select(datetime_literal("2024-06-01 12:00", pl.Datetime("us", "Europe/Oslo"))).item()
    assert naive.hour == 12 and naive.utcoffset().total_seconds() == 7200
    # a naive column has no zone: the literal's own wall time is used
    assert pl.select(datetime_literal("2024-06-01T12:00:00+02:00", pl.Datetime("ms"))).item() == datetime(2024, 6, 1, 12)
    assert pl.select(datetime_literal("2024-06-01", pl.Date)).dtypes == [pl.Datetime("us")]


def test_formula_compares_offset_literal_with_zoned_column():
    df = pl.DataFrame({"t": [datetime(2024, 6, 1, 11), datetime(2024, 6, 1, 13)]}).with_columns(pl.col("t").dt.replace_time_zone("Europe/Oslo"))
    e, _, _ = compile_formula('t > "2024-06-01T12:00:00+02:00"', df.schema)
    assert df.select(e)["t"].to_list() == [False, True]


# ------------------------------------------------------------------ E2 / E3 fits
def test_polynomial_fit_is_stable_on_epoch_seconds_and_predicts_consistently():
    x = 1.7e9 + np.linspace(0, 86400, 3000)
    y = 3 + 2 * (x - 1.7e9) / 86400 - 5 * ((x - 1.7e9) / 86400) ** 2
    lf = pl.DataFrame({"x": x, "y": y}).lazy()
    f = fits.fit_frame(lf, "x", "y", "polynomial", degree=2)[0]
    pred = lf.with_columns(fits.predict_expr(f.kind, f.params, pl.col("x")).alias("p")).collect()
    assert float((pred["p"] - pred["y"]).abs().max()) < 1e-6 and f.r2 == pytest.approx(1.0)
    assert np.allclose(fits.predict_arrays(f.kind, f.params, x), pred["p"].to_numpy())
    assert "z = (x" in f.equation


def test_polynomial_needs_enough_distinct_x():
    lf = pl.DataFrame({"x": [1.0, 1.0, 2.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0, 4.0, 5.0]}).lazy()
    with pytest.raises(ValueError, match="distinct x"):
        fits.fit_frame(lf, "x", "y", "polynomial", degree=3)


def test_power_fit_is_least_squares_in_y_and_reports_convergence():
    rng = np.random.default_rng(2)
    xs = np.linspace(0.5, 20, 400); ys = 1.5 * xs ** 0.7 + rng.normal(0, 0.3, 400)
    f = fits.fit_frame(pl.DataFrame({"x": xs, "y": ys}).lazy(), "x", "y", "power")[0]
    assert f.params == pytest.approx([1.5, 0.7], rel=0.05) and f.converged and f.iterations >= 1
    # log-space fit on positive rows only would be worse in y: compare SSE
    b, la = np.polyfit(np.log(xs[ys > 0]), np.log(ys[ys > 0]), 1)
    sse_log = float(np.sum((np.exp(la) * xs ** b - ys) ** 2))
    sse_fit = float(np.sum((fits.predict_arrays("power", f.params, xs) - ys) ** 2))
    assert sse_fit <= sse_log + 1e-9


def test_positive_x_shapes_refuse_zero_and_predict_null_outside():
    with pytest.raises(ValueError, match="above zero"):
        fits.fit_frame(pl.DataFrame({"x": [0.0, 1.0, 2.0], "y": [1.0, 2.0, 3.0]}).lazy(), "x", "y", "logarithmic")
    out = pl.select(fits.predict_expr("power", [2.0, 1.0], pl.Series("x", [-1.0, 0.0, 2.0])))
    assert out.to_series().to_list() == [None, None, 4.0]
    assert np.isnan(fits.predict_arrays("logarithmic", [1.0, 0.0], np.array([0.0]))[0])


def test_group_fits_compare_groups_as_text_and_predict_warns_per_group(tmp_path):
    rows = [{"day": date(2024, 6, d), "x": float(x), "y": float(d * x)} for d in (1, 2) for x in range(1, 11)]
    p = pipe_with(tmp_path, pl.DataFrame(rows))
    f = p.add_node("fit_curve", params={"x": "x", "y": "y", "kind": "linear", "group": "day"}); p.connect("src", f.id)
    st = Executor(p).run()[f.id]
    assert st.status == "done", st.error
    assert sorted(x["group"] for x in st.report["fits"]) == ["2024-06-01", "2024-06-02"]
    new = pl.DataFrame({"day": [date(2024, 6, 1), date(2024, 6, 2)], "x": [5.0, 50.0]})
    new.write_parquet(tmp_path / "new.parquet")
    p.add_node("load_file", params={"path": "new.parquet"}, id="new")
    pr = p.add_node("predict"); p.connect("new", pr.id, "data"); p.connect(f.id, pr.id, "model")
    res = Executor(p).run()[pr.id]
    assert res.status == "done", res.error
    assert pl.read_parquet(res.output)["predicted_y"].to_list() == pytest.approx([5.0, 100.0])
    assert any("1 rows are outside" in m for m in res.messages) and "equations" in res.report


# ------------------------------------------------------------------ E4 summarise_around
@pytest.fixture
def dense(tmp_path):
    t0 = datetime(2024, 6, 1)
    log = pl.DataFrame({"t": [t0 + timedelta(minutes=i) for i in range(180)], "v": [float(i) for i in range(180)]})
    samples = pl.DataFrame({"when": [t0 + timedelta(minutes=m) for m in (60, 70, 80)], "lab": [1.0, 2.0, 3.0]})
    log.write_parquet(tmp_path / "log.parquet"); samples.write_parquet(tmp_path / "s.parquet")
    p = Pipeline(); p.path = tmp_path / "p.json"
    p.add_node("load_file", params={"path": "log.parquet"}, id="log"); p.add_node("load_file", params={"path": "s.parquet"}, id="s")
    return p


@pytest.mark.parametrize("side,count,means", [("before", 60, [30.5, 40.5, 50.5]), ("after", 60, [89.5, 99.5, 109.5]), ("around", 61, [60.0, 70.0, 80.0])])
def test_summarise_around_sees_the_whole_window_when_samples_overlap(dense, side, count, means):
    a = dense.add_node("summarise_around", params={"window": "1h", "side": side, "columns": ["v"], "stats": ["count", "mean"]})
    dense.connect("s", a.id, "samples"); dense.connect("log", a.id, "log")
    out = run_one(dense, a.id)
    assert out["v_count"].to_list() == [count] * 3 and out["v_mean"].to_list() == pytest.approx(means)
    assert out.columns == ["when", "lab", "v_count", "v_mean"]


def test_summarise_around_rejects_unknown_stat_and_side(dense):
    a = dense.add_node("summarise_around", params={"window": "1h", "stats": ["mode"]})
    dense.connect("s", a.id, "samples"); dense.connect("log", a.id, "log")
    assert "Unknown statistic" in failed(dense, a.id)


# ------------------------------------------------------------------ E5 fix_values
def test_fix_values_keeps_date_type_and_refuses_lossy_integer(tmp_path):
    df = pl.DataFrame({"d": [date(2024, 1, 1), date(2024, 1, 2)], "n": [1, 2]})
    p = pipe_with(tmp_path, df)
    f = p.add_node("fix_values", params={"fixes": [{"row": 2, "column": "d", "value": "2024-02-02"}]}); p.connect("src", f.id)
    out = run_one(p, f.id)
    assert out.schema["d"] == pl.Date and out["d"][1] == date(2024, 2, 2)
    p.set_params(f.id, fixes=[{"row": 1, "column": "n", "value": "12.7"}])
    assert "decimals" in failed(p, f.id)
    p.set_params(f.id, fixes=[{"row": 0, "column": "n", "value": "1"}])
    assert "numbered from 1" in failed(p, f.id)


# ------------------------------------------------------------------ E6 temp names
def test_take_sample_and_outliers_tolerate_reserved_looking_columns(tmp_path):
    df = pl.DataFrame({"__i": list(range(10)), "v": [1.0] * 9 + [100.0], "__bad_v": [0] * 10})
    p = pipe_with(tmp_path, df)
    s = p.add_node("take_sample", params={"mode": "every", "every": 5}); p.connect("src", s.id)
    assert run_one(p, s.id)["__i"].to_list() == [0, 5]
    o = p.add_node("remove_outliers", params={"columns": ["v"], "method": "zscore", "threshold": 2, "action": "flag"}); p.connect("src", o.id)
    assert run_one(p, o.id)["is_outlier"].sum() == 1
    p.set_params(o.id, flag_column="v")
    assert "already a column" in failed(p, o.id)
    assert temp_name("row", {"__dancr_row": 1}) == "__dancr_row1"


# ------------------------------------------------------------------ E7 formulas
@pytest.fixture
def fdf():
    return pl.DataFrame({"a": [1.0, 2.0, 3.0, None], "s": ["x", "y", "z", None], "Cutoff": [5.0, 6.0, 7.0, 8.0],
                         "t": [datetime(2024, 1, 1), datetime(2024, 6, 1), datetime(2025, 1, 1), None], "b": [True, False, True, None]})


def ev(df, src, inputs=None):
    e, kind, _ = compile_formula(src, df.schema, inputs)
    return df.select(e.alias("r"))["r"].to_list(), kind


@pytest.mark.parametrize("src,expected", [
    ("-2^2", [4.0] * 4), ("2^-1", [0.5] * 4), ("2^3^2", [512.0] * 4), ("-a^2", [1.0, 4.0, 9.0, None]), ("3 - -2", [5] * 4), ("2*-a", [-2.0, -4.0, -6.0, None]),
    ("LEAD(a)", [2.0, 3.0, None, None]), ("PCT_CHANGE(a)", [None, 100.0, 50.0, None]), ("ROLLING_SUM(a, 2)", [1.0, 3.0, 5.0, 3.0]),
    ("ROLLING_MAX(a, 2)", [1.0, 2.0, 3.0, 3.0]), ("ROLLING_MIN(a, 3)", [1.0, 1.0, 1.0, 2.0]), ("CUMMAX(a)", [1.0, 2.0, 3.0, None]), ("CUMMIN(a)", [1.0, 1.0, 1.0, None]),
    ("RANK(a)", [3, 2, 1, None]), ("RANK(a, 1)", [1, 2, 3, None]), ("PERCENTILE(a, 50)", [2.0] * 4), ("CLIP(a, 1.5, 2.5)", [1.5, 2.0, 2.5, None]),
    ("LEFT(s, 1) & RIGHT(s, 1)", ["xx", "yy", "zz", ""]), ("MID(\"hello\", 2, 3)", ["ell"] * 4), ("REPLACE(s, \"x\", \"q\")", ["q", "y", "z", None]),
    ("CONTAINS(s, \"y\")", [False, True, False, None]), ("STARTSWITH(s, \"z\")", [False, False, True, None]), ("ENDSWITH(s, \"x\")", [True, False, False, None]),
    ("TEXT(t, \"%Y\")", ["2024", "2024", "2025", None]), ("VALUE(\"1,200\") + a", [1201.0, 1202.0, 1203.0, None]),
    ("YEAR(DATE(\"01/02/2024\", \"%d/%m/%Y\"))", [2024] * 4), ("MONTH(DATE(\"01/02/2024\", \"%d/%m/%Y\"))", [2] * 4),
    ("WEEKDAY(t)", [1, 6, 3, None]), ("DAYOFYEAR(t)", [1, 153, 1, None]), ("FILL_FORWARD(a)", [1.0, 2.0, 3.0, 3.0]),
    ("INTERPOLATE(IF(a = 2, NULL, a))", [1.0, 2.0, 3.0, None]), ("IFNULL(a, 0)", [1.0, 2.0, 3.0, 0.0]),
    ("ROUND(ATAN2(1, 1) * 4, 4)", [3.1416] * 4), ("LOG(8, 2)", [3.0] * 4), ("POW(2, a)", [2.0, 4.0, 8.0, None]), ("SIGN(-a)", [-1.0, -1.0, -1.0, None]),
    ("ROUND(SIN(PI() / 2) + COS(0) + TAN(0), 6)", [2.0] * 4), ("ROUND(ASIN(1) + ACOS(1) + ATAN(0), 4)", [1.5708] * 4),
    ("IF(b, 1, 0)", [1, 0, 1, 0]), ("COALESCE(NULL, a, 9)", [1.0, 2.0, 3.0, 9.0]), ("SECONDS_BETWEEN(t, t)", [0.0, 0.0, 0.0, None]),
])
def test_formula_functions(fdf, src, expected):
    got, _ = ev(fdf, src)
    for g, e in zip(got, expected):
        if isinstance(e, float) and g is not None:
            assert g == pytest.approx(e, rel=1e-6), src
        else:
            assert g == e, src


def test_column_wins_over_input_of_the_same_name(fdf):
    assert ev(fdf, "cutoff", {"cutoff": 99})[0] == [5.0, 6.0, 7.0, 8.0]
    assert ev(fdf, "a + limit", {"Limit": 99})[0] == [100.0, 101.0, 102.0, None]


@pytest.mark.parametrize("src,fragment", [
    ('IF(a > 1, "hi", 5)', "same kind"), ('IFNULL(a, "x")', "same kind"), ("s % 2", "Expected a number"), ('TEXT(a, "%Y")', "needs a date/time"),
    ("DATE(s, a)", "plain text in quotes"), ("PERCENTILE(a, 150)", "between 0 and 1"), ("ROUND(a, 1.7)", "whole number"), ("LEFT(s, TRUE)", "plain number"),
])
def test_formula_type_errors(fdf, src, fragment):
    msg = check_formula(src, fdf.schema)
    assert msg is not None and fragment in msg, (src, msg)


# ------------------------------------------------------------------ E8 conditions
def test_conditions_cover_every_op_and_duration_columns():
    df = pl.DataFrame({"n": [1.0, 2.0, None], "s": ["Alpha", "beta", None], "b": [True, False, None],
                       "d": [timedelta(seconds=30), timedelta(minutes=2), None]})
    m = lambda rule: df.filter(rule_mask(df.schema, rule)).height
    assert m({"column": "d", "op": "gt", "value": "1m"}) == 1 and m({"column": "d", "op": "between", "value": "10s", "value2": "3m"}) == 2
    assert [k for k, _ in ops_for_kind("duration")][:3] == ["eq", "ne", "gt"]
    assert m({"column": "s", "op": "contains", "value": "alp"}) == 1 and m({"column": "s", "op": "contains", "value": "alp", "case_sensitive": True}) == 0
    assert m({"column": "s", "op": "not_contains", "value": "e"}) == 2 and m({"column": "s", "op": "starts", "value": "b"}) == 1
    assert m({"column": "s", "op": "ends", "value": "TA"}) == 1 and m({"column": "s", "op": "ne", "value": "beta"}) == 2   # a blank is not equal to "beta" (as in Excel)
    assert m({"column": "n", "op": "le", "value": "1"}) == 1 and m({"column": "n", "op": "in", "value": [1, 2]}) == 2 and m({"column": "n", "op": "in", "value": "1,000; 2"}) == 1
    assert m({"column": "b", "op": "true"}) == 1 and m({"column": "b", "op": "false"}) == 1 and m({"column": "b", "op": "eq", "value": "yes"}) == 1
    assert m({"column": "d", "op": "empty"}) == 1 and m({"column": "n", "op": "not_empty"}) == 2
    with pytest.raises(ValueError, match="does not apply"):
        rule_mask(df.schema, {"column": "n", "op": "contains", "value": "1"})
    with pytest.raises(ValueError, match="not a number"):
        number_from_text("abc", "x")
    assert text_to_bool(" Yes ") and text_to_bool("t") and not text_to_bool("no")


def test_parse_duration_forms():
    assert parse_duration("5 min") == ("5m", 300.0) and parse_duration("1h30m") == ("1h30m", 5400.0)
    assert parse_duration("1.5h") == ("90m", 5400.0) and parse_duration("250ms")[1] == 0.25
    for bad in ("", "1 fortnight", "abc"):
        with pytest.raises(ValueError):
            parse_duration(bad)


# ------------------------------------------------------------------ E9 load_file
def test_load_sheet_zero_header_dupes_and_dash_values(tmp_path):
    f = tmp_path / "d.csv"
    good = "".join(f"{i},{i},2024-01-{i + 1:02d}\n" for i in range(9))
    f.write_text("a, a ,when\n1,-,2024-01-01\n2,3,not a date\n" + good)
    p = Pipeline(); p.path = tmp_path / "p.json"
    p.add_node("load_file", params={"path": str(f)}, id="src")
    st = Executor(p).run()["src"]
    assert st.status == "done", st.error
    df = pl.read_parquet(st.output)
    assert df.columns == ["a", "a_2", "when"] and df["a_2"].to_list()[:2] == [None, 3.0]      # "-" is a blank among numbers
    assert any("Read 'a_2' as numbers" in m for m in st.messages)
    assert any("only differed by spaces" in m for m in st.messages)
    assert df.schema["when"] == pl.Datetime("us") and any("do not match and become blank" in m for m in st.messages)
    xl = tmp_path / "b.xlsx"; pl.DataFrame({"x": [1]}).write_excel(xl)
    p.set_params("src", path=str(xl), sheet="0")
    assert "numbered from 1" in failed(p, "src")


# ------------------------------------------------------------------ E10 combine
def test_combine_drops_blank_times_and_matches_int_to_float_keys(tmp_path):
    a = pl.DataFrame({"t": [datetime(2024, 1, 1), None, datetime(2024, 1, 3)], "k": [1, 2, 3]})
    b = pl.DataFrame({"t": [datetime(2024, 1, 1), datetime(2024, 1, 3)], "k": [1.0, 3.0], "v": ["p", "q"]})
    p = pipe_with(tmp_path, a); b.write_parquet(tmp_path / "b.parquet"); p.add_node("load_file", params={"path": "b.parquet"}, id="b")
    j = p.add_node("combine", params={"method": "nearest_time"}); p.connect("src", j.id, "left"); p.connect("b", j.id, "right")
    assert run_one(p, j.id).height == 2
    p.set_params(j.id, method="match", on=["k"], how="inner")
    out = run_one(p, j.id)
    assert out["v"].to_list() == ["p", "q"]


# ------------------------------------------------------------------ E11 / E12 / E13 time-series
def test_rolling_time_window_centred_and_named_consistently(dense):
    r = dense.add_node("rolling", params={"columns": ["v"], "window": "2m", "stat": "mean", "centered": True}); dense.connect("log", r.id)
    out = run_one(dense, r.id)
    assert out["v_mean_2m"].head(3).to_list() == [0.5, 1.0, 2.0]      # [t-1m, t+1m], like a centred 3-row window
    dense.set_params(r.id, window="3")
    assert run_one(dense, r.id)["v_mean_3"].head(3).to_list() == [0.5, 1.0, 2.0]
    dense.set_params(r.id, window="2m", centered=False)
    assert run_one(dense, r.id)["v_mean_2m"].head(3).to_list() == [0.0, 0.5, 1.5]


def test_regular_grid_is_lazy_and_handles_zoned_time(dense):
    g = dense.add_node("regular_grid", params={"every": "30s", "method": "nearest"}); dense.connect("log", g.id)
    out = run_one(dense, g.id)
    assert out.height == 359 and (out["t"].diff().drop_nulls().dt.total_seconds() == 30).all()
    ct = dense.add_node("calculate", params={"formulas": [{"name": "t", "expr": "t"}]}); dense.connect("log", ct.id)   # no-op step to attach a zoned copy
    zoned = pl.read_parquet(Executor(dense).state("log").output).with_columns(pl.col("t").dt.replace_time_zone("Europe/Oslo"))
    zoned.write_parquet(dense.path.parent / "z.parquet"); dense.add_node("load_file", params={"path": "z.parquet"}, id="z")
    g2 = dense.add_node("regular_grid", params={"every": "1h"}); dense.connect("z", g2.id)
    out = run_one(dense, g2.id)
    assert out.schema["t"] == pl.Datetime("us", "Europe/Oslo") and out.height == 3


def test_find_gaps_counts_at_least_one_missing_reading_and_bucket_count_clash(tmp_path):
    t0 = datetime(2024, 1, 1)
    times = [t0 + timedelta(seconds=s) for s in (0, 1, 2, 3.3, 4.3, 5.3)]
    p = pipe_with(tmp_path, pl.DataFrame({"t": times, "v": [1.0] * 6}))
    g = p.add_node("find_gaps", params={"factor": 1.2}); p.connect("src", g.id)
    out = run_one(p, g.id)
    assert out.height == 1 and out["missing_readings"][0] == 1
    tb = p.add_node("time_buckets", params={"every": "1m", "count_column": "v"}); p.connect("src", tb.id)
    assert "already used" in failed(p, tb.id)


# ------------------------------------------------------------------ E14-E17 smaller node fixes
def test_chart_validates_every_named_column_and_export_accepts_txt(tmp_path):
    p = pipe_with(tmp_path, pl.DataFrame({"x": [1.0, 2.0], "g": ["a", "b"]}))
    c = p.add_node("chart", params={"kind": "bar", "category": "nope", "value": "x"}); p.connect("src", c.id)
    assert "nope" in failed(p, c.id)
    p.set_params(c.id, kind="line", x="x", series=[{"column": "x"}], color_by="missing")
    assert "missing" in failed(p, c.id)
    p.set_params(c.id, kind="histogram", column="x", split_by="g")
    assert Executor(p).run(targets=[c.id])[c.id].status == "done"
    e = p.add_node("export", params={"path": "out.txt"}); p.connect("src", e.id)
    run_one(p, "src"); assert Executor(p).run(targets=[e.id])[e.id].status == "done" and (tmp_path / "out.txt").read_text().startswith("x,g")
    p.set_params(e.id, path="out.json")
    assert ".tsv, .txt" in failed(p, e.id)


def test_group_summary_duplicate_by_and_alias_with_several_stats(tmp_path):
    p = pipe_with(tmp_path, pl.DataFrame({"g": ["a", "a", "b"], "v": [1.0, 3.0, 5.0]}))
    s = p.add_node("group_summary", params={"by": ["g", "g"]}); p.connect("src", s.id)
    assert "listed twice" in failed(p, s.id)
    p.set_params(s.id, by=["g"], aggregations=[{"column": "v", "stats": ["min", "max"], "alias": "val"}])
    assert run_one(p, s.id).columns == ["g", "val_min", "val_max"]


def test_fix_missing_reports_skipped_columns_and_change_type_keeps_fractional_seconds(tmp_path):
    p = pipe_with(tmp_path, pl.DataFrame({"n": [1.0, None, 3.0], "s": ["a", None, "c"], "e": [1717200000.5, 1717200001.25, 0.0]}))
    fm = p.add_node("fix_missing", params={"method": "interpolate"}); p.connect("src", fm.id)
    st = Executor(p).run(targets=[fm.id])[fm.id]
    assert st.status == "done" and any("not number columns" in m and "s" in m for m in st.messages)
    assert pl.read_parquet(st.output)["n"].to_list() == [1.0, 2.0, 3.0]
    ct = p.add_node("change_type", params={"columns": ["e"], "to": "datetime", "epoch_unit": "s"}); p.connect("src", ct.id)
    out = run_one(p, ct.id)
    assert out["e"][0].microsecond == 500000 and out["e"][1].microsecond == 250000


def test_calculate_warns_when_replacing_a_column_and_enter_data_names_blank_columns(tmp_path):
    p = pipe_with(tmp_path, pl.DataFrame({"x": [1.0]}))
    c = p.add_node("calculate", params={"formulas": [{"name": "x", "expr": "x * 2"}]}); p.connect("src", c.id)
    st = Executor(p).run(targets=[c.id])[c.id]
    assert st.status == "done" and any("replaces the existing" in m for m in st.messages)
    e = p.add_node("enter_data", params={"columns": [{"name": "  ", "type": "number"}, {"name": "b", "type": "bool"}], "rows": [["1", "yes"], ["2", "no"]]})
    out = run_one(p, e.id)
    assert out.columns == ["column_1", "b"] and out["b"].to_list() == [True, False]


def test_report_ignores_malformed_block_index(tmp_path):
    p = pipe_with(tmp_path, pl.DataFrame({"x": [1.0, 2.0]}))
    r = p.add_node("report", params={"title": "T", "path": "r.html", "pdf": False, "blocks": [{"type": "item", "index": "zero"}, {"type": "heading", "text": "H"}]})
    p.connect("src", r.id, "items")
    assert Executor(p).run()[r.id].status == "done" and "<h2>H</h2>" in (tmp_path / "r.html").read_text()


# ------------------------------------------------------------------ E18 downsampling
def test_line_data_is_m4_and_ignores_non_finite_and_blank_x():
    n = 50_000
    x = np.arange(n, dtype=float); y = np.sin(x / 300) * 10
    y[100:200] = np.nan; y[3000] = np.inf; y[7000] = -np.inf
    xs = x.copy(); xs[10] = np.nan
    lf = pl.DataFrame({"x": xs, "y": y}).lazy()
    d = lod.line_data(lf, "x", ["y"], width_px=200)
    s = d.series[0]
    finite = y[np.isfinite(y) & np.isfinite(xs)]
    assert d.mode == "envelope" and d.rows_in_range == n - 1
    assert np.all(np.diff(s.x) >= 0) and np.all(np.isfinite(s.y))
    assert s.y.min() == finite.min() and s.y.max() == finite.max()
    assert 2 * 200 < len(s.x) <= 4 * 200
    # first/last of each bucket present: the very first and last points of the data
    assert s.x[0] == 0.0 and s.x[-1] == n - 1
