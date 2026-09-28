"""Analysis steps: fits and predictions, limit checks, outliers, summaries and group summaries."""
import json
from datetime import date, datetime

import numpy as np
import polars as pl
import pytest

from conftest import run_one
from dancr.core import Pipeline, fits
from dancr.core.executor import Executor
from dancr.core.fits import fit_arrays, predict_arrays


def pipe_from_csv(path, tmp_path):
    p = Pipeline()
    p.path = tmp_path / "p.json"
    p.add_node("load_file", params={"path": str(path)}, id="src")
    return p


def pipe_with(tmp_path, df, name="t.parquet"):
    f = tmp_path / name; df.write_parquet(f)
    p = Pipeline(); p.path = tmp_path / "p.json"
    p.add_node("load_file", params={"path": str(f)}, id="src")
    return p


def step(p, type_, params, after="src", port="in", id_="s"):
    p.add_node(type_, params=params, id=id_)
    p.connect(after, id_, port)
    return id_


def _pipe(df: pl.DataFrame, tmp_path, name="in.parquet") -> Pipeline:
    df.write_parquet(tmp_path / name)
    p = Pipeline("t")
    p.add_node("load_file", "Load", {"path": name}, id="src")
    p.path = tmp_path / "p.json"
    return p


def _step(p: Pipeline, key: str, params: dict, port: str | None = None, src: str = "src") -> str:
    nid = p.add_node(key, params=params).id
    p.connect(src, nid, port)
    return nid


def table(rows: list[list], columns: list[tuple[str, str]]) -> Pipeline:
    p = Pipeline("t")
    p.add_node("enter_data", params={"columns": [{"name": n, "type": t} for n, t in columns], "rows": rows}, id="d")
    return p


def after(p: Pipeline, type_key: str, params: dict, src: str = "d", nid: str = "s") -> pl.DataFrame:
    p.add_node(type_key, params=params, id=nid)
    p.connect(src, nid)
    return Executor(p).preview(nid)[0]


def failed(p, node_id):
    st = Executor(p).run(targets=[node_id])[node_id]
    assert st.status == "failed", "expected a failure"
    return st.error


def test_fit_curve_recovers_relationship(pipe, truth):
    j = pipe.add_node("combine", params={"method": "nearest_time", "tolerance": "30ms"}); pipe.connect("a", j.id, "left"); pipe.connect("b", j.id, "right")
    o = pipe.add_node("remove_outliers", params={"columns": ["pressure_psi_2"], "method": "rolling", "window": 51, "threshold": 6}); pipe.connect(j.id, o.id)
    c = pipe.add_node("fit_curve", params={"x": "pressure_psi", "y": "pressure_psi_2", "kind": "linear"}); pipe.connect(o.id, c.id)
    st = Executor(pipe).run(targets=[c.id])[c.id]
    assert st.status == "done", st.error
    rep = st.report
    assert rep["r_squared"] > 0.9 and "pressure_psi_2_fitted" in [c["name"] for c in st.columns]
    assert abs(rep["parameters"][0] - truth["b_slope"]) < 0.05


def test_summarize_and_group_summary(small_csv, tmp_path):
    p = pipe_from_csv(small_csv, tmp_path)
    s = p.add_node("summarize"); p.connect("src", s.id)
    df = run_one(p, s.id)
    assert df.filter(pl.col("column") == "x")["mean"][0] == 5.5
    assert df.filter(pl.col("column") == "name")["distinct"][0] == 2
    assert df.filter(pl.col("column") == "t")["earliest"][0].startswith("2024-01-01")
    g = p.add_node("group_summary", params={"by": ["name"], "default_stats": ["mean", "count"], "count_column": "rows"}); p.connect("src", g.id)
    df = run_one(p, g.id)
    assert df.filter(pl.col("name") == "a")["rows"][0] == 5


@pytest.mark.parametrize("values", [[5.0] * 8, [0.0] * 8, [0.1] * 7 + [None], [1e9 + 0.1] * 8, [3.0]])
@pytest.mark.parametrize("method", ["zscore", "iqr"])
def test_a_constant_column_has_no_outliers(tmp_path, values, method):
    p = pipe_with(tmp_path, pl.DataFrame({"v": values}))
    n = step(p, "remove_outliers", {"columns": ["v"], "method": method, "threshold": 3, "action": "remove"})
    st = Executor(p).run()[n]
    assert st.status == "done" and st.rows == len(values)


def test_zscore_still_finds_a_real_outlier(tmp_path):
    p = pipe_with(tmp_path, pl.DataFrame({"v": [10.0] * 30 + [11.0] * 30 + [500.0]}))
    n = step(p, "remove_outliers", {"columns": ["v"], "method": "zscore", "threshold": 3, "action": "flag"})
    out = pl.read_parquet(Executor(p).run()[n].output)
    assert out["is_outlier"].sum() == 1 and out.filter(pl.col("is_outlier"))["v"].to_list() == [500.0]


def test_limits_on_an_all_blank_column_check_nothing(tmp_path):
    p = pipe_with(tmp_path, pl.DataFrame({"v": pl.Series([None, None, float("nan")], dtype=pl.Float64)}))
    n = step(p, "check_limits", {"column": "v", "min": "0", "max": "10"})
    st = Executor(p).run()[n]
    assert st.status == "done"
    assert st.report["verdict"] == "NOTHING CHECKED" and st.report["checked"] == 0
    assert st.messages[0].startswith("NOTHING CHECKED")


def test_limits_refuse_a_minimum_above_the_maximum(tmp_path):
    p = pipe_with(tmp_path, pl.DataFrame({"v": [1.0, 2.0]}))
    p.set_input("upper", 1)
    n = step(p, "check_limits", {"column": "v", "min": "5", "max": "upper"})
    st = Executor(p).run()[n]
    assert st.status == "failed" and "minimum (5) is above the maximum (1)" in st.error


def test_limits_still_pass_and_fail(tmp_path):
    p = pipe_with(tmp_path, pl.DataFrame({"v": [1.0, None, 12.0]}))
    n = step(p, "check_limits", {"column": "v", "min": "0", "max": "10"})
    st = Executor(p).run()[n]
    assert (st.report["verdict"], st.report["outside"], st.report["checked"]) == ("FAIL", 1, 2)


@pytest.mark.parametrize("fit_groups,predict_groups", [
    (["1", "2"], [1, 2]), ([1.0, 2.0], [1, 2]), ([1, 2], [1.0, 2.0]), ([1, 2], ["1", "2"]), (["a", "b"], ["a", "b"]),
])
def test_predict_finds_the_group_fit_whatever_the_group_type(tmp_path, fit_groups, predict_groups):
    g1, g2 = fit_groups
    p = pipe_with(tmp_path, pl.DataFrame({"g": [g1] * 3 + [g2] * 3, "x": [1.0, 2.0, 3.0] * 2, "y": [2.0, 4.0, 6.0, 3.0, 6.0, 9.0]}))
    step(p, "fit_curve", {"x": "x", "y": "y", "group": "g"}, id_="fit")
    new = tmp_path / "new.parquet"
    pl.DataFrame({"g": predict_groups, "x": [10.0, 10.0]}).write_parquet(new)
    p.add_node("load_file", params={"path": str(new)}, id="new")
    step(p, "predict", {"x": "x", "group": "g", "output": "yhat"}, after="new", port="data", id_="pred")
    p.connect("fit", "pred", "model")
    st = Executor(p).run()["pred"]
    assert st.status == "done", st.error
    assert pl.read_parquet(st.output)["yhat"].to_list() == pytest.approx([20.0, 30.0])


@pytest.mark.parametrize("kind,fn,params", [
    ("linear", lambda x: 2.5 * x + 1, [2.5, 1.0]),
    ("saturating", lambda x: 10 * x / (3 + x), [10.0, 3.0]),
    ("exponential", lambda x: 2 * np.exp(0.3 * x), [2.0, 0.3]),
    ("power", lambda x: 1.5 * x ** 0.7, [1.5, 0.7]),
    ("logarithmic", lambda x: 3 * np.log(x) + 2, [3.0, 2.0]),
])
def test_fit_kinds_recover_parameters(kind, fn, params):
    rng = np.random.default_rng(0)
    xs = np.linspace(0.5, 20, 400)
    ys = fn(xs) * (1 + rng.normal(0, 0.01, 400))
    p, pred = fit_arrays(kind, xs, ys)
    assert np.allclose(p, params, rtol=0.05), (kind, p)
    assert np.allclose(predict_arrays(kind, p, xs), pred)


def test_fit_node_per_group_and_predict(tmp_path):
    rng = np.random.default_rng(1)
    rows = []
    for season, (a, b) in {"summer": (12.0, 2.0), "winter": (6.0, 4.0)}.items():
        for x in np.linspace(0.2, 15, 150):
            rows.append({"season": season, "loading": float(x), "removal": float(a * x / (b + x) * (1 + rng.normal(0, 0.02)))})
    df = pl.DataFrame(rows)
    p = pipe_with(tmp_path, df)
    f = p.add_node("fit_curve", params={"x": "loading", "y": "removal", "kind": "saturating", "group": "season"}); p.connect("src", f.id)
    st = Executor(p).run()[f.id]
    assert st.status == "done", st.error
    fits = {x["group"]: x for x in st.report["fits"]}
    assert fits["summer"]["params"][0] == pytest.approx(12.0, rel=0.05) and fits["winter"]["params"][1] == pytest.approx(4.0, rel=0.1)
    # predict for a proposed loading per season
    new = pl.DataFrame({"season": ["summer", "winter"], "loading": [5.0, 5.0]})
    new.write_parquet(tmp_path / "new.parquet")
    p.add_node("load_file", params={"path": "new.parquet"}, id="new")
    pr = p.add_node("predict", params={"x": "loading", "group": "season", "output": "expected_removal"})
    p.connect("new", pr.id, "data"); p.connect(f.id, pr.id, "model")
    out = run_one(p, pr.id)
    assert out["expected_removal"][0] == pytest.approx(12 * 5 / 7, rel=0.05) and out["expected_removal"][1] == pytest.approx(6 * 5 / 9, rel=0.1)
    # beyond-range caution
    far = pl.DataFrame({"season": ["summer"], "loading": [100.0]}); far.write_parquet(tmp_path / "new.parquet")
    st = Executor(p).run(force=True)[pr.id]
    assert any("outside the range" in m for m in st.messages)


def test_predict_requires_model(tmp_path):
    p = pipe_with(tmp_path, pl.DataFrame({"x": [1.0]}))
    pr = p.add_node("predict"); p.connect("src", pr.id, "data")
    assert "Fit a curve" in Executor(p).run()[pr.id].error


def test_check_limits_with_inputs(tmp_path):
    df = pl.DataFrame({"tp": [0.5, 1.2, 0.9, 3.0, None]})
    p = pipe_with(tmp_path, df)
    p.set_input("permit limit", 1.0, "mg/L")
    c = p.add_node("check_limits", params={"column": "tp", "max": "permit limit"}); p.connect("src", c.id)
    st = Executor(p).run()[c.id]
    assert st.status == "done", st.error
    assert st.report["outside"] == 2 and st.report["blank"] == 1 and st.report["verdict"] == "FAIL"
    assert pl.read_parquet(st.output)["tp_ok"].to_list() == [True, False, True, False, None]   # a blank is not checked
    p.set_input("permit limit", 5.0)
    st2 = Executor(p).run()[c.id]
    assert not st2.from_cache and st2.report["outside"] == 0 and st2.report["verdict"] == "PASS"   # the input changed -> recomputed
    p.set_params(c.id, action="remove")
    assert run_one(p, c.id).height == 4


def test_predict_finds_the_fit_on_either_port(tmp_path):
    from dancr.core.executor import Executor
    xs = np.linspace(1, 10, 50)
    pl.DataFrame({"x": xs, "y": 2 * xs + 1}).write_csv(tmp_path / "d.csv")
    pl.DataFrame({"x": [20.0, 30.0]}).write_csv(tmp_path / "new.csv")
    p = Pipeline("p"); p.path = tmp_path / "p.json"
    p.add_node("load_file", params={"path": str(tmp_path / "d.csv")}, id="src")
    p.add_node("load_file", params={"path": str(tmp_path / "new.csv")}, id="new")
    p.add_node("fit_curve", params={"x": "x", "y": "y", "kind": "linear"}, id="fit"); p.connect("src", "fit")
    p.add_node("predict", params={}, id="pr")
    # no port given: the Fit step is routed to the model port, the table to the data port
    assert p.connect("fit", "pr").port == "model"
    assert p.connect("new", "pr").port == "data"
    res = Executor(p).run()
    assert res["pr"].status == "done", res["pr"].error
    # wired the wrong way round by hand: still works
    p.disconnect("fit", "pr"); p.disconnect("new", "pr")
    p.connect("fit", "pr", "data"); p.connect("new", "pr", "model")
    res = Executor(p).run()
    assert res["pr"].status == "done", res["pr"].error
    df = Executor(p).frame("pr").collect()
    assert np.allclose(df["predicted_y"].to_list(), [41.0, 61.0], rtol=1e-6)


def test_group_summary_count_column_clash_is_friendly(tmp_path):
    p = Pipeline("g")
    p.add_node("enter_data", id="e", params={
        "columns": [{"name": "g", "type": "text"}, {"name": "v", "type": "number"}],
        "rows": [["a", 1.0], ["a", 2.0], ["b", 3.0]]})
    p.add_node("group_summary", id="s", params={
        "by": ["g"], "default_stats": ["mean", "max"], "count_column": "v_mean"})
    p.connect("e", "s")
    p.path = tmp_path / "g.json"
    res = Executor(p).run()
    assert res["s"].status == "failed"
    assert "already used" in (res["s"].error or "")


def test_iqr_outliers_use_excel_quartiles(tmp_path):
    vals = list(range(1, 10)) + [100]
    p = _pipe(pl.DataFrame({"v": [float(v) for v in vals]}), tmp_path)
    n = _step(p, "remove_outliers", {"columns": ["v"], "method": "iqr", "iqr_factor": 1.5, "action": "remove"})
    q1, q3 = np.quantile(vals, [0.25, 0.75])
    keep = [v for v in vals if q1 - 1.5 * (q3 - q1) <= v <= q3 + 1.5 * (q3 - q1)]
    assert run_one(p, n)["v"].to_list() == keep


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


@pytest.mark.parametrize("method", ["zscore", "iqr"])
def test_one_blank_does_not_make_every_row_an_outlier(method):
    p = table([[float(i % 10)] for i in range(200)] + [[None]], [("v", "number")])
    p.add_node("calculate", params={"formulas": [{"name": "r", "expr": "SQRT([v] - 20)"}]}, id="c0"); p.connect("d", "c0")   # NaN
    df = after(p, "remove_outliers", {"columns": ["r"], "method": method}, src="c0")
    assert df.height == 201


def test_clipping_whole_numbers_to_fractional_limits():
    p = table([[1], [2], [3], [4]], [("n", "number")])
    p.add_node("change_type", params={"columns": ["n"], "to": "integer"}, id="i"); p.connect("d", "i")
    df = after(p, "remove_outliers", {"columns": ["n"], "method": "range", "min": "1.5", "max": "3.5", "action": "clip"}, src="i")
    assert df["n"].to_list() == [1.5, 2.0, 3.0, 3.5]


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


def test_group_summary_duplicate_by_and_alias_with_several_stats(tmp_path):
    p = pipe_with(tmp_path, pl.DataFrame({"g": ["a", "a", "b"], "v": [1.0, 3.0, 5.0]}))
    s = p.add_node("group_summary", params={"by": ["g", "g"]}); p.connect("src", s.id)
    assert "listed twice" in failed(p, s.id)
    p.set_params(s.id, by=["g"], aggregations=[{"column": "v", "stats": ["min", "max"], "alias": "val"}])
    assert run_one(p, s.id).columns == ["g", "val_min", "val_max"]


def test_fit_curve_ignores_inf(tmp_path):
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [2.0, 4.0, float("inf"), 8.0]})
    p = pipe_with(tmp_path, df)
    c = p.add_node("fit_curve", params={"x": "x", "y": "y"}); p.connect("src", c.id)
    st = Executor(p).run()[c.id]
    assert st.status == "done" and st.report["parameters"][0] == pytest.approx(2.0) and st.report["points"] == 3
    json.loads(json.dumps(st.to_dict()), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))


@pytest.mark.parametrize("a,b", [(10, 2), (5, 0.5), (-3, 1)])
def test_a_saturating_fit_on_exact_data_settles(a, b):
    x = np.linspace(0.1, 50, 5000)
    lf = pl.LazyFrame({fits.X: x, fits.Y: a * x / (b + x)})
    (fa, fb), _, converged = fits.fit_lazy(lf, "saturating")
    assert converged and fa == pytest.approx(a) and fb == pytest.approx(b)


def test_a_text_group_code_needs_its_exact_text(tmp_path):
    p = pipe_with(tmp_path, pl.DataFrame({"g": ["1"] * 3 + ["2"] * 3, "x": [1.0, 2.0, 3.0] * 2, "y": [2.0, 4.0, 6.0, 3.0, 6.0, 9.0]}))
    step(p, "fit_curve", {"x": "x", "y": "y", "group": "g"}, id_="fit")
    new = tmp_path / "new.parquet"
    pl.DataFrame({"g": ["01", "2"], "x": [10.0, 10.0]}).write_parquet(new)     # "01" is another code, not group 1
    p.add_node("load_file", params={"path": str(new)}, id="new")
    step(p, "predict", {"x": "x", "group": "g", "output": "yhat"}, after="new", port="data", id_="pred")
    p.connect("fit", "pred", "model")
    st = Executor(p).run()["pred"]
    assert st.status == "done", st.error
    assert pl.read_parquet(st.output)["yhat"].to_list() == [None, pytest.approx(30.0)]


@pytest.mark.parametrize("kind", ["linear", "logarithmic", "power", "exponential", "saturating", "polynomial"])
@pytest.mark.parametrize("x", [0.1, 123.456, 1e9 + 0.1])
def test_every_shape_refuses_a_single_x_value(kind, x):
    """With one x value any parameters 'fit'; the nonlinear shapes (and a line whose centred sum of squares came
    out a hair above zero) used to report a meaningless equation instead of saying so."""
    lf = pl.DataFrame({"x": [x] * 7, "y": [1.0, 2, 3, 4, 5, 6, 7]}).lazy()
    with pytest.raises(ValueError, match="All x values are the same"):
        fits.fit_frame(lf, "x", "y", kind)
