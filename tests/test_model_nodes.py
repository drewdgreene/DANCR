"""Fit, predict, limits, inputs, type-in tables, fix values, workbook, summarise-around."""

from datetime import datetime

import numpy as np
import polars as pl
import pytest

from dancr.core import Pipeline, PipelineError
from dancr.core.executor import Executor
from dancr.core.fits import fit_arrays, predict_arrays
from conftest import run_one


def pipe_with(tmp_path, df, name="t.parquet"):
    f = tmp_path / name; df.write_parquet(f)
    p = Pipeline(); p.path = tmp_path / "p.json"
    p.add_node("load_file", params={"path": str(f)}, id="src")
    return p


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
    assert st.report["outside"] == 3 and st.report["verdict"] == "FAIL"
    assert pl.read_parquet(st.output)["tp_ok"].to_list() == [True, False, True, False, False]
    p.set_input("permit limit", 5.0)
    st2 = Executor(p).run()[c.id]
    assert not st2.from_cache and st2.report["outside"] == 1        # the input changed -> recomputed
    p.set_params(c.id, action="remove")
    assert run_one(p, c.id).height == 4


def test_inputs_in_formulas_and_filters(tmp_path):
    df = pl.DataFrame({"flow": [100.0, 200.0]})
    p = pipe_with(tmp_path, df)
    p.set_input("belt area", 12.5, "m²"); p.set_input("cutoff", 150)
    c = p.add_node("calculate", params={"formulas": [{"name": "per_area", "expr": "flow / [belt area]"}]}); p.connect("src", c.id)
    assert run_one(p, c.id)["per_area"].to_list() == [8.0, 16.0]
    k = p.add_node("keep_rows", params={"conditions": {"match": "all", "rules": [{"column": "flow", "op": "gt", "value": "cutoff"}]}}); p.connect("src", k.id)
    assert run_one(p, k.id)["flow"].to_list() == [200.0]
    ex = Executor(p)
    assert set(ex.inputs_used(c.id)) == {"belt area"} and set(ex.inputs_used(k.id)) == {"cutoff"} and ex.inputs_used("src") == {}
    p.save(); q = Pipeline.load(p.path)
    assert q.input_values() == {"belt area": 12.5, "cutoff": 150}
    with pytest.raises(PipelineError):
        p.set_input("1bad", 3)


def test_enter_data_and_fix_values(tmp_path):
    p = Pipeline(); p.path = tmp_path / "p.json"
    e = p.add_node("enter_data", params={"columns": [{"name": "specimen", "type": "text"}, {"name": "width_mm", "type": "number"}, {"name": "tested", "type": "datetime"}],
                                         "rows": [["A1", "12.5", "2024-06-01"], ["A2", "1,200", "2024-06-02"], ["A3", "", ""]]})
    df = run_one(p, e.id)
    assert df["width_mm"].to_list() == [12.5, 1200.0, None] and df.schema["tested"] == pl.Datetime("us")
    f = p.add_node("fix_values", params={"fixes": [{"row": 2, "column": "width_mm", "value": "12.0", "was": "1,200", "note": "typo"}]}); p.connect(e.id, f.id)
    out = run_one(p, f.id)
    assert out["width_mm"].to_list() == [12.5, 12.0, None]
    p.save(); assert Pipeline.load(p.path).nodes[e.id].params["rows"][1][1] == "1,200"


def test_workbook_and_summarise_around(pipe, tmp_path):
    tb = pipe.add_node("time_buckets", params={"every": "5m"}); pipe.connect("a", tb.id)
    g = pipe.add_node("find_gaps"); pipe.connect("a", g.id)
    w = pipe.add_node("workbook", params={"path": "out/book.xlsx"}); pipe.connect(tb.id, w.id); pipe.connect(g.id, w.id)
    st = Executor(pipe).run()[w.id]
    assert st.status == "done", st.error
    import fastexcel
    names = fastexcel.read_excel(str(tmp_path / "out" / "book.xlsx")).sheet_names
    assert names == ["Average over time", "Find gaps"]
    # samples every 10 minutes; summarise the 10 minutes before each
    samples = pl.DataFrame({"when": [datetime(2024, 6, 1, 0, 10), datetime(2024, 6, 1, 0, 20)], "lab_value": [1.0, 2.0]})
    samples.write_parquet(tmp_path / "s.parquet")
    pipe.add_node("load_file", params={"path": "s.parquet"}, id="s")
    ar = pipe.add_node("summarise_around", params={"window": "10m", "columns": ["pressure_psi"], "stats": ["mean", "count"]})
    pipe.connect("s", ar.id, "samples"); pipe.connect("a", ar.id, "log")
    out = run_one(pipe, ar.id)
    assert out.columns == ["when", "lab_value", "pressure_psi_mean", "pressure_psi_count"]
    a = pl.read_parquet(Executor(pipe).state("a").output)
    expected = a.filter((pl.col("time") > datetime(2024, 6, 1, 0, 0)) & (pl.col("time") <= datetime(2024, 6, 1, 0, 10))).height
    assert out["pressure_psi_count"][0] == expected and expected > 10000


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
