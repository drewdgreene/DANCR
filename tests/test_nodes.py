
import polars as pl
import pytest

from dancr.core import Pipeline
from dancr.core.executor import Executor
from conftest import run_one


def pipe_from_csv(path, tmp_path):
    p = Pipeline()
    p.path = tmp_path / "p.json"
    p.add_node("load_file", params={"path": str(path)}, id="src")
    return p


def test_load_detects_dates_and_numbers(small_csv, tmp_path):
    p = pipe_from_csv(small_csv, tmp_path)
    df = run_one(p, "src")
    assert df.schema["t"] == pl.Datetime("us")
    assert df.schema["x"] == pl.Float64
    assert df["name"][9] is None


def test_load_separator_sniff_and_skip_rows(tmp_path):
    f = tmp_path / "semi.txt"
    f.write_text("junk line\nanother junk\na;b;c\n1;2;x\n3;4;y\n")
    p = Pipeline(); p.path = tmp_path / "p.json"
    p.add_node("load_file", params={"path": str(f), "skip_rows": 2}, id="src")
    df = run_one(p, "src")
    assert df.columns == ["a", "b", "c"] and df["b"].to_list() == [2, 4]


def test_load_excel(tmp_path, small_df):
    f = tmp_path / "book.xlsx"
    small_df.write_excel(f)
    p = Pipeline(); p.path = tmp_path / "p.json"
    p.add_node("load_file", params={"path": str(f)}, id="src")
    df = run_one(p, "src")
    assert len(df) == 10 and df.schema["t"] == pl.Datetime("us")


def test_load_missing_file_is_friendly(tmp_path):
    p = Pipeline(); p.path = tmp_path / "p.json"
    p.add_node("load_file", params={"path": "nope.csv"}, id="src")
    st = Executor(p).run()["src"]
    assert st.status == "failed" and "File not found" in st.error


def test_keep_rows_conditions(small_csv, tmp_path):
    p = pipe_from_csv(small_csv, tmp_path)
    k = p.add_node("keep_rows", params={"conditions": {"match": "all", "rules": [
        {"column": "x", "op": "ge", "value": "3"}, {"column": "name", "op": "eq", "value": "A"}]}})
    p.connect("src", k.id)
    df = run_one(p, k.id)
    assert df["x"].to_list() == [3.0, 5.0, 7.0, 9.0]          # case-insensitive text match
    p.set_params(k.id, conditions={"match": "any", "rules": [{"column": "name", "op": "empty"}, {"column": "x", "op": "between", "value": "1", "value2": "2"}]})
    assert run_one(p, k.id)["x"].to_list() == [1.0, 2.0, 10.0]
    p.set_params(k.id, mode="remove", conditions={"match": "all", "rules": [{"column": "t", "op": "lt", "value": "2024-01-01 00:00:08"}]})
    assert run_one(p, k.id)["x"].to_list() == [9.0, 10.0]
    p.set_params(k.id, mode="keep", conditions={"match": "all", "rules": []}, formula="x * 2 > 15 and name != \"a\"")
    assert run_one(p, k.id)["x"].to_list() == [8.0]


def test_keep_rows_bad_value_message(small_csv, tmp_path):
    p = pipe_from_csv(small_csv, tmp_path)
    k = p.add_node("keep_rows", params={"conditions": {"match": "all", "rules": [{"column": "x", "op": "gt", "value": "abc"}]}})
    p.connect("src", k.id)
    st = Executor(p).run()[k.id]
    assert st.status == "failed" and "not a number" in st.error


def test_choose_columns_and_rename(small_csv, tmp_path):
    p = pipe_from_csv(small_csv, tmp_path)
    c = p.add_node("choose_columns", params={"columns": ["x", "t"], "rename": {"x": "value"}})
    p.connect("src", c.id)
    assert run_one(p, c.id).columns == ["value", "t"]
    p.set_params(c.id, mode="drop", columns=["name"], rename={})
    assert run_one(p, c.id).columns == ["t", "x", "y"]


def test_sort(small_csv, tmp_path):
    p = pipe_from_csv(small_csv, tmp_path)
    s = p.add_node("sort", params={"columns": ["y"], "descending": True})
    p.connect("src", s.id)
    assert run_one(p, s.id)["y"][0] == 100.0


def test_calculate_chain_and_errors(small_csv, tmp_path):
    p = pipe_from_csv(small_csv, tmp_path)
    c = p.add_node("calculate", params={"formulas": [{"name": "double", "expr": "x * 2"}, {"name": "quad", "expr": "double * 2"}]})
    p.connect("src", c.id)
    df = run_one(p, c.id)
    assert df["quad"].to_list()[:3] == [4.0, 8.0, 12.0]
    p.set_params(c.id, formulas=[{"name": "bad", "expr": "nope + 1"}])
    st = Executor(p).run()[c.id]
    assert st.status == "failed" and "no column or input called 'nope'" in st.error


def test_fix_missing_and_change_type(tmp_path):
    f = tmp_path / "m.csv"
    f.write_text("id,val,when\n1,,01/02/2024\n2,\"1,500\",03/02/2024\n3,7,\n")
    p = Pipeline(); p.path = tmp_path / "p.json"
    p.add_node("load_file", params={"path": str(f), "parse_dates": False}, id="src")
    ct = p.add_node("change_type", params={"columns": ["val"], "to": "number"}); p.connect("src", ct.id)
    df = run_one(p, ct.id)
    assert df["val"].to_list() == [None, 1500.0, 7.0]
    fm = p.add_node("fix_missing", params={"method": "value", "value": "0", "columns": ["val"]}); p.connect(ct.id, fm.id)
    assert run_one(p, fm.id)["val"].to_list() == [0.0, 1500.0, 7.0]
    p.set_params(fm.id, method="forward", columns=[])
    assert run_one(p, fm.id)["when"].to_list() == ["01/02/2024", "03/02/2024", "03/02/2024"]
    dt = p.add_node("change_type", params={"columns": ["when"], "to": "datetime", "date_format": "%d/%m/%Y"}); p.connect("src", dt.id)
    out = run_one(p, dt.id)
    assert out.schema["when"] == pl.Datetime("us") and out["when"][0].month == 2


def test_stack_and_dedupe(small_csv, tmp_path):
    p = pipe_from_csv(small_csv, tmp_path)
    s = p.add_node("stack", params={"label_column": "src", "labels": ["one", "two"]})
    p.add_node("load_file", params={"path": str(small_csv)}, id="src2")
    p.connect("src", s.id); p.connect("src2", s.id)
    df = run_one(p, s.id)
    assert len(df) == 20 and df["src"].unique().sort().to_list() == ["one", "two"]
    d = p.add_node("remove_duplicates", params={"columns": ["x"]}); p.connect(s.id, d.id)
    assert len(run_one(p, d.id)) == 10


def test_take_sample(small_csv, tmp_path):
    p = pipe_from_csv(small_csv, tmp_path)
    s = p.add_node("take_sample", params={"mode": "every", "every": 3}); p.connect("src", s.id)
    assert run_one(p, s.id)["x"].to_list() == [1.0, 4.0, 7.0, 10.0]
    p.set_params(s.id, mode="last", rows=2)
    assert run_one(p, s.id)["x"].to_list() == [9.0, 10.0]


def test_combine_match_and_nearest_time(tmp_path, small_df):
    a = tmp_path / "a.csv"; b = tmp_path / "b.csv"
    small_df.select(["t", "x"]).write_csv(a)
    small_df.select(["t", "y"]).with_columns((pl.col("t") + pl.duration(milliseconds=300)).alias("t")).write_csv(b)
    p = Pipeline(); p.path = tmp_path / "p.json"
    p.add_node("load_file", params={"path": str(a)}, id="a"); p.add_node("load_file", params={"path": str(b)}, id="b")
    j = p.add_node("combine", params={"method": "nearest_time", "tolerance": "500ms"}); p.connect("a", j.id, "left"); p.connect("b", j.id, "right")
    df = run_one(p, j.id)
    assert len(df) == 10 and df["y"][0] == pytest.approx(2.1)
    p.set_params(j.id, tolerance="100ms")
    assert run_one(p, j.id)["y"].null_count() == 10
    p.set_params(j.id, method="match", on=["t"], how="inner")
    assert len(run_one(p, j.id)) == 0
    p.set_params(j.id, method="side_by_side")
    df = run_one(p, j.id)
    assert df.columns == ["t", "x", "t_2", "y"]


def test_time_buckets(pipe, truth):
    tb = pipe.add_node("time_buckets", params={"every": "1m", "default_stats": ["mean", "max"], "count_column": "n"})
    pipe.connect("a", tb.id)
    df = run_one(pipe, tb.id)
    assert df.columns == ["time", "pressure_psi_mean", "pressure_psi_max", "temp_c_mean", "temp_c_max", "n"]
    assert df["n"].max() == 1200                          # 20 Hz * 60 s
    assert len(df) == pytest.approx(truth["hours"] * 60, abs=1)
    pipe.set_params(tb.id, every="30s", default_stats=["mean"], aggregations=[{"column": "pressure_psi", "stats": ["min"], "alias": "pmin"}])
    df = run_one(pipe, tb.id)
    assert "pmin" in df.columns and "temp_c" not in df.columns


def test_find_gaps_matches_truth(pipe, truth):
    g = pipe.add_node("find_gaps"); pipe.connect("a", g.id)
    df = run_one(pipe, g.id)
    assert len(df) == len(truth["gaps"])
    assert df["gap_seconds"][0] == pytest.approx(truth["gaps"][0]["seconds"], abs=0.1)
    assert df["missing_readings"][0] == truth["gaps"][0]["rows"]


def test_rolling_rate_and_outliers(pipe, truth):
    r = pipe.add_node("rolling", params={"columns": ["pressure_psi"], "stat": "median", "window": "5"}); pipe.connect("b", r.id)
    df = run_one(pipe, r.id)
    assert "pressure_psi_median_5" in df.columns
    pipe.set_params(r.id, window="2s", replace=True)
    assert run_one(pipe, r.id).columns == ["time", "pressure_psi", "temp_c"]
    rate = pipe.add_node("rate_of_change", params={"columns": ["pressure_psi"], "per": "m"}); pipe.connect("a", rate.id)
    df = run_one(pipe, rate.id)
    assert "pressure_psi_per_minute" in df.columns and df["pressure_psi_per_minute"].abs().median() < 5.0
    o = pipe.add_node("remove_outliers", params={"columns": ["pressure_psi"], "method": "rolling", "window": 51, "threshold": 6}); pipe.connect("b", o.id)
    st = Executor(pipe).run(targets=[o.id])
    removed = st["b"].rows - st[o.id].rows
    assert 0 < removed <= truth["spikes_b"] * 3
    pipe.set_params(o.id, action="flag")
    df = run_one(pipe, o.id)
    assert df["is_outlier"].sum() == removed


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


def test_regular_grid(pipe):
    g = pipe.add_node("regular_grid", params={"every": "1s", "method": "nearest"}); pipe.connect("a", g.id)
    df = run_one(pipe, g.id)
    assert (df["time"].diff().drop_nulls().dt.total_seconds() == 1).all()


def test_export_and_chart_pass_through(pipe, tmp_path):
    e = pipe.add_node("export", params={"path": "out/a.csv"}); pipe.connect("a", e.id)
    c = pipe.add_node("chart", params={"kind": "line", "x": "time", "series": [{"column": "pressure_psi"}]}); pipe.connect(e.id, c.id)
    res = Executor(pipe).run()
    assert (tmp_path / "out" / "a.csv").exists()
    assert res[c.id].status == "done" and res[c.id].rows == res["a"].rows
    pipe.set_params(c.id, series=[{"column": "nope"}])
    assert Executor(pipe).run()[c.id].status == "failed"
