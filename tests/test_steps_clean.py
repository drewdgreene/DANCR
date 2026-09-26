"""Cleaning steps: keep rows, choose columns, sort, calculate, fix values, fill blanks, change type, take a sample."""
from datetime import date, datetime

import polars as pl
import pytest

from conftest import run_one
from dancr.core import Pipeline
from dancr.core.dtypes import temp_name
from dancr.core.executor import Executor


def pipe_from_csv(path, tmp_path):
    p = Pipeline()
    p.path = tmp_path / "p.json"
    p.add_node("load_file", params={"path": str(path)}, id="src")
    return p


def pipe_with(tmp_path, df, name="t.parquet", id_="src"):
    f = tmp_path / name
    df.write_parquet(f)
    p = Pipeline(); p.path = tmp_path / "p.json"
    p.add_node("load_file", params={"path": str(f)}, id=id_)
    return p


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
    assert run_one(p, k.id)["x"].to_list() == [8.0, 10.0]           # as in Excel, a blank name is not "a"


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


def test_take_sample(small_csv, tmp_path):
    p = pipe_from_csv(small_csv, tmp_path)
    s = p.add_node("take_sample", params={"mode": "every", "every": 3}); p.connect("src", s.id)
    assert run_one(p, s.id)["x"].to_list() == [1.0, 4.0, 7.0, 10.0]
    p.set_params(s.id, mode="last", rows=2)
    assert run_one(p, s.id)["x"].to_list() == [9.0, 10.0]


def test_a_correction_whose_cell_moved_is_not_written(tmp_path):
    p = pipe_with(tmp_path, pl.DataFrame({"v": [1, 2, 3]}))
    p.add_node("fix_values", params={"fixes": [{"row": 2, "column": "v", "value": "20", "was": "2"}]}, id="f")
    p.connect("src", "f")
    assert pl.read_parquet(Executor(p).run()["f"].output)["v"].to_list() == [1, 20, 3]
    pl.DataFrame({"v": [0, 1, 2, 3]}).write_parquet(tmp_path / "t.parquet")     # a row was added at the top
    st = Executor(p).run()["f"]
    assert st.status == "failed"
    assert "row 2, v: it holds '1' now but held '2'" in st.error and "Nothing was changed" in st.error


def test_corrections_of_the_same_cell_follow_each_other(tmp_path):
    p = pipe_with(tmp_path, pl.DataFrame({"v": [1.0, 2.0], "t": [datetime(2024, 1, 1), None], "s": ["a", None]}))
    p.add_node("fix_values", params={"fixes": [
        {"row": 1, "column": "v", "value": "5", "was": "1.0"},
        {"row": 1, "column": "v", "value": "6", "was": "5"},                    # made on the first correction's result
        {"row": 1, "column": "t", "value": "2024-02-01", "was": "2024-01-01 00:00:00"},
        {"row": 2, "column": "s", "value": "b", "was": None},                   # the cell was blank
        {"row": 2, "column": "v", "value": "9"},                                 # no "was": not checked
    ]}, id="f")
    p.connect("src", "f")
    st = Executor(p).run()["f"]
    assert st.status == "done", st.error
    out = pl.read_parquet(st.output)
    assert out["v"].to_list() == [6.0, 9.0] and out["s"].to_list() == ["a", "b"] and out["t"][0] == datetime(2024, 2, 1)


def test_previews_do_not_put_corrections_on_sampled_rows(tmp_path):
    p = pipe_with(tmp_path, pl.DataFrame({"v": list(range(1000))}))
    p.add_node("fix_values", params={"fixes": [{"row": 3, "column": "v", "value": "-1", "was": "2"}]}, id="f")
    p.connect("src", "f")
    ex = Executor(p)
    ex.run(targets=["src"])
    df, res, kind = ex.preview("f", rows=100)          # every 10th row of the table
    assert kind == "spread" and -1 not in df["v"].to_list()
    assert "not shown in this preview" in " ".join(res.messages)
    df, res, kind = ex.preview("f", rows=5000)         # the whole table
    assert kind == "all" and df["v"][2] == -1


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


def test_whole_number_conversion_rounds_like_excel(tmp_path):
    p = _pipe(pl.DataFrame({"s": ["2.5", "3.5", "-2.5"]}), tmp_path)
    n = _step(p, "change_type", {"columns": ["s"], "to": "integer"})
    assert run_one(p, n)["s"].to_list() == [3, 4, -3]


def test_categorical_columns_convert(tmp_path):
    p = _pipe(pl.DataFrame({"c": pl.Series(["1,5", "2"], dtype=pl.Categorical)}), tmp_path)
    n = _step(p, "change_type", {"columns": ["c"], "to": "number"})
    assert run_one(p, n)["c"].to_list() == [1.5, 2.0]


def test_corrections_that_changed_nothing_are_reported(tmp_path):
    p = _pipe(pl.DataFrame({"v": [1, 2]}), tmp_path)
    n = _step(p, "fix_values", {"fixes": [{"row": 5, "column": "v", "value": "9", "was": "1"},
                                          {"row": 1, "column": "gone", "value": "1"},
                                          {"row": 2, "column": "v", "value": "9007199254740993", "was": "7"}]})
    st = Executor(p).run()[n]
    # corrections that no longer fit their cell fail the step rather than overwrite the wrong cell
    assert st.status == "failed" and "row 5, v: the table has only 2 rows" in st.error and "held '7'" in st.error
    p.set_params(n, fixes=[{"row": 1, "column": "gone", "value": "1"},
                           {"row": 2, "column": "v", "value": "9007199254740993", "was": "2"}])
    st = Executor(p).run()[n]
    assert "no longer in the table" in " ".join(st.messages)
    assert pl.read_parquet(st.output)["v"].to_list() == [1, 9007199254740993]


def test_nan_is_a_blank_for_fill_blanks(tmp_path):
    p = _pipe(pl.DataFrame({"v": [1.0, float("nan"), None]}), tmp_path)
    n = _step(p, "fix_missing", {"method": "zero"})
    assert run_one(p, n)["v"].to_list() == [1.0, 0.0, 0.0]


def test_explicit_zero_is_not_the_default(tmp_path):
    p = _pipe(pl.DataFrame({"v": list(range(1000))}), tmp_path)
    n = _step(p, "take_sample", {"mode": "random", "fraction": 0})
    assert run_one(p, n).height == 0


def test_long_ids_written_as_text_become_exact_integers():
    p = table([["12345678901234567"], ["1,000"]], [("id", "text")])
    df = after(p, "change_type", {"columns": ["id"], "to": "integer"})
    assert df["id"].to_list() == [12345678901234567, 1000]


def test_fix_missing_value_keeps_an_integer_column_integer(tmp_path):
    p = pipe_with(tmp_path, pl.DataFrame({"n": [1, None, 3]}))
    p.add_node("fix_missing", params={"method": "value", "value": "5"}, id="fm")
    p.connect("src", "fm")
    out = run_one(p, "fm")
    assert out.schema["n"] == pl.Int64 and out["n"].to_list() == [1, 5, 3]


def test_fix_missing_value_refuses_decimals_for_an_integer_column(tmp_path):
    p = pipe_with(tmp_path, pl.DataFrame({"n": [1, None, 3]}))
    p.add_node("fix_missing", params={"method": "value", "value": "5.5"}, id="fm")
    p.connect("src", "fm")
    st = Executor(p).run(targets=["fm"])["fm"]
    assert st.status == "failed" and "decimals" in st.error


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


def test_fix_missing_keeps_types(tmp_path):
    df = pl.DataFrame({"t": [datetime(2024, 1, 1), None], "b": [True, None], "v": [1.0, None], "s": ["a", None]})
    p = pipe_with(tmp_path, df)
    f = p.add_node("fix_missing", params={"method": "zero"}); p.connect("src", f.id)
    out = run_one(p, f.id)
    assert out.schema["t"] == pl.Datetime("us") and out.schema["b"] == pl.Boolean and out["v"][1] == 0.0 and out["s"][1] == ""
    p.set_params(f.id, method="value", value="2024-02-01", columns=["t"])
    assert run_one(p, f.id)["t"][1] == datetime(2024, 2, 1)


def test_misc_node_validation(tmp_path):
    df = pl.DataFrame({"Pressure": [1.0, 2.0, 3.0], "x": ["a", "b", "c"]})
    p = pipe_with(tmp_path, df)
    with pytest.raises(ValueError, match="at least 1"):     # declared bounds are enforced when the setting is applied
        p.add_node("take_sample", params={"mode": "first", "rows": 0})
    c = p.add_node("change_type", params={"columns": ["pressure"], "to": "text"}); p.connect("src", c.id)
    assert run_one(p, c.id).schema["Pressure"] == pl.Utf8            # case-insensitive fallback honoured
    r = p.add_node("choose_columns", params={"rename": {"x": "Pressure"}}); p.connect("src", r.id)
    assert "already a column" in Executor(p).run()[r.id].error
    st = p.add_node("stack", params={"label_column": "x"}); p.connect("src", st.id)
    assert "already a column" in Executor(p).run()[st.id].error
    tb = p.add_node("time_buckets", params={"every": "1m", "default_stats": ["mean", "bogus"]})
    p.add_node("calculate", params={"formulas": [{"name": "t", "expr": 'DATE("2024-01-01")'}]}, id="cal"); p.connect("src", "cal"); p.connect("cal", tb.id)
    assert "Unknown statistic" in Executor(p).run()[tb.id].error
