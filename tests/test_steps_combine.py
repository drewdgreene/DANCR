"""Combining tables: match on keys, nearest time, and stacking."""
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from conftest import run_one
from dancr.core import Pipeline
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


def test_stack_and_dedupe(small_csv, tmp_path):
    p = pipe_from_csv(small_csv, tmp_path)
    s = p.add_node("stack", params={"label_column": "src", "labels": ["one", "two"]})
    p.add_node("load_file", params={"path": str(small_csv)}, id="src2")
    p.connect("src", s.id); p.connect("src2", s.id)
    df = run_one(p, s.id)
    assert len(df) == 20 and df["src"].unique().sort().to_list() == ["one", "two"]
    d = p.add_node("remove_duplicates", params={"columns": ["x"]}); p.connect(s.id, d.id)
    assert len(run_one(p, d.id)) == 10


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


def test_combine_matches_text_keys_ignoring_case_and_keeps_them_as_written(tmp_path):
    p = pipe_with(tmp_path, pl.DataFrame({"code": ["abc", "Def", "x", None], "v": [1, 2, 3, 4]}), "a.parquet", "a")
    pl.DataFrame({"code": ["ABC", "def", "zz"], "w": [10, 20, 30]}).write_parquet(tmp_path / "b.parquet")
    p.add_node("load_file", params={"path": str(tmp_path / "b.parquet")}, id="b")
    p.add_node("combine", params={"method": "match", "on": ["code"], "how": "left"}, id="c")
    p.connect("a", "c", "left"); p.connect("b", "c", "right")
    out = pl.read_parquet(Executor(p).run()["c"].output)
    assert out.columns == ["code", "v", "w"]
    assert out.rows() == [("abc", 1, 10), ("Def", 2, 20), ("x", 3, None), (None, 4, None)]
    p.set_params("c", how="outer")
    out = pl.read_parquet(Executor(p).run()["c"].output)
    assert sorted(out["code"].to_list(), key=str) == sorted(["abc", "Def", "x", None, "zz"], key=str)


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


def test_a_date_matched_to_a_zoned_time_uses_the_local_clock():
    left = pl.LazyFrame({"day": [datetime(2024, 6, 1), datetime(2024, 6, 2)]}).with_columns(pl.col("day").cast(pl.Date))
    right = (pl.LazyFrame({"t": [datetime(2024, 6, 1, 0, 20), datetime(2024, 6, 2, 0, 20)], "v": [10, 20]})
             .with_columns(pl.col("t").dt.replace_time_zone("Europe/Oslo")))
    from dancr.core.nodes.combine import _combine
    from dancr.core.registry import Ctx
    out = _combine(Ctx(Path("."), "c", "c"), {"left": [left], "right": [right]},
                   {"method": "nearest_time", "left_time": "day", "right_time": "t", "tolerance": "30m"}).frame.collect()
    assert out["v"].to_list() == [10, 20]


def test_tables_in_different_time_zones_stack():
    from dancr.core.nodes.basic import _stack
    from dancr.core.registry import Ctx
    a = pl.LazyFrame({"t": [datetime(2024, 1, 1, 12)]}).with_columns(pl.col("t").dt.replace_time_zone("Europe/Oslo"))
    b = pl.LazyFrame({"t": [datetime(2024, 1, 1, 12)]})
    res = _stack(Ctx(Path("."), "s", "s"), {"tables": [a, b]}, {})
    assert res.frame.collect()["t"].to_list()[0].hour == 11 and res.messages


def test_combine_drops_blank_times_and_matches_int_to_float_keys(tmp_path):
    a = pl.DataFrame({"t": [datetime(2024, 1, 1), None, datetime(2024, 1, 3)], "k": [1, 2, 3]})
    b = pl.DataFrame({"t": [datetime(2024, 1, 1), datetime(2024, 1, 3)], "k": [1.0, 3.0], "v": ["p", "q"]})
    p = pipe_with(tmp_path, a); b.write_parquet(tmp_path / "b.parquet"); p.add_node("load_file", params={"path": "b.parquet"}, id="b")
    j = p.add_node("combine", params={"method": "nearest_time"}); p.connect("src", j.id, "left"); p.connect("b", j.id, "right")
    assert run_one(p, j.id).height == 2
    p.set_params(j.id, method="match", on=["k"], how="inner")
    out = run_one(p, j.id)
    assert out["v"].to_list() == ["p", "q"]


def test_join_on_int64_and_uint64_keys_is_exact(tmp_path):
    left = pl.DataFrame({"k": pl.Series([2 ** 53, 2 ** 53 + 1], dtype=pl.Int64), "x": [1, 2]})
    right = pl.DataFrame({"k": pl.Series([2 ** 53 + 1], dtype=pl.UInt64), "y": [9]})
    left.write_parquet(tmp_path / "l.parquet"); right.write_parquet(tmp_path / "r.parquet")
    p = Pipeline("t")
    p.add_node("load_file", "L", {"path": "l.parquet"}, id="l")
    p.add_node("load_file", "R", {"path": "r.parquet"}, id="r")
    p.add_node("combine", params={"method": "match", "on": ["k"], "right_on": ["k"], "how": "left"}, id="j")
    p.connect("l", "j", "left"); p.connect("r", "j", "right")
    p.path = tmp_path / "p.json"
    df = run_one(p, "j")
    assert df["y"].to_list() == [None, 9]
    assert df.schema["k"] == pl.Int64                     # the key keeps the first table's type


def test_combine_time_mismatch_and_clash(tmp_path):
    a = pl.DataFrame({"time": pl.datetime_range(datetime(2024, 1, 1), datetime(2024, 1, 1, 0, 0, 3), "1s", eager=True), "v": [1, 2, 3, 4]})
    b = pl.DataFrame({"ts": pl.datetime_range(datetime(2024, 1, 1), datetime(2024, 1, 1, 0, 0, 3), "1s", eager=True, time_unit="ms"),
                      "time": ["x"] * 4, "w": [5, 6, 7, 8]}).with_columns(pl.col("ts").dt.replace_time_zone("UTC"))
    a.write_parquet(tmp_path / "a.parquet"); b.write_parquet(tmp_path / "b.parquet")
    p = Pipeline(); p.path = tmp_path / "p.json"
    p.add_node("load_file", params={"path": "a.parquet"}, id="a"); p.add_node("load_file", params={"path": "b.parquet"}, id="b")
    j = p.add_node("combine", params={"method": "nearest_time", "left_time": "time", "right_time": "ts"}); p.connect("a", j.id, "left"); p.connect("b", j.id, "right")
    out = run_one(p, j.id)
    assert out["w"].to_list() == [5, 6, 7, 8] and "time_2" in out.columns
    p.set_params(j.id, method="match", on=["time"], right_on=["ts"], how="inner")
    assert run_one(p, j.id).height == 4
