"""Steps give the right answer (review 2026-09-24, phase 2)."""
from datetime import datetime, timedelta

import polars as pl
import pytest

from dancr.core import Pipeline
from dancr.core.executor import Executor
from dancr.core.expr import compile_formula
from dancr.core.timeutil import detect_datetime_format, day_month_ambiguous
from tests.conftest import run_one


def _pipe_on(df: pl.DataFrame, tmp_path, name: str = "in.parquet") -> Pipeline:
    df.write_parquet(tmp_path / name)
    p = Pipeline("t")
    p.add_node("load_file", "Load", {"path": name}, id="src")
    p.path = tmp_path / "p.json"
    return p


# ------------------------------------------------------------------ centred time windows
def _brute_centred(ts: list[datetime], vs: list[float], half: timedelta, fn) -> list[float]:
    return [fn([v for tt, v in zip(ts, vs) if t - half <= tt <= t + half]) for t in ts]


@pytest.mark.parametrize("stat,fn", [("sum", sum), ("mean", lambda xs: sum(xs) / len(xs)), ("max", max)])
def test_centred_time_window_is_centred(tmp_path, stat, fn):
    base = datetime(2024, 1, 1)
    ts = [base + timedelta(seconds=s) for s in (0, 1, 2, 3, 3, 4, 5.5, 6, 9)]   # repeated and irregular times
    vs = [0.0, 0.0, 0.0, 10.0, 4.0, 0.0, 2.0, 1.0, 7.0]
    p = _pipe_on(pl.DataFrame({"t": ts, "v": vs}), tmp_path)
    p.add_node("rolling", params={"columns": ["v"], "stat": stat, "window": "3s", "centered": True}, id="r")
    p.connect("src", "r")
    got = run_one(p, "r")[f"v_{stat}_3s"].to_list()
    assert got == pytest.approx(_brute_centred(ts, vs, timedelta(seconds=1.5), fn))


def test_centred_and_trailing_windows_differ(tmp_path):
    ts = [datetime(2024, 1, 1) + timedelta(seconds=i) for i in range(7)]
    p = _pipe_on(pl.DataFrame({"t": ts, "v": [0.0, 0, 0, 10, 0, 0, 0]}), tmp_path)
    p.add_node("rolling", params={"columns": ["v"], "stat": "sum", "window": "3s", "centered": True}, id="c")
    p.add_node("rolling", params={"columns": ["v"], "stat": "sum", "window": "3s", "centered": False}, id="t")
    p.connect("src", "c"); p.connect("src", "t")
    assert run_one(p, "c")["v_sum_3s"].to_list() == [0, 0, 10, 10, 10, 0, 0]
    assert run_one(p, "t")["v_sum_3s"].to_list() == [0, 0, 0, 10, 10, 10, 0]


def test_centred_time_window_replace_keeps_column_order(tmp_path):
    ts = [datetime(2024, 1, 1) + timedelta(seconds=i) for i in range(4)]
    p = _pipe_on(pl.DataFrame({"v": [1.0, 2, 3, 4], "t": ts, "w": [0.0, 0, 0, 0]}), tmp_path)
    p.add_node("rolling", params={"columns": ["v"], "stat": "mean", "window": "2s", "replace": True}, id="r")
    p.connect("src", "r")
    df = run_one(p, "r")
    assert df.columns == ["v", "t", "w"]
    assert df["v"].to_list() == [1.5, 2.0, 3.0, 3.5]


# ------------------------------------------------------------------ day/month dates
def test_ambiguous_dates_default_to_month_first():
    s = pl.Series(["01/05/2024", "02/06/2024", "03/07/2024"])
    assert detect_datetime_format(s) == "%m/%d/%Y"
    assert detect_datetime_format(s, day_first=True) == "%d/%m/%Y"
    assert day_month_ambiguous(s, "%m/%d/%Y")
    assert detect_datetime_format(pl.Series(["13/05/2024", "01/05/2024"])) == "%d/%m/%Y"   # a 13th decides
    assert detect_datetime_format(pl.Series(["05/13/2024", "01/05/2024"])) == "%m/%d/%Y"
    assert not day_month_ambiguous(pl.Series(["13/05/2024"]), "%d/%m/%Y")


def _minute_log(tmp_path, fmt: str) -> Pipeline:
    ts = pl.datetime_range(datetime(2024, 1, 5), datetime(2024, 1, 20), "1m", eager=True)
    df = pl.DataFrame({"when": ts.dt.strftime(fmt), "v": range(len(ts))})
    df.write_csv(tmp_path / "log.csv")
    p = Pipeline("t")
    p.add_node("load_file", "Load", {"path": "log.csv"}, id="src")
    p.path = tmp_path / "p.json"
    return p


def test_us_log_whose_first_rows_are_ambiguous_is_read_month_first(tmp_path):
    p = _minute_log(tmp_path, "%m/%d/%Y %H:%M")
    st = Executor(p).run()["src"]
    df = pl.read_parquet(st.output)
    assert df["when"].null_count() == 0
    assert df["when"][0] == datetime(2024, 1, 5)
    assert any("whole file fits month/day" in m or "month/day" in m for m in st.messages)


def test_european_log_whose_first_rows_are_ambiguous_is_read_day_first_over_the_whole_file(tmp_path):
    p = _minute_log(tmp_path, "%d/%m/%Y %H:%M")     # the first 2000 rows are all 05/01/2024 (5 January)
    st = Executor(p).run()["src"]
    df = pl.read_parquet(st.output)
    assert df["when"].null_count() == 0
    assert df["when"][0] == datetime(2024, 1, 5)
    assert any("whole file fits day/month" in m for m in st.messages)


def test_genuinely_ambiguous_dates_say_how_they_were_read(tmp_path):
    pl.DataFrame({"d": ["01/05/2024", "02/06/2024"], "v": [1, 2]}).write_csv(tmp_path / "x.csv")
    p = Pipeline("t"); p.add_node("load_file", "L", {"path": "x.csv"}, id="src"); p.path = tmp_path / "p.json"
    st = Executor(p).run()["src"]
    assert pl.read_parquet(st.output)["d"][0] == datetime(2024, 1, 5)
    assert any("read as month/day" in m and "Day comes before month" in m for m in st.messages)
    p.set_params("src", day_first=True)
    assert pl.read_parquet(Executor(p).run()["src"].output)["d"][0] == datetime(2024, 5, 1)


# ------------------------------------------------------------------ integer arithmetic
@pytest.mark.parametrize("formula,expected", [
    ("u - w", -2), ("i8 + i8", 200), ("i8 * i8", 10000), ("-u", -1), ("LEN(t) - 5", -2), ("u64 + 1", 2.0 ** 63 + 1),
])
def test_integer_arithmetic_does_not_wrap(formula, expected):
    df = pl.DataFrame({"u": pl.Series([1], dtype=pl.UInt32), "w": pl.Series([3], dtype=pl.UInt32),
                       "i8": pl.Series([100], dtype=pl.Int8), "u64": pl.Series([2 ** 63], dtype=pl.UInt64), "t": ["abc"]})
    e, _, _ = compile_formula(formula, dict(df.schema), {})
    assert df.select(e.alias("r"))["r"][0] == pytest.approx(expected)


def test_whole_numbers_stay_whole():
    df = pl.DataFrame({"a": pl.Series([2], dtype=pl.Int32)})
    e, _, _ = compile_formula("a * 3", dict(df.schema), {})
    assert df.select(e.alias("r")).schema["r"] == pl.Int64


# ------------------------------------------------------------------ join keys
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


# ------------------------------------------------------------------ blanks and "does not equal"
@pytest.mark.parametrize("op,value,kept", [
    ("ne", "bad", ["ok", None, "ok"]),
    ("not_contains", "ba", ["ok", None, "ok"]),
])
def test_blank_cells_are_not_equal_to_a_value(tmp_path, op, value, kept):
    p = _pipe_on(pl.DataFrame({"s": ["ok", "bad", None, "ok"]}), tmp_path)
    p.add_node("keep_rows", params={"conditions": {"match": "all", "rules": [{"column": "s", "op": op, "value": value}]}}, id="k")
    p.connect("src", "k")
    assert run_one(p, "k")["s"].to_list() == kept


def test_blank_numbers_are_not_equal_to_a_value(tmp_path):
    p = _pipe_on(pl.DataFrame({"n": [1.0, 2.0, None]}), tmp_path)
    p.add_node("keep_rows", params={"conditions": {"match": "all", "rules": [{"column": "n", "op": "ne", "value": "2"}]}}, id="k")
    p.connect("src", "k")
    assert run_one(p, "k")["n"].to_list() == [1.0, None]


def test_uint64_arithmetic_stays_exact():
    df = pl.DataFrame({"u": pl.Series([2 ** 60 + 1, 2 ** 60 + 3, 2 ** 64 - 1], dtype=pl.UInt64)})
    e, _, _ = compile_formula("u + 1", dict(df.schema), {})
    assert df.select(e.alias("r"))["r"].to_list() == [2 ** 60 + 2, 2 ** 60 + 4, 2 ** 64]
    e, _, _ = compile_formula("-u", dict(df.schema), {})
    assert df.select(e.alias("r"))["r"][2] == -(2 ** 64 - 1)
