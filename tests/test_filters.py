"""Filter rules follow the same Excel rules as formulas; a condition without its value is left out, or refused."""
from datetime import datetime, timedelta

import polars as pl
import pytest

from conftest import run_one
from dancr.core import Pipeline
from dancr.core.conditions import build_mask, incomplete_rules, ops_for_kind, rule_mask
from dancr.core.dtypes import number_from_text, text_to_bool
from dancr.core.executor import Executor
from dancr.core.expr import compile_formula


DF = pl.DataFrame({"s": ["abc", "ABC", None, "", "x"], "n": [1.0, None, 3.0, float("nan"), 4.0]})


def keep(rule, df=DF):
    return df.select(rule_mask(df.schema, rule).fill_null(False).alias("m"))["m"].to_list()


def formula(src, df=DF):
    e, _, _ = compile_formula(src, df.schema)
    return df.select(e.fill_null(False).alias("m"))["m"].to_list()


def table(rows: list[list], columns: list[tuple[str, str]]) -> Pipeline:
    p = Pipeline("t")
    p.add_node("enter_data", params={"columns": [{"name": n, "type": t} for n, t in columns], "rows": rows}, id="d")
    return p


def after(p: Pipeline, type_key: str, params: dict, src: str = "d", nid: str = "s") -> pl.DataFrame:
    p.add_node(type_key, params=params, id=nid)
    p.connect(src, nid)
    return Executor(p).preview(nid)[0]


def _pipe(df: pl.DataFrame, tmp_path, name: str = "in.parquet") -> Pipeline:
    df.write_parquet(tmp_path / name)
    p = Pipeline("t")
    p.add_node("load_file", "Load", {"path": name}, id="src")
    p.path = tmp_path / "p.json"
    return p


@pytest.mark.parametrize("op,value,src", [
    ("eq", "abc", 's = "abc"'), ("ne", "abc", 's <> "abc"'), ("ne", "x", 's <> "X"'), ("empty", None, 's = ""'),
])
def test_filter_rules_and_formulas_agree_on_text(op, value, src):
    assert keep({"column": "s", "op": op, "value": value}) == formula(src)


@pytest.mark.parametrize("op,value,src", [("eq", "3", "n = 3"), ("ne", "3", "n <> 3"), ("gt", "2", "n > 2"), ("le", "3", "n <= 3")])
def test_filter_rules_and_formulas_agree_on_numbers(op, value, src):
    assert keep({"column": "n", "op": op, "value": value}) == formula(src)


def test_text_equality_is_case_insensitive_and_a_blank_differs():
    assert keep({"column": "s", "op": "eq", "value": "ABC"}) == [True, True, False, False, False]
    assert keep({"column": "s", "op": "ne", "value": "abc"}) == [False, False, True, True, True]


def test_blank_cells_are_asked_for_with_is_empty():
    assert keep({"column": "s", "op": "empty"}) == [False, False, True, True, False]
    assert keep({"column": "s", "op": "not_empty"}) == [True, True, False, False, True]
    assert keep({"column": "n", "op": "empty"}) == [False, True, False, True, False]     # NaN is blank


@pytest.mark.parametrize("op", ["eq", "ne", "gt", "lt", "ge", "le", "contains", "not_contains", "starts", "ends", "in"])
def test_a_condition_without_its_value_is_unfinished(op):
    col = "s" if op in ("contains", "not_contains", "starts", "ends", "in") else "n"
    rule = {"column": col, "op": op, "value": ""}
    assert incomplete_rules({"rules": [rule]}) == [rule]
    with pytest.raises(ValueError, match="enter a value"):
        build_mask(DF.schema, {"rules": [rule]})


def test_between_needs_both_ends():
    with pytest.raises(ValueError, match="both ends"):
        rule_mask(DF.schema, {"column": "n", "op": "between", "value": "1", "value2": ""})


def test_keep_rows_says_a_rule_without_a_value_is_left_out(tmp_path):
    f = tmp_path / "t.parquet"
    DF.write_parquet(f)
    p = Pipeline(); p.path = tmp_path / "p.json"
    p.add_node("load_file", params={"path": str(f)}, id="src")
    p.add_node("keep_rows", params={"conditions": {"match": "all", "rules": [{"column": "n", "op": "gt", "value": ""}]}}, id="k")
    p.connect("src", "k")
    st = Executor(p).run()["k"]          # made from a column's menu, before the value is typed
    assert st.status == "done" and st.rows == DF.height
    assert any("no value yet" in m for m in st.messages)
    p.set_params("k", conditions={"match": "all", "rules": [{"column": "s", "op": "eq", "value": ""}]})
    st = Executor(p).run()["k"]          # 'equals' is the first condition a new rule offers
    assert st.status == "done" and st.rows == DF.height
    p.set_params("k", conditions={"match": "all", "rules": [{"column": "s", "op": "empty"}]})
    st = Executor(p).run()["k"]
    assert st.status == "done" and st.rows == 2


def test_is_one_of_splits_numbers_on_commas():
    df = pl.DataFrame({"n": [100.0, 200.0, 100200300.0, 1000.0]})
    assert df.filter(rule_mask(df.schema, {"column": "n", "op": "in", "value": "100,200,300"})).height == 2
    assert df.filter(rule_mask(df.schema, {"column": "n", "op": "in", "value": "1,000; 7"})).height == 1


def test_nan_is_a_blank_in_filters_and_statistics():
    p = table([[1.0], [2.0], [3.0], [100.0], [-1.0]], [("v", "number")])
    p.add_node("calculate", params={"formulas": [{"name": "w", "expr": "IF([v] < 0, SQRT([v]), [v])"}]}, id="c0"); p.connect("d", "c0")
    kept = after(p, "keep_rows", {"conditions": {"match": "all", "rules": [{"column": "w", "op": "gt", "value": 50}]}}, src="c0", nid="k")
    assert kept["w"].to_list() == [100.0]
    g = after(p, "group_summary", {"columns": ["w"], "default_stats": ["median"]}, src="c0", nid="g")
    assert g["w"].to_list() == [2.5]


def test_conditions_in_is_case_insensitive_like_eq():
    from dancr.core.conditions import rule_mask

    df = pl.DataFrame({"s": ["OK", "ok", "no"]})
    m = lambda rule: df.select(rule_mask(df.schema, rule).alias("m"))["m"].to_list()
    assert m({"column": "s", "op": "eq", "value": "ok"}) == [True, True, False]
    assert m({"column": "s", "op": "in", "value": "ok"}) == [True, True, False]
    assert m({"column": "s", "op": "in", "value": "ok", "case_sensitive": True}) == [False, True, False]


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


@pytest.mark.parametrize("op,value,kept", [
    ("ne", "bad", ["ok", None, "ok"]),
    ("not_contains", "ba", ["ok", None, "ok"]),
])
def test_blank_cells_are_not_equal_to_a_value(tmp_path, op, value, kept):
    p = _pipe(pl.DataFrame({"s": ["ok", "bad", None, "ok"]}), tmp_path)
    p.add_node("keep_rows", params={"conditions": {"match": "all", "rules": [{"column": "s", "op": op, "value": value}]}}, id="k")
    p.connect("src", "k")
    assert run_one(p, "k")["s"].to_list() == kept


def test_blank_numbers_are_not_equal_to_a_value(tmp_path):
    p = _pipe(pl.DataFrame({"n": [1.0, 2.0, None]}), tmp_path)
    p.add_node("keep_rows", params={"conditions": {"match": "all", "rules": [{"column": "n", "op": "ne", "value": "2"}]}}, id="k")
    p.connect("src", "k")
    assert run_one(p, "k")["n"].to_list() == [1.0, None]


def test_conditions_tz_categorical_thousands():
    df = pl.DataFrame({"t": pl.datetime_range(datetime(2024, 1, 1), datetime(2024, 1, 3), "1d", eager=True),
                       "c": ["a", "", None], "n": [1000.0, 2500.0, 7.0]})
    tz = df.with_columns(pl.col("t").dt.replace_time_zone("Europe/Oslo"), pl.col("c").cast(pl.Categorical))
    m = rule_mask(tz.schema, {"column": "t", "op": "gt", "value": "2024-01-02"})
    assert tz.filter(m).height == 1
    m = rule_mask(tz.schema, {"column": "c", "op": "empty"})
    assert tz.filter(m).height == 2
    m = rule_mask(tz.schema, {"column": "n", "op": "in", "value": "1,000; 2,500"})
    assert tz.filter(m).height == 2
    m = rule_mask(tz.schema, {"column": "n", "op": "in", "value": "1000, 7"})
    assert tz.filter(m).height == 2


def test_a_year_must_be_whole_and_finite():
    df = pl.DataFrame({"d": [datetime(2024, 1, 1)]})
    with pytest.raises(ValueError, match="whole year"):
        rule_mask(df.schema, {"column": "d", "op": "year", "value": "2024.7"})
    with pytest.raises(ValueError):
        rule_mask(df.schema, {"column": "d", "op": "year", "value": "inf"})


def test_a_reversed_between_range_is_refused_not_silently_empty():
    with pytest.raises(ValueError, match="before it starts"):
        rule_mask(DF.schema, {"column": "n", "op": "between", "value": 10, "value2": 2})
