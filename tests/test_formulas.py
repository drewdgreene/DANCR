"""The formula language: Excel's answers, names, types, functions, blanks and errors."""
from datetime import date, datetime, timedelta

import polars as pl
import pytest

from dancr.core import Pipeline
from dancr.core.executor import Executor
from dancr.core.expr import FUNCTIONS, FormulaError, check_formula, compile_formula


DF = pl.DataFrame({"s": ["abc", "ABC", None, "x"], "n": [1.0, None, 3.0, 4.0],
                   "d": [date(2024, 6, 2), date(2024, 6, 3), date(2024, 6, 8), None]})


def evaluate(src, df=DF):
    e, _, _ = compile_formula(src, df.schema)
    return df.select(e.alias("r"))["r"].to_list()


def one(src):
    return evaluate(src)[0]


@pytest.fixture
def df():
    return pl.DataFrame({
        "Pressure A": [1.0, 2.0, 3.0, 4.0], "Pressure B": [1.1, 2.1, 2.9, 4.2],
        "t": pl.datetime_range(pl.datetime(2024, 1, 1), pl.datetime(2024, 1, 1, 0, 0, 3), "1s", eager=True),
        "name": ["a", "b", "c", None], "flag": [True, False, True, False],
    })


def _f(formula: str, df: pl.DataFrame) -> list:
    e, _, _ = compile_formula(formula, dict(df.schema), {})
    return df.select(e.alias("r"))["r"].to_list()


def table(rows: list[list], columns: list[tuple[str, str]]) -> Pipeline:
    p = Pipeline("t")
    p.add_node("enter_data", params={"columns": [{"name": n, "type": t} for n, t in columns], "rows": rows}, id="d")
    return p


def after(p: Pipeline, type_key: str, params: dict, src: str = "d", nid: str = "s") -> pl.DataFrame:
    p.add_node(type_key, params=params, id=nid)
    p.connect(src, nid)
    return Executor(p).preview(nid)[0]


def calc(p: Pipeline, expr: str) -> list:
    if "c" in p.nodes:
        p.remove_node("c")
    return after(p, "calculate", {"formulas": [{"name": "r", "expr": expr}]}, nid="c")["r"].to_list()


@pytest.fixture
def fdf():
    return pl.DataFrame({"a": [1.0, 2.0, 3.0, None], "s": ["x", "y", "z", None], "Cutoff": [5.0, 6.0, 7.0, 8.0],
                         "t": [datetime(2024, 1, 1), datetime(2024, 6, 1), datetime(2025, 1, 1), None], "b": [True, False, True, None]})


def ev(df, src, inputs=None):
    e, kind, _ = compile_formula(src, df.schema, inputs)
    return df.select(e.alias("r"))["r"].to_list(), kind


@pytest.mark.parametrize("src,expected", [
    ("2^3^2", 64.0), ("-2^2", 4.0), ("2^-1", 0.5), ("2^-3^2", 1 / 64),     # ^ works left to right, as in Excel
    ("ROUND(2.4999999999, 0)", 2.0), ("ROUND(2.675, 2)", 2.68), ("ROUND(1.005, 2)", 1.01), ("ROUND(-2.5, 0)", -3.0),
    ("ROUND(1234.5, -2)", 1200.0), ("ROUND(0.285, 2)", 0.29),
    ("CONCAT(5.0)", "5"), ("CONCAT(NULL, \"x\")", "x"), ("CONCAT(2.5, \"-\", 3)", "2.5-3"),
    ("(0.1 + 0.2) & \"\"", "0.3"), ("1/3 & \"\"", "0.333333333333333"), ("1e20 & \"\"", "1e+20"), ("TEXT(0.1 + 0.2)", "0.3"),
    ("MOD(-3, 2)", 1), ("MOD(3, -2)", -1), ("MOD(5, 0)", None), ("5 % 0", None),
    ("REPLACE(\"abcdef\", 2, 3, \"X\")", "aXef"), ("REPLACE(\"abc\", 4, 0, \"d\")", "abcd"),
    ("SUBSTITUTE(\"a-b-c\", \"-\", \"+\")", "a+b+c"), ("SUBSTITUTE(\"a-b-c\", \"-\", \"+\", 2)", "a-b+c"),
    ("SUBSTITUTE(\"a-b-c\", \"-\", \"+\", 3)", "a-b-c"), ("SUBSTITUTE(\"abc\", \"\", \"+\")", "abc"),
    ("\"abc\" = \"ABC\"", True), ("\"abc\" <> \"ABC\"", False), ("\"a\" < \"B\"", True),
])
def test_excel_answers(src, expected):
    got = one(src)
    assert got == pytest.approx(expected) if isinstance(expected, float) else got == expected


def test_text_comparison_ignores_case_and_treats_blank_as_empty_text():
    assert evaluate('s = "abc"') == [True, True, False, False]
    assert evaluate('s <> "a"') == [True, True, True, True]                 # a blank cell differs from "a"
    assert evaluate('s = ""') == [False, False, True, False]                # and equals empty text
    assert evaluate("n <> 3") == [True, True, False, True]                  # a blank number differs from 3 too


def test_mod_and_remainder_agree():
    df = pl.DataFrame({"a": [7.0, -7.0, 5.0], "b": [2.0, 2.0, 0.0]})
    assert evaluate("MOD(a, b)", df) == evaluate("a % b", df) == [1.0, 1.0, None]


def test_weekday_counts_from_sunday_unless_told_otherwise():
    # 2 June 2024 is a Sunday, 3 June a Monday, 8 June a Saturday
    assert evaluate("WEEKDAY(d)") == [1, 2, 7, None]
    assert evaluate("WEEKDAY(d, 2)") == [7, 1, 6, None]
    assert evaluate("WEEKDAY(d, 3)") == [6, 0, 5, None]
    assert evaluate("WEEKDAY(d, 11)") == [7, 1, 6, None] and evaluate("WEEKDAY(d, 17)") == [1, 2, 7, None]
    with pytest.raises(FormulaError, match="return type"):
        evaluate("WEEKDAY(d, 4)")


def test_replace_and_substitute_are_excels():
    assert FUNCTIONS["REPLACE"][:2] == (4, 4)
    with pytest.raises(FormulaError, match="takes 4"):
        evaluate('REPLACE(s, "a", "b")')
    with pytest.raises(FormulaError, match="start must be 1"):
        evaluate('REPLACE(s, 0, 1, "b")')
    assert evaluate('SUBSTITUTE(s, "b", "_")') == ["a_c", "ABC", None, "x"]    # SUBSTITUTE matches case, as Excel's


def test_rolling_formulas_centre_the_window_like_the_smooth_step():
    df = pl.DataFrame({"v": [1.0, 2.0, 3.0, 4.0, 5.0]})
    assert evaluate("ROLLING_MEAN(v, 3)", df) == [1.5, 2.0, 3.0, 4.0, 4.5]
    assert evaluate("ROLLING_MEAN(v, 3, FALSE)", df) == [1.0, 1.5, 2.0, 3.0, 4.0]


@pytest.mark.parametrize("src,expected", [
    ("[Pressure B] - [Pressure A]", [0.1, 0.1, -0.1, 0.2]),
    ("`Pressure A` * 2 + 1", [3, 5, 7, 9]),
    ("PressureA ^ 2", [1, 4, 9, 16]),                 # forgiving name match
    ("pressure_a + 0", [1, 2, 3, 4]),
    ("-[Pressure A]", [-1, -2, -3, -4]),
    ("ROUND(SQRT([Pressure A]) * 10, 1)", [10.0, 14.1, 17.3, 20.0]),
    ("MIN([Pressure A], [Pressure B])", [1.0, 2.0, 2.9, 4.0]),
    ("AVERAGE([Pressure A], [Pressure B])", [1.05, 2.05, 2.95, 4.1]),
    ("IF([Pressure A] > 2, 1, 0)", [0, 0, 1, 1]),
    ("IF(name = \"a\" or name = \"c\", \"yes\", \"no\")", ["yes", "no", "yes", "no"]),
    ("not flag", [False, True, False, True]),
    ("ELAPSED(t, \"s\")", [0.0, 1.0, 2.0, 3.0]),
    ("SECOND(t)", [0, 1, 2, 3]),
    ("DIFF([Pressure A])", [None, 1.0, 1.0, 1.0]),
    ("LAG([Pressure A])", [None, 1.0, 2.0, 3.0]),
    ("ROLLING_MEAN([Pressure A], 2, FALSE)", [1.0, 1.5, 2.5, 3.5]),
    ("CUMSUM([Pressure A])", [1.0, 3.0, 6.0, 10.0]),
    ("UPPER(name) & \"!\"", ["A!", "B!", "C!", "!"]),
    ("LEN(name)", [1, 1, 1, None]),
    ("ISBLANK(name)", [False, False, False, True]),
    ("COALESCE(name, \"?\")", ["a", "b", "c", "?"]),
    ("10 % 3", [1] * 4),
    ("t > \"2024-01-01 00:00:01\"", [False, False, True, True]),
    ("MOD([Pressure A], 2) = 0", [False, True, False, True]),
    ("ROW()", [1, 2, 3, 4]),
    ("[Pressure A] / 0", [None] * 4),                 # no answer, as Excel's #DIV/0!: a blank, not infinity
])
def test_values(df, src, expected):
    got, _ = ev(df, src)
    for g, e in zip(got, expected):
        if isinstance(e, float) and g is not None:
            assert g == pytest.approx(e, rel=1e-6), src
        else:
            assert g == e, src


def test_column_aggregates_broadcast(df):
    out = df.with_columns(compile_formula("[Pressure A] - AVERAGE([Pressure A])", df.schema)[0].alias("r"))
    assert out["r"].to_list() == [-1.5, -0.5, 0.5, 1.5]
    out = df.with_columns(compile_formula("MEDIAN([Pressure B])", df.schema)[0].alias("m"))
    assert out["m"][0] == pytest.approx(2.5)
    out = df.with_columns(compile_formula("STDEV([Pressure A])", df.schema)[0].alias("s"))
    assert out["s"][0] == pytest.approx(1.29099, rel=1e-4)


def test_kinds(df):
    assert ev(df, "[Pressure A] > 1")[1] == "true/false"
    assert ev(df, "name & \"x\"")[1] == "text"
    assert ev(df, "t")[1] == "date/time"
    assert ev(df, "t - t")[1] == "duration"


@pytest.mark.parametrize("src,fragment", [
    ("Presure * 2", "no column or input called 'Presure'"),
    ("FOO(1)", "Unknown function FOO"),
    ("ROUND(1, 2, 3)", "takes 1 to 2"),
    ("1 +", "ends too early"),
    ("IF(1, 2)", "takes 3"),
    ("[Pressure A] + name", "arithmetic on a text"),
    ("(1 + 2", "Expected ')'"),
    ("1 2", "Unexpected"),
    ("", "empty"),
    ("YEAR(name)", "date/time column"),
])
def test_errors(df, src, fragment):
    msg = check_formula(src, df.schema)
    assert msg is not None and fragment in msg, (src, msg)


def test_no_python_eval_possible(df):
    for evil in ["__import__('os')", "os.system('x')", "exec('1')", "eval('1')"]:
        assert check_formula(evil, df.schema) is not None


def test_zscore_constant_column_is_null_not_inf(tmp_path):
    p = Pipeline("z")
    p.add_node("enter_data", id="e", params={
        "columns": [{"name": "x", "type": "number"}], "rows": [[5.0], [5.0], [5.0]]})
    p.add_node("calculate", id="c", params={"formulas": [{"name": "z", "expr": "ZSCORE(x)"}]})
    p.connect("e", "c")
    p.path = tmp_path / "z.json"
    res = Executor(p).run()
    assert res["c"].status == "done", res["c"].error
    out = pl.read_parquet(res["c"].output)
    assert out["z"].null_count() == out.height == 3
    assert out["z"].is_nan().sum() == 0


def test_round_is_excel_round():
    df = pl.DataFrame({"x": [2.5, -2.5, 1.005, 1234.5, 0.125]})
    assert _f("ROUND(x)", df) == [3, -3, 1, 1235, 0]
    assert _f("ROUND(x, 2)", df) == [2.5, -2.5, 1.01, 1234.5, 0.13]
    assert _f("ROUND(x, -2)", df) == [0, 0, 0, 1200, 0]


def test_percentile_and_quartiles_interpolate():
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    assert _f("PERCENTILE(x, 0.5)", df)[0] == 2.5
    assert _f("PERCENTILE(x, 25)", df)[0] == 1.75
    from dancr.views.stats import exact_quantiles
    assert exact_quantiles(df.lazy(), pl.col("x")) == [1.75, 2.5, 3.25]


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


def test_formulas_behave_like_excel():
    p = table([["01/02/2024", 1, 0], ["12/25/2024", 2, 2], ["2024-03-04 10:00", 4, 4], [None, 3, 1]],
              [("s", "text"), ("a", "number"), ("b", "number")])
    assert calc(p, "DATE([s])") == [datetime(2024, 1, 2), datetime(2024, 12, 25), datetime(2024, 3, 4, 10), None]
    assert calc(p, "[a] / [b]") == [None, 1.0, 1.0, 3.0]            # #DIV/0! is a blank, not infinity
    assert calc(p, "RANK([a])") == [4, 3, 1, 2]                      # the largest is 1
    assert calc(p, "RANK([a], 1)") == [1, 2, 4, 3]


def test_negative_text_lengths_are_refused_plainly():
    p = table([["abc"]], [("s", "text")])
    with pytest.raises(Exception, match="cannot be negative"):
        calc(p, "LEFT([s], -1)")


def test_date_with_a_format_on_a_time_column_is_rejected():
    from dancr.core.expr import check_formula

    msg = check_formula('DATE(t, "%Y-%m-%d")', {"t": pl.Datetime("us")})
    assert msg is not None and "already" in msg
    assert check_formula('DATE("01/02/2024", "%d/%m/%Y")', {}) is None


def test_formula_compares_offset_literal_with_zoned_column():
    df = pl.DataFrame({"t": [datetime(2024, 6, 1, 11), datetime(2024, 6, 1, 13)]}).with_columns(pl.col("t").dt.replace_time_zone("Europe/Oslo"))
    e, _, _ = compile_formula('t > "2024-06-01T12:00:00+02:00"', df.schema)
    assert df.select(e)["t"].to_list() == [False, True]


@pytest.mark.parametrize("src,expected", [
    ("-2^2", [4.0] * 4), ("2^-1", [0.5] * 4), ("2^3^2", [64.0] * 4), ("-a^2", [1.0, 4.0, 9.0, None]), ("3 - -2", [5] * 4), ("2*-a", [-2.0, -4.0, -6.0, None]),
    ("LEAD(a)", [2.0, 3.0, None, None]), ("PCT_CHANGE(a)", [None, 100.0, 50.0, None]), ("ROLLING_SUM(a, 2, FALSE)", [1.0, 3.0, 5.0, 3.0]),
    ("ROLLING_MAX(a, 2, FALSE)", [1.0, 2.0, 3.0, 3.0]), ("ROLLING_MIN(a, 3, FALSE)", [1.0, 1.0, 1.0, 2.0]), ("CUMMAX(a)", [1.0, 2.0, 3.0, None]), ("CUMMIN(a)", [1.0, 1.0, 1.0, None]),
    ("RANK(a)", [3, 2, 1, None]), ("RANK(a, 1)", [1, 2, 3, None]), ("PERCENTILE(a, 50)", [2.0] * 4), ("CLIP(a, 1.5, 2.5)", [1.5, 2.0, 2.5, None]),
    ("LEFT(s, 1) & RIGHT(s, 1)", ["xx", "yy", "zz", ""]), ("MID(\"hello\", 2, 3)", ["ell"] * 4), ("SUBSTITUTE(s, \"x\", \"q\")", ["q", "y", "z", None]),
    ("CONTAINS(s, \"y\")", [False, True, False, None]), ("STARTSWITH(s, \"z\")", [False, False, True, None]), ("ENDSWITH(s, \"x\")", [True, False, False, None]),
    ("TEXT(t, \"%Y\")", ["2024", "2024", "2025", None]), ("VALUE(\"1,200\") + a", [1201.0, 1202.0, 1203.0, None]),
    ("YEAR(DATE(\"01/02/2024\", \"%d/%m/%Y\"))", [2024] * 4), ("MONTH(DATE(\"01/02/2024\", \"%d/%m/%Y\"))", [2] * 4),
    ("WEEKDAY(t)", [2, 7, 4, None]), ("WEEKDAY(t, 2)", [1, 6, 3, None]), ("DAYOFYEAR(t)", [1, 153, 1, None]), ("FILL_FORWARD(a)", [1.0, 2.0, 3.0, 3.0]),
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


def test_uint64_arithmetic_stays_exact():
    df = pl.DataFrame({"u": pl.Series([2 ** 60 + 1, 2 ** 60 + 3, 2 ** 64 - 1], dtype=pl.UInt64)})
    e, _, _ = compile_formula("u + 1", dict(df.schema), {})
    assert df.select(e.alias("r"))["r"].to_list() == [2 ** 60 + 2, 2 ** 60 + 4, 2 ** 64]
    e, _, _ = compile_formula("-u", dict(df.schema), {})
    assert df.select(e.alias("r"))["r"][2] == -(2 ** 64 - 1)


def test_formula_unicode_escape_and_recursion():
    df = pl.DataFrame({"Größe": [1.0], "s": ["a"]})
    e, k, _ = compile_formula("Größe * 2", df.schema)
    assert df.select(e)[0, 0] == 2.0
    e, k, _ = compile_formula('"Größe\\n"', df.schema)            # as in Excel, a backslash is just a character
    assert df.select(e)[0, 0] == "Größe\\n"
    assert "too long" in check_formula("(" * 400 + "1" + ")" * 400, df.schema)
    assert "too long" in check_formula(" + ".join(["Größe"] * 3000), df.schema)


def test_formula_time_vs_number_rejected_and_tz_literals():
    df = pl.DataFrame({"t": pl.datetime_range(datetime(2024, 1, 1), datetime(2024, 1, 3), "1d", eager=True)})
    assert "plain number" in check_formula("t > 20240102", df.schema)
    tz = df.with_columns(pl.col("t").dt.replace_time_zone("UTC"))
    e, _, _ = compile_formula('t > "2024-01-02"', tz.schema)
    assert tz.select(e)["t"].to_list() == [False, False, True]
    e, _, _ = compile_formula("SECONDS_BETWEEN(t, t)", tz.schema)
    assert tz.select(e)[0, 0] == 0.0
    d = df.with_columns(pl.col("t").cast(pl.Date))
    e, _, _ = compile_formula('t >= "2024-01-02"', d.schema)
    assert d.select(e)["t"].to_list() == [False, True, True]


def test_formula_ambiguous_squash_and_text_math():
    df = pl.DataFrame({"a b": [1.0], "a_b": [2.0], "name": ["x"]})
    assert "could mean" in check_formula("a__b * 2", df.schema)
    assert "number" in check_formula("ABS(name)", df.schema)
    assert "negate" in check_formula("-name", df.schema)


def test_replace_from_columns_is_blank_where_excel_gives_value_error():
    df = pl.DataFrame({"s": ["x-y-z"] * 5, "st": [1.0, 0.0, -1.0, 3.0, 2.9], "n": [1, 1, 1, -1, 2]})
    assert _f('REPLACE([s], [st], [n], "Z")', df) == ["Z-y-z", None, None, None, "xZ-z"]


def test_text_and_round_keep_extreme_numbers_exact():
    df = pl.DataFrame({"x": [1e-300, 1e300, 0.1, -0.0, float("inf")]})
    assert _f('[x] & ""', df) == ["1e-300", "1e+300", "0.1", "0", ""]
    assert _f("ROUND([x], 2)", df)[:4] == [0.0, 1e300, 0.1, 0.0]
    assert str(_f("ROUND([x] * -1, 0)", pl.DataFrame({"x": [0.2]}))[0]) == "0.0"      # no -0


@pytest.mark.parametrize("src", ["s / 2", "2 / s", "d / 2", "s ^ 2", "2 ^ d", "(n > 1) / 2", "2 ^ (n > 1)"])
def test_divide_and_power_refuse_text_dates_and_true_false_like_the_other_arithmetic(src):
    with pytest.raises(FormulaError, match="Cannot do arithmetic"):
        compile_formula(src, DF.schema)
    assert check_formula(src, DF.schema) is not None
    assert evaluate("n / 2") == [0.5, None, 1.5, 2.0] and evaluate("n ^ 2") == [1.0, None, 9.0, 16.0]


# ---------------------------------------------------------------- durations, COUNT, POW, ROUND
DD = pl.DataFrame({"t1": [datetime(2024, 1, 1)], "t2": [datetime(2024, 1, 2)]}).with_columns(
    pl.col("t1").cast(pl.Datetime("us")), pl.col("t2").cast(pl.Datetime("us")))


def test_duration_compares_with_a_text_span():
    assert evaluate('(t2 - t1) > "1d"', DD) == [False]
    assert evaluate('(t2 - t1) >= "1d"', DD) == [True]
    assert evaluate('(t2 - t1) < "12h"', DD) == [False]


def test_duration_compared_with_a_plain_number_is_refused_not_a_polars_error():
    msg = check_formula("(t2 - t1) > 5", DD.schema)
    assert msg is not None and "duration" in msg.lower()


def test_duration_divided_by_a_number_stays_a_duration():
    assert evaluate("(t2 - t1) / 2", DD) == [timedelta(hours=12)]


def test_duration_divided_by_a_duration_is_a_plain_ratio():
    assert evaluate("(t2 - t1) / (t2 - t1)", DD) == [1.0]


def test_pow_and_caret_agree_when_there_is_no_answer():
    assert evaluate("POW(-1, 0.5)") == [None]
    assert evaluate("(-1)^0.5") == [None]


def test_count_counts_only_non_empty_values():
    df = pl.DataFrame({"v": [1.0, float("nan"), None, 3.0], "s": ["a", "", None, "  "]})
    assert evaluate("COUNT(v)", df) == [2]
    assert evaluate("COUNT(s)", df) == [1]
    assert evaluate("COUNT(v)", df) == evaluate("SUM(IF(ISBLANK(v), 0, 1))", df)


def test_round_with_extreme_digits_is_sane():
    assert evaluate("ROUND(1.5, 400)") == [1.5]      # no fractional place is left: the value is unchanged
    assert evaluate("ROUND(1.5, -1000)") == [0.0]    # rounds to a magnitude no double can hold
