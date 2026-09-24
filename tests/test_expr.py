import polars as pl
import pytest

from dancr.core.expr import compile_formula, check_formula


@pytest.fixture
def df():
    return pl.DataFrame({
        "Pressure A": [1.0, 2.0, 3.0, 4.0], "Pressure B": [1.1, 2.1, 2.9, 4.2],
        "t": pl.datetime_range(pl.datetime(2024, 1, 1), pl.datetime(2024, 1, 1, 0, 0, 3), "1s", eager=True),
        "name": ["a", "b", "c", None], "flag": [True, False, True, False],
    })


def ev(df, src):
    e, kind, used = compile_formula(src, df.schema)
    return df.select(e.alias("r"))["r"].to_list(), kind


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
    ("ROLLING_MEAN([Pressure A], 2)", [1.0, 1.5, 2.5, 3.5]),
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
