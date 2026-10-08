"""Shape and description steps: pivot (rows into columns), describe (a data dictionary), and fuzzy key matching."""
import polars as pl

from dancr.core import Pipeline
from dancr.core.executor import Executor


def run(p: Pipeline, node_id: str):
    st = Executor(p).run(targets=[node_id])[node_id]
    assert st.status == "done", st.error
    return pl.read_parquet(st.output), st


def data(p, cols, rows, nid="d"):
    p.add_node("enter_data", params={"columns": [{"name": n, "type": t} for n, t in cols], "rows": rows}, id=nid)
    return nid


def test_pivot_spreads_values_into_columns(tmp_path):
    p = Pipeline("p"); p.path = tmp_path / "p.json"
    src = data(p, [("site", "text"), ("month", "text"), ("v", "number")],
               [["A", "Jan", 1], ["A", "Feb", 2], ["B", "Jan", 3]])
    p.add_node("pivot", params={"index": ["site"], "columns": "month", "values": "v", "agg": "sum"}, id="piv")
    p.connect(src, "piv")
    df, st = run(p, "piv")
    assert "Jan" in df.columns and "Feb" in df.columns and "v" not in df.columns
    a = df.filter(pl.col("site") == "A").row(0, named=True)
    b = df.filter(pl.col("site") == "B").row(0, named=True)
    assert a["Jan"] == 1 and a["Feb"] == 2
    assert b["Jan"] == 3 and b["Feb"] == 0            # totals fill blanks with 0
    assert st.report["new_columns"] == 2


def test_pivot_counts_rows_when_no_value_column(tmp_path):
    p = Pipeline("p"); p.path = tmp_path / "p.json"
    src = data(p, [("site", "text"), ("day", "text")],
               [["A", "Mon"], ["A", "Mon"], ["A", "Tue"], ["B", "Mon"]])
    p.add_node("pivot", params={"index": ["site"], "columns": "day"}, id="piv")
    p.connect(src, "piv")
    df, _ = run(p, "piv")
    assert df.filter(pl.col("site") == "A").row(0, named=True)["Mon"] == 2


def test_describe_builds_a_data_dictionary(tmp_path):
    p = Pipeline("p"); p.path = tmp_path / "p.json"
    p.set_column_meta("yield", unit="q/ha")
    src = data(p, [("plot", "text"), ("yield", "number")], [["P1", 10.0], ["P2", None], ["P3", 30.0]])
    p.add_node("describe_dataset", params={"definitions": {"yield": "grain yield"}}, id="dsc")
    p.connect(src, "dsc")
    df, st = run(p, "dsc")
    for c in ("column", "role", "type", "unit", "definition", "missing", "distinct", "min", "max", "mean", "example"):
        assert c in df.columns, c
    y = df.filter(pl.col("column") == "yield").row(0, named=True)
    assert y["role"] == "number" and y["unit"] == "q/ha" and y["definition"] == "grain yield"
    assert y["missing"] == 1 and y["mean"] == 20.0
    assert y["distinct"] == 2                    # the blank is missing, not a distinct value
    assert st.report["columns"] == 2 and st.report["with_blanks"] == 1


def test_fuzzy_combine_tidies_and_matches(tmp_path):
    p = Pipeline("p"); p.path = tmp_path / "p.json"
    left = data(p, [("name", "text"), ("n", "number")], [["Johnston, IA", 1], ["Ames  IA", 2], ["Nowhere", 3]], "l")
    right = data(p, [("site", "text"), ("code", "text")], [["Johnston IA", "J"], ["Ames IA", "A"]], "r")
    p.add_node("combine", params={"method": "fuzzy", "left_key": "name", "right_key": "site",
                                  "algorithm": "normalized", "how": "left"}, id="c")
    p.connect(left, "c", "left"); p.connect(right, "c", "right")
    df, st = run(p, "c")
    got = {r["name"]: r["code"] for r in df.iter_rows(named=True)}
    assert got["Johnston, IA"] == "J" and got["Ames  IA"] == "A" and got["Nowhere"] is None
    assert abs(st.report["match_percent"] - 100 * 2 / 3) < 1e-9


def test_fuzzy_nearest_refuses_too_many_keys_on_either_side(tmp_path):
    p = Pipeline("p"); p.path = tmp_path / "p.json"
    left = data(p, [("name", "text")], [["a"], ["b"], ["c"]], "l")
    right = data(p, [("site", "text")], [["a"], ["b"]], "r")
    p.add_node("combine", params={"method": "fuzzy", "left_key": "name", "right_key": "site",
                                  "algorithm": "nearest", "max_candidates": 2}, id="c")
    p.connect(left, "c", "left"); p.connect(right, "c", "right")
    st = Executor(p).run(targets=["c"])["c"]
    assert st.status == "failed" and "first table has" in (st.error or "").lower()


def test_fuzzy_combine_nearest_finds_a_close_key(tmp_path):
    p = Pipeline("p"); p.path = tmp_path / "p.json"
    left = data(p, [("name", "text")], [["DesMoines"]], "l")
    right = data(p, [("site", "text"), ("code", "text")], [["Des Moines", "D"]], "r")
    p.add_node("combine", params={"method": "fuzzy", "left_key": "name", "right_key": "site",
                                  "algorithm": "nearest", "threshold": 0.8}, id="c")
    p.connect(left, "c", "left"); p.connect(right, "c", "right")
    df, st = run(p, "c")
    row = df.row(0, named=True)
    assert row["code"] == "D" and row["match_score"] > 0.8
