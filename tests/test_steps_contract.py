"""Stewardship: a data contract (columns, rules, references) and a before/after diff."""
import json

import polars as pl

from dancr.core import Pipeline
from dancr.core.executor import Executor


def run(p: Pipeline, node_id: str):
    st = Executor(p).run(targets=[node_id])[node_id]
    assert st.status == "done", st.error
    return pl.read_parquet(st.output), st


def test_contract_reports_column_and_rule_issues(tmp_path):
    p = Pipeline("c"); p.path = tmp_path / "c.json"
    p.add_node("enter_data", params={
        "columns": [{"name": "plot_id", "type": "text"}, {"name": "yield", "type": "number"}],
        "rows": [["P1", 10.0], ["P2", None], ["P2", 300.0]]}, id="d")
    contract = {"columns": {"plot_id": {"unique": True}, "yield": {"kind": "number", "max": 200}},
                "rules": [{"name": "nonnegative", "expr": "yield >= 0"}]}
    p.add_node("check_contract", params={"contract": json.dumps(contract), "output": "issues"}, id="k")
    p.connect("d", "k", "in")
    df, st = run(p, "k")
    checks = set(df["check"].to_list())
    assert {"unique", "maximum"} <= checks
    assert st.report["errors"] >= 2 and st.report["rows"] == 3


def test_contract_unique_counts_a_null_as_missing(tmp_path):
    p = Pipeline("c"); p.path = tmp_path / "c.json"
    p.add_node("enter_data", params={"columns": [{"name": "k", "type": "text"}],
                                     "rows": [["a"], ["b"], ["b"], [None]]}, id="d")
    p.add_node("check_contract", params={"contract": json.dumps({"columns": {"k": {"unique": True}}}),
                                         "output": "issues"}, id="k")
    p.connect("d", "k", "in")
    df, _st = run(p, "k")
    assert "unique" in set(df["check"].to_list())          # b is repeated; the blank must not hide it


def test_contract_allowed_compares_numbers_as_numbers(tmp_path):
    p = Pipeline("c"); p.path = tmp_path / "c.json"
    p.add_node("enter_data", params={
        "columns": [{"name": "code", "type": "number"}],
        "rows": [[1.0], [2.0], [3.0]]}, id="d")
    contract = {"columns": {"code": {"allowed": [1, 2]}}}
    p.add_node("check_contract", params={"contract": json.dumps(contract), "output": "issues"}, id="k")
    p.connect("d", "k", "in")
    df, st = run(p, "k")
    issues = [r for r in df.iter_rows(named=True) if r["check"] == "allowed values"]
    assert len(issues) == 1 and issues[0]["failing_rows"] == 1   # only 3.0 is outside; 1.0 and 2.0 are allowed


def test_contract_infers_a_draft_when_none_is_given(tmp_path):
    p = Pipeline("c"); p.path = tmp_path / "c.json"
    p.add_node("enter_data", params={"columns": [{"name": "x", "type": "number"}],
                                     "rows": [[1], [2], [3]]}, id="d")
    p.add_node("check_contract", params={}, id="k"); p.connect("d", "k", "in")
    st = Executor(p).run(targets=["k"])["k"]
    assert st.status == "done"
    inferred = st.report["inferred_contract"]["columns"]["x"]
    assert inferred["kind"] == "number" and inferred["min"] == 1 and inferred["max"] == 3
    assert inferred["required"] is True


def test_contract_reference_integrity(tmp_path):
    p = Pipeline("c"); p.path = tmp_path / "c.json"
    p.add_node("enter_data", params={"columns": [{"name": "pid", "type": "text"}],
                                     "rows": [["P1"], ["P9"]]}, id="d")
    p.add_node("enter_data", params={"columns": [{"name": "pid", "type": "text"}],
                                     "rows": [["P1"], ["P2"]]}, id="ref")
    contract = {"references": [{"column": "pid", "to_node": "ref", "to_column": "pid"}]}
    p.add_node("check_contract", params={"contract": json.dumps(contract), "output": "issues"}, id="k")
    p.connect("d", "k", "in"); p.connect("ref", "k", "references")
    df, st = run(p, "k")
    assert st.report["errors"] == 1
    assert df.row(0, named=True)["check"] == "reference"


def test_diff_tables_reports_added_removed_changed(tmp_path):
    p = Pipeline("c"); p.path = tmp_path / "c.json"
    p.add_node("enter_data", params={"columns": [{"name": "id", "type": "text"}, {"name": "v", "type": "number"}],
                                     "rows": [["A", 1.0], ["B", 2.0], ["C", 3.0]]}, id="a")
    p.add_node("enter_data", params={"columns": [{"name": "id", "type": "text"}, {"name": "v", "type": "number"}],
                                     "rows": [["A", 1.0], ["B", 9.0], ["D", 4.0]]}, id="b")
    p.add_node("diff_tables", params={"key": ["id"]}, id="df")
    p.connect("a", "df", "a"); p.connect("b", "df", "b")
    df, st = run(p, "df")
    kinds = {}
    for r in df.iter_rows(named=True):
        kinds.setdefault(r["change_type"], []).append(r)
    assert st.report["changed"] == 1 and st.report["added"] == 1 and st.report["removed"] == 1
    changed = kinds["changed"][0]
    assert changed["column"] == "v" and changed["before"] == "2.0" and changed["after"] == "9.0"
    assert changed["delta"] == 7.0


def test_diff_tables_survives_an_after_column_that_looks_suffixed(tmp_path):
    # the after table already has a real "v_after": the join's own suffix must not collide with it
    p = Pipeline("c"); p.path = tmp_path / "c.json"
    p.add_node("enter_data", params={"columns": [{"name": "id", "type": "text"}, {"name": "v", "type": "number"}],
                                     "rows": [["A", 1.0]]}, id="a")
    p.add_node("enter_data", params={"columns": [{"name": "id", "type": "text"}, {"name": "v", "type": "number"},
                                                 {"name": "v_after", "type": "number"}],
                                     "rows": [["A", 9.0, 100.0]]}, id="b")
    p.add_node("diff_tables", params={"key": ["id"]}, id="df")
    p.connect("a", "df", "a"); p.connect("b", "df", "b")
    df, st = run(p, "df")
    assert st.report["changed"] == 1
    row = next(r for r in df.iter_rows(named=True) if r["column"] == "v")
    assert row["before"] == "1.0" and row["after"] == "9.0"
