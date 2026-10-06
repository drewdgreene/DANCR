"""Governance: sensitivity labels, redaction, and retrieval that withholds restricted passages."""
import polars as pl

from dancr.core import Pipeline
from dancr.core.executor import Executor
from dancr.core.rag import search_knowledge


def run(p: Pipeline, node_id: str):
    st = Executor(p).run(targets=[node_id])[node_id]
    assert st.status == "done", st.error
    return pl.read_parquet(st.output), st


def test_label_constant_and_from_a_column(tmp_path):
    p = Pipeline("g"); p.path = tmp_path / "g.json"
    p.add_node("enter_data", params={"columns": [{"name": "level", "type": "text"}],
                                     "rows": [["public"], ["restricted"]]}, id="d")
    p.add_node("label_sensitivity", params={"level": "internal"}, id="const"); p.connect("d", "const", "in")
    p.add_node("label_sensitivity", params={"column": "level"}, id="fromcol"); p.connect("d", "fromcol", "in")
    out1, st1 = run(p, "const")
    out2, _ = run(p, "fromcol")
    assert out1["sensitivity"].to_list() == ["internal", "internal"]
    assert out2["sensitivity"].to_list() == ["public", "restricted"]
    assert st1.report["level"] == "internal"


def test_redact_mask_and_hash(tmp_path):
    p = Pipeline("g"); p.path = tmp_path / "g.json"
    p.add_node("enter_data", params={"columns": [{"name": "name", "type": "text"}, {"name": "n", "type": "number"}],
                                     "rows": [["Alice", 1], ["Bob", 2]]}, id="d")
    p.add_node("redact", params={"columns": ["name"], "method": "mask", "keep": 2}, id="mask"); p.connect("d", "mask", "in")
    p.add_node("redact", params={"columns": ["name"], "method": "hash"}, id="hash"); p.connect("d", "hash", "in")
    p.add_node("redact", params={"columns": ["name"], "method": "drop"}, id="drop"); p.connect("d", "drop", "in")
    m, _ = run(p, "mask")
    h, _ = run(p, "hash")
    d, _ = run(p, "drop")
    assert m["name"].to_list() == ["Al****", "Bo****"]
    assert h["name"][0] != "Alice" and len(h["name"][0]) == 12
    assert "name" not in d.columns and "n" in d.columns


def test_search_withholds_restricted_by_default(tmp_path):
    p = Pipeline("g"); p.path = tmp_path / "g.json"
    p.add_node("enter_data", params={
        "columns": [{"name": "text", "type": "text"}, {"name": "sensitivity", "type": "text"}],
        "rows": [["drought summary for farmers", "public"],
                 ["drought yield forecast internal", "confidential"],
                 ["drought patent draft", "restricted"]]}, id="d")
    p.add_node("build_index", params={}, id="idx"); p.connect("d", "idx", "items")
    Executor(p).run(targets=["idx"])
    governed = search_knowledge(p, "drought", k=5)
    assert governed["count"] == 1 and governed["withheld"] == 2
    assert all(h["sensitivity"] == "public" for h in governed["hits"])
    open_search = search_knowledge(p, "drought", k=5, allow_restricted=True)
    assert open_search["count"] == 3 and open_search["withheld"] == 0
