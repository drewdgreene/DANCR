"""Retrieval: an offline index over a project's own text, and search over it."""
import json
import subprocess
import sys

import numpy as np
import polars as pl
import pytest

from dancr.core import Pipeline
from dancr.core.executor import Executor
from dancr.core.rag import chunks, embed, search_knowledge


def make(tmp_path):
    p = Pipeline("rag"); p.path = tmp_path / "rag.json"
    p.add_node("enter_data", params={
        "columns": [{"name": "doc", "type": "text"}, {"name": "text", "type": "text"}],
        "rows": [["t1", "drought tolerance improved in hybrid wheat"],
                 ["t2", "nitrogen use efficiency and yield stability in maize"],
                 ["t3", "drought stress reduced canopy temperature in 2025 trials"]]}, id="d")
    p.add_node("build_index", params={"chunk_chars": 60}, id="idx")
    p.connect("d", "idx", "items")
    p.add_node("retrieve", params={"query": "drought", "k": 2}, id="r")
    p.connect("idx", "r")
    p.save()
    return p


def test_embedding_is_deterministic_and_ranks_by_overlap():
    a = embed("drought stress field trials")
    b = embed("drought stress field trials")
    c = embed("nitrogen use efficiency")
    assert np.allclose(a, b)
    assert float(a @ b) > float(a @ c)


def test_chunks_overlap_and_drop_blanks():
    assert chunks("", 10, 2) == []
    pieces = chunks("abcdefghij", 4, 2)
    assert pieces[0] == "abcd" and pieces[1] == "cdef" and pieces[-1].endswith("j")


def test_build_index_and_retrieve(tmp_path):
    p = make(tmp_path)
    res = Executor(p).run()
    assert res["idx"].status == "done" and res["r"].status == "done", res["idx"].error or res["r"].error
    df = pl.read_parquet(res["r"].output)
    assert list(df.columns[:2]) == ["rank", "score"]
    assert df.height == 2
    assert df.row(0, named=True)["doc"] in ("t1", "t3")
    assert res["r"].report["hits"] == 2


def test_search_knowledge_headless(tmp_path):
    p = make(tmp_path)
    out = search_knowledge(p, "drought", k=2)
    assert out["kind"] == "dancr.search" and out["count"] == 2 and out["top_score"] > 0
    assert all("text" in h and "score" in h for h in out["hits"])


def test_search_needs_an_index(tmp_path):
    p = Pipeline("p"); p.path = tmp_path / "p.json"
    p.add_node("enter_data", params={"columns": [{"name": "x", "type": "text"}], "rows": [["a"]]}, id="d")
    with pytest.raises(ValueError, match="no search index"):
        search_knowledge(p, "x")


def test_mcp_search_knowledge(tmp_path, mcp_root):
    import dancr.mcp_server as srv
    pj = tmp_path / "p.json"
    srv.create_pipeline(str(pj))
    srv.add_node(str(pj), "enter_data", {"columns": [{"name": "text", "type": "text"}],
                                         "rows": [["drought stress trials"], ["nitrogen management"]]}, node_id="d")
    srv.add_node(str(pj), "build_index", {}, node_id="idx", after="d", port="items")
    srv.run_pipeline(str(pj))
    out = json.loads(srv.search_knowledge(str(pj), "drought", k=1))
    assert out["count"] == 1 and out["hits"][0]["rank"] == 1


def test_incremental_index_reuses_unchanged_documents(tmp_path):
    data = tmp_path / "d.csv"

    def write(rows):
        pl.DataFrame({"id": [r[0] for r in rows], "text": [r[1] for r in rows]}).write_csv(data)

    write([["a", "drought tolerance in hybrid wheat"], ["b", "nitrogen use efficiency"], ["c", "drought stress trials"]])
    p = Pipeline("i"); p.path = tmp_path / "i.json"
    p.add_node("load_file", params={"path": "d.csv"}, id="d")
    p.add_node("build_index", params={"id_column": "id", "index_path": "idx.parquet", "chunk_chars": 60}, id="idx")
    p.connect("d", "idx", "items")
    p.save()

    r1 = Executor(p).run()["idx"]
    assert r1.report["index"]["added"] == 3 and r1.report["index"]["embedded_chunks"] == 3
    assert (tmp_path / "idx.parquet").is_file()

    # change one document, add one, remove one, leave one unchanged
    write([["a", "drought tolerance improved in hybrid wheat"], ["c", "drought stress trials"], ["d", "new maize line"]])
    r2 = Executor(p).run(force=True)["idx"]
    ch = r2.report["index"]
    assert ch["added"] == 1 and ch["changed"] == 1 and ch["unchanged"] == 1 and ch["removed"] == 1
    assert ch["reused_chunks"] == 1 and ch["embedded_chunks"] == 2
    assert pl.read_parquet(tmp_path / "idx.parquet").height == 3


def make_two(tmp_path):
    p = Pipeline("f"); p.path = tmp_path / "f.json"
    p.add_node("enter_data", params={"columns": [{"name": "text", "type": "text"}],
                                     "rows": [["alpha drought wheat"], ["alpha2 maize"]]}, id="a")
    p.add_node("enter_data", params={"columns": [{"name": "text", "type": "text"}],
                                     "rows": [["beta nitrogen"]]}, id="b")
    p.add_node("build_index", params={"index_path": "idx.parquet"}, id="idx")
    p.connect("a", "idx", "items"); p.connect("b", "idx", "items")
    p.save()
    return p


def test_build_index_skips_datasets_from_a_changed_feed(tmp_path):
    p = make_two(tmp_path)
    r1 = Executor(p).run()["idx"]
    assert r1.report["index"]["added"] == 3 and r1.report["skipped_datasets"] == []

    # a --changed feed lists only b; a is unchanged and must be skipped (carried from the index, not read)
    (tmp_path / "feed.json").write_text(json.dumps(
        {"changes": {"changed": ["b"]}, "documents": [{"node": "b", "content_hash": "x"}]}))
    p.set_params("idx", changed_path="feed.json"); p.save()
    r2 = Executor(p).run(force=True)["idx"]
    assert r2.report["skipped_datasets"] == ["a"]
    assert r2.report["index"]["carried_chunks"] == 2      # a's two passages carried over
    assert r2.report["index"]["embedded_chunks"] == 0     # b is unchanged, so reused
    out = search_knowledge(p, "alpha drought")
    assert any(h["chunk_id"].startswith("a:") for h in out["hits"])


def test_build_index_skips_from_a_full_context_when_hashes_match(tmp_path):
    p = make_two(tmp_path)
    Executor(p).run()["idx"]
    ex = Executor(p)
    full = {"documents": [{"node": "a", "content_hash": ex.safe_hash("a")},
                          {"node": "b", "content_hash": ex.safe_hash("b")}]}
    (tmp_path / "full.json").write_text(json.dumps(full))
    p.set_params("idx", changed_path="full.json"); p.save()
    r2 = Executor(p).run(force=True)["idx"]
    assert r2.report["skipped_datasets"] == ["a", "b"]
    assert r2.report["index"]["removed"] == 0 and r2.report["index"]["carried_chunks"] == 3
    assert r2.report["index"]["embedded_chunks"] == 0


def test_changed_feed_reads_jsonl_including_a_single_line(tmp_path):
    from dancr.core.rag import read_context_feed
    one = json.dumps({"node": "cites", "content_hash": "abc", "changes": {"changed": ["cites"]}})
    is_changed, listed = read_context_feed(one)            # a one-document JSON Lines export is one JSON line
    assert is_changed and listed == {"cites": "abc"}
    two = one + "\n" + json.dumps({"node": "dict", "content_hash": "d"})
    is_changed2, listed2 = read_context_feed(two)
    assert is_changed2 and listed2 == {"cites": "abc", "dict": "d"}


def test_build_index_skips_from_a_one_line_jsonl_changed_feed(tmp_path):
    p = make_two(tmp_path)
    Executor(p).run()["idx"]
    (tmp_path / "feed.jsonl").write_text(
        json.dumps({"node": "b", "content_hash": "x", "changes": {"changed": ["b"]}}) + "\n")
    p.set_params("idx", changed_path="feed.jsonl"); p.save()
    r2 = Executor(p).run(force=True)["idx"]
    assert r2.report["skipped_datasets"] == ["a"] and r2.report["index"]["carried_chunks"] == 2


def test_cli_search(tmp_path):
    make(tmp_path)                                        # saves rag.json next to the data
    r = subprocess.run([sys.executable, "-m", "dancr.cli", "search", "rag.json", "drought", "--k", "1"],
                       capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert "1 passage(s)" in r.stdout and "drought" in r.stdout
