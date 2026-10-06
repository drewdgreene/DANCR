"""Hybrid retrieval (C1): the deterministic default is unchanged, BM25 and fusion are opt-in and recorded."""
import json
import subprocess
import sys

import numpy as np
import pytest

from dancr.core import Pipeline
from dancr.core.executor import Executor
from dancr.core.rag import (EMBEDDER_ID, bm25_scores, embed, load_index_meta, rank, search_knowledge)

DOCS = ["drought tolerance improved in hybrid wheat",
        "nitrogen use efficiency and yield stability in maize",
        "drought stress reduced canopy temperature in 2025 trials"]


def make(tmp_path):
    p = Pipeline("rag")
    p.path = tmp_path / "rag.json"
    p.add_node("enter_data", params={"columns": [{"name": "doc", "type": "text"}, {"name": "text", "type": "text"}],
                                     "rows": [[f"t{i + 1}", d] for i, d in enumerate(DOCS)]}, id="d")
    p.add_node("build_index", params={"chunk_chars": 60, "index_path": "idx.parquet"}, id="idx")
    p.connect("d", "idx", "items")
    p.add_node("retrieve", params={"query": "drought", "k": 2}, id="r")
    p.connect("idx", "r")
    p.save()
    return p


def test_bm25_is_deterministic_and_finds_the_matching_passage():
    s1 = bm25_scores(DOCS, "drought")
    s2 = bm25_scores(DOCS, "drought")
    assert np.allclose(s1, s2)
    assert s1.argmax() in (0, 2) and s1[1] == 0        # the maize passage has no 'drought'


def test_default_lexical_ranking_is_byte_identical_to_the_old_computation():
    vectors = [embed(d).tolist() for d in DOCS]
    order, scores, prov = rank("drought", texts=DOCS, vectors=vectors, retriever="lexical", k=3)
    mat = np.asarray(vectors, dtype=np.float32)
    expected = mat @ embed("drought", mat.shape[1])
    assert np.allclose(scores, expected)
    assert prov["embedder"] == EMBEDDER_ID and prov["retrievers"] == ["lexical"]


def test_hybrid_scores_are_bounded_and_rank_the_match_first():
    vectors = [embed(d).tolist() for d in DOCS]
    order, scores, prov = rank("drought", texts=DOCS, vectors=vectors, retriever="hybrid", k=3)
    assert 0.0 <= float(scores.min()) and float(scores.max()) <= 1.0
    assert order[0] in (0, 2)
    assert prov["retrievers"] == ["lexical", "bm25"] and prov["fusion"]["method"] == "weighted_minmax"


def test_bm25_without_vectors_is_a_valid_fallback():
    order, scores, prov = rank("drought", texts=DOCS, vectors=None, retriever="lexical", k=3)
    assert prov["retrievers"] == ["bm25"] and prov["embedder"] is None
    assert order[0] in (0, 2)


def test_unknown_retriever_is_refused():
    with pytest.raises(ValueError, match="Unknown retriever"):
        rank("x", texts=DOCS, vectors=None, retriever="semantic")


def test_build_index_writes_an_embedder_sidecar(tmp_path):
    p = make(tmp_path)
    res = Executor(p).run()
    assert res["idx"].report["embedder"] == EMBEDDER_ID
    meta = load_index_meta(tmp_path / "idx.parquet")
    assert meta["embedder"] == EMBEDDER_ID and meta["dim"] == 256


def test_search_knowledge_reports_provenance(tmp_path):
    p = make(tmp_path)
    out = search_knowledge(p, "drought", k=2)                       # default lexical, unchanged behaviour
    assert out["count"] == 2 and out["provenance"]["retriever"] == "lexical"
    assert out["provenance"]["embedder"] == EMBEDDER_ID
    hy = search_knowledge(p, "drought", k=2, retriever="hybrid")
    assert hy["provenance"]["retrievers"] == ["lexical", "bm25"]
    # deterministic: the same query gives the same answer
    assert json.dumps(out) == json.dumps(search_knowledge(p, "drought", k=2))


def test_cli_search_retriever(tmp_path):
    make(tmp_path)
    r = subprocess.run([sys.executable, "-m", "dancr.cli", "search", "rag.json", "drought", "--k", "1",
                        "--retriever", "bm25"], capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert "drought" in r.stdout


def test_mcp_search_knowledge_retriever(tmp_path, mcp_root):
    pj = tmp_path / "p.json"
    import dancr.mcp_server as srv
    srv.create_pipeline(str(pj))
    srv.add_node(str(pj), "enter_data", {"columns": [{"name": "text", "type": "text"}],
                                         "rows": [["drought stress trials"], ["nitrogen management"]]}, node_id="d")
    srv.add_node(str(pj), "build_index", {}, node_id="idx", after="d", port="items")
    srv.run_pipeline(str(pj))
    out = json.loads(srv.search_knowledge(str(pj), "drought", k=1, retriever="hybrid"))
    assert out["count"] == 1 and out["provenance"]["retriever"] == "hybrid"
