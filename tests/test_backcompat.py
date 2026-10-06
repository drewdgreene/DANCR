"""Backward compatibility: old projects, old indexes and old attestations keep loading.

The roadmap's hard constraints require that an existing ``{"dancr": 2}`` file still
loads, an index built by an earlier DANCR still reads and merges, and an
attestation without the newer evidence block still verifies. These tests pin that.
"""
import json

import polars as pl

from dancr.core import Pipeline
from dancr.core.executor import Executor
from dancr.core.rag import load_index, merge_index
from dancr.headless import build_attestation, dump_attestation, verify_pipeline

# A minimal project exactly as an earlier DANCR wrote it: only the keys the format guarantees.
LEGACY_PROJECT = {
    "dancr": 2,
    "name": "legacy",
    "nodes": [{"id": "src", "type": "load_file", "title": "src", "x": 0, "y": 0,
               "params": {"path": "data.csv"}}],
    "edges": [], "notes": [], "answers": [], "inputs": [], "columns": {},
    "meta": {"future_thing": {"kept": True}},
}


def test_a_dancr2_project_still_loads_and_round_trips(tmp_path):
    (tmp_path / "data.csv").write_text("a,b\n1,2\n")
    (tmp_path / "legacy.json").write_text(json.dumps(LEGACY_PROJECT))
    p = Pipeline.load(tmp_path / "legacy.json")
    assert "src" in p.nodes and p.nodes["src"].type == "load_file"
    # unknown meta keys survive a load/save (the documented forward-compatibility rule)
    assert p.meta.get("future_thing") == {"kept": True}
    p.save()
    again = json.loads((tmp_path / "legacy.json").read_text())
    assert again["dancr"] == 2 and again["meta"]["future_thing"] == {"kept": True}


def test_an_index_without_the_dataset_column_still_loads_and_merges(tmp_path):
    """An index written before the incremental 'dataset' column existed must not crash a rebuild."""
    old = pl.DataFrame({
        "doc_key": ["k"], "content_hash": ["h"], "chunk_id": ["k::0"], "chunk_index": [0],
        "text": ["hello world"], "vector": [[0.0] * 256],
    })
    path = tmp_path / "idx.parquet"
    old.write_parquet(path)
    loaded = load_index(path)
    assert loaded is not None and loaded.height == 1
    out, changes = merge_index(loaded, [{"doc_key": "k", "content_hash": "h2", "text": "new text", "meta": {}}],
                               dim=256, chunk_chars=100, overlap=10)
    assert changes["changed"] == 1 and out.height >= 1


def _project(tmp_path):
    (tmp_path / "data").mkdir(exist_ok=True)
    pl.DataFrame({"region": ["N", "S", "N", "S"], "amount": [1.0, 2.0, 3.0, 4.0]}).write_csv(tmp_path / "data" / "m.csv")
    p = Pipeline("m")
    p.path = tmp_path / "m.json"
    p.add_node("load_file", params={"path": "data/m.csv"}, id="src")
    p.add_node("group_summary", params={"by": ["region"]}, id="g")
    p.connect("src", "g")
    p.save()
    return p


def test_an_old_attestation_without_extra_still_verifies(tmp_path):
    p = _project(tmp_path)
    ex = Executor(p)
    ex.run()
    att = build_attestation(p, ex)
    assert "extra" not in att
    (tmp_path / "old.json").write_text(dump_attestation(att))
    res = verify_pipeline(p, tmp_path / "old.json")
    assert res["verdict"] == "verified" and res["ok"] is True
    # a record with no 'extra' key is not compared against anything: no extra check at all
    assert not any(c["scope"] == "extra" for c in res["checks"])


def test_an_extra_evidence_block_round_trips_and_verifies(tmp_path):
    p = _project(tmp_path)
    ex = Executor(p)
    ex.run()
    evidence = {"scenarios": [{"id": "s1", "plan_hash": "abc", "output_hash": "def"}]}
    att = build_attestation(p, ex, extra=evidence)
    assert att["extra"] == evidence
    (tmp_path / "att.json").write_text(dump_attestation(att))

    # supplying the same evidence verifies
    assert verify_pipeline(p, tmp_path / "att.json", extra=evidence)["verdict"] == "verified"

    # a reference carrying evidence, verified without recomputing it: a notice, not a false mismatch
    res = verify_pipeline(p, tmp_path / "att.json")
    assert res["verdict"] == "verified" and any(n["scope"] == "extra" for n in res["notices"])

    # different evidence is a mismatch
    other = verify_pipeline(p, tmp_path / "att.json", extra={"scenarios": [{"id": "s2"}]})
    assert other["verdict"] == "mismatch"
    assert any(m["scope"] == "extra" for m in other["mismatches"])


def test_extra_is_folded_into_the_attestation_hash(tmp_path):
    p = _project(tmp_path)
    ex = Executor(p)
    ex.run()
    a = build_attestation(p, ex)
    b = build_attestation(p, ex, extra={"scenarios": [{"id": "s1"}]})
    assert a["attestation_hash"] != b["attestation_hash"]
