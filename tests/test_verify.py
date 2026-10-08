"""Attestations and verification: a result reproduces, or it does not.

The record is deterministic; a changed source, a changed answer, a deleted file and a changed engine each
give the right verdict; and the MCP surface is confined the same way every other write tool is.
"""
import json
from pathlib import Path

import polars as pl
import pytest
from mcp.server.mcpserver.exceptions import ToolError

from dancr.core import Pipeline
from dancr.core.executor import Executor
from dancr.core.verify import output_hash
from dancr.headless import build_attestation, dump_attestation, verify_pipeline


def make_project(tmp_path):
    (tmp_path / "data").mkdir(exist_ok=True)
    src = tmp_path / "data" / "m.csv"
    pl.DataFrame({"region": ["N", "S", "N", "S"], "amount": [1.0, 2.0, 3.0, 4.0]}).write_csv(src)
    p = Pipeline("m")
    p.path = tmp_path / "m.json"
    p.add_node("load_file", params={"path": "data/m.csv"}, id="src")
    p.add_node("group_summary", params={"by": ["region"]}, id="g")
    p.connect("src", "g")
    p.add_node("report", params={"title": "R", "path": "r.html", "pdf": False}, id="rep")
    p.connect("g", "rep")
    p.save()
    return p, src


def record(p, path):
    att = build_attestation(p, Executor(p))
    Path(path).write_text(dump_attestation(att))
    return att


def test_record_then_verify_is_verified(tmp_path):
    p, _src = make_project(tmp_path)
    att = record(p, tmp_path / "m.attestation.json")
    assert att["kind"] == "dancr.attestation"
    # a stored result has a content hash; a sink (the report) is recorded but not bit-verified
    by_id = {n["id"]: n for n in att["nodes"]}
    assert by_id["g"]["output_hash"] and by_id["rep"]["materialize"] is False and by_id["rep"]["output_hash"] is None

    res = verify_pipeline(p, tmp_path / "m.attestation.json")
    assert res["ok"] is True and res["verdict"] == "verified" and res["mismatches"] == []


def test_a_changed_source_is_a_mismatch(tmp_path):
    p, src = make_project(tmp_path)
    record(p, tmp_path / "a.json")
    pl.DataFrame({"region": ["N", "S", "N", "S"], "amount": [1.0, 2.0, 3.0, 999.0]}).write_csv(src)
    res = verify_pipeline(p, tmp_path / "a.json", mode="rerun")
    assert res["ok"] is False and res["verdict"] == "mismatch"
    assert any(m["scope"] == "node" and m["name"] == "output_hash" for m in res["mismatches"])


def test_strict_sources_turns_a_source_change_into_a_mismatch(tmp_path):
    p, src = make_project(tmp_path)
    record(p, tmp_path / "a.json")
    pl.DataFrame({"region": ["N", "S", "N", "S"], "amount": [1.0, 2.0, 3.0, 5.0]}).write_csv(src)

    loose = verify_pipeline(p, tmp_path / "a.json", mode="rerun")
    assert any(n["scope"] == "source" for n in loose["notices"])

    strict = verify_pipeline(p, tmp_path / "a.json", mode="rerun", strict_sources=True)
    assert any(m["scope"] == "source" for m in strict["mismatches"])


def test_a_deleted_source_is_incomplete(tmp_path):
    p, src = make_project(tmp_path)
    record(p, tmp_path / "a.json")
    src.unlink()
    res = verify_pipeline(p, tmp_path / "a.json")
    assert res["verdict"] == "incomplete" and res["incomplete"]


def test_a_changed_engine_is_reported_not_confused_with_a_data_change(tmp_path, monkeypatch):
    p, _src = make_project(tmp_path)
    record(p, tmp_path / "a.json")
    monkeypatch.setattr("dancr.core.verify.CODE_FINGERPRINT", "0123456789ab")
    res = verify_pipeline(p, tmp_path / "a.json")
    assert res["verdict"] == "engine-changed"
    assert any(n["scope"] == "engine" for n in res["notices"])


def test_a_changed_answer_spec_is_a_mismatch(tmp_path):
    p, _src = make_project(tmp_path)
    p.add_answer("Q", "g", view="table", spec={"recipe": "breakdown", "table": "src", "by": ["src", "region"]})
    p.save()
    record(p, tmp_path / "a.json")
    p.answer("answer_1").spec = {"recipe": "top", "table": "src", "by": ["src", "region"]}
    p.save()
    res = verify_pipeline(p, tmp_path / "a.json")
    assert any(m["scope"] == "answer" for m in res["mismatches"])


def test_the_record_is_deterministic(tmp_path):
    p, _src = make_project(tmp_path)
    a1 = build_attestation(p, Executor(p))
    a2 = build_attestation(p, Executor(p))
    assert a1["attestation_hash"] == a2["attestation_hash"]
    assert a1["nodes"] == a2["nodes"]


def test_output_hash_is_order_independent(tmp_path):
    p1 = Pipeline("d1"); p1.path = tmp_path / "d1.json"
    p1.add_node("enter_data", params={"columns": [{"name": "a", "type": "number"}], "rows": [[1], [2], [3]]}, id="d")
    p2 = Pipeline("d2"); p2.path = tmp_path / "d2.json"
    p2.add_node("enter_data", params={"columns": [{"name": "a", "type": "number"}], "rows": [[3], [1], [2]]}, id="d")
    assert output_hash(Executor(p1), "d") == output_hash(Executor(p2), "d")


def test_verify_against_a_plain_run_manifest(tmp_path):
    """A run manifest has no per-report or output-hash fields; verifying against it must not read that as a
    mismatch -- it checks plan hashes, rows and sources."""
    import json
    from dancr.core.fair import run_manifest
    p, _src = make_project(tmp_path)
    ex = Executor(p)
    ex.run()
    (tmp_path / "man.json").write_text(json.dumps(run_manifest(p, ex)))
    res = verify_pipeline(p, tmp_path / "man.json", mode="rerun")
    assert res["verdict"] == "verified"


def test_a_crate_carries_a_verifiable_attestation(tmp_path):
    import zipfile
    from dancr.headless import package_rocrate
    p, _src = make_project(tmp_path)
    ex = Executor(p)
    ex.run()
    crate = tmp_path / "crate.zip"
    package_rocrate(p, ex, out=crate, copy="metadata", zip=True)
    with zipfile.ZipFile(crate) as z:
        assert "dancr-attestation.json" in z.namelist()
        att = json.loads(z.read("dancr-attestation.json"))
    assert att["kind"] == "dancr.attestation"
    assert verify_pipeline(p, att)["verdict"] == "verified"


def test_mcp_record_and_verify_roundtrip_is_confined(tmp_path, mcp_root):
    import dancr.mcp_server as srv
    (tmp_path / "data").mkdir()
    pl.DataFrame({"a": [1, 2, 3]}).write_csv(tmp_path / "data" / "s.csv")
    pj = tmp_path / "p.json"
    srv.create_pipeline(str(pj))
    srv.add_node(str(pj), "load_file", {"path": "data/s.csv"}, node_id="src")

    rec = json.loads(srv.record_attestation(str(pj), "att.json"))
    assert rec["ok"] and Path(rec["path"]).name == "att.json"

    ver = json.loads(srv.verify_pipeline(str(pj), "att.json"))
    assert ver["verdict"] == "verified"

    # a write outside the pipeline file's folder is refused, same as every other MCP write
    with pytest.raises(ToolError):
        srv.record_attestation(str(pj), str(tmp_path.parent / "evil.json"))


def test_a_changed_row_count_is_a_mismatch(tmp_path):
    # tamper only the recorded row count, leaving the output hash equal: this must not pass as "verified"
    p, _src = make_project(tmp_path)
    att = record(p, tmp_path / "a.json")
    for n in att["nodes"]:
        if n["id"] == "g":
            n["rows"] = 999
    Path(tmp_path / "a.json").write_text(dump_attestation(att))
    res = verify_pipeline(p, tmp_path / "a.json")
    assert res["verdict"] == "mismatch"
    assert any(m["name"] == "rows" for m in res["mismatches"])


def test_a_result_that_cannot_be_reread_is_not_verified(tmp_path, monkeypatch):
    # the attestation recorded an output hash; if this run cannot produce one, that is not a confirmation
    p, _src = make_project(tmp_path)
    record(p, tmp_path / "a.json")
    monkeypatch.setattr("dancr.core.verify.output_hash", lambda ex, nid: None)
    res = verify_pipeline(p, tmp_path / "a.json")
    assert res["verdict"] == "incomplete"
    assert any("could not be re-read" in s for s in res["incomplete"])
