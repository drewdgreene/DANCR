"""Scenario sets (F2): deterministic specs, a headless runner, and attestation evidence."""
import json
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

from dancr.core import Pipeline
from dancr.core.scenarios import from_spec, monte_carlo, sensitivity, sweep
from dancr.headless import build_attestation, run_scenarios, scenario_set, verify_pipeline


def make(tmp_path):
    pl.DataFrame({"amount": [10.0, 20.0, 30.0]}).write_csv(tmp_path / "m.csv")
    p = Pipeline("s")
    p.path = tmp_path / "s.json"
    p.add_node("load_file", params={"path": "m.csv"}, id="src")
    p.add_node("calculate", params={"formulas": [{"name": "scaled", "expr": "[amount] * [factor]"}]}, id="calc")
    p.connect("src", "calc")
    p.set_input("factor", 1)
    p.save()
    return p


def test_sweep_is_deterministic_and_sorted():
    a = sweep({}, {"factor": [1, 2, 3]})
    b = sweep({}, {"factor": [1, 2, 3]})
    assert a == b and [s["id"] for s in a] == ["factor=1", "factor=2", "factor=3"]
    grid = sweep({}, {"b": [1, 2], "a": ["x", "y"]})
    assert [s["inputs"] for s in grid] == [{"a": "x", "b": 1}, {"a": "x", "b": 2},
                                           {"a": "y", "b": 1}, {"a": "y", "b": 2}]


def test_monte_carlo_is_seeded_and_sensitivity_is_one_at_a_time():
    assert monte_carlo({}, {"x": {"min": 0, "max": 1}}, n=5, seed=7) == monte_carlo({}, {"x": {"min": 0, "max": 1}}, n=5, seed=7)
    assert monte_carlo({}, {"x": {"min": 0, "max": 1}}, n=5, seed=7) != monte_carlo({}, {"x": {"min": 0, "max": 1}}, n=5, seed=8)
    s = sensitivity({"factor": 1}, {"factor": [1, 2, 3]})
    assert s[0]["inputs"]["factor"] == 1 and {x["inputs"]["factor"] for x in s} == {1, 2, 3}


def test_from_spec_dispatch():
    assert from_spec({"base": {"x": 1}}) == [{"id": "base", "inputs": {"x": 1}}]
    assert len(from_spec({"sweep": {"x": [1, 2]}})) == 2
    with pytest.raises(ValueError):
        from_spec({})


def test_sweep_refuses_an_unbounded_grid_but_a_limit_bounds_it():
    with pytest.raises(ValueError, match="limit"):
        sweep({}, {"a": list(range(400)), "b": list(range(400))})       # 160,000 scenarios
    assert len(sweep({}, {"a": list(range(400)), "b": list(range(400))}, limit=5)) == 5


def test_run_scenarios_writes_outputs_and_hashes(tmp_path):
    p = make(tmp_path)
    rec = run_scenarios(p, {"sweep": {"factor": [1, 2, 3]}}, out_dir="out")
    assert rec["count"] == 3 and rec["ok"] is True
    assert [s["id"] for s in rec["scenarios"]] == ["factor=1", "factor=2", "factor=3"]
    assert len({s["plan_hash"] for s in rec["scenarios"]}) == 3      # the input is in the plan hash
    assert all(s["output_hash"] for s in rec["scenarios"])
    combined = pl.read_csv(rec["combined"])
    assert "scenario" in combined.columns
    assert combined.filter(pl.col("scenario") == "factor=2")["scaled"].to_list() == [20.0, 40.0, 60.0]
    # the project file itself is never changed
    assert Pipeline.load(tmp_path / "s.json").input_values()["factor"] == 1


def test_run_scenarios_is_deterministic(tmp_path):
    p = make(tmp_path)
    a = run_scenarios(p, {"sweep": {"factor": [1, 2]}}, out_dir="o1")
    b = run_scenarios(p, {"sweep": {"factor": [1, 2]}}, out_dir="o2")
    assert [(s["id"], s["plan_hash"], s["output_hash"]) for s in a["scenarios"]] == \
           [(s["id"], s["plan_hash"], s["output_hash"]) for s in b["scenarios"]]


def test_scenario_evidence_round_trips_through_attestation(tmp_path):
    p = make(tmp_path)
    rec = run_scenarios(p, {"sweep": {"factor": [1, 2]}}, out_dir="out")
    att = build_attestation(p, extra=rec["evidence"])
    assert att["extra"] == rec["evidence"]
    assert verify_pipeline(p, att, extra=rec["evidence"])["verdict"] == "verified"
    assert verify_pipeline(p, att, extra={"scenarios": [{"id": "other"}]})["verdict"] == "mismatch"


def test_scenario_set_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="unique"):
        scenario_set([{"id": "a", "inputs": {}}, {"id": "a", "inputs": {}}])


def test_a_scenario_id_with_a_path_separator_is_written_safely(tmp_path):
    p = make(tmp_path)
    rec = run_scenarios(p, [{"id": "../evil/x", "inputs": {"factor": 2}}], out_dir="out")
    assert rec["ok"] and rec["scenarios"][0]["id"] == "../evil/x"       # the logical id is kept
    out = Path(rec["scenarios"][0]["output"]).resolve()
    assert out.parent == (tmp_path / "out").resolve()                  # but it cannot escape the folder


def test_cli_scenarios(tmp_path):
    make(tmp_path)
    (tmp_path / "spec.json").write_text(json.dumps({"sweep": {"factor": [1, 2]}}))
    r = subprocess.run([sys.executable, "-m", "dancr.cli", "scenarios", "s.json", "--set", "spec.json",
                        "--out-dir", "out"], capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert "2 scenario(s)" in r.stdout


def test_cli_verify_with_scenarios(tmp_path):
    make(tmp_path)
    (tmp_path / "spec.json").write_text(json.dumps({"sweep": {"factor": [1, 2]}}))
    rec = subprocess.run([sys.executable, "-m", "dancr.cli", "verify", "s.json", "--record", "att.json",
                          "--scenarios", "spec.json"], capture_output=True, text=True, cwd=tmp_path)
    assert rec.returncode == 0, rec.stderr
    ver = subprocess.run([sys.executable, "-m", "dancr.cli", "verify", "s.json", "--manifest", "att.json",
                          "--scenarios", "spec.json"], capture_output=True, text=True, cwd=tmp_path)
    assert ver.returncode == 0 and "VERIFIED" in ver.stdout, ver.stdout + ver.stderr


def test_mcp_run_scenarios(tmp_path, mcp_root):
    import dancr.mcp_server as srv
    pj = tmp_path / "p.json"
    (tmp_path / "m.csv").write_text("amount\n10\n20\n")
    srv.create_pipeline(str(pj))
    srv.add_node(str(pj), "load_file", {"path": "m.csv"}, node_id="src")
    srv.add_node(str(pj), "calculate", {"formulas": [{"name": "scaled", "expr": "[amount] * [factor]"}]},
                 node_id="calc", after="src")
    srv.set_input(str(pj), "factor", 1)
    out = json.loads(srv.run_scenarios(str(pj), spec={"sweep": {"factor": [1, 2]}}, out_dir="out"))
    assert out["count"] == 2 and out["ok"]
