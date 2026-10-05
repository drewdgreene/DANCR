"""Batch runs: one step applied over many files, an output each plus a combined table, through the Python API
(`dancr.run_batch`), the CLI (`dancr batch`) and MCP (`run_batch`)."""
import json
import subprocess
import sys

import polars as pl
import pytest

from dancr import headless as hl
from dancr.core import Pipeline


def run(*args, cwd=None):
    r = subprocess.run([sys.executable, "-m", "dancr.cli", *args], capture_output=True, text=True, cwd=cwd)
    return r.returncode, r.stdout, r.stderr


@pytest.fixture()
def project(tmp_path):
    d = tmp_path / "files"; d.mkdir()
    pl.DataFrame({"x": [1, 2, 3], "y": [1.0, 2.0, 3.0]}).write_csv(d / "run1.csv")
    pl.DataFrame({"x": [4, 5], "y": [4.0, 5.0]}).write_csv(d / "run2.csv")
    pl.DataFrame({"x": [6], "y": [6.0]}).write_csv(d / "run3.csv")
    p = Pipeline("p"); p.path = tmp_path / "p.json"
    p.add_node("load_file", title="Source", params={"path": "files/run1.csv"}, id="src")
    p.add_node("calculate", title="Doubled", params={"formulas": [{"name": "z", "expr": "[y] * 2"}]}, id="calc")
    p.connect("src", "calc")
    p.save()
    return p


def test_batch_writes_outputs_and_combined(project, tmp_path):
    rec = hl.run_batch(project, [str(project.directory / "files" / "*.csv")], target="calc", out_dir="results")
    assert rec["ok"] and rec["count"] == 3
    results = project.directory / "results"
    assert {p.name for p in results.iterdir()} >= {"run1.csv", "run2.csv", "run3.csv", "combined.csv"}
    combined = pl.read_csv(results / "combined.csv").sort(["source_file", "x"])
    assert combined.height == 6
    assert set(combined["source_file"]) == {"run1.csv", "run2.csv", "run3.csv"}
    assert combined["z"].to_list() == [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
    assert rec["combined"] == str(results / "combined.csv")


def test_batch_rerun_is_served_from_cache(project):
    first = hl.run_batch(project, ["files/*.csv"], target="calc", out_dir="out")
    second = hl.run_batch(project, ["files/*.csv"], target="calc", out_dir="out")
    assert second["ok"] and second["count"] == 3
    assert first["combined"] == second["combined"]


def test_batch_target_defaults_to_last_step(project):
    rec = hl.run_batch(project, ["files/*.csv"], target="", out_dir="out")
    assert rec["target"] == "calc" and rec["ok"]


def test_batch_refuses_output_outside_project(project):
    with pytest.raises(ValueError):
        hl.run_batch(project, ["files/*.csv"], target="calc", out_dir="/tmp/elsewhere-batch")


def test_batch_manifest_written(project):
    out = project.directory / "out"
    hl.run_batch(project, ["files/*.csv"], target="calc", out_dir=out, manifest=out / "manifest.json")
    man = json.loads((out / "manifest.json").read_text())
    assert man["kind"] == "dancr.batch" and man["count"] == 3 and man["target"] == "calc"


def test_batch_can_skip_combined(project):
    out = project.directory / "out"
    rec = hl.run_batch(project, ["files/*.csv"], target="calc", out_dir=out, combined=False)
    assert rec["combined"] is None
    assert not (out / "combined.csv").exists()


def test_cli_batch(project):
    code, out, err = run("batch", str(project.path), "--files", "files/*.csv", "--out-dir", "res", "--node", "calc")
    assert code == 0, err
    assert (project.directory / "res" / "combined.csv").exists()


def test_mcp_run_batch(tmp_path, mcp_root):
    import dancr.mcp_server as srv
    d = tmp_path / "files"; d.mkdir()
    for i in range(2):
        pl.DataFrame({"x": [i]}).write_csv(d / f"f{i}.csv")
    pj = tmp_path / "p.json"
    srv.create_pipeline(str(pj))
    srv.add_node(str(pj), "load_file", {"path": "files/f0.csv"}, node_id="src")
    out = json.loads(srv.run_batch(str(pj), ["files/*.csv"], "out", node_id="src"))
    assert out["ok"] and out["count"] == 2
    assert (tmp_path / "out" / "combined.csv").exists()
