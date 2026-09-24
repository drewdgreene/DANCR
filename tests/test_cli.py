import json
import subprocess
import sys

import pytest


def run(*args, cwd=None):
    r = subprocess.run([sys.executable, "-m", "dancr.cli", *args], capture_output=True, text=True, cwd=cwd)
    return r.returncode, r.stdout, r.stderr


def test_cli_flow(probe_dir, tmp_path):
    pj = tmp_path / "p.json"
    code, out, err = run("new", str(pj)); assert code == 0
    code, out, _ = run("--json", "add", str(pj), "load_file", "--id", "a", "--set", f"path={probe_dir / 'probe_A.csv'}")
    assert code == 0 and json.loads(out)["id"] == "a"
    code, out, _ = run("--json", "add", str(pj), "time_buckets", "--id", "tb", "--after", "a", "--set", "every=1m")
    assert code == 0
    code, out, _ = run("--json", "run", str(pj))
    assert code == 0, out
    data = json.loads(out)
    assert data["ok"] and {n["node_id"] for n in data["nodes"]} == {"a", "tb"}
    code, out, _ = run("--json", "schema", str(pj), "tb")
    assert "pressure_psi" in out
    code, out, _ = run("--json", "sample", str(pj), "tb", "--rows", "2")
    assert len(json.loads(out)) == 2
    code, out, _ = run("--json", "stats", str(pj), "tb")
    assert code == 0 and "pressure_psi" in out
    png = tmp_path / "c.png"
    code, out, _ = run("chart", str(pj), "tb", "--out", str(png), "--x", "time", "--y", "pressure_psi")
    assert code == 0 and png.exists()
    code, out, _ = run("show", str(pj))
    assert "✓" in out
    code, out, _ = run("set", str(pj), "tb", "every=banana")
    assert code == 2
    code, out, _ = run("--json", "remove", str(pj), "tb")
    assert code == 0
    code, out, _ = run("status", str(pj), "a")
    assert code == 0 and "[a]" in out and "tb" not in out


def test_cli_nodes_and_formulas():
    code, out, _ = run("--json", "nodes")
    assert code == 0 and any(t["key"] == "combine" for t in json.loads(out))
    code, out, _ = run("formulas")
    assert "ROLLING_MEAN" in out


def test_cli_errors_are_json(tmp_path):
    code, out, _ = run("--json", "show", str(tmp_path / "missing.json"))
    assert code == 2 and "error" in json.loads(out)


def test_cli_unknown_node_id(probe_dir, tmp_path):
    pj = tmp_path / "p.json"
    run("new", str(pj))
    run("add", str(pj), "load_file", "--id", "a", "--set", f"path={probe_dir / 'probe_A.csv'}")
    for cmd in (["status"], ["schema"], ["sample"], ["remove"], ["run"], ["rename"]):
        extra = ["New"] if cmd == ["rename"] else []
        code, out, err = run(*cmd, str(pj), "nosuch", *extra)
        assert code == 2 and "No step called 'nosuch'" in err and "['a']" in err, (cmd, err)
        code, out, err = run("--json", *cmd, str(pj), "nosuch", *extra)
        assert code == 2 and "No step called 'nosuch'" in json.loads(out)["error"], (cmd, out)
    code, out, err = run("connect", str(pj), "a", "nosuch")
    assert code == 2 and "No step called 'nosuch'" in err
    code, out, err = run("add", str(pj), "sort", "--after", "nosuch")
    assert code == 2 and "No step called 'nosuch'" in err


def test_cli_run_json_reports_problems(probe_dir, tmp_path):
    pj = tmp_path / "p.json"
    run("new", str(pj))
    run("add", str(pj), "load_file", "--id", "a", "--set", f"path={probe_dir / 'probe_A.csv'}")
    run("add", str(pj), "sort", "--id", "s")            # not connected: a configuration problem
    code, out, _ = run("--json", "run", str(pj), "a")
    assert code == 0, out
    data = json.loads(out)
    assert data["ok"] and isinstance(data["problems"], list) and data["problems"]
    assert json.loads(run("--json", "show", str(pj))[1])["problems"] == data["problems"]


def test_mcp_writes_stay_inside_the_project_folder(probe_dir, tmp_path):
    from mcp.server.mcpserver.exceptions import ToolError
    from dancr.mcp_server import create_pipeline, add_node, run_pipeline, export_node, render_chart, _inside_project
    from dancr.core import Pipeline
    proj = tmp_path / "proj"; proj.mkdir()
    pj = proj / "p.json"
    create_pipeline(str(pj))
    add_node(str(pj), "load_file", {"path": str(probe_dir / "probe_A.csv")}, node_id="a")
    run_pipeline(str(pj))
    p = Pipeline.load(pj)
    assert _inside_project(p, "out/x.csv") == proj / "out" / "x.csv"
    assert _inside_project(p, str(proj / "sub" / "y.parquet")) == proj / "sub" / "y.parquet"
    for bad in (str(tmp_path / "escape.csv"), "../escape.csv", str(proj / "sub" / ".." / ".." / "escape.csv")):
        with pytest.raises(ToolError, match="inside the project folder"):
            _inside_project(p, bad)
    with pytest.raises(ToolError, match="inside the project folder"):
        export_node(str(pj), "a", str(tmp_path / "escape.csv"))
    with pytest.raises(ToolError, match="inside the project folder"):
        render_chart(str(pj), "a", out_png="../escape.png", x="time", y=["pressure_psi"])
    assert not (tmp_path / "escape.csv").exists() and not (tmp_path / "escape.png").exists()
    export_node(str(pj), "a", "sub/ok.csv")
    assert (proj / "sub" / "ok.csv").exists()
