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


def test_mcp_writes_stay_inside_the_project_folder(probe_dir, tmp_path, mcp_root):
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


def test_mcp_steps_cannot_write_outside_the_project_folder(probe_dir, tmp_path, mcp_root):
    from mcp.server.mcpserver.exceptions import ToolError
    from dancr.mcp_server import create_pipeline, add_node, run_pipeline, set_params
    proj = tmp_path / "proj"; proj.mkdir()
    pj = str(proj / "p.json")
    create_pipeline(pj)
    add_node(pj, "load_file", {"path": str(probe_dir / "probe_A.csv")}, node_id="a")
    for i, bad in enumerate(("../escaped.csv", str(tmp_path / "escaped.csv"))):
        add_node(pj, "export", {"path": bad}, node_id=f"ex{i}", after="a")
    add_node(pj, "report", {"title": "R", "path": str(tmp_path / "r.html"), "pdf": False}, node_id="rep", after="a", port="items")
    add_node(pj, "workbook", {"path": "../book.xlsx"}, node_id="wb", after="a")
    out = json.loads(run_pipeline(pj))
    assert set(out["failed"]) == {"ex0", "ex1", "rep", "wb"}
    assert all("inside the project folder" in out["nodes"][n]["error"] for n in out["failed"])
    assert not list(tmp_path.glob("escaped.csv")) and not (tmp_path / "r.html").exists() and not (tmp_path / "book.xlsx").exists()
    set_params(pj, "ex0", {"path": "fine.csv"})
    assert json.loads(run_pipeline(pj, node_ids=["ex0"]))["ok"]
    assert (proj / "fine.csv").exists()


def test_mcp_creates_pipelines_only_under_its_root(tmp_path, mcp_root):
    from mcp.server.mcpserver.exceptions import ToolError
    from dancr.mcp_server import create_pipeline, build_template
    outside = tmp_path.parent / "elsewhere.json"
    with pytest.raises(ToolError, match="only be created inside"):
        create_pipeline(str(outside))
    with pytest.raises(ToolError, match="only be created inside"):
        build_template(str(tmp_path / ".." / "t.json"), "compare")
    with pytest.raises(ToolError, match=r"\.json"):
        create_pipeline(str(tmp_path / "p.txt"))
    assert not outside.exists()
    create_pipeline("relative.json")                           # relative to the root
    assert (tmp_path / "relative.json").exists()


def test_bad_template_name_writes_nothing(tmp_path, mcp_root):
    from mcp.server.mcpserver.exceptions import ToolError
    from dancr.mcp_server import build_template
    proj = tmp_path / "proj"; proj.mkdir()
    with pytest.raises(ToolError, match="Unknown template"):
        build_template(str(proj / "x.json"), "bogus")
    assert list(proj.iterdir()) == []
    code, _, _ = run("template", "bogus", str(proj / "y.json"))
    assert code != 0 and list(proj.iterdir()) == []


def test_templates_store_the_data_path_relative_to_the_project(tmp_path, mcp_root):
    from dancr.mcp_server import build_template
    from dancr.core import Pipeline
    pj = tmp_path / "t.json"
    build_template(str(pj), "compare")
    p = Pipeline.load(pj)
    load = next(n for n in p.nodes.values() if n.type == "load_file")
    assert load.params["path"] == "sample_data.csv"


def test_mcp_render_chart_overrides_and_validates(probe_dir, tmp_path, mcp_root):
    from mcp.server.mcpserver.exceptions import ToolError
    from dancr.mcp_server import create_pipeline, add_node, render_chart, get_schema
    pj = str(tmp_path / "p.json")
    create_pipeline(pj)
    add_node(pj, "load_file", {"path": str(probe_dir / "probe_A.csv")}, node_id="a")
    add_node(pj, "chart", {"kind": "scatter", "x": "time", "series": [{"column": "pressure_psi"}]}, node_id="c", after="a")
    render_chart(pj, "c", kind="histogram", column="pressure_psi", out_png="h.png")
    assert (tmp_path / "h.png").exists()
    with pytest.raises(ToolError, match="nosuch"):
        render_chart(pj, "a", x="time", y=["nosuch"])
    with pytest.raises(ToolError, match="No node"):
        get_schema(pj, "zzz")
