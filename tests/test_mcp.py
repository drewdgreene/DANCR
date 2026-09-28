"""The MCP server: its tools, where it may write, what it refuses, and the protocol over stdio."""
import asyncio
import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

import polars as pl
import pytest
from mcp.server.mcpserver.exceptions import ToolError

import dancr.mcp_server as srv
from dancr.core import Pipeline


def run(*args, cwd=None):
    r = subprocess.run([sys.executable, "-m", "dancr.cli", *args], capture_output=True, text=True, cwd=cwd)
    return r.returncode, r.stdout, r.stderr


@pytest.fixture
def project(tmp_path, mcp_root) -> Path:
    (tmp_path / "data").mkdir()
    pl.DataFrame({"a": [1, 2, 3]}).write_csv(tmp_path / "data" / "source.csv")
    pj = tmp_path / "p.json"
    srv.create_pipeline(str(pj))
    srv.add_node(str(pj), "load_file", {"path": "data/source.csv"}, node_id="src")
    return pj


ROOT = Path(__file__).resolve().parent.parent


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
    # refused when the step is added or changed ...
    for bad in ("../escaped.csv", str(tmp_path / "escaped.csv")):
        with pytest.raises(ToolError, match="inside the project folder"):
            add_node(pj, "export", {"path": bad}, after="a")
    add_node(pj, "export", {"path": "fine.csv"}, node_id="ex0", after="a")
    with pytest.raises(ToolError, match="inside the project folder"):
        set_params(pj, "ex0", {"path": "../escaped.csv"})
    # ... and when it runs, for a project file edited by other means
    from dancr.core import Pipeline
    p = Pipeline.load(pj)
    p.set_params("ex0", path="../escaped.csv")
    p.add_node("export", params={"path": str(tmp_path / "escaped.csv")}, id="ex1"); p.connect("a", "ex1")
    p.add_node("report", params={"title": "R", "path": str(tmp_path / "r.html"), "pdf": False}, id="rep"); p.connect("a", "rep", "items")
    p.add_node("workbook", params={"path": "../book.xlsx"}, id="wb"); p.connect("a", "wb")
    p.save()
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
    with pytest.raises(ToolError, match="No step called"):
        get_schema(pj, "zzz")


def test_export_node_refuses_to_overwrite_the_data_the_project_reads(project, tmp_path):
    before = (tmp_path / "data" / "source.csv").read_text()
    with pytest.raises(ToolError, match="reads its data from that file"):
        srv.export_node(str(project), "src", "data/source.csv")
    with pytest.raises(ToolError, match="reads its data"):
        srv.render_chart(str(project), "src", out_png="data/../data/source.csv", x="a", y=["a"])
    assert (tmp_path / "data" / "source.csv").read_text() == before
    out = srv.export_node(str(project), "src", "data/copy.csv")         # any other file in the folder is fine
    assert Path(json.loads(out)["path"]).exists()


def test_export_steps_cannot_point_at_a_source_file(project, tmp_path):
    with pytest.raises(ToolError, match="reads its data"):
        srv.add_node(str(project), "export", {"path": "data/source.csv"}, after="src")
    assert "export_1" not in Pipeline.load(project).nodes


def test_mcp_never_writes_into_the_dancr_folder(project, tmp_path):
    srv.run_pipeline(str(project))
    cached = next((tmp_path / ".dancr" / "cache").rglob("*.parquet"))
    before = cached.read_bytes()
    for bad in (str(cached.relative_to(tmp_path)), ".DANCR/cache/x.csv", "sub/.dancr/x.csv"):
        with pytest.raises(ToolError, match=".dancr folder"):
            srv.export_node(str(project), "src", bad)
    with pytest.raises(ToolError, match=".dancr folder"):
        srv.render_chart(str(project), "src", out_png=".dancr/x.png", x="a", y=["a"])
    with pytest.raises(ToolError, match=".dancr folder"):
        srv.add_node(str(project), "export", {"path": ".dancr/out.csv"}, after="src")
    with pytest.raises(ToolError, match=".dancr folder"):
        srv.create_pipeline(str(tmp_path / ".dancr" / "versions" / "p" / "x.json"))
    assert cached.read_bytes() == before


def test_steps_run_from_mcp_cannot_write_into_the_dancr_folder(tmp_path):
    from dancr.core.executor import Executor
    p = Pipeline()
    p.add_node("enter_data", id="d", params={"columns": [{"name": "a", "type": "number"}], "rows": [[1]]})
    p.add_node("export", id="ex", params={"path": ".dancr/out.csv"})
    p.connect("d", "ex")
    p.save(tmp_path / "p.json")
    st = Executor(p, output_root=tmp_path).run()["ex"]
    assert st.status == "failed" and ".dancr folder" in st.error
    assert not (tmp_path / ".dancr" / "out.csv").exists()


def test_a_step_changed_elsewhere_is_refused_before_it_runs(project, tmp_path):
    p = Pipeline.load(project)                          # e.g. edited by hand or by the command line
    p.add_node("export", params={"path": "data/source.csv"}, id="x"); p.connect("src", "x")
    p.save()
    before = (tmp_path / "data" / "source.csv").read_text()
    out = json.loads(srv.run_pipeline(str(project)))
    assert out["failed"] == ["x"] and "reads its data" in out["nodes"]["x"]["error"]
    assert out["nodes"]["src"]["status"] == "done"      # the rest runs
    assert (tmp_path / "data" / "source.csv").read_text() == before
    with pytest.raises(ToolError, match="reads its data"):
        srv.get_sample(str(project), "x")
    srv.get_sample(str(project), "src")                 # reading a step that does not need it still works


def test_create_pipeline_overwrites_only_pipelines(tmp_path, mcp_root):
    (tmp_path / "package.json").write_text('{"name": "web"}')
    with pytest.raises(ToolError, match="not a DANCR project"):
        srv.create_pipeline(str(tmp_path / "package.json"), overwrite=True)
    with pytest.raises(ToolError, match="not a DANCR project"):
        srv.create_pipeline(str(tmp_path / "package.json"))
    assert (tmp_path / "package.json").read_text() == '{"name": "web"}'
    pj = tmp_path / "p.json"
    srv.create_pipeline(str(pj))
    srv.add_node(str(pj), "enter_data", node_id="e")
    srv.create_pipeline(str(pj), overwrite=True)
    assert Pipeline.load(pj).nodes == {}


def test_home_folder_root_needs_to_be_named(tmp_path, monkeypatch):
    monkeypatch.setattr(srv, "ROOT", tmp_path.resolve())
    srv.create_pipeline("p.json")
    srv.add_node("p.json", "enter_data", node_id="e")
    monkeypatch.setattr(srv, "ROOT_REFUSED", True)       # as if started in the home folder
    with pytest.raises(ToolError, match="--root"):
        srv.create_pipeline("q.json")
    with pytest.raises(ToolError, match="--root"):
        srv.rename_node("p.json", "e", "changed")
    assert srv.formula_reference()                      # reading tools keep working, on existing pipelines too
    assert "e" in srv.describe_pipeline("p.json")


def test_main_refuses_home_and_drive_roots_only_without_root(tmp_path, monkeypatch):
    ran = []
    monkeypatch.setattr(srv.mcp, "run", lambda transport: ran.append(transport))
    monkeypatch.setattr(srv, "ROOT_REFUSED", False)
    for start, refused in ((Path.home(), True), (Path.home().parent, True), (Path(Path.home().anchor), True),
                           (tmp_path, False)):
        monkeypatch.setattr(srv, "ROOT", start.resolve())
        srv.main()
        assert srv.ROOT_REFUSED is refused, start
    monkeypatch.setattr(srv, "ROOT", Path.home().resolve())
    srv.main(str(Path.home()))                          # named on purpose: allowed
    assert srv.ROOT_REFUSED is False


def test_parallel_mcp_edits_are_all_kept(tmp_path, mcp_root):
    from dancr.mcp_server import create_pipeline, add_node
    from dancr.core import Pipeline
    pj = str(tmp_path / "p.json")
    create_pipeline(pj)
    errors = []

    def add(i):
        try:
            add_node(pj, "enter_data", node_id=f"n{i}")
        except Exception as e:  # noqa: BLE001
            errors.append(e)
    threads = [threading.Thread(target=add, args=(i,)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert len(Pipeline.load(pj).nodes) == 16


def test_open_in_gui_refuses_steps_that_write_elsewhere(tmp_path, mcp_root, monkeypatch):
    from mcp.server.mcpserver.exceptions import ToolError
    import dancr.cli
    from dancr.mcp_server import create_pipeline, open_in_gui
    from dancr.core import Pipeline
    launched = []
    monkeypatch.setattr(dancr.cli, "launch_gui", lambda path, wait=False: launched.append(path))
    pj = tmp_path / "p.json"
    create_pipeline(str(pj))
    p = Pipeline.load(pj)
    p.add_node("enter_data", id="e"); p.add_node("export", params={"path": str(tmp_path.parent / "leak.csv")}, id="x"); p.connect("e", "x")
    p.save()
    with pytest.raises(ToolError, match="inside the project folder"):
        open_in_gui(str(pj))
    assert launched == []


@pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
def test_symlinks_do_not_escape(tmp_path, mcp_root):
    from mcp.server.mcpserver.exceptions import ToolError
    from dancr.mcp_server import create_pipeline, add_node, export_node
    from dancr.core.registry import Ctx
    outside = tmp_path.parent / f"outside-{tmp_path.name}"
    outside.mkdir()
    (tmp_path / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ToolError):
        create_pipeline(str(tmp_path / "link" / "p.json"))
    pj = str(tmp_path / "p.json")
    create_pipeline(pj)
    add_node(pj, "enter_data", node_id="e", params={"columns": [{"name": "a", "type": "number"}], "rows": [[1]]})
    with pytest.raises(ToolError):
        add_node(pj, "export", {"path": "link/x.csv"}, after="e")
    with pytest.raises(ToolError):
        export_node(pj, "e", "link/x.csv")
    with pytest.raises(ValueError):
        Ctx(tmp_path, "n", "n", output_root=tmp_path).resolve_output("link/x.csv")
    assert not list(outside.iterdir())


def test_mcp_reading_tools_check_their_arguments(tmp_path, mcp_root):
    from mcp.server.mcpserver.exceptions import ToolError
    from dancr.mcp_server import create_pipeline, add_node, get_sample, remove_input, set_input, inspect_file
    pj = str(tmp_path / "p.json")
    create_pipeline(pj)
    add_node(pj, "enter_data", node_id="e", params={"columns": [{"name": "a", "type": "number"}], "rows": [[1]]})
    with pytest.raises(ToolError, match="between 1 and 500"):
        get_sample(pj, "e", rows=0)
    with pytest.raises(ToolError, match="no column called 'b'"):
        get_sample(pj, "e", columns=["b"])
    with pytest.raises(ToolError, match="No input"):
        remove_input(pj, "nosuch")
    with pytest.raises(ToolError, match="not a list"):
        set_input(pj, "x", {"a": 1})
    pl.DataFrame({"z": [1]}).write_csv(tmp_path / "d.csv")
    assert json.loads(inspect_file("d.csv"))["columns"][0]["name"] == "z"      # relative to the root


def test_the_mcp_protocol_over_stdio(tmp_path):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def session():
        params = StdioServerParameters(command=sys.executable, args=["-m", "dancr.cli", "mcp", "--root", str(tmp_path)],
                                       env={**os.environ})
        async with stdio_client(params) as (r, w):
            async with ClientSession(r, w) as s:
                await s.initialize()
                names = {t.name for t in (await s.list_tools()).tools}
                bad = await s.call_tool("create_pipeline", {"path": str(tmp_path.parent / "out.json")})
                good = await s.call_tool("build_template", {"path": "t.json", "template": "fit"})
                run = await s.call_tool("run_pipeline", {"path": "t.json"})
                return names, bad, good, run
    names, bad, good, run = asyncio.run(session())
    documented = set(re.findall(r"`(\w+)`", (ROOT / "AGENTS.md").read_text().split("- **CLI**")[0].split("Tools:")[1]))
    assert documented <= names
    assert bad.is_error and "only be created inside" in bad.content[0].text
    assert not good.is_error and json.loads(run.content[0].text)["ok"]


def test_mcp_run_and_status_share_cli_field_names(probe_dir, tmp_path, mcp_root):
    from dancr.mcp_server import create_pipeline, add_node, run_pipeline, node_status

    pj = tmp_path / "p.json"
    create_pipeline(str(pj))
    add_node(str(pj), "load_file", {"path": str(probe_dir / "probe_A.csv")}, node_id="a")
    data = json.loads(run_pipeline(str(pj)))
    assert "elapsed" in data and "cache_dir" in data and isinstance(data["failed"], list)
    assert data["nodes"]["a"]["node_id"] == "a" and "elapsed" in data["nodes"]["a"]
    st = json.loads(node_status(str(pj), "a"))
    assert st["node_id"] == "a" and "elapsed" in st and st["columns"][0].keys() == {"name", "dtype"}


def test_a_second_name_for_a_data_file_is_still_refused(mcp_root):
    import os
    data = mcp_root / "data.csv"
    data.write_text("a\n1\n2\n")
    os.link(data, mcp_root / "DATA-link.csv")          # as A.CSV is a.csv on a Mac or Windows disk
    pj = mcp_root / "p.json"
    srv.create_pipeline(str(pj))
    srv.add_node(str(pj), "load_file", node_id="src", params={"path": "data.csv"})
    with pytest.raises(ToolError, match="reads its data"):
        srv.export_node(str(pj), "src", "DATA-link.csv")
    assert data.read_text() == "a\n1\n2\n"
