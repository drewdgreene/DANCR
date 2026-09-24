"""The CLI and the MCP server behave as AGENTS.md says (audit 2026-09-24, A11, A12, I1–I9, T3)."""
import asyncio
import json
import os
import re
import shlex
import subprocess
import sys
import threading
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parent.parent


def cli(*args, cwd=None):
    r = subprocess.run([sys.executable, "-m", "dancr.cli", *args], capture_output=True, text=True, cwd=cwd)
    return r.returncode, r.stdout, r.stderr


# ------------------------------------------------------------------ AGENTS.md recipes, run as written
def _recipes() -> dict[str, list[str]]:
    text = (ROOT / "AGENTS.md").read_text()
    out = {}
    for title, body in re.findall(r"## Recipe: (.+?)\n.*?```bash\n(.*?)```", text, re.S):
        body = body.replace("\\\n", " ")
        out[title] = [ln.split("  #")[0].strip() for ln in body.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    return out


def _logs(folder: Path) -> None:
    n = 3000
    t = [datetime(2024, 6, 1) + timedelta(seconds=10 * i) for i in range(n)]
    rng = np.random.default_rng(3)
    v = 100 + np.sin(np.arange(n) / 200)
    pl.DataFrame({"time": t, "value": v + rng.normal(0, 0.01, n)}).write_csv(folder / "log_A.csv")
    pl.DataFrame({"time": [x + timedelta(milliseconds=20) for x in t], "value": 1.02 * v - 2 + rng.normal(0, 0.05, n)}).write_csv(folder / "log_B.csv")
    t2 = [datetime(2024, 6, 1) + timedelta(minutes=10 * i) for i in range(3000)]
    pl.DataFrame({"time": t2, "value": np.linspace(1, 3, 3000), "temperature": np.linspace(10, 20, 3000)}).write_csv(folder / "log.csv")


@pytest.mark.parametrize("title", list(_recipes()))
def test_agents_md_recipes_run_as_written(title, tmp_path):
    _logs(tmp_path)
    for line in _recipes()[title]:
        args = shlex.split(line)
        assert args[0] == "dancr"
        if args[1] == "open":
            continue                                    # would start the window
        code, out, err = cli(*args[1:], cwd=tmp_path)
        assert code == 0, f"{line}\n{out}\n{err}"
        if args[1:3] == ["--json", "run"]:
            assert json.loads(out)["ok"], out


# ------------------------------------------------------------------ CLI details
def test_cli_run_and_status_have_the_documented_shape(tmp_path):
    pj = str(tmp_path / "t.json")
    assert cli("template", "fit", pj)[0] == 0
    code, out, _ = cli("run", pj, "--json")                  # --json after the command works too
    data = json.loads(out)
    assert code == 0 and set(data) >= {"ok", "failed", "nodes", "problems", "elapsed", "cache_dir"}
    nid = next(iter(data["nodes"]))
    rec = data["nodes"][nid]
    assert set(rec) >= {"node_id", "title", "type", "status", "rows", "columns", "error", "messages", "report", "elapsed", "from_cache"}
    code, out, _ = cli("--json", "status", pj, nid)
    assert json.loads(out)["node_id"] == nid


def test_cli_errors(tmp_path):
    code, out, _ = cli("--json", "frobnicate")
    assert code == 2 and "error" in json.loads(out)
    code, out, _ = cli("--json", "template", "compare")
    assert code == 2 and "Name the project file" in json.loads(out)["error"]
    code, out, _ = cli("--json", "template", "compare", str(tmp_path / "t.json"), "--data", "nosuch.csv")
    assert code == 2 and "No data file" in json.loads(out)["error"] and not (tmp_path / "t.json").exists()
    pj = str(tmp_path / "p.json")
    cli("new", pj)
    code, out, _ = cli("--json", "inputs", pj, "area")
    assert code == 2 and "value" in json.loads(out)["error"]
    code, out, _ = cli("--json", "inputs", pj, "--remove", "nosuch")
    assert code == 2


# ------------------------------------------------------------------ MCP
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
