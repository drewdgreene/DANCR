"""The command line: its commands, JSON output, exit codes, errors and the recipes AGENTS.md documents."""
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from dancr.core import Pipeline


def run(*args, cwd=None):
    r = subprocess.run([sys.executable, "-m", "dancr.cli", *args], capture_output=True, text=True, cwd=cwd)
    return r.returncode, r.stdout, r.stderr


ROOT = Path(__file__).resolve().parent.parent


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
    assert data["ok"] and set(data["nodes"]) == {"a", "tb"} and data["nodes"]["tb"]["node_id"] == "tb"
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


@pytest.mark.parametrize("title", list(_recipes()))
def test_agents_md_recipes_run_as_written(title, tmp_path):
    _logs(tmp_path)
    for line in _recipes()[title]:
        args = shlex.split(line)
        assert args[0] == "dancr"
        if args[1] == "open":
            continue                                    # would start the window
        code, out, err = run(*args[1:], cwd=tmp_path)
        assert code == 0, f"{line}\n{out}\n{err}"
        if args[1:3] == ["--json", "run"]:
            assert json.loads(out)["ok"], out


def test_cli_run_and_status_have_the_documented_shape(tmp_path):
    pj = str(tmp_path / "t.json")
    assert run("template", "fit", pj)[0] == 0
    code, out, _ = run("run", pj, "--json")                  # --json after the command works too
    data = json.loads(out)
    assert code == 0 and set(data) >= {"ok", "failed", "nodes", "problems", "elapsed", "cache_dir"}
    nid = next(iter(data["nodes"]))
    rec = data["nodes"][nid]
    assert set(rec) >= {"node_id", "title", "type", "status", "rows", "columns", "error", "messages", "report", "elapsed", "from_cache"}
    code, out, _ = run("--json", "status", pj, nid)
    assert json.loads(out)["node_id"] == nid


def test_cli_errors(tmp_path):
    code, out, _ = run("--json", "frobnicate")
    assert code == 2 and "error" in json.loads(out)
    code, out, _ = run("--json", "template", "compare")
    assert code == 2 and "Name the project file" in json.loads(out)["error"]
    code, out, _ = run("--json", "template", "compare", str(tmp_path / "t.json"), "--data", "nosuch.csv")
    assert code == 2 and "No data file" in json.loads(out)["error"] and not (tmp_path / "t.json").exists()
    pj = str(tmp_path / "p.json")
    run("new", pj)
    code, out, _ = run("--json", "inputs", pj, "area")
    assert code == 2 and "value" in json.loads(out)["error"]
    code, out, _ = run("--json", "inputs", pj, "--remove", "nosuch")
    assert code == 2


def test_cli_template_inputs_columns(tmp_path):
    from dancr.cli import main
    import io, contextlib
    out = tmp_path / "t.json"
    main(["template", "fit", str(out)])
    assert out.exists() and (tmp_path / "sample_data.csv").exists()
    main(["inputs", str(out), "k", "2.5", "--unit", "x"])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main(["--json", "inputs", str(out)])
    assert json.loads(buf.getvalue()) == [{"name": "k", "value": 2.5, "unit": "x", "note": ""}]
    main(["columns", str(out), "value A", "--label", "Value A", "--unit", "kPa"])
    p = Pipeline.load(out)
    assert p.column_title("value A") == "Value A (kPa)"


def test_a_step_that_fails_when_read_exits_1(tmp_path):
    pj = tmp_path / "p.json"
    p = Pipeline("p"); p.add_node("load_file", params={"path": "missing.csv"}, id="l"); p.save(pj)
    r = subprocess.run([sys.executable, "-m", "dancr.cli", "sample", str(pj), "l", "--run"], capture_output=True, text=True)
    assert r.returncode == 1


def test_cli_run_json_reports_failed_nodes(probe_dir, tmp_path):
    import subprocess
    import sys

    pj = tmp_path / "p.json"
    subprocess.run([sys.executable, "-m", "dancr.cli", "new", str(pj)], capture_output=True, text=True)
    subprocess.run([sys.executable, "-m", "dancr.cli", "add", str(pj), "load_file", "--id", "a",
                    "--set", f"path={probe_dir / 'probe_A.csv'}"], capture_output=True, text=True)
    r = subprocess.run([sys.executable, "-m", "dancr.cli", "--json", "run", str(pj)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["failed"] == []
    subprocess.run([sys.executable, "-m", "dancr.cli", "add", str(pj), "calculate", "--id", "bad", "--after", "a",
                    "--params", '{"formulas":[{"name":"z","expr":"no_such_column * 2"}]}'], capture_output=True, text=True)
    r = subprocess.run([sys.executable, "-m", "dancr.cli", "--json", "run", str(pj)], capture_output=True, text=True)
    assert r.returncode == 1
    assert json.loads(r.stdout)["failed"] == ["bad"]


@pytest.mark.skipif(sys.platform == "win32" or os.geteuid() == 0, reason="needs a folder this user cannot write")
def test_listing_needs_no_lock_so_a_read_only_folder_works(tmp_path):
    pj = tmp_path / "p.json"
    p = Pipeline(); p.set_input("area", 12.5); p.save(pj)
    tmp_path.chmod(0o555)
    try:
        for cmd in (["inputs", str(pj)], ["columns", str(pj)], ["show", str(pj)]):
            code, out, err = run("--json", *cmd)
            assert code == 0, (cmd, err)
    finally:
        tmp_path.chmod(0o755)
