"""Smoke test of a frozen build: python packaging/smoke.py path/to/dancr-cli [work folder]

Checks what the frozen app does in ways the test suite cannot: the command line runs a template, a report
writes its PDF through the frozen pdf helper, `dancr-cli mcp` answers an MCP handshake and lists its tools, and
the window program (DANCR, next to dancr-cli) starts, opens a project and keeps running.
Exits non-zero with a message at the first thing that does not work."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path


def cli(exe: str, *args: str) -> str:
    r = subprocess.run([exe, *args], capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        sys.exit(f"dancr-cli {' '.join(args)} failed ({r.returncode}):\n{r.stdout}\n{r.stderr}")
    return r.stdout


def check_report_pdf(exe: str, work: Path) -> None:
    proj = work / "report.json"
    cli(exe, "template", "report", str(proj), "--force")
    nodes = json.loads(cli(exe, "--json", "show", str(proj)))["nodes"]
    rep = next(n for n in nodes if n["type"] == "report")
    cli(exe, "set", str(proj), rep["id"], "pdf=true")
    run = json.loads(cli(exe, "--json", "run", str(proj)))
    if not run["ok"]:
        sys.exit(f"the report project failed: {run['failed']}")
    pdf = Path(run["nodes"][rep["id"]]["report"]["pdf"])
    if not pdf.is_file() or pdf.read_bytes()[:5] != b"%PDF-":
        sys.exit(f"the report's PDF was not written: {pdf}")
    print(f"report PDF ok: {pdf}")


def check_mcp(exe: str, work: Path) -> None:
    msgs = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "smoke", "version": "1"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    ]
    p = subprocess.Popen([exe, "mcp", "--root", str(work)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True)
    watchdog = threading.Timer(120, p.kill)        # a server that never answers ends the read, not the CI job
    watchdog.start()
    try:
        replies = {}
        assert p.stdin is not None and p.stdout is not None
        p.stdin.write(json.dumps(msgs[0]) + "\n"); p.stdin.flush()
        replies[1] = json.loads(p.stdout.readline() or "null")
        for m in msgs[1:]:
            p.stdin.write(json.dumps(m) + "\n")
        p.stdin.flush()
        replies[2] = json.loads(p.stdout.readline() or "null")
    finally:
        watchdog.cancel()
        p.kill()
        _, err = p.communicate(timeout=30)
    if not replies.get(1) or "result" not in replies[1]:
        sys.exit(f"dancr-cli mcp did not answer initialize: {replies.get(1)}\n{err}")
    tools = {t["name"] for t in ((replies.get(2) or {}).get("result") or {}).get("tools", [])}
    if not {"create_pipeline", "run_pipeline", "render_chart"} <= tools:
        sys.exit(f"dancr-cli mcp listed the wrong tools: {sorted(tools)}\n{err}")
    print(f"MCP ok: {len(tools)} tools")


def check_window(exe: str, work: Path) -> None:
    """The window program has no console, so it is checked by running it (offscreen) on a project: a missing Qt
    plugin or a module the window alone imports ends it at once with an error, a working window keeps running."""
    gui = Path(exe).with_name("DANCR.exe" if exe.lower().endswith(".exe") else "DANCR")
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
    p = subprocess.Popen([str(gui), str(work / "t.json")], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True)
    try:
        time.sleep(20)
        if p.poll() is not None:
            out, _ = p.communicate(timeout=30)
            log = cli(exe, "log").strip()
            sys.exit(f"the window program ended by itself ({p.returncode}):\n{out}\n{log}")
    finally:
        if p.poll() is None:
            p.terminate()
            p.wait(timeout=30)
    print("window ok")


def main() -> None:
    exe = sys.argv[1]
    work = Path(sys.argv[2] if len(sys.argv) > 2 else "smoke").resolve()
    work.mkdir(parents=True, exist_ok=True)
    print(cli(exe, "--version").strip())
    cli(exe, "template", "compare", str(work / "t.json"), "--force")
    if not json.loads(cli(exe, "--json", "run", str(work / "t.json")))["ok"]:
        sys.exit("the compare template failed to run")
    check_report_pdf(exe, work)
    check_mcp(exe, work)
    check_window(exe, work)


if __name__ == "__main__":
    main()
