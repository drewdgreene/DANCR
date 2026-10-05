"""Watching a project and its data: `dancr.watch` / `dancr watch` reruns when a source or the project changes."""
import subprocess
import sys
import threading
import time

import polars as pl
import pytest

from dancr import headless as hl
from dancr.core import Pipeline


def run(*args, cwd=None):
    r = subprocess.run([sys.executable, "-m", "dancr.cli", *args], capture_output=True, text=True, cwd=cwd)
    return r.returncode, r.stdout, r.stderr


@pytest.fixture()
def proj(tmp_path):
    pl.DataFrame({"x": [1, 2, 3]}).write_csv(tmp_path / "d.csv")
    p = Pipeline("p"); p.path = tmp_path / "p.json"
    p.add_node("load_file", title="Data", params={"path": "d.csv"}, id="src")
    p.save()
    return p


def test_watch_state_tracks_sources(proj):
    state = hl._watch_state(Pipeline.load(proj.path))
    assert str((proj.directory / "d.csv").resolve()) in state
    (proj.directory / "d.csv").write_text("x\n9\n")
    assert hl._watch_state(Pipeline.load(proj.path)) != state


def test_watch_once_runs(proj):
    rec = hl.watch(proj.path, once=True, interval=0.05)
    assert rec["runs"] == 1 and rec["ok"] and rec["kind"] == "dancr.watch"


def test_watch_reruns_when_a_source_changes(proj):
    events: list[dict] = []
    stop = threading.Event()
    t = threading.Thread(target=lambda: hl.watch(proj.path, interval=0.05, on_event=events.append, stop=stop), daemon=True)
    t.start()
    deadline = time.time() + 15
    while not any(e["type"] == "watch_ran" for e in events) and time.time() < deadline:
        time.sleep(0.02)
    assert any(e["type"] == "watch_started" for e in events)
    (proj.directory / "d.csv").write_text("x\n9\n9\n9\n")
    while not any(e["type"] == "watch_ran" and e.get("why") == "changed" for e in events) and time.time() < deadline:
        time.sleep(0.02)
    stop.set(); t.join(timeout=5)
    assert any(e["type"] == "watch_changed" for e in events)
    assert any(e["type"] == "watch_ran" and e.get("why") == "changed" for e in events)


def test_cli_watch_once(proj):
    code, out, err = run("watch", str(proj.path), "--once")
    assert code == 0, err
