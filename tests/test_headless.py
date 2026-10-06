"""One writer at a time: the project file's lock, shared by the window, the command line and the MCP server."""
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from dancr import headless as hl
from dancr.core import Pipeline

HOLD = """
import sys, time
from dancr.headless import project_lock
with project_lock(sys.argv[1]):
    print("held", flush=True)
    sys.stdin.readline()
"""


def _holder(path: Path) -> subprocess.Popen:
    p = subprocess.Popen([sys.executable, "-c", HOLD, str(path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    assert p.stdout.readline().strip() == "held"
    return p


def test_the_lock_is_held_across_processes_and_dies_with_its_holder(tmp_path):
    pj = tmp_path / "p.json"
    Pipeline("p").save(pj)
    holder = _holder(pj)
    try:
        with pytest.raises(hl.ProjectBusy, match="being changed by another program"):
            with hl.project_lock(pj, wait=0.3):
                pass
    finally:
        holder.kill(); holder.wait()                     # killed, not released: the system frees the lock
    with hl.project_lock(pj, wait=5):
        pass


def test_a_command_waits_for_the_lock_then_fails_plainly(tmp_path, monkeypatch):
    from dancr.cli import main
    pj = tmp_path / "p.json"
    Pipeline("p").save(pj)
    monkeypatch.setattr(hl, "LOCK_WAIT", 0.3)
    holder = _holder(pj)
    try:
        with pytest.raises(SystemExit) as e:
            main(["add", str(pj), "enter_data"])
        assert e.value.code == 2
    finally:
        holder.stdin.write("\n"); holder.stdin.flush(); holder.wait()
    main(["add", str(pj), "enter_data", "--id", "e"])
    assert "e" in Pipeline.load(pj).nodes


def test_threads_in_one_process_take_turns(tmp_path):
    pj = tmp_path / "p.json"
    Pipeline("p").save(pj)

    def add(i):
        with hl.editing(pj) as p:
            p.add_node("enter_data", id=f"n{i}")
    threads = [threading.Thread(target=add, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(Pipeline.load(pj).nodes) == 8


def test_an_edit_is_not_saved_over_a_change_made_without_the_lock(tmp_path):
    pj = tmp_path / "p.json"
    Pipeline("p").save(pj)
    with pytest.raises(hl.ProjectBusy, match="nothing was saved"):
        with hl.editing(pj) as p:
            p.add_node("enter_data", id="mine")
            other = Pipeline.load(pj); other.add_node("enter_data", id="theirs"); other.save()   # a text editor, say
    assert list(Pipeline.load(pj).nodes) == ["theirs"]


def test_an_edit_that_changes_nothing_does_not_write(tmp_path):
    pj = tmp_path / "p.json"
    Pipeline("p").save(pj)
    before = pj.stat().st_mtime_ns
    with hl.editing(pj):
        pass
    assert pj.stat().st_mtime_ns == before


def test_a_deferred_edit_holds_no_lock_while_it_computes(tmp_path):
    """A long agent call (a model turn, a build) must not block the window: the lock is only held to save."""
    pj = tmp_path / "p.json"
    Pipeline("p").save(pj)
    inside = threading.Event()
    release = threading.Event()
    got = []

    def compute(p):
        inside.set()
        assert release.wait(5)                   # while this blocks, another writer must be able to take the lock
        p.rename_node(next(iter(p.nodes)), "later") if p.nodes else None
        return "done"

    t = threading.Thread(target=lambda: got.append(hl.editing_deferred(pj, compute)))
    t.start()
    assert inside.wait(5)
    with hl.project_lock(pj, wait=1.0):          # succeeds only because editing_deferred released the lock
        pass
    release.set()
    t.join(5)
    assert got == ["done"]


def test_a_deferred_edit_aborts_when_the_file_changed_meanwhile(tmp_path):
    pj = tmp_path / "p.json"
    p = Pipeline("p")
    p.add_node("load_file", params={"path": "data.csv"}, id="src")
    p.save(pj)

    def compute(pp):
        other = Pipeline.load(pj)                # another writer lands while we compute
        other.rename_node("src", "renamed elsewhere")
        other.save()
        return "result"

    with pytest.raises(hl.ProjectBusy):
        hl.editing_deferred(pj, compute)
    assert Pipeline.load(pj).nodes["src"].title == "renamed elsewhere"   # the other change is not clobbered


def test_the_lock_nests_on_one_thread_and_excludes_others(tmp_path):
    pj = tmp_path / "p.json"
    Pipeline().save(pj)
    seen = []

    def other():
        try:
            with hl.project_lock(pj, wait=0.2):
                seen.append("other got it")
        except hl.ProjectBusy:
            seen.append("other waited")
    with hl.project_lock(pj):
        with hl.project_lock(pj, wait=0.2):              # a save inside an edit
            t = threading.Thread(target=other); t.start(); t.join()
    assert seen == ["other waited"]
    with hl.project_lock(pj, wait=0.2):
        pass
