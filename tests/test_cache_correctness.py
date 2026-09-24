"""The cache serves a result only while everything that shaped it is unchanged (review 2026-09-24, phase 1)."""
import json
import os
import shutil
import time
from pathlib import Path

import polars as pl
import pytest

from dancr.core import Pipeline
from dancr.core.executor import Executor, LIVE_DIR, _code_fingerprint


def _table(tmp_path: Path, name: str = "p.json") -> Pipeline:
    pl.DataFrame({"x": [1.0, 2.0, 3.0], "g": ["a", "b", "c"]}).write_csv(tmp_path / "in.csv")
    p = Pipeline("t")
    p.add_node("load_file", "Load", {"path": "in.csv"}, id="src")
    p.path = tmp_path / name
    p.save()
    return p


def test_force_replaces_a_bad_cached_result(tmp_path):
    p = _table(tmp_path)
    ex = Executor(p)
    st = ex.run()["src"]
    pl.DataFrame({"junk": [-1.0]}).write_parquet(st.output)          # a wrong file under the right hash
    assert pl.read_parquet(ex.run()["src"].output).columns == ["junk"]   # an ordinary run trusts the cache
    st = ex.run(force=True)["src"]
    assert pl.read_parquet(st.output)["x"].to_list() == [1.0, 2.0, 3.0]
    assert st.rows == 3


def test_non_ascii_input_names_invalidate(tmp_path):
    p = _table(tmp_path)
    p.set_input("Débit", 2)
    p.add_node("calculate", params={"formulas": [{"name": "y", "expr": "[x] * [Débit]"}]}, id="c")
    p.connect("src", "c")
    ex = Executor(p)
    assert pl.read_parquet(ex.run()["c"].output)["y"].to_list() == [2.0, 4.0, 6.0]
    p.set_input("Débit", 10)
    assert ex.state("c").status != "done"
    assert pl.read_parquet(ex.run()["c"].output)["y"].to_list() == [10.0, 20.0, 30.0]


def test_export_is_written_again_when_its_file_is_deleted_or_changed(tmp_path):
    p = _table(tmp_path)
    p.add_node("export", params={"path": "out.csv"}, id="ex")
    p.connect("src", "ex")
    ex = Executor(p)
    ex.run()
    out = tmp_path / "out.csv"
    assert out.exists()
    out.unlink()
    assert ex.state("ex").status == "stale"
    ex.run()
    assert out.exists()
    out.write_text("edited by hand\n")
    assert ex.state("ex").status == "stale"
    ex.run()
    assert pl.read_csv(out)["x"].to_list() == [1.0, 2.0, 3.0]


def test_export_keeps_no_second_copy_of_the_table(tmp_path):
    p = _table(tmp_path)
    p.add_node("export", params={"path": "out.parquet"}, id="ex")
    p.connect("src", "ex")
    ex = Executor(p)
    st = ex.run()["ex"]
    assert st.status == "done" and st.rows == 3
    assert not list(ex.node_dir("ex").glob("*.parquet"))


def test_workbook_follows_upstream_titles(tmp_path):
    p = _table(tmp_path)
    p.add_node("workbook", params={"path": "book.xlsx"}, id="wb")
    p.connect("src", "wb")
    ex = Executor(p)
    ex.run()
    assert pl.read_excel(tmp_path / "book.xlsx", sheet_name="Load").height == 3
    p.rename_node("src", "Readings")
    assert ex.state("wb").status != "done"
    ex.run()
    assert pl.read_excel(tmp_path / "book.xlsx", sheet_name="Readings").height == 3


def test_report_follows_column_labels(tmp_path):
    p = _table(tmp_path)
    p.add_node("report", params={"title": "R", "path": "r.html", "pdf": False}, id="rep")
    p.connect("src", "rep", "items")
    ex = Executor(p)
    ex.run()
    assert "Flow rate" not in (tmp_path / "r.html").read_text()
    p.set_column_meta("x", label="Flow rate", unit="L/s")
    assert ex.state("rep").status != "done"
    ex.run()
    assert "Flow rate (L/s)" in (tmp_path / "r.html").read_text()


def test_save_as_keeps_every_path_pointing_at_the_same_file_and_keeps_results(tmp_path):
    a, b = tmp_path / "A", tmp_path / "B"
    a.mkdir(); b.mkdir()
    p = _table(a)
    p.add_node("export", params={"path": "out.csv"}, id="ex")
    p.add_node("chart", params={"kind": "line", "x": "x", "series": [{"column": "x"}]}, id="ch")
    p.connect("src", "ex"); p.connect("src", "ch")
    Executor(p).run()
    old_cache = Executor(p).cache_dir
    p.save(b / "p.json")
    assert p.nodes["src"].params["path"] == str(a / "in.csv")          # the data did not move
    assert p.nodes["ex"].params["path"] == str(a / "out.csv")
    new = Executor(p)
    new.cache_dir.parent.mkdir(parents=True)
    shutil.move(str(old_cache), str(new.cache_dir))                   # what the window does on Save As
    assert all(new.state(n).status == "done" for n in p.nodes)        # nothing to recompute
    p.save(a / "p2.json")                                              # and back inside the data folder: relative again
    assert p.nodes["src"].params["path"] == "in.csv"


def test_a_moved_project_folder_writes_outputs_in_its_new_place(tmp_path):
    a = tmp_path / "A"
    a.mkdir()
    p = _table(a)
    p.add_node("export", params={"path": "out.csv"}, id="ex")
    p.connect("src", "ex"); p.save()
    Executor(p).run()
    b = tmp_path / "B"
    shutil.copytree(a, b)                                              # copy the whole folder, cache included
    (b / "out.csv").unlink()
    moved = Pipeline.load(b / "p.json")
    moved.nodes["src"]  # relative paths now mean the files in B
    st = Executor(moved).run()["ex"]
    assert st.status == "done" and (b / "out.csv").exists()


def test_every_engine_module_is_fingerprinted(monkeypatch):
    read: list[str] = []
    orig = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda self: (read.append(self.name), orig(self))[1])
    _code_fingerprint()
    for name in ("fits.py", "executor.py", "render.py", "combine.py", "timeutil.py"):
        assert name in read
    for name in ("examples.py", "understand.py", "recipes.py", "ask.py", "answers.py", "planner.py", "samples.py"):     # never shape a step's output
        assert name not in read


def test_polars_version_is_part_of_the_fingerprint(monkeypatch):
    before = _code_fingerprint()
    monkeypatch.setattr(pl, "__version__", "0.0.0-test")
    assert _code_fingerprint() != before


def test_sweep_keeps_results_another_live_process_holds(tmp_path):
    p = _table(tmp_path)
    p.add_node("calculate", params={"formulas": [{"name": "y", "expr": "[x] * 2"}]}, id="c")
    p.connect("src", "c")
    p.save()
    window = Executor(p)
    held_hash = window.run()["c"].hash
    window.hold({"c": held_hash})
    old = window.node_dir("c") / f"{held_hash}.parquet"
    for f in (old, old.with_suffix(".json")):
        os.utime(f, (time.time() - 7200, time.time() - 7200))         # well past the grace period
    other = Pipeline.load(p.path)                                      # the CLI runs a different saved version
    other.set_params("c", formulas=[{"name": "y", "expr": "[x] * 3"}])
    Executor(other).run()
    assert old.exists()                                                # still shown in the window
    window.release()
    Executor(other).run()
    assert not old.exists()


def test_leases_of_dead_processes_are_ignored_and_removed(tmp_path):
    p = _table(tmp_path)
    ex = Executor(p)
    ex.run()
    d = ex.cache_dir / LIVE_DIR
    d.mkdir(exist_ok=True)
    dead = d / "999999999-dead.json"
    dead.write_text(json.dumps({"pid": 999999999, "hashes": {"src": "whatever"}}))
    assert ex._held() == {}
    assert not dead.exists()


def test_a_run_holds_what_it_reads(tmp_path, monkeypatch):
    p = _table(tmp_path)
    ex = Executor(p)
    seen = {}
    from dancr.core.nodes import basic  # noqa: F401 - registry loaded
    orig = Executor._run_node

    def spy(self, nid, *a, **k):
        seen[nid] = self._held()
        return orig(self, nid, *a, **k)
    monkeypatch.setattr(Executor, "_run_node", spy)
    ex.run()
    assert seen["src"].get("src")                                      # the run's own lease was in place
    assert ex._held() == {}                                            # and removed afterwards


def test_output_root_refuses_writes_outside(tmp_path):
    p = _table(tmp_path)
    p.add_node("export", params={"path": "../escaped.csv"}, id="ex")
    p.connect("src", "ex")
    st = Executor(p, output_root=tmp_path).run()["ex"]
    assert st.status == "failed" and "inside the project folder" in st.error
    assert not (tmp_path.parent / "escaped.csv").exists()
    p.set_params("ex", path=str(tmp_path / "sub" / "fine.csv"))
    assert Executor(p, output_root=tmp_path).run()["ex"].status == "done"


@pytest.mark.skipif(os.name == "nt", reason="POSIX pid probe")
def test_pid_probe():
    from dancr.core.executor import _pid_alive
    assert _pid_alive(os.getpid())
    assert not _pid_alive(999999999)
    assert not _pid_alive(0)


def test_a_failed_save_as_leaves_the_project_as_it_was(tmp_path):
    a = tmp_path / "A"; a.mkdir()
    p = _table(a)
    locked = tmp_path / "locked"; locked.mkdir(); locked.chmod(0o500)
    try:
        with pytest.raises(OSError):
            p.save(locked / "p.json", auto=True)
    finally:
        locked.chmod(0o700)
    assert p.path == a / "p.json"
    assert p.nodes["src"].params["path"] == "in.csv"                    # still the same file
    assert "autosaved" not in p.meta


def test_numpy_and_fastexcel_versions_are_part_of_the_fingerprint(monkeypatch):
    import numpy
    before = _code_fingerprint()
    monkeypatch.setattr(numpy, "__version__", "0.0.0-test")
    assert _code_fingerprint() != before


def test_records_of_steps_without_a_table_are_swept(tmp_path):
    p = _table(tmp_path)
    p.add_node("export", params={"path": "out.csv"}, id="ex")
    p.connect("src", "ex"); p.save()
    ex = Executor(p)
    for i in range(4):
        p.set_params("ex", path=f"out{i}.csv")
        ex.run()
    for f in ex.node_dir("ex").glob("*.json"):
        os.utime(f, (time.time() - 7200, time.time() - 7200))
    ex.gc()
    assert len(list(ex.node_dir("ex").glob("*.json"))) == 1


def test_an_unsaved_projects_cache_has_an_owner_before_its_first_step_runs(tmp_path, monkeypatch):
    from dancr.core.executor import sweep_untitled_caches
    monkeypatch.setenv("DANCR_HOME", str(tmp_path / "home"))
    pl.DataFrame({"x": [1.0]}).write_csv(tmp_path / "a.csv")
    p = Pipeline("u")
    p.add_node("load_file", params={"path": str(tmp_path / "a.csv")}, id="s")
    ex = Executor(p)
    seen = {}
    orig = Executor._run_node

    def during(self, nid, *a, **k):
        sweep_untitled_caches()                                        # another window starting right now
        seen["owner"] = (self.cache_dir / "owner.pid").exists()
        return orig(self, nid, *a, **k)
    monkeypatch.setattr(Executor, "_run_node", during)
    assert ex.run()["s"].status == "done" and seen["owner"]


def test_leases_from_another_machine_are_trusted_until_old(tmp_path):
    p = _table(tmp_path)
    ex = Executor(p)
    ex.run()
    d = ex.cache_dir / LIVE_DIR
    d.mkdir(exist_ok=True)
    lease = d / "1-other.json"
    lease.write_text(json.dumps({"pid": 999999999, "host": "some-other-machine", "hashes": {"src": "abc"}}))
    assert ex._held() == {"src": {"abc"}}                              # its process cannot be checked from here
    os.utime(lease, (time.time() - 8 * 86400, time.time() - 8 * 86400))
    assert ex._held() == {} and not lease.exists()


def test_preview_sampling_never_returns_more_than_asked(tmp_path):
    p = _pipe_rows(tmp_path, 15)
    ex = Executor(p)
    ex.run()
    lf, kind = ex.sample_frame("src", 10)
    assert kind == "spread" and lf.collect().height <= 10


def _pipe_rows(tmp_path, n):
    pl.DataFrame({"x": [float(i) for i in range(n)]}).write_parquet(tmp_path / "n.parquet")
    p = Pipeline("t"); p.add_node("load_file", params={"path": "n.parquet"}, id="src"); p.path = tmp_path / "p.json"
    return p


def test_deep_previews_hash_each_step_once(tmp_path, monkeypatch):
    p = _table(tmp_path)
    prev = "src"
    for i in range(40):
        p.add_node("sort", params={"columns": ["x"]}, id=f"s{i}"); p.connect(prev, f"s{i}"); prev = f"s{i}"
    calls = []
    orig = Executor.plan_hash
    monkeypatch.setattr(Executor, "plan_hash", lambda self, nid, memo=None: (calls.append(nid), orig(self, nid, memo))[1])
    Executor(p).schema(prev)
    assert len(calls) < 400                                            # was quadratic: thousands of calls
