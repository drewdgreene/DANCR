"""The cache: what goes into a step's hash, where results live, leases, sweeping, and when a result is stale."""
import json
import os
import shutil
import time
from pathlib import Path

import fastexcel
import polars as pl
import pytest

from dancr.core import Pipeline
from dancr.core.executor import Executor, LIVE_DIR, _code_fingerprint, default_cache_dir, engine_files


def pipe_on_parquet(tmp_path, df=None, name="p.json"):
    f = tmp_path / "t.parquet"
    if not f.exists():
        (df if df is not None else pl.DataFrame({"x": [1.0, 2.0, 3.0]})).write_parquet(f)
    p = Pipeline(); p.path = tmp_path / name
    p.add_node("load_file", params={"path": str(f)}, id="src")
    return p


def table(rows: list[list], columns: list[tuple[str, str]]) -> Pipeline:
    p = Pipeline("t")
    p.add_node("enter_data", params={"columns": [{"name": n, "type": t} for n, t in columns], "rows": rows}, id="d")
    return p


def saved_table(tmp_path: Path, name: str = "p.json") -> Pipeline:
    pl.DataFrame({"x": [1.0, 2.0, 3.0], "g": ["a", "b", "c"]}).write_csv(tmp_path / "in.csv")
    p = Pipeline("t")
    p.add_node("load_file", "Load", {"path": "in.csv"}, id="src")
    p.path = tmp_path / name
    p.save()
    return p


def test_the_fingerprint_follows_what_steps_import():
    paths = [f.as_posix() for f in engine_files()]
    # must follow the modules a run executes, into packages (expr, geo) and the node packages (inquiry)
    for must in ("core/executor.py", "core/expr/__init__.py", "core/expr/_parse.py", "core/fits.py",
                 "core/geo/__init__.py", "core/geo/_world.py", "views/render.py", "views/stats.py",
                 "nodes/combine.py", "nodes/inquiry/quality.py", "views/lod.py"):
        assert any(p.endswith(must) for p in paths), must
    for never in ("understand.py", "recipes.py", "ask.py", "answers.py", "planner.py", "samples.py", "examples.py",
                  "mainwindow.py", "cli.py", "mcp_server.py", "headless.py"):
        assert not any(p.endswith("/" + never) for p in paths), never
    assert not any("/ui/" in f.as_posix() for f in engine_files())


def test_only_setting_values_name_inputs(tmp_path):
    p = pipe_on_parquet(tmp_path)
    p.set_input("max", 5); p.set_input("value", 1); p.set_input("limit", 2)
    p.add_node("check_limits", params={"column": "x", "min": "0", "max": "limit"}, id="c")
    p.connect("src", "c")
    ex = Executor(p)
    assert ex.inputs_used("c") == {"limit": 2}                         # the key "max" is not the input "max"
    h = ex.plan_hash("c")
    p.set_input("max", 50)
    assert Executor(p).plan_hash("c") == h


def test_a_step_is_handed_only_the_inputs_its_hash_covers(tmp_path):
    p = pipe_on_parquet(tmp_path)
    p.set_input("factor", 3); p.set_input("other", 9)
    p.add_node("calculate", params={"formulas": [{"name": "y", "expr": "x * [Factor]"}]}, id="c")
    p.connect("src", "c")
    ex = Executor(p)
    assert ex._ctx("c", preview=False).inputs == {"factor": 3}
    assert pl.read_parquet(ex.run()["c"].output)["y"].to_list() == [3.0, 6.0, 9.0]


def test_a_report_draws_its_charts_limit_lines_from_inputs(tmp_path):
    p = pipe_on_parquet(tmp_path)
    p.set_input("ceiling", 2.5)
    p.add_node("chart", params={"kind": "line", "series": [{"column": "x"}], "limits": [{"value": "ceiling", "label": "top"}]}, id="ch")
    p.connect("src", "ch")
    p.add_node("report", params={"title": "R", "path": "r.html"}, id="r")
    p.connect("ch", "r", "items")
    ex = Executor(p)
    assert ex.inputs_used("r") == {"ceiling": 2.5}
    assert ex.run()["r"].status == "done"


def test_two_projects_in_one_folder_never_share_results(tmp_path):
    a, b = Pipeline(), Pipeline()
    a.path, b.path = tmp_path / "plant.json", tmp_path / "plant.dancr.json"
    c = Pipeline(); c.path = tmp_path / "plant.dancr"
    dirs = {default_cache_dir(x) for x in (a, b, c)}
    assert len(dirs) == 3


def test_a_same_size_replacement_with_the_old_time_is_noticed(tmp_path):
    f = tmp_path / "in.csv"
    f.write_text("x\n1\n2\n")
    p = Pipeline(); p.path = tmp_path / "p.json"
    p.add_node("load_file", params={"path": "in.csv"}, id="src")
    first = Executor(p).run()["src"]
    st = f.stat()
    f.write_text("x\n7\n8\n")                                          # same size
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns))                   # and the old modification time (cp -p)
    again = Executor(p).run()["src"]
    assert again.hash != first.hash and pl.read_parquet(again.output)["x"].to_list() == [7, 8]


def test_a_big_source_is_sampled_not_read_whole(tmp_path, monkeypatch):
    from dancr.core import executor
    f = tmp_path / "big.bin"
    f.write_bytes(os.urandom(1_000_000))
    read = []
    real_open = open

    def spy(path, mode="r", *a, **k):
        fh = real_open(path, mode, *a, **k)
        orig = fh.read
        fh.read = lambda n=-1: (read.append(n), orig(n))[1]
        return fh
    monkeypatch.setattr(executor, "open", spy, raising=False)
    executor._content_sample(f, f.stat().st_size)
    assert read and all(0 < n <= 64 * 1024 for n in read)


def test_a_sweep_puts_back_a_result_a_new_run_took_meanwhile(tmp_path, monkeypatch):
    p = pipe_on_parquet(tmp_path)
    p.add_node("calculate", params={"formulas": [{"name": "y", "expr": "x * 2"}]}, id="c")
    p.connect("src", "c"); p.save()
    old_hash = Executor(p).run()["c"].hash
    old = Executor(p).node_dir("c") / f"{old_hash}.parquet"
    p.set_params("c", formulas=[{"name": "y", "expr": "x * 3"}])
    ex = Executor(p)
    ex.run()                                                           # the old result is kept a while for undo
    assert old.exists()
    calls = []
    real = Executor._held

    def held(self):
        calls.append(1)
        # another process starts a run of the old version between the sweep's two looks at the leases
        return real(self) if len(calls) == 1 else {"c": {old_hash}}
    monkeypatch.setattr(Executor, "_held", held)
    ex.gc(grace_seconds=0)
    assert old.exists() and old.with_suffix(".json").exists()
    monkeypatch.setattr(Executor, "_held", real)
    ex.gc(grace_seconds=0)
    assert not old.exists()
    assert not [f for f in ex.node_dir("c").iterdir() if ".gc." in f.name]


def test_an_input_named_on_its_own_line_of_a_formula_reruns_the_step(tmp_path):
    p = table([[1], [2]], [("x", "number")])
    p.path = tmp_path / "p.json"
    p.set_input("k", 2)
    p.add_node("calculate", params={"formulas": [{"name": "y", "expr": "[x] *\nk"}]}, id="c"); p.connect("d", "c")
    before = Executor(p).plan_hash("c")
    p.set_input("k", 10)
    assert Executor(p).plan_hash("c") != before


def test_force_replaces_a_bad_cached_result(tmp_path):
    p = saved_table(tmp_path)
    ex = Executor(p)
    st = ex.run()["src"]
    pl.DataFrame({"junk": [-1.0]}).write_parquet(st.output)          # a wrong file under the right hash
    assert pl.read_parquet(ex.run()["src"].output).columns == ["junk"]   # an ordinary run trusts the cache
    st = ex.run(force=True)["src"]
    assert pl.read_parquet(st.output)["x"].to_list() == [1.0, 2.0, 3.0]
    assert st.rows == 3


def test_non_ascii_input_names_invalidate(tmp_path):
    p = saved_table(tmp_path)
    p.set_input("Débit", 2)
    p.add_node("calculate", params={"formulas": [{"name": "y", "expr": "[x] * [Débit]"}]}, id="c")
    p.connect("src", "c")
    ex = Executor(p)
    assert pl.read_parquet(ex.run()["c"].output)["y"].to_list() == [2.0, 4.0, 6.0]
    p.set_input("Débit", 10)
    assert ex.state("c").status != "done"
    assert pl.read_parquet(ex.run()["c"].output)["y"].to_list() == [10.0, 20.0, 30.0]


def test_export_is_written_again_when_its_file_is_deleted_or_changed(tmp_path):
    p = saved_table(tmp_path)
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
    p = saved_table(tmp_path)
    p.add_node("export", params={"path": "out.parquet"}, id="ex")
    p.connect("src", "ex")
    ex = Executor(p)
    st = ex.run()["ex"]
    assert st.status == "done" and st.rows == 3
    assert not list(ex.node_dir("ex").glob("*.parquet"))


def test_workbook_follows_upstream_titles(tmp_path):
    p = saved_table(tmp_path)
    p.add_node("workbook", params={"path": "book.xlsx"}, id="wb")
    p.connect("src", "wb")
    ex = Executor(p)
    ex.run()
    assert fastexcel.read_excel(str(tmp_path / "book.xlsx")).load_sheet("Load").height == 3
    p.rename_node("src", "Readings")
    assert ex.state("wb").status != "done"
    ex.run()
    assert fastexcel.read_excel(str(tmp_path / "book.xlsx")).load_sheet("Readings").height == 3


def test_report_follows_column_labels(tmp_path):
    p = saved_table(tmp_path)
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
    p = saved_table(a)
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
    p = saved_table(a)
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
    p = saved_table(tmp_path)
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
    p = saved_table(tmp_path)
    ex = Executor(p)
    ex.run()
    d = ex.cache_dir / LIVE_DIR
    d.mkdir(exist_ok=True)
    dead = d / "999999999-dead.json"
    dead.write_text(json.dumps({"pid": 999999999, "hashes": {"src": "whatever"}}))
    assert ex._held() == {}
    assert not dead.exists()


def test_a_run_holds_what_it_reads(tmp_path, monkeypatch):
    p = saved_table(tmp_path)
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


@pytest.mark.skipif(os.name == "nt", reason="POSIX pid probe")
def test_pid_probe():
    from dancr.core.executor import _pid_alive
    assert _pid_alive(os.getpid())
    assert not _pid_alive(999999999)
    assert not _pid_alive(0)


def test_numpy_and_fastexcel_versions_are_part_of_the_fingerprint(monkeypatch):
    import numpy
    before = _code_fingerprint()
    monkeypatch.setattr(numpy, "__version__", "0.0.0-test")
    assert _code_fingerprint() != before


def test_records_of_steps_without_a_table_are_swept(tmp_path):
    p = saved_table(tmp_path)
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
    p = saved_table(tmp_path)
    ex = Executor(p)
    ex.run()
    d = ex.cache_dir / LIVE_DIR
    d.mkdir(exist_ok=True)
    lease = d / "1-other.json"
    lease.write_text(json.dumps({"pid": 999999999, "host": "some-other-machine", "hashes": {"src": "abc"}}))
    assert ex._held() == {"src": {"abc"}}                              # its process cannot be checked from here
    os.utime(lease, (time.time() - 8 * 86400, time.time() - 8 * 86400))
    assert ex._held() == {} and not lease.exists()



def test_leases_from_a_sandbox_on_this_machine_are_not_judged_by_pid(tmp_path):
    """A Flatpak sandbox numbers its processes apart from the host: pid 999999999 there may well be alive."""
    from dancr.core import executor
    p = saved_table(tmp_path)
    ex = Executor(p)
    ex.run()
    d = ex.cache_dir / LIVE_DIR
    d.mkdir(exist_ok=True)
    host = executor._HOST.split(" ")[0]
    lease = d / "1-sandbox.json"
    lease.write_text(json.dumps({"pid": 999999999, "host": f"{host} pid:[1]", "hashes": {"src": "abc"}}))
    assert ex._held() == {"src": {"abc"}}

def test_deep_previews_hash_each_step_once(tmp_path, monkeypatch):
    p = saved_table(tmp_path)
    prev = "src"
    for i in range(40):
        p.add_node("sort", params={"columns": ["x"]}, id=f"s{i}"); p.connect(prev, f"s{i}"); prev = f"s{i}"
    calls = []
    orig = Executor.plan_hash
    monkeypatch.setattr(Executor, "plan_hash", lambda self, nid, memo=None: (calls.append(nid), orig(self, nid, memo))[1])
    Executor(p).schema(prev)
    assert len(calls) < 400                                            # was quadratic: thousands of calls


def test_hash_not_memoised_across_edits(pipe):
    ex = Executor(pipe); ex.run()
    assert ex.state("a").status == "done"
    pipe.set_params("a", skip_rows=1)
    assert ex.state("a").status == "stale"


def test_gc_survives_missing_files(pipe):
    ex = Executor(pipe); ex.run()
    nd = ex.node_dir("a")
    for f in nd.glob("*.parquet"):
        f.unlink()
    ex.gc()  # must not raise
    assert ex.state("a").status == "stale"


def test_a_frozen_app_uses_the_fingerprint_its_build_baked_in(monkeypatch, tmp_path):
    from dancr.core import executor
    baked = tmp_path / "fingerprint.txt"
    baked.write_text("abc123def456\n")
    monkeypatch.setattr(executor.sys, "frozen", True, raising=False)
    monkeypatch.setattr(executor, "BAKED_FINGERPRINT", baked)
    assert _code_fingerprint() == "abc123def456"
    baked.unlink()
    with pytest.raises(RuntimeError, match="incomplete"):
        _code_fingerprint()


def test_clearing_the_cache_keeps_what_another_live_process_holds(tmp_path):
    p = saved_table(tmp_path)
    p.add_node("calculate", params={"formulas": [{"name": "y", "expr": "[x] * 2"}]}, id="c")
    p.connect("src", "c")
    p.save()
    window = Executor(p)
    states = window.run()
    window.hold({"c": states["c"].hash})
    held = window.node_dir("c") / f"{states['c'].hash}.parquet"
    src = window.node_dir("src") / f"{states['src'].hash}.parquet"
    Executor(Pipeline.load(p.path)).clear_cache()                        # dancr clear-cache while the window shows c
    assert held.exists() and not src.exists()
    window.clear_cache()                                                 # the window's own clear lets go of its hold
    assert not held.exists()


def test_a_read_holds_the_result_until_it_is_done(tmp_path):
    from dancr import headless as hl
    p = saved_table(tmp_path)
    ex = Executor(p)
    h = ex.run()["src"].hash
    with hl.result_frame(p, Executor(p), "src", run=False) as lf:
        Executor(Pipeline.load(p.path)).clear_cache()                    # another process clears meanwhile
        assert lf.collect().height > 0
    Executor(Pipeline.load(p.path)).clear_cache()
    assert not (ex.node_dir("src") / f"{h}.parquet").exists()
