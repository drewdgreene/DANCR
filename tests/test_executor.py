import time
import polars as pl

from dancr.core import Pipeline
from dancr.core.executor import Executor


def test_cache_and_invalidation(pipe, probe_dir):
    k = pipe.add_node("keep_rows", params={"conditions": {"match": "all", "rules": [{"column": "pressure_psi", "op": "gt", "value": "2250"}]}})
    pipe.connect("a", k.id)
    ex = Executor(pipe)
    r1 = ex.run()
    assert all(s.status == "done" and not s.from_cache for s in r1.values())
    r2 = ex.run()
    assert all(s.from_cache for s in r2.values())
    # change a setting downstream: only that node recomputes
    pipe.set_params(k.id, conditions={"match": "all", "rules": [{"column": "pressure_psi", "op": "lt", "value": "2250"}]})
    r3 = ex.run()
    assert r3["a"].from_cache and r3["b"].from_cache and not r3[k.id].from_cache
    assert r3[k.id].rows + r2[k.id].rows == r3["a"].rows
    # touching the source file invalidates everything downstream
    src = probe_dir / "probe_A.csv"
    src.touch()
    time.sleep(0.01)
    r4 = ex.run()
    assert not r4["a"].from_cache and not r4[k.id].from_cache and r4["b"].from_cache
    # gc keeps only current outputs
    ex.gc(keep_per_node=0)
    parquets = list(ex.node_dir(k.id).glob("*.parquet"))
    assert len(parquets) == 1


def test_failed_node_blocks_downstream_and_retries(pipe):
    c = pipe.add_node("calculate", params={"formulas": [{"name": "z", "expr": "nope"}]}); pipe.connect("a", c.id)
    s = pipe.add_node("sort", params={"columns": ["z"]}); pipe.connect(c.id, s.id)
    ex = Executor(pipe)
    r = ex.run()
    assert r[c.id].status == "failed" and r[s.id].status == "failed" and "Waiting on" in r[s.id].error
    assert ex.state(c.id).status == "failed"
    pipe.set_params(c.id, formulas=[{"name": "z", "expr": "pressure_psi * 2"}])
    r = ex.run()
    assert r[c.id].status == "done" and r[s.id].status == "done"


def test_schema_without_running(pipe):
    c = pipe.add_node("calculate", params={"formulas": [{"name": "z", "expr": "pressure_psi * 2"}]}); pipe.connect("a", c.id)
    ex = Executor(pipe)
    sch = ex.schema(c.id)
    assert sch is not None and "z" in sch and sch["z"] == pl.Float64
    df, res, kind = ex.preview(c.id, rows=100)
    assert len(df) == 100 and "z" in df.columns


def test_run_targets_only_needed(pipe):
    k = pipe.add_node("keep_rows"); pipe.connect("a", k.id)
    ex = Executor(pipe)
    r = ex.run(targets=[k.id])
    assert set(r) == {"a", k.id}


def test_untitled_cache_lives_on_a_real_disk_and_is_swept(probe_dir, tmp_path, monkeypatch):
    import os
    from dancr.core.executor import sweep_untitled_caches
    from dancr.logsetup import untitled_cache_root
    monkeypatch.setenv("DANCR_HOME", str(tmp_path))
    p = Pipeline("scratch")
    p.add_node("load_file", "A", {"path": str(probe_dir / "probe_A.csv")}, id="a")
    ex = Executor(p)
    assert ex.cache_dir.is_relative_to(untitled_cache_root())          # the user cache folder, never the system temp folder
    ex.run()
    assert (ex.cache_dir / "owner.pid").read_text() == str(os.getpid())
    dead = untitled_cache_root() / "untitled-deadbeef"; dead.mkdir(parents=True); (dead / "owner.pid").write_text("999999999")
    stale = untitled_cache_root() / "untitled-noowner"; stale.mkdir()
    assert sweep_untitled_caches() == 2
    assert ex.cache_dir.exists() and not dead.exists() and not stale.exists()


def test_disk_full_is_reported_as_disk_full(probe_dir, tmp_path, monkeypatch):

    p = Pipeline("d"); p.add_node("load_file", "A", {"path": str(probe_dir / "probe_A.csv")}, id="a"); p.path = tmp_path / "d.json"
    ex = Executor(p)
    monkeypatch.setattr(pl.LazyFrame, "sink_parquet", lambda self, *a, **k: (_ for _ in ()).throw(OSError("underlying IO error: Disk quota exceeded (os error 122)")))
    st = ex.run()["a"]
    assert st.status == "failed" and "full" in st.error and "memory" not in st.error.lower()
    assert str(ex.cache_dir) in st.error
