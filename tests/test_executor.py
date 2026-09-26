"""Running a project: what runs, failures, previews and samples, concurrent runs and the disk."""
import os
import threading
from pathlib import Path

import polars as pl
import pytest

from dancr.core import Pipeline, PipelineError
from dancr.core.executor import Executor


def saved_table(tmp_path: Path, name: str = "p.json") -> Pipeline:
    pl.DataFrame({"x": [1.0, 2.0, 3.0], "g": ["a", "b", "c"]}).write_csv(tmp_path / "in.csv")
    p = Pipeline("t")
    p.add_node("load_file", "Load", {"path": "in.csv"}, id="src")
    p.path = tmp_path / name
    p.save()
    return p


def _pipe_rows(tmp_path, n):
    pl.DataFrame({"x": [float(i) for i in range(n)]}).write_parquet(tmp_path / "n.parquet")
    p = Pipeline("t"); p.add_node("load_file", params={"path": "n.parquet"}, id="src"); p.path = tmp_path / "p.json"
    return p


def pipe_with(tmp_path, df, name="t.parquet"):
    f = tmp_path / name
    df.write_parquet(f)
    p = Pipeline()
    p.path = tmp_path / "p.json"
    p.add_node("load_file", params={"path": str(f)}, id="src")
    return p


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
    st = src.stat()
    os.utime(src, ns=(st.st_atime_ns, st.st_mtime_ns + 2_000_000_000))   # a clearly later time, whatever the filesystem's resolution
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


def test_output_root_refuses_writes_outside(tmp_path):
    p = saved_table(tmp_path)
    p.add_node("export", params={"path": "../escaped.csv"}, id="ex")
    p.connect("src", "ex")
    st = Executor(p, output_root=tmp_path).run()["ex"]
    assert st.status == "failed" and "inside the project folder" in st.error
    assert not (tmp_path.parent / "escaped.csv").exists()
    p.set_params("ex", path=str(tmp_path / "sub" / "fine.csv"))
    assert Executor(p, output_root=tmp_path).run()["ex"].status == "done"


def test_preview_sampling_never_returns_more_than_asked(tmp_path):
    p = _pipe_rows(tmp_path, 15)
    ex = Executor(p)
    ex.run()
    lf, kind = ex.sample_frame("src", 10)
    assert kind == "spread" and lf.collect().height <= 10


def test_schema_and_preview_handle_a_diamond_without_a_false_loop(tmp_path):
    """Two branches sharing one not-yet-run ancestor must not be mistaken for a loop."""
    p = pipe_with(tmp_path, pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]}))
    p.add_node("calculate", params={"formulas": [{"name": "u", "expr": "x + 1"}]}, id="c1")
    p.connect("src", "c1")
    p.add_node("calculate", params={"formulas": [{"name": "v", "expr": "y + 1"}]}, id="c2")
    p.connect("src", "c2")
    p.add_node("combine", params={"method": "match", "on": ["x"], "right_on": ["x"], "how": "inner"}, id="cb")
    p.connect("c1", "cb", "left")
    p.connect("c2", "cb", "right")
    ex = Executor(p)
    schema = ex.schema("cb")
    assert schema is not None and {"x", "y", "u", "v"} <= set(schema)
    df, _res, _kind = ex.preview("cb")
    assert df.height == 3


def test_preview_failure_is_typed_and_friendly(tmp_path):
    """A node that cannot run on a sample must raise PreviewUnavailable, never a raw traceback."""
    from dancr.core.executor import PreviewUnavailable

    p = pipe_with(tmp_path, pl.DataFrame({"x": [1.0], "y": [2.0]}))
    p.add_node("fit_curve", params={"x": "x", "y": "y", "kind": "linear"}, id="fit")
    p.connect("src", "fit")
    with pytest.raises(PreviewUnavailable) as ei:
        Executor(p).preview("fit")
    assert "at least two points" in str(ei.value)


def test_concurrent_runs_do_not_corrupt(probe_dir, tmp_path):
    p = Pipeline(); p.add_node("load_file", params={"path": str(probe_dir / "probe_A.csv")}, id="a")
    p.add_node("time_buckets", params={"every": "1m"}, id="tb"); p.connect("a", "tb")
    p.save(tmp_path / "p.json")
    errors = []

    def go():
        try:
            res = Executor(Pipeline.load(tmp_path / "p.json")).run(force=True)
            if any(s.status != "done" for s in res.values()):
                errors.append([s.error for s in res.values() if s.error])
        except Exception as e:
            errors.append(repr(e))
    ts = [threading.Thread(target=go) for _ in range(3)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert not errors, errors
    ex = Executor(Pipeline.load(tmp_path / "p.json"))
    st = ex.state("tb")
    assert st.status == "done"
    assert pl.read_parquet(st.output).height == st.rows      # file is fully readable


def test_run_targets_unknown_raises(pipe):
    with pytest.raises(PipelineError, match="No step called"):
        Executor(pipe).run(targets=["nope"])


def test_meta_written_atomically_and_preview_spread(pipe):
    ex = Executor(pipe); ex.run()
    assert not list(ex.node_dir("a").glob("*.tmp.json"))
    lf, kind = ex.sample_frame("a", 1000)
    assert kind == "spread"
    df = lf.collect()
    assert 900 <= len(df) <= 1100
    span = (df["time"].max() - df["time"].min()).total_seconds()
    assert span > 60 * 25       # covers most of the 30-minute file, not just the head


def test_column_stats_in_meta(pipe):
    ex = Executor(pipe); res = ex.run()
    cs = res["a"].column_stats
    assert cs["pressure_psi"]["nulls"] == 0 and cs["pressure_psi"]["min"] < cs["pressure_psi"]["max"]
    assert "min" in cs["time"]
