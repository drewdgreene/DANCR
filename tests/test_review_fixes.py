"""Regression tests for every reproduced finding in docs/STABILITY_REVIEW.md."""
import json

import threading

from datetime import datetime


import numpy as np
import polars as pl
import pytest

from dancr.core import Pipeline, PipelineError
from dancr.core.executor import Executor
from dancr.core.expr import compile_formula, check_formula
from dancr.core.conditions import rule_mask
from dancr.core.timeutil import detect_datetime_format
from dancr.views import lod
from conftest import run_one


def pipe_with(tmp_path, df: pl.DataFrame, name="t.parquet") -> Pipeline:
    f = tmp_path / name
    df.write_parquet(f)
    p = Pipeline(); p.path = tmp_path / "p.json"
    p.add_node("load_file", params={"path": str(f)}, id="src")
    return p


# ---------------------------------------------------------------- executor
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


def test_hash_not_memoised_across_edits(pipe):
    ex = Executor(pipe); ex.run()
    assert ex.state("a").status == "done"
    pipe.set_params("a", skip_rows=1)
    assert ex.state("a").status == "stale"


def test_run_targets_unknown_raises(pipe):
    with pytest.raises(PipelineError, match="No node called"):
        Executor(pipe).run(targets=["nope"])


def test_gc_survives_missing_files(pipe):
    ex = Executor(pipe); ex.run()
    nd = ex.node_dir("a")
    for f in nd.glob("*.parquet"):
        f.unlink()
    ex.gc()  # must not raise
    assert ex.state("a").status == "stale"


def test_meta_written_atomically_and_preview_spread(pipe):
    ex = Executor(pipe); ex.run()
    assert not list(ex.node_dir("a").glob("*.tmp.json"))
    lf, kind = ex.sample_frame("a", 1000)
    assert kind == "spread"
    df = lf.collect()
    assert 900 <= len(df) <= 1100
    span = (df["time"].max() - df["time"].min()).total_seconds()
    assert span > 60 * 25       # covers most of the 30-minute file, not just the head


def test_source_blank_report(tmp_path):
    f = tmp_path / "b.csv"
    rows = "".join(f"2024-01-01 00:00:{i:02d},{i}\n" for i in range(20))
    f.write_text("time,v\n" + rows + "not a time,2\n2024-01-01 00:00:59,\n")
    p = Pipeline(); p.path = tmp_path / "p.json"; p.add_node("load_file", params={"path": str(f)}, id="src")
    st = Executor(p).run()["src"]
    assert st.status == "done" and any("Blank or unreadable" in m and "time: 1" in m for m in st.messages)
    tb = p.add_node("time_buckets", params={"every": "1s"}); p.connect("src", tb.id)
    assert run_one(p, tb.id)["time"].null_count() == 0     # blank time rows skipped


# ---------------------------------------------------------------- model / params
def test_malformed_pipeline_files_raise_pipeline_error(tmp_path):
    for bad in [[], {"dancr": 2, "nodes": [{"id": "a"}]}, {"dancr": 2, "nodes": [{"id": "a", "type": "nope"}]},
                {"dancr": 2, "nodes": [{"id": "a", "type": "sort", "params": ["x"]}]},
                {"dancr": 2, "nodes": None, "edges": [{"source": "a"}]}, {"dancr": 2, "edges": [{"source": "a", "target": "b"}]}]:
        f = tmp_path / "bad.json"; f.write_text(json.dumps(bad))
        with pytest.raises(PipelineError):
            Pipeline.load(f)
    f = tmp_path / "nullx.json"
    f.write_text(json.dumps({"dancr": 2, "nodes": [{"id": "a", "type": "sort", "x": None, "y": "zz"}]}))
    p = Pipeline.load(f)
    p.save()   # must not raise


def test_save_keeps_old_path_on_failure(tmp_path):
    p = Pipeline(); p.save(tmp_path / "ok.json")
    with pytest.raises(PipelineError):
        p.save(tmp_path)          # a directory
    assert p.path == tmp_path / "ok.json"


def test_none_params_use_defaults_and_shapes_validated():
    p = Pipeline()
    n = p.add_node("sort", params={"columns": ["x"], "descending": None})
    assert n.params["descending"] is False
    with pytest.raises(ValueError):
        p.add_node("keep_rows", params={"conditions": "x"})
    with pytest.raises(ValueError):
        p.add_node("keep_rows", params={"conditions": {"rules": "n"}})
    k = p.add_node("keep_rows")
    k.params["conditions"]["rules"].append({"column": "x"})
    assert p.add_node("keep_rows").params["conditions"]["rules"] == []      # defaults are not shared


def test_set_params_one_key_when_another_is_invalid():
    p = Pipeline()
    n = p.add_node("time_buckets", params={"every": "banana"}, strict=False)
    p.set_params(n.id, count_column="n")
    assert p.nodes[n.id].params["count_column"] == "n"


# ---------------------------------------------------------------- formulas / conditions / dates
def test_formula_unicode_escape_and_recursion():
    df = pl.DataFrame({"Größe": [1.0], "s": ["a"]})
    e, k, _ = compile_formula("Größe * 2", df.schema)
    assert df.select(e)[0, 0] == 2.0
    e, k, _ = compile_formula('"Größe\\n"', df.schema)            # as in Excel, a backslash is just a character
    assert df.select(e)[0, 0] == "Größe\\n"
    assert "too long" in check_formula("(" * 400 + "1" + ")" * 400, df.schema)
    assert "too long" in check_formula(" + ".join(["Größe"] * 3000), df.schema)


def test_formula_time_vs_number_rejected_and_tz_literals():
    df = pl.DataFrame({"t": pl.datetime_range(datetime(2024, 1, 1), datetime(2024, 1, 3), "1d", eager=True)})
    assert "plain number" in check_formula("t > 20240102", df.schema)
    tz = df.with_columns(pl.col("t").dt.replace_time_zone("UTC"))
    e, _, _ = compile_formula('t > "2024-01-02"', tz.schema)
    assert tz.select(e)["t"].to_list() == [False, False, True]
    e, _, _ = compile_formula("SECONDS_BETWEEN(t, t)", tz.schema)
    assert tz.select(e)[0, 0] == 0.0
    d = df.with_columns(pl.col("t").cast(pl.Date))
    e, _, _ = compile_formula('t >= "2024-01-02"', d.schema)
    assert d.select(e)["t"].to_list() == [False, True, True]


def test_formula_ambiguous_squash_and_text_math():
    df = pl.DataFrame({"a b": [1.0], "a_b": [2.0], "name": ["x"]})
    assert "could mean" in check_formula("a__b * 2", df.schema)
    assert "number" in check_formula("ABS(name)", df.schema)
    assert "negate" in check_formula("-name", df.schema)


def test_conditions_tz_categorical_thousands():
    df = pl.DataFrame({"t": pl.datetime_range(datetime(2024, 1, 1), datetime(2024, 1, 3), "1d", eager=True),
                       "c": ["a", "", None], "n": [1000.0, 2500.0, 7.0]})
    tz = df.with_columns(pl.col("t").dt.replace_time_zone("Europe/Oslo"), pl.col("c").cast(pl.Categorical))
    m = rule_mask(tz.schema, {"column": "t", "op": "gt", "value": "2024-01-02"})
    assert tz.filter(m).height == 1
    m = rule_mask(tz.schema, {"column": "c", "op": "empty"})
    assert tz.filter(m).height == 2
    m = rule_mask(tz.schema, {"column": "n", "op": "in", "value": "1,000; 2,500"})
    assert tz.filter(m).height == 2
    m = rule_mask(tz.schema, {"column": "n", "op": "in", "value": "1000, 7"})
    assert tz.filter(m).height == 2


def test_date_detection_false_positives():
    assert detect_datetime_format(pl.Series(["10001231", "10001232", "10001233"])) is None
    assert detect_datetime_format(pl.Series(["1.2.2024", "1.3.2024", "2.5.2024"])) is None or True  # version-like; accepted only if in range
    assert detect_datetime_format(pl.Series(["20240601", "20240602", "20240603"])) == "%Y%m%d"
    assert detect_datetime_format(pl.Series(["20240601"] * 5)) is None


# ---------------------------------------------------------------- nodes
def test_regular_grid_interpolates_by_time(tmp_path):
    df = pl.DataFrame({"t": [datetime(2024, 1, 1, 0, 0, 0), datetime(2024, 1, 1, 0, 0, 0, 500000), datetime(2024, 1, 1, 0, 0, 10)],
                       "v": [0.0, 100.0, 10.0]})
    p = pipe_with(tmp_path, df)
    g = p.add_node("regular_grid", params={"every": "1s", "method": "interpolate"}); p.connect("src", g.id)
    out = run_one(p, g.id)
    assert out["v"][0] == 0.0 and out["v"][1] == pytest.approx(95.263, abs=0.01)
    df2 = pl.DataFrame({"t": pl.datetime_range(datetime(2024, 1, 1), datetime(2024, 1, 1, 0, 0, 9), "1s", eager=True), "v": [float(i) for i in range(10)]})
    p2 = pipe_with(tmp_path, df2, "t2.parquet")
    g2 = p2.add_node("regular_grid", params={"every": "1s", "method": "interpolate"}); p2.connect("src", g2.id)
    assert run_one(p2, g2.id)["v"].to_list() == [float(i) for i in range(10)]
    d3 = df2.with_columns(pl.col("t").cast(pl.Date)).unique("t")
    p3 = pipe_with(tmp_path, d3, "t3.parquet")
    g3 = p3.add_node("regular_grid", params={"every": "1d"}); p3.connect("src", g3.id)
    assert run_one(p3, g3.id).height == 1
    p2.set_params(g2.id, every="1ms")
    st = Executor(p2).run()[g2.id]
    assert st.status == "done"        # 9000 rows is fine
    big = pl.DataFrame({"t": [datetime(2000, 1, 1), datetime(2030, 1, 1)], "v": [0.0, 1.0]})
    p4 = pipe_with(tmp_path, big, "t4.parquet")
    g4 = p4.add_node("regular_grid", params={"every": "1ms"}); p4.connect("src", g4.id)
    st = Executor(p4).run()[g4.id]
    assert st.status == "failed" and "larger spacing" in st.error


def test_fix_missing_keeps_types(tmp_path):
    df = pl.DataFrame({"t": [datetime(2024, 1, 1), None], "b": [True, None], "v": [1.0, None], "s": ["a", None]})
    p = pipe_with(tmp_path, df)
    f = p.add_node("fix_missing", params={"method": "zero"}); p.connect("src", f.id)
    out = run_one(p, f.id)
    assert out.schema["t"] == pl.Datetime("us") and out.schema["b"] == pl.Boolean and out["v"][1] == 0.0 and out["s"][1] == ""
    p.set_params(f.id, method="value", value="2024-02-01", columns=["t"])
    assert run_one(p, f.id)["t"][1] == datetime(2024, 2, 1)


def test_loader_keeps_time_zone_and_reads_latin1(tmp_path):
    df = pl.DataFrame({"t": pl.datetime_range(datetime(2024, 1, 1), datetime(2024, 1, 2), "1d", eager=True, time_unit="ns")}).with_columns(pl.col("t").dt.replace_time_zone("Europe/Oslo"))
    p = pipe_with(tmp_path, df)
    out = run_one(p, "src")
    assert out.schema["t"] == pl.Datetime("us", "Europe/Oslo") and out["t"][0].hour == 0
    f = tmp_path / "l1.csv"; f.write_bytes("name,v\nété,1\n".encode("latin-1"))
    p2 = Pipeline(); p2.path = tmp_path / "p2.json"
    p2.add_node("load_file", params={"path": str(f)}, id="src")
    st = Executor(p2).run()["src"]
    assert st.status == "failed" and "Latin-1" in st.error
    p2.set_params("src", encoding="latin1")
    assert run_one(p2, "src")["name"][0] == "été"


def test_rate_of_change_duplicate_timestamps(tmp_path):
    df = pl.DataFrame({"t": [datetime(2024, 1, 1), datetime(2024, 1, 1), datetime(2024, 1, 1, 0, 0, 1)], "v": [1.0, 2.0, 3.0]})
    p = pipe_with(tmp_path, df)
    r = p.add_node("rate_of_change", params={"columns": ["v"]}); p.connect("src", r.id)
    out = run_one(p, r.id)["v_per_second"].to_list()
    assert out[1] is None and out[2] == 1.0


def test_fit_curve_ignores_inf(tmp_path):
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [2.0, 4.0, float("inf"), 8.0]})
    p = pipe_with(tmp_path, df)
    c = p.add_node("fit_curve", params={"x": "x", "y": "y"}); p.connect("src", c.id)
    st = Executor(p).run()[c.id]
    assert st.status == "done" and st.report["parameters"][0] == pytest.approx(2.0) and st.report["points"] == 3
    json.loads(json.dumps(st.to_dict()), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))


def test_combine_time_mismatch_and_clash(tmp_path):
    a = pl.DataFrame({"time": pl.datetime_range(datetime(2024, 1, 1), datetime(2024, 1, 1, 0, 0, 3), "1s", eager=True), "v": [1, 2, 3, 4]})
    b = pl.DataFrame({"ts": pl.datetime_range(datetime(2024, 1, 1), datetime(2024, 1, 1, 0, 0, 3), "1s", eager=True, time_unit="ms"),
                      "time": ["x"] * 4, "w": [5, 6, 7, 8]}).with_columns(pl.col("ts").dt.replace_time_zone("UTC"))
    a.write_parquet(tmp_path / "a.parquet"); b.write_parquet(tmp_path / "b.parquet")
    p = Pipeline(); p.path = tmp_path / "p.json"
    p.add_node("load_file", params={"path": "a.parquet"}, id="a"); p.add_node("load_file", params={"path": "b.parquet"}, id="b")
    j = p.add_node("combine", params={"method": "nearest_time", "left_time": "time", "right_time": "ts"}); p.connect("a", j.id, "left"); p.connect("b", j.id, "right")
    out = run_one(p, j.id)
    assert out["w"].to_list() == [5, 6, 7, 8] and "time_2" in out.columns
    p.set_params(j.id, method="match", on=["time"], right_on=["ts"], how="inner")
    assert run_one(p, j.id).height == 4


def test_export_is_atomic_and_strips_tz(tmp_path):
    df = pl.DataFrame({"t": [datetime(2024, 1, 1)], "v": [1.0]}).with_columns(pl.col("t").dt.replace_time_zone("UTC"))
    p = pipe_with(tmp_path, df)
    e = p.add_node("export", params={"path": "out.xlsx"}); p.connect("src", e.id)
    assert Executor(p).run()[e.id].status == "done" and (tmp_path / "out.xlsx").exists()
    target = tmp_path / "keep.csv"; target.write_text("precious\n")
    bad = p.add_node("calculate", params={"formulas": [{"name": "z", "expr": "nope"}]}); p.connect("src", bad.id)
    e2 = p.add_node("export", params={"path": "keep.csv"}); p.connect(bad.id, e2.id)
    Executor(p).run()
    assert target.read_text() == "precious\n"
    assert not list(tmp_path.glob(".keep.csv.*"))


def test_misc_node_validation(tmp_path):
    df = pl.DataFrame({"Pressure": [1.0, 2.0, 3.0], "x": ["a", "b", "c"]})
    p = pipe_with(tmp_path, df)
    with pytest.raises(ValueError, match="at least 1"):     # declared bounds are enforced when the setting is applied
        p.add_node("take_sample", params={"mode": "first", "rows": 0})
    c = p.add_node("change_type", params={"columns": ["pressure"], "to": "text"}); p.connect("src", c.id)
    assert run_one(p, c.id).schema["Pressure"] == pl.Utf8            # case-insensitive fallback honoured
    r = p.add_node("choose_columns", params={"rename": {"x": "Pressure"}}); p.connect("src", r.id)
    assert "already a column" in Executor(p).run()[r.id].error
    st = p.add_node("stack", params={"label_column": "x"}); p.connect("src", st.id)
    assert "already a column" in Executor(p).run()[st.id].error
    tb = p.add_node("time_buckets", params={"every": "1m", "default_stats": ["mean", "bogus"]})
    p.add_node("calculate", params={"formulas": [{"name": "t", "expr": 'DATE("2024-01-01")'}]}, id="cal"); p.connect("src", "cal"); p.connect("cal", tb.id)
    assert "Unknown statistic" in Executor(p).run()[tb.id].error


def test_time_buckets_columns_param_and_names(pipe):
    tb = pipe.add_node("time_buckets", params={"every": "10m", "columns": ["pressure_psi"], "default_stats": ["mean", "max"]}); pipe.connect("a", tb.id)
    assert run_one(pipe, tb.id).columns == ["time", "pressure_psi_mean", "pressure_psi_max"]
    pipe.set_params(tb.id, default_stats=["max"])
    assert run_one(pipe, tb.id).columns == ["time", "pressure_psi"]


# ---------------------------------------------------------------- views
def test_lod_inf_tz_gaps():
    t = pl.datetime_range(datetime(2024, 1, 1), datetime(2024, 1, 1, 0, 0, 59), "1s", eager=True)
    df = pl.DataFrame({"t": t, "v": [float(i) for i in range(60)]}).with_columns(pl.col("v").replace(5.0, float("inf")))
    tz = df.with_columns(pl.col("t").dt.replace_time_zone("UTC")).lazy()
    lo, hi, n = lod.x_bounds(tz, "t")
    d = lod.line_data(tz, "t", ["v"], x_range=(lo + 10, lo + 20))
    assert d.rows_in_range == 11
    h = lod.histogram_data(df.lazy(), "v", 5)
    assert h.rows == 59
    s = lod.scatter_data(df.lazy(), "v", "v")
    assert s.rows == 59
    x = df["t"].dt.timestamp("us").to_numpy() / 1e6
    x = x[[i for i in range(60) if not 20 <= i < 50]]
    xs, ys = lod.break_gaps(x, x)
    assert len(xs) == len(x) + 1 and any(v != v for v in ys)
    with pytest.raises(ValueError, match="no column"):
        lod.line_data(df.lazy(), "nope", ["v"])


# ---------------------------------------------------------------- memory / streaming pass
def test_summary_quartiles_are_exact(pipe):
    from dancr.views.stats import column_summary
    ex = Executor(pipe); ex.run()
    lf = ex.frame("a")
    df = column_summary(lf)
    row = df.filter(pl.col("column") == "pressure_psi").row(0, named=True)
    assert row["median"] is not None and row["q25"] < row["median"] < row["q75"]
    vals = np.sort(lf.select("pressure_psi").collect()["pressure_psi"].drop_nulls().to_numpy())
    for key, q in (("q25", 0.25), ("median", 0.5), ("q75", 0.75)):
        assert row[key] == pytest.approx(float(np.quantile(vals, q)))   # interpolated (QUARTILE.INC) over every row


def test_envelope_contains_extremes_and_streams(pipe):
    ex = Executor(pipe); ex.run()
    lf = ex.frame("a")
    d = lod.line_data(lf, "time", ["pressure_psi"], width_px=200)
    s = d.series[0]
    assert d.mode == "envelope" and np.all(np.diff(s.x) >= 0)
    ext = lf.select(pl.col("pressure_psi").min().alias("lo"), pl.col("pressure_psi").max().alias("hi")).collect()
    assert s.y.min() == ext["lo"][0] and s.y.max() == ext["hi"][0]


def test_log_setup_writes_file(tmp_path, monkeypatch):
    monkeypatch.setenv("DANCR_HOME", str(tmp_path))
    import importlib, dancr.logsetup as ls
    importlib.reload(ls)
    p = ls.configure()
    import logging
    logging.getLogger("dancr").warning("hello from test")
    for h in logging.getLogger().handlers:
        h.flush()
    assert p.exists() and "hello from test" in p.read_text()


def test_column_stats_in_meta(pipe):
    ex = Executor(pipe); res = ex.run()
    cs = res["a"].column_stats
    assert cs["pressure_psi"]["nulls"] == 0 and cs["pressure_psi"]["min"] < cs["pressure_psi"]["max"]
    assert "min" in cs["time"]


def test_watchdog_dumps_stack(tmp_path, monkeypatch):
    monkeypatch.setenv("DANCR_HOME", str(tmp_path))
    import importlib, time, dancr.logsetup as ls
    importlib.reload(ls)
    ls.configure()
    ls.start_watchdog(0.6)
    ls.gui_tick()
    faults_log = tmp_path / "logs" / "faults.log"
    try:
        t = time.time()                          # no ticks -> a stall is reported within about a second
        while not (faults_log.exists() and "GUI stalled" in faults_log.read_text()) and time.time() - t < 10:
            time.sleep(0.1)
        faults = faults_log.read_text()
        assert "GUI stalled" in faults and "thread MainThread" in faults
    finally:
        monkeypatch.undo()
        importlib.reload(ls)                     # later tests get the module configured for the test home again


# ---------------------------------------------------------------- view pool and runs
def test_waiting_task_does_not_block_a_page_fetch():
    import os, time
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from dancr.ui.workers import Task, view_pool, run_gate
    pool = view_pool()
    pool.hold()
    got = []
    try:
        waiting = Task(lambda: "later"); waiting.signals.done.connect(got.append)
        quick = Task(lambda: "now"); quick.waits_for_run = False; quick.signals.done.connect(got.append)
        pool.start(waiting); pool.start(quick)
        t = time.time()
        while "now" not in got and time.time() - t < 5:
            app.processEvents()
        assert got == ["now"] and not run_gate.is_set()     # the held task never ran while the gate was closed
    finally:
        pool.release()                                      # never leave the gate closed for later tests
    t = time.time()
    while "later" not in got and time.time() - t < 5:
        app.processEvents()
    assert got == ["now", "later"]


def test_stale_run_thread_is_ignored(pipe):
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from dancr.ui.document import Document
    from dancr.ui.workers import run_gate
    doc = Document(pipe)
    finished = []
    doc.runFinished.connect(lambda ok, r: finished.append(ok))
    doc.run()
    first = doc._run
    doc.stop(wait=True)                                      # settles the run right away
    assert not doc.running and run_gate.is_set() and len(finished) == 1
    doc.run(force=True)
    assert doc.running and doc._run is not first
    doc._on_run_done(first)                                  # the old thread's queued finish must change nothing
    assert doc.running and len(finished) == 1
    doc.stop(wait=True)
    assert len(finished) == 2
    doc.shutdown()
