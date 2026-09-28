"""Time steps: buckets, regular grids, rolling windows, rates of change, gaps and summarising around samples."""
from datetime import datetime, timedelta

import polars as pl
import pytest

from conftest import run_one
from dancr.core import Pipeline
from dancr.core.executor import Executor


def _pipe(df: pl.DataFrame, tmp_path, name="in.parquet") -> Pipeline:
    df.write_parquet(tmp_path / name)
    p = Pipeline("t")
    p.add_node("load_file", "Load", {"path": name}, id="src")
    p.path = tmp_path / "p.json"
    return p


def _step(p: Pipeline, key: str, params: dict, port: str | None = None, src: str = "src") -> str:
    nid = p.add_node(key, params=params).id
    p.connect(src, nid, port)
    return nid


def pipe_with(tmp_path, df, name="t.parquet"):
    f = tmp_path / name
    df.write_parquet(f)
    p = Pipeline()
    p.path = tmp_path / "p.json"
    p.add_node("load_file", params={"path": str(f)}, id="src")
    return p


def failed(p, node_id):
    st = Executor(p).run(targets=[node_id])[node_id]
    assert st.status == "failed", "expected a failure"
    return st.error


@pytest.fixture
def dense(tmp_path):
    t0 = datetime(2024, 6, 1)
    log = pl.DataFrame({"t": [t0 + timedelta(minutes=i) for i in range(180)], "v": [float(i) for i in range(180)]})
    samples = pl.DataFrame({"when": [t0 + timedelta(minutes=m) for m in (60, 70, 80)], "lab": [1.0, 2.0, 3.0]})
    log.write_parquet(tmp_path / "log.parquet"); samples.write_parquet(tmp_path / "s.parquet")
    p = Pipeline(); p.path = tmp_path / "p.json"
    p.add_node("load_file", params={"path": "log.parquet"}, id="log"); p.add_node("load_file", params={"path": "s.parquet"}, id="s")
    return p


def _brute_centred(ts: list[datetime], vs: list[float], half: timedelta, fn) -> list[float]:
    return [fn([v for tt, v in zip(ts, vs) if t - half <= tt <= t + half]) for t in ts]


def test_time_buckets(pipe, truth):
    tb = pipe.add_node("time_buckets", params={"every": "1m", "default_stats": ["mean", "max"], "count_column": "n"})
    pipe.connect("a", tb.id)
    df = run_one(pipe, tb.id)
    assert df.columns == ["time", "pressure_psi_mean", "pressure_psi_max", "temp_c_mean", "temp_c_max", "n"]
    assert df["n"].max() == 1200                          # 20 Hz * 60 s
    assert len(df) == pytest.approx(truth["hours"] * 60, abs=1)
    pipe.set_params(tb.id, every="30s", default_stats=["mean"], aggregations=[{"column": "pressure_psi", "stats": ["min"], "alias": "pmin"}])
    df = run_one(pipe, tb.id)
    assert "pmin" in df.columns and "temp_c" not in df.columns


def test_find_gaps_matches_truth(pipe, truth):
    g = pipe.add_node("find_gaps"); pipe.connect("a", g.id)
    df = run_one(pipe, g.id)
    assert len(df) == len(truth["gaps"])
    assert df["gap_seconds"][0] == pytest.approx(truth["gaps"][0]["seconds"], abs=0.1)
    assert df["missing_readings"][0] == truth["gaps"][0]["rows"]


def test_rolling_rate_and_outliers(pipe, truth):
    r = pipe.add_node("rolling", params={"columns": ["pressure_psi"], "stat": "median", "window": "5"}); pipe.connect("b", r.id)
    df = run_one(pipe, r.id)
    assert "pressure_psi_median_5" in df.columns
    pipe.set_params(r.id, window="2s", replace=True)
    assert run_one(pipe, r.id).columns == ["time", "pressure_psi", "temp_c"]
    rate = pipe.add_node("rate_of_change", params={"columns": ["pressure_psi"], "per": "m"}); pipe.connect("a", rate.id)
    df = run_one(pipe, rate.id)
    assert "pressure_psi_per_minute" in df.columns and df["pressure_psi_per_minute"].abs().median() < 5.0
    o = pipe.add_node("remove_outliers", params={"columns": ["pressure_psi"], "method": "rolling", "window": 51, "threshold": 6}); pipe.connect("b", o.id)
    st = Executor(pipe).run(targets=[o.id])
    removed = st["b"].rows - st[o.id].rows
    assert 0 < removed <= truth["spikes_b"] * 3
    pipe.set_params(o.id, action="flag")
    df = run_one(pipe, o.id)
    assert df["is_outlier"].sum() == removed


def test_regular_grid(pipe):
    g = pipe.add_node("regular_grid", params={"every": "1s", "method": "nearest"}); pipe.connect("a", g.id)
    df = run_one(pipe, g.id)
    assert (df["time"].diff().drop_nulls().dt.total_seconds() == 1).all()


def test_regular_grid_keeps_its_last_tick(tmp_path):
    t0 = datetime(2024, 1, 1)
    p = _pipe(pl.DataFrame({"t": [t0, t0 + timedelta(milliseconds=300)], "v": [1.0, 2.0]}), tmp_path)
    n = _step(p, "regular_grid", {"every": "100ms", "method": "nearest"})
    assert run_one(p, n).height == 4


def test_summarise_around_samples_one_window_apart(tmp_path):
    base = datetime(2024, 1, 1)
    samples = pl.DataFrame({"at": [base, base + timedelta(hours=2), None]})
    log = pl.DataFrame({"t": [base + timedelta(hours=1)], "v": [5.0]})
    samples.write_parquet(tmp_path / "s.parquet"); log.write_parquet(tmp_path / "l.parquet")
    p = Pipeline("t")
    p.add_node("load_file", params={"path": "s.parquet"}, id="s"); p.add_node("load_file", params={"path": "l.parquet"}, id="l")
    p.add_node("summarise_around", params={"window": "2h", "side": "around", "stats": ["mean", "count"]}, id="a")
    p.connect("s", "a", "samples"); p.connect("l", "a", "log"); p.path = tmp_path / "p.json"
    df = run_one(p, "a")
    assert df.height == 3                                           # the sample without a time is kept
    assert df["v_mean"].to_list()[:2] == [5.0, 5.0]                 # both ends of a closed window see the log row
    assert df["v_count"].to_list()[2] == 0


def test_first_and_last_in_buckets_follow_time(tmp_path):
    base = datetime(2024, 1, 1)
    p = _pipe(pl.DataFrame({"t": [base + timedelta(minutes=30), base], "v": [2.0, 1.0]}), tmp_path)
    n = _step(p, "time_buckets", {"every": "1h", "default_stats": ["first", "last"]})
    df = run_one(p, n)
    assert (df["v_first"][0], df["v_last"][0]) == (1.0, 2.0)


def test_rows_without_a_time_are_reported(tmp_path):
    p = _pipe(pl.DataFrame({"t": [datetime(2024, 1, 1), None], "v": [1.0, 2.0]}), tmp_path)
    n = _step(p, "time_buckets", {"every": "1h"})
    assert any("1 rows with a blank t were left out" in m for m in Executor(p).run()[n].messages)


def test_duplicate_aggregations_are_explained(tmp_path):
    p = _pipe(pl.DataFrame({"t": [datetime(2024, 1, 1)], "v": [1.0]}), tmp_path)
    n = _step(p, "time_buckets", {"every": "1h", "aggregations": [{"column": "v", "stats": ["mean"]}, {"column": "v", "stats": ["mean"]}]})
    assert "would both be called" in Executor(p).run()[n].error


def test_two_tables_on_the_same_grid_share_their_ticks():
    p = Pipeline("t")
    for nid, off in (("a", 0.12), ("b", 0.45)):
        p.add_node("enter_data", params={"columns": [{"name": "time", "type": "datetime"}, {"name": "v", "type": "number"}],
                                         "rows": [[f"2024-01-01 00:00:{off + i:06.3f}", i] for i in range(5)]}, id=nid)
        p.add_node("regular_grid", params={"every": "1s"}, id="g" + nid); p.connect(nid, "g" + nid)
    p.add_node("combine", params={"method": "match", "on": ["time"]}, id="c")
    p.connect("ga", "c", "left"); p.connect("gb", "c", "right")
    assert Executor(p).preview("c")[0].height == 4


def test_regular_grid_keeps_a_zoned_instant(tmp_path):
    t0 = datetime(2024, 1, 1, 0, 0)
    zoned = (pl.DataFrame({"t": [t0, t0.replace(hour=1)], "v": [1.0, 2.0]})
             .with_columns(pl.col("t").dt.replace_time_zone("Europe/Oslo")))
    zoned.write_parquet(tmp_path / "z.parquet")
    p = Pipeline()
    p.path = tmp_path / "p.json"
    p.add_node("load_file", params={"path": "z.parquet"}, id="z")
    p.add_node("regular_grid", params={"every": "1h", "method": "nearest"}, id="g")
    p.connect("z", "g")
    out = run_one(p, "g")
    assert [str(x) for x in out["t"].to_list()] == ["2024-01-01 00:00:00+01:00", "2024-01-01 01:00:00+01:00"]
    assert out["v"].to_list() == [1.0, 2.0]


def test_rate_of_change_returns_a_node_result(tmp_path):
    t0 = datetime(2024, 1, 1)
    df = pl.DataFrame({"t": [t0.replace(second=s) for s in range(5)], "v": [1.0, 2.0, 4.0, 8.0, 16.0]})
    p = pipe_with(tmp_path, df)
    p.add_node("rate_of_change", params={"columns": ["v"], "per": "s"}, id="r")
    p.connect("src", "r")
    out = run_one(p, "r")
    assert "v_per_second" in out.columns


@pytest.mark.parametrize("side,count,means", [("before", 60, [30.5, 40.5, 50.5]), ("after", 60, [89.5, 99.5, 109.5]), ("around", 61, [60.0, 70.0, 80.0])])
def test_summarise_around_sees_the_whole_window_when_samples_overlap(dense, side, count, means):
    a = dense.add_node("summarise_around", params={"window": "1h", "side": side, "columns": ["v"], "stats": ["count", "mean"]})
    dense.connect("s", a.id, "samples"); dense.connect("log", a.id, "log")
    out = run_one(dense, a.id)
    assert out["v_count"].to_list() == [count] * 3 and out["v_mean"].to_list() == pytest.approx(means)
    assert out.columns == ["when", "lab", "v_count", "v_mean"]


def test_summarise_around_rejects_unknown_stat_and_side(dense):
    a = dense.add_node("summarise_around", params={"window": "1h", "stats": ["mode"]})
    dense.connect("s", a.id, "samples"); dense.connect("log", a.id, "log")
    assert "Unknown statistic" in failed(dense, a.id)


def test_rolling_time_window_centred_and_named_consistently(dense):
    r = dense.add_node("rolling", params={"columns": ["v"], "window": "2m", "stat": "mean", "centered": True}); dense.connect("log", r.id)
    out = run_one(dense, r.id)
    assert out["v_mean_2m"].head(3).to_list() == [0.5, 1.0, 2.0]      # [t-1m, t+1m], like a centred 3-row window
    dense.set_params(r.id, window="3")
    assert run_one(dense, r.id)["v_mean_3"].head(3).to_list() == [0.5, 1.0, 2.0]
    dense.set_params(r.id, window="2m", centered=False)
    assert run_one(dense, r.id)["v_mean_2m"].head(3).to_list() == [0.0, 0.5, 1.5]


def test_regular_grid_is_lazy_and_handles_zoned_time(dense):
    g = dense.add_node("regular_grid", params={"every": "30s", "method": "nearest"}); dense.connect("log", g.id)
    out = run_one(dense, g.id)
    assert out.height == 359 and (out["t"].diff().drop_nulls().dt.total_seconds() == 30).all()
    ct = dense.add_node("calculate", params={"formulas": [{"name": "t", "expr": "t"}]}); dense.connect("log", ct.id)   # no-op step to attach a zoned copy
    zoned = pl.read_parquet(Executor(dense).state("log").output).with_columns(pl.col("t").dt.replace_time_zone("Europe/Oslo"))
    zoned.write_parquet(dense.path.parent / "z.parquet"); dense.add_node("load_file", params={"path": "z.parquet"}, id="z")
    g2 = dense.add_node("regular_grid", params={"every": "1h"}); dense.connect("z", g2.id)
    out = run_one(dense, g2.id)
    assert out.schema["t"] == pl.Datetime("us", "Europe/Oslo") and out.height == 3


def test_find_gaps_counts_at_least_one_missing_reading_and_bucket_count_clash(tmp_path):
    t0 = datetime(2024, 1, 1)
    times = [t0 + timedelta(seconds=s) for s in (0, 1, 2, 3.3, 4.3, 5.3)]
    p = pipe_with(tmp_path, pl.DataFrame({"t": times, "v": [1.0] * 6}))
    g = p.add_node("find_gaps", params={"factor": 1.2}); p.connect("src", g.id)
    out = run_one(p, g.id)
    assert out.height == 1 and out["missing_readings"][0] == 1
    tb = p.add_node("time_buckets", params={"every": "1m", "count_column": "v"}); p.connect("src", tb.id)
    assert "already used" in failed(p, tb.id)


@pytest.mark.parametrize("stat,fn", [("sum", sum), ("mean", lambda xs: sum(xs) / len(xs)), ("max", max)])
def test_centred_time_window_is_centred(tmp_path, stat, fn):
    base = datetime(2024, 1, 1)
    ts = [base + timedelta(seconds=s) for s in (0, 1, 2, 3, 3, 4, 5.5, 6, 9)]   # repeated and irregular times
    vs = [0.0, 0.0, 0.0, 10.0, 4.0, 0.0, 2.0, 1.0, 7.0]
    p = _pipe(pl.DataFrame({"t": ts, "v": vs}), tmp_path)
    p.add_node("rolling", params={"columns": ["v"], "stat": stat, "window": "3s", "centered": True}, id="r")
    p.connect("src", "r")
    got = run_one(p, "r")[f"v_{stat}_3s"].to_list()
    assert got == pytest.approx(_brute_centred(ts, vs, timedelta(seconds=1.5), fn))


def test_centred_and_trailing_windows_differ(tmp_path):
    ts = [datetime(2024, 1, 1) + timedelta(seconds=i) for i in range(7)]
    p = _pipe(pl.DataFrame({"t": ts, "v": [0.0, 0, 0, 10, 0, 0, 0]}), tmp_path)
    p.add_node("rolling", params={"columns": ["v"], "stat": "sum", "window": "3s", "centered": True}, id="c")
    p.add_node("rolling", params={"columns": ["v"], "stat": "sum", "window": "3s", "centered": False}, id="t")
    p.connect("src", "c"); p.connect("src", "t")
    assert run_one(p, "c")["v_sum_3s"].to_list() == [0, 0, 10, 10, 10, 0, 0]
    assert run_one(p, "t")["v_sum_3s"].to_list() == [0, 0, 0, 10, 10, 10, 0]


def test_centred_time_window_replace_keeps_column_order(tmp_path):
    ts = [datetime(2024, 1, 1) + timedelta(seconds=i) for i in range(4)]
    p = _pipe(pl.DataFrame({"v": [1.0, 2, 3, 4], "t": ts, "w": [0.0, 0, 0, 0]}), tmp_path)
    p.add_node("rolling", params={"columns": ["v"], "stat": "mean", "window": "2s", "replace": True}, id="r")
    p.connect("src", "r")
    df = run_one(p, "r")
    assert df.columns == ["v", "t", "w"]
    assert df["v"].to_list() == [1.5, 2.0, 3.0, 3.5]


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


def test_rate_of_change_duplicate_timestamps(tmp_path):
    df = pl.DataFrame({"t": [datetime(2024, 1, 1), datetime(2024, 1, 1), datetime(2024, 1, 1, 0, 0, 1)], "v": [1.0, 2.0, 3.0]})
    p = pipe_with(tmp_path, df)
    r = p.add_node("rate_of_change", params={"columns": ["v"]}); p.connect("src", r.id)
    out = run_one(p, r.id)["v_per_second"].to_list()
    assert out[1] is None and out[2] == 1.0


def test_time_buckets_columns_param_and_names(pipe):
    tb = pipe.add_node("time_buckets", params={"every": "10m", "columns": ["pressure_psi"], "default_stats": ["mean", "max"]}); pipe.connect("a", tb.id)
    assert run_one(pipe, tb.id).columns == ["time", "pressure_psi_mean", "pressure_psi_max"]
    pipe.set_params(tb.id, default_stats=["max"])
    assert run_one(pipe, tb.id).columns == ["time", "pressure_psi"]


def test_regular_grid_follows_the_calendar_across_daylight_saving(tmp_path):
    """A daily grid stays on local midnight across a clock change, and an hourly grid keeps every real hour
    (Python's wall-clock arithmetic on zoned times put the days at 01:00 and lost the repeated hour)."""
    spring = pl.datetime_range(datetime(2024, 3, 29), datetime(2024, 4, 2), "1h", time_zone="Europe/London", eager=True)
    p = _pipe(pl.DataFrame({"t": spring, "v": range(len(spring))}), tmp_path)
    n = _step(p, "regular_grid", {"every": "1d", "method": "nearest"})
    assert [x.hour for x in run_one(p, n)["t"].to_list()] == [0, 0, 0, 0, 0]
    autumn = pl.datetime_range(datetime(2024, 10, 26, 20), datetime(2024, 10, 27, 4), "1h", time_zone="Europe/London", eager=True)
    p = _pipe(pl.DataFrame({"t": autumn, "v": range(len(autumn))}), tmp_path, "autumn.parquet")
    n = _step(p, "regular_grid", {"every": "1h", "method": "nearest"})
    out = run_one(p, n)
    assert out.height == len(autumn) and out["v"].to_list() == list(range(len(autumn)))
