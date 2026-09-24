"""Regression tests for the audit fixes: durations, count-column clashes, ZSCORE, report tables,
synth parquet streaming."""
import pytest

import polars as pl

from dancr.core import Pipeline
from dancr.core.executor import Executor
from dancr.core.timeutil import parse_duration


# ------------------------------------------------------------------ durations
def test_parse_duration_forms():
    assert parse_duration("5 min") == ("5m", 300.0)
    assert parse_duration("1.5h") == ("90m", 5400.0)


@pytest.mark.parametrize("bad", ["0s", "0m", "0h", "0d", "0w", "0"])
def test_parse_duration_rejects_non_positive(bad):
    with pytest.raises(ValueError):
        parse_duration(bad)


# ------------------------------------------------------------------ ZSCORE
def test_zscore_constant_column_is_null_not_inf(tmp_path):
    p = Pipeline("z")
    p.add_node("enter_data", id="e", params={
        "columns": [{"name": "x", "type": "number"}], "rows": [[5.0], [5.0], [5.0]]})
    p.add_node("calculate", id="c", params={"formulas": [{"name": "z", "expr": "ZSCORE(x)"}]})
    p.connect("e", "c")
    p.path = tmp_path / "z.json"
    res = Executor(p).run()
    assert res["c"].status == "done", res["c"].error
    out = pl.read_parquet(res["c"].output)
    assert out["z"].null_count() == out.height == 3
    assert out["z"].is_nan().sum() == 0


# ------------------------------------------------------- group_summary count clash
def test_group_summary_count_column_clash_is_friendly(tmp_path):
    p = Pipeline("g")
    p.add_node("enter_data", id="e", params={
        "columns": [{"name": "g", "type": "text"}, {"name": "v", "type": "number"}],
        "rows": [["a", 1.0], ["a", 2.0], ["b", 3.0]]})
    p.add_node("group_summary", id="s", params={
        "by": ["g"], "default_stats": ["mean", "max"], "count_column": "v_mean"})
    p.connect("e", "s")
    p.path = tmp_path / "g.json"
    res = Executor(p).run()
    assert res["s"].status == "failed"
    assert "already used" in (res["s"].error or "")


# ------------------------------------------------------------------ report table
def test_report_table_says_how_many_rows_are_shown():
    from dancr.core.nodes.report import _table_html
    df = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
    html = _table_html(df, 2, 5)
    assert "Showing the first 2 of 5 rows" in html


# ------------------------------------------------------------------ synth parquet
def test_synth_parquet_streams_and_matches_csv(tmp_path):
    from dancr.synth import write_dataset
    csv = write_dataset(tmp_path / "csv", hours=0.05, rate=20.0, seed=3, fmt="csv")
    pq = write_dataset(tmp_path / "pq", hours=0.05, rate=20.0, seed=3, fmt="parquet")
    assert pq["rows_a"] == csv["rows_a"] and pq["rows_b"] == csv["rows_b"]
    assert pl.read_parquet(tmp_path / "pq" / "probe_A.parquet").height == pq["rows_a"]
    assert pl.read_parquet(tmp_path / "pq" / "probe_B.parquet").height == pq["rows_b"]


def test_synth_low_rate_does_not_crash(tmp_path):
    from dancr.synth import write_dataset
    t = write_dataset(tmp_path, hours=1.0, rate=0.001, seed=1, fmt="parquet")
    assert pl.read_parquet(tmp_path / "probe_A.parquet").height == t["rows_a"]


# ------------------------------------------------------------------ map dot grid
def test_map_dot_grid_has_zoom_independent_spacing():
    from PySide6.QtCore import QRectF
    from dancr.ui.canvas import dot_grid
    rect = QRectF(-500, -300, 1000, 600)
    for scale in (0.15, 0.4, 1.0, 2.5):
        minor, major = dot_grid(rect, scale)
        assert minor and major
        xs = sorted({round(p.x(), 6) for p in minor})
        gap = min(b - a for a, b in zip(xs, xs[1:]))
        assert 10 <= gap * scale <= 40          # minor dots stay readable on screen at every zoom
        assert all(round(p.x(), 6) in xs for p in major)  # major dots sit on the minor grid
    assert dot_grid(rect, 0.0) == ([], [])
