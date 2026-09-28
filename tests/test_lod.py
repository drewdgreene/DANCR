"""Chart level of detail: envelopes, bounds and gaps, whatever order or values the rows have."""
import math
from datetime import datetime

import numpy as np
import polars as pl
import pytest

from dancr.core.executor import Executor
from dancr.views import lod


def test_envelope_of_rows_not_sorted_by_x_never_doubles_back():
    n = 50_000
    x = np.arange(n, dtype=float)
    y = np.sin(x / 2000)
    y[12_345] = 9.0                                         # one spike that must survive
    rows = pl.DataFrame({"x": x, "y": y})
    shuffled = pl.concat([rows.slice(n // 2), rows.slice(0, n // 2)]).sample(fraction=1.0, shuffle=True, seed=2)
    a = lod.line_data(rows.lazy(), "x", ["y"], width_px=300).series[0]
    b = lod.line_data(shuffled.lazy(), "x", ["y"], width_px=300).series[0]
    assert np.all(np.diff(b.x) >= 0)
    np.testing.assert_array_equal(a.x, b.x)
    np.testing.assert_array_equal(a.y, b.y)
    assert b.x[0] == 0 and b.x[-1] == n - 1 and b.y.max() == 9.0


def test_values_that_are_not_finite_are_left_out_of_the_envelope():
    n = 20_000
    x = np.arange(n, dtype=float)[::-1].copy()              # descending
    y = np.ones(n)
    y[::3] = np.nan
    y[1] = np.inf
    s = lod.line_data(pl.DataFrame({"x": x, "y": y}).lazy(), "x", ["y"], width_px=100).series[0]
    assert s.mode == "envelope" and np.all(np.isfinite(s.y)) and set(s.y) == {1.0}
    assert np.all(np.diff(s.x) >= 0)


def test_an_infinite_x_does_not_squash_a_chart():
    from dancr.views.lod import x_bounds
    lf = pl.LazyFrame({"x": [0.0, 1.0, 2.0, math.inf]})
    lo, hi, _ = x_bounds(lf, "x")
    assert (lo, hi) == (0.0, 2.0)


def test_line_data_is_m4_and_ignores_non_finite_and_blank_x():
    n = 50_000
    x = np.arange(n, dtype=float); y = np.sin(x / 300) * 10
    y[100:200] = np.nan; y[3000] = np.inf; y[7000] = -np.inf
    xs = x.copy(); xs[10] = np.nan
    lf = pl.DataFrame({"x": xs, "y": y}).lazy()
    d = lod.line_data(lf, "x", ["y"], width_px=200)
    s = d.series[0]
    finite = y[np.isfinite(y) & np.isfinite(xs)]
    assert d.mode == "envelope" and d.rows_in_range == n - 1
    assert np.all(np.diff(s.x) >= 0) and np.all(np.isfinite(s.y))
    assert s.y.min() == finite.min() and s.y.max() == finite.max()
    assert 2 * 200 < len(s.x) <= 4 * 200
    # first/last of each bucket present: the very first and last points of the data
    assert s.x[0] == 0.0 and s.x[-1] == n - 1


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


def test_group_values_info_brings_back_only_the_shown_groups(monkeypatch):
    # many distinct values: only the shown groups leave the query, and what is left out is still counted
    n = 20_000
    keys = [f"k{i:05d}" for i in range(n)] + ["b", "b", "b", "a", "a", "a", None, None, None, "c", "c"]
    lf = pl.LazyFrame({"g": keys})
    heights = []
    real = lod._collect
    monkeypatch.setattr(lod, "_collect", lambda q: heights.append((df := real(q)).height) or df)
    shown, others, other_rows = lod.group_values_info(lf, "g", limit=4)
    assert max(heights) <= 4
    assert shown == ["a", "b", None, "c"]                   # most frequent first, ties by value, blank last among equals
    assert others == n and other_rows == n
    assert lod.group_values_info(pl.LazyFrame({"g": ["x", "y"]}), "g", limit=12) == (["x", "y"], 0, 0)
    assert lod.group_values_info(pl.LazyFrame({"g": pl.Series([], dtype=pl.Utf8)}), "g") == ([], 0, 0)


def test_envelope_contains_extremes_and_streams(pipe):
    ex = Executor(pipe); ex.run()
    lf = ex.frame("a")
    d = lod.line_data(lf, "time", ["pressure_psi"], width_px=200)
    s = d.series[0]
    assert d.mode == "envelope" and np.all(np.diff(s.x) >= 0)
    ext = lf.select(pl.col("pressure_psi").min().alias("lo"), pl.col("pressure_psi").max().alias("hi")).collect()
    assert s.y.min() == ext["lo"][0] and s.y.max() == ext["hi"][0]
