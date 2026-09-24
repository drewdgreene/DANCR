import numpy as np
import polars as pl

from dancr.core.executor import Executor
from dancr.views import lod, table, stats, render


def test_line_envelope_and_zoom(pipe):
    ex = Executor(pipe); ex.run()
    lf = ex.frame("a")
    d = lod.line_data(lf, "time", ["pressure_psi"], width_px=300)
    assert d.mode == "envelope" and d.axis.kind == "time"
    s = d.series[0]
    assert len(s.x) <= 1200 and np.all(np.diff(s.x) >= 0)      # up to four points per pixel column
    # envelope must contain the true min/max
    df = lf.select(pl.col("pressure_psi").min().alias("lo"), pl.col("pressure_psi").max().alias("hi")).collect()
    assert s.y.min() == df["lo"][0] and s.y.max() == df["hi"][0]
    z = lod.line_data(lf, "time", ["pressure_psi"], x_range=(d.axis.lo, d.axis.lo + 30), width_px=300)
    assert z.mode == "raw" and 590 <= z.rows_in_range <= 610
    r = lod.line_data(lf, None, ["pressure_psi"], width_px=100)
    assert r.axis.kind == "index"


def test_scatter_hist_bar(pipe):
    ex = Executor(pipe); ex.run()
    lf = ex.frame("a")
    sd = lod.scatter_data(lf, "temp_c", "pressure_psi", width_px=50, height_px=40)
    assert sd.mode == "density" and sd.density.shape == (50, 40) and sd.density.sum() == sd.rows
    hd = lod.histogram_data(lf, "pressure_psi", bins=10)
    assert hd.counts.sum() == hd.rows == ex.state("a").rows
    bd = lod.bar_data(lf.with_columns((pl.col("temp_c") > 2.75).alias("warm")), "warm", "pressure_psi", "mean")
    assert set(bd.labels) <= {"true", "false"}


def test_table_pager_and_format(pipe):
    ex = Executor(pipe); ex.run()
    tp = table.TablePager(ex.frame("a"), page_size=100)
    assert tp.rows == ex.state("a").rows
    assert tp.value(150, 1) == ex.frame("a").slice(150, 1).collect()[0, 1]
    assert table.format_value(1234567.891) == "1,234,567.89"
    assert table.format_value(0.1 + 0.2) == "0.3"
    assert table.format_value(None) == ""
    assert table.format_value(float("nan")) == ""


def test_column_summary_and_render(pipe, tmp_path):
    ex = Executor(pipe); ex.run()
    lf = ex.frame("a")
    df = stats.column_summary(lf)
    assert df["column"].to_list() == ["time", "pressure_psi", "temp_c"]
    out = render.render_chart(lf, {"kind": "line", "x": "time", "series": [{"column": "pressure_psi"}], "title": "t"}, tmp_path / "c.png", width=400, height=300)
    assert out.exists() and out.stat().st_size > 1000
    for kind, extra in [("scatter", {"x": "temp_c", "series": [{"column": "pressure_psi"}]}), ("histogram", {"column": "pressure_psi"}),
                        ("bar", {"category": "temp_c", "value": "pressure_psi", "stat": "count"})]:
        render.render_chart(lf, {"kind": kind, **extra}, tmp_path / f"{kind}.png", width=300, height=200)
