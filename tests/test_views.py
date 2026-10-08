"""Chart queries, table formatting, statistics and the shared palette."""
import numpy as np
import polars as pl
import pytest

from dancr.core.executor import Executor
from dancr.views import lod, render, stats, table


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


def test_colour_by_counts_rows_once_and_says_what_it_leaves_out():
    from dancr.views.chartquery import query_one
    n = 2000
    lf = pl.LazyFrame({"x": [float(i) for i in range(n)], "y": [1.0] * n,
                       "g": [None if i % 50 == 0 else f"g{i % 15}" for i in range(n)]})
    cd = query_one(lf, dict(lf.collect_schema()), {"kind": "line", "x": "x", "series": [{"column": "y"}], "color_by": "g"})
    labels = [g for g, _ in cd.groups]
    assert len(labels) == 12 and f"of {n:,} rows" in cd.summary()
    assert "not shown" in cd.note


def test_split_and_colour_by_a_true_false_column(tmp_path):
    from dancr.views.chartquery import query_panels
    lf = pl.LazyFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 2.0, 3.0, 4.0], "ok": [True, False, True, False]})
    schema = dict(lf.collect_schema())
    panels = query_panels(lf, schema, {"kind": "line", "x": "x", "series": [{"column": "y"}], "split_by": "ok"})
    assert len(panels) == 2 and all(cd.line.series[0].x.size == 2 for _, cd in panels)


def test_series_palette_is_shared_between_the_view_and_the_renderer():
    from dancr.views import palette, render

    assert render.PALETTE == palette.SERIES_COLORS
    assert palette.series_color(0) == palette.SERIES_COLORS[0]
    assert palette.series_color(3, {"color": "#000000"}) == "#000000"
    assert palette.series_color(len(palette.SERIES_COLORS)) == palette.SERIES_COLORS[0]


def test_column_title_is_formatted_in_one_place():
    from dancr.views.table import column_title

    assert column_title("x", {"x": {"label": "Pressure", "unit": "psi"}}) == "Pressure (psi)"
    assert column_title("x", {"x": {"label": "Pressure"}}) == "Pressure"
    assert column_title("x", None) == "x"
    assert column_title("", None) == ""


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


def test_a_dropped_series_does_not_shift_the_others_colour_and_label():
    from dancr.views.chartquery import query_one
    lf = pl.LazyFrame({"x": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0], "c": [3.0, 2.0, 1.0]})
    spec = {"kind": "line", "x": "x",
            "series": [{"column": "gone", "label": "Gone"}, {"column": "b", "label": "Bee"},
                       {"column": "c", "label": "Cee"}]}
    cd = query_one(lf, dict(lf.collect_schema()), spec)
    assert cd.ys == ["b", "c"]                                  # the vanished series is left out
    assert [s.get("label") for s in cd.series_specs] == ["Bee", "Cee"]   # and the rest keep their labels
