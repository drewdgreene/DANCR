"""The chart page: chips, panels, axes and previews, and a chart that follows its input."""
import time
from datetime import datetime

import polars as pl

from dancr.core.samples import write_sample
from helpers import pump, settle, wait_run


def test_chart_page_chips_and_add_to_report(window, app, sample):
    window._add_load_node(str(sample), None); wait_run(window, app)
    window.steps.chart_columns(["value A", "value B"])
    cid = window._current
    wait_run(window, app); pump(app, 1200)
    assert window.chart.panels[0].items, "chart drew nothing"
    assert "value A" in window.chart.y_chip.text()
    window.chart._set({"kind": "histogram", "column": "value A"})
    pump(app, 1200)
    assert "Histogram" in window.chart.kind_chip.text() and window.chart.panels[0].items
    window.chart._toggle_series("temperature", True)
    assert [s["column"] for s in window.doc.pipeline.nodes[cid].params["series"]][-1] == "temperature"
    window.steps.add_to_report(cid)
    rid = window._current
    assert window.doc.pipeline.nodes[rid].type == "report" and window.pages.currentWidget() is window.report
    assert window.doc.pipeline.inputs_of(rid) == {"items": [cid]}
    assert window.report.blocks.count() == 1
    window.report.title.setText("Test report"); window.report._commit_text()
    assert window.doc.pipeline.nodes[rid].params["title"] == "Test report" and window.doc.pipeline.nodes[rid].title == "Test report"
    window.doc.set_params(rid, {"blocks": [{"type": "heading", "text": "Section"}, {"type": "item", "index": 0}]})
    pump(app, 100)
    assert window.report.blocks.count() == 2
    window.report.blocks.setCurrentRow(0); window.report._remove_block()
    assert window.doc.pipeline.nodes[rid].params["blocks"] == [{"type": "item", "node": cid}]


def test_chart_clears_when_its_step_is_removed(window, app, sample):
    window._add_load_node(str(sample), None); wait_run(window, app)
    window.steps.chart_columns(["value A"]); wait_run(window, app); pump(app, 800)
    cid = window._current
    assert window.chart.nid == cid
    window.doc.remove_nodes([cid]); pump(app, 100)
    assert window.chart.nid is None            # the deleted step's binding is dropped, not left stale


def test_split_by_panels(window, app, tmp_path):
    from dancr.views.render import render_chart
    sample = write_sample(tmp_path / "big", rows=40_000)        # spans both sites
    window._add_load_node(str(sample), None); wait_run(window, app)
    window.steps.chart_columns(["value A"])
    cid = window._current
    window.chart._set({"split_by": "location"})
    wait_run(window, app); pump(app, 1500)
    assert len(window.chart.panels) == 2
    assert all(p.items for p in window.chart.panels)
    assert window.chart.panels[1].plot.getViewBox().linkedView(0) is window.chart.panels[0].plot.getViewBox()
    assert "2 panels" in window.chart.info.text()
    window.chart._set({"split_by": ""}); pump(app, 1200)
    assert len(window.chart.panels) == 1
    # the PNG renderer used by reports draws the same panels
    from dancr.core.executor import Executor
    lf = Executor(window.doc.pipeline).frame(window.current_table())
    png = render_chart(lf, {**window.doc.pipeline.nodes[cid].params, "split_by": "location"}, tmp_path / "split.png", width=800, height=400)
    assert png.exists() and png.stat().st_size > 5000


def test_window_can_shrink_on_chart_page(window, app, sample):
    window._add_load_node(str(sample), None); wait_run(window, app)
    window.steps.chart_columns(["value A", "value B"]); wait_run(window, app); pump(app, 800)
    assert window.minimumSizeHint().width() < 700
    window.resize(760, 600); pump(app, 200)
    assert window.width() <= 780


def test_chart_chip_menus_are_reused_not_recreated(window, app, sample):
    """Every refresh rebuilds the chip menus in place; a fresh QMenu per refresh would leak."""
    window._add_load_node(str(sample), None); wait_run(window, app)
    window.steps.chart_columns(["value A"])
    wait_run(window, app); pump(app, 800)
    menu = window.chart.kind_menu
    assert window.chart.kind_chip.menu() is menu
    window.chart.refresh(); window.chart.refresh(); pump(app, 50)
    assert window.chart.kind_menu is menu and window.chart.kind_chip.menu() is menu


def test_chart_preview_that_cannot_be_built_shows_a_message(window, app, tmp_path):
    """Opening a chart whose input cannot be previewed (here a one-row fit) must not crash; it shows
    a calm 'Preview unavailable' overlay instead of a background-task traceback."""
    import polars as pl
    one = tmp_path / "one.parquet"
    pl.DataFrame({"x": [1.0], "y": [2.0]}).write_parquet(one)
    window.doc.set_auto_run(False)
    src = window.doc.add_node("load_file", 0, 0, params={"path": str(one)}, title="One")
    fit = window.doc.add_node("fit_curve", 200, 0, params={"x": "x", "y": "y", "kind": "linear"}, title="Fit")
    window.doc.connect(src, fit)
    ch = window.doc.add_node("chart", 400, 0, params={"kind": "scatter", "x": "x", "series": [{"column": "y"}]}, title="Chart")
    window.doc.connect(fit, ch)
    window.show_node(ch)
    t = time.time()
    while time.time() - t < 5 and "couldn't build a preview" not in window.chart.overlay.text().lower():
        app.processEvents(); time.sleep(0.02)
    assert "couldn't build a preview" in window.chart.overlay.text().lower()


def test_a_bar_charts_labels_do_not_stay(window, app, tmp_path):
    doc = window.doc
    pl.DataFrame({"c": ["a", "b"], "v": [1.0, 2.0]}).write_csv(tmp_path / "b.csv")
    src = doc.add_node("load_file", 0, 0, params={"path": str(tmp_path / "b.csv")})
    ch = doc.add_node("chart", 300, 0, params={"kind": "bar", "category": "c", "value": "v"}, connect_from=src)
    wait_run(window, app)
    window.show_node(ch)
    settle(app, lambda: window.chart.plot.getAxis("bottom")._tickLevels is not None)
    doc.set_params(ch, {"kind": "line", "x": "v", "series": [{"column": "v"}]})
    settle(app, lambda: window.chart.plot.getAxis("bottom")._tickLevels is None)


def test_zoned_axis_labels_follow_daylight_saving(app):
    from dancr.ui.chartview import ZonedDateAxis
    axis = ZonedDateAxis("Europe/Oslo")
    before = datetime(2024, 3, 31, 0, 30, tzinfo=__import__("zoneinfo").ZoneInfo("UTC")).timestamp()   # 01:30 CET
    after = before + 2 * 3600                                                                              # 04:30 CEST
    axis.zoomLevel = type("Z", (), {"tickSpecs": [type("S", (), {"spacing": 1, "format": "%H:%M"})()]})()
    assert axis.tickStrings([before, after], 1, 1) == ["01:30", "04:30"]


def test_a_chart_follows_when_its_input_is_rewired(app, tmp_path):
    from dancr.ui.mainwindow import MainWindow
    pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]}).write_csv(tmp_path / "a.csv")
    pl.DataFrame({"x": [1.0, 2.0], "z": [5.0, 6.0]}).write_csv(tmp_path / "b.csv")
    w = MainWindow(); w.show()
    doc = w.doc
    a = doc.add_node("load_file", 0, 0, params={"path": str(tmp_path / "a.csv")}, title="Table A")
    b = doc.add_node("load_file", 0, 100, params={"path": str(tmp_path / "b.csv")}, title="Table B")
    c = doc.add_node("chart", 300, 0, params={"kind": "line", "x": "x"})
    doc.connect(a, c)
    w.chart.set_node(c)
    settle(app, lambda: "Table A" in w.chart.source.text())
    doc.undo.beginMacro("rewire")
    doc.disconnect(next(e for e in doc.pipeline.edges if e.target == c))
    doc.connect(b, c)
    doc.undo.endMacro()
    settle(app, lambda: "Table B" in w.chart.source.text() and "z" in w.chart.schema)
    doc.undo.setClean(); w.close()
