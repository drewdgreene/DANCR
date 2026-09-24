"""The window does what the person asked (audit 2026-09-24, A13, A14, W1–W7, W10–W16)."""
import time
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QApplication

from helpers import pump, settle, wait_run


@pytest.fixture
def window(app, tmp_path):
    from PySide6.QtCore import QSettings
    from dancr.ui.mainwindow import MainWindow
    QSettings().setValue("tour_shown", True)
    w = MainWindow()
    w.show(); app.processEvents()
    yield w


def _csv(tmp_path, name="a.csv", n=50):
    pl.DataFrame({"t": [datetime(2024, 1, 1) + timedelta(minutes=i) for i in range(n)], "v": np.arange(n, dtype=float)}).write_csv(tmp_path / name)
    return str(tmp_path / name)


def test_delete_removes_the_selected_arrow_not_the_step_on_screen(window, app, tmp_path):
    doc = window.doc
    a = doc.add_node("load_file", 0, 0, params={"path": _csv(tmp_path)})
    s = doc.add_node("sort", 300, 0, params={"columns": ["v"]})
    doc.connect(a, s)
    window.show_node(s)
    pump(app, 50)
    window.scene.clearSelection()
    next(iter(window.scene.edges.values())).setSelected(True)
    window.delete_current()
    assert s in doc.pipeline.nodes and doc.pipeline.edges == []


def test_fixing_a_value_needs_the_real_rows(app):
    from dancr.ui.grid import Grid
    from dancr.views.table import TablePager
    g = Grid()
    lf = pl.LazyFrame({"v": [1.0, 2.0]})
    g.model.set_pager(TablePager(lf, rows=2), in_memory=True)       # a preview
    assert not g.can_fix_values()
    g.model.set_pager(TablePager(lf, rows=2), in_memory=False)      # the step's computed rows
    assert g.can_fix_values()


def test_a_stopped_run_is_not_reported_as_done(window, app, tmp_path, monkeypatch):
    from dancr.core.executor import Executor
    doc = window.doc
    doc.set_auto_run(False)
    doc.add_node("load_file", 0, 0, params={"path": _csv(tmp_path)})
    slow = Executor._run_node

    def crawl(self, *a, **k):
        time.sleep(0.5)
        return slow(self, *a, **k)
    monkeypatch.setattr(Executor, "_run_node", crawl)
    doc.add_node("sort", 300, 0, params={"columns": ["v"]}, connect_from=next(iter(doc.pipeline.nodes)))
    window.run()
    settle(app, lambda: doc.running)
    doc.stop()
    settle(app, lambda: not doc.running)
    pump(app, 50)
    assert doc.last_run_outcome == "stopped" and "Done" not in window.status.currentMessage()


def test_events_of_a_replaced_run_are_ignored(app, tmp_path):
    from dancr.ui.document import Document
    from dancr.core.executor import NodeState
    doc = Document()
    doc.add_node("enter_data", 0, 0, params={"columns": [{"name": "a", "type": "number"}], "rows": [[1]]})
    doc.run()
    old = doc._run
    doc.stop(wait=True)
    doc._run = None                                          # the project was replaced; that run is over
    seen = []
    doc.nodeState.connect(lambda *a: seen.append(a))
    old.event.emit({"type": "node_finished", "state": NodeState("ghost", status="failed")})
    app.processEvents()
    assert seen == [] and "ghost" not in doc._states_cache
    doc.undo.setClean(); doc.shutdown()


def test_split_and_colour_by_a_true_false_column(tmp_path):
    from dancr.views.chartquery import query_panels
    lf = pl.LazyFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 2.0, 3.0, 4.0], "ok": [True, False, True, False]})
    schema = dict(lf.collect_schema())
    panels = query_panels(lf, schema, {"kind": "line", "x": "x", "series": [{"column": "y"}], "split_by": "ok"})
    assert len(panels) == 2 and all(cd.line.series[0].x.size == 2 for _, cd in panels)


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


def test_a_theme_switch_during_a_run_keeps_stop_working(window, app, tmp_path, monkeypatch):
    from dancr.ui.mainwindow import MainWindow
    monkeypatch.setattr(type(window.doc), "running", property(lambda self: True))
    w2 = MainWindow(doc=window.doc)
    assert w2.a_stop.isEnabled() and not w2.a_run.isEnabled()
    monkeypatch.undo()
    w2.dispose_for_theme()


def test_escape_does_not_stop_a_run(window):
    assert window.a_stop.shortcut() != QKeySequence("Escape")


def test_the_wizard_starts_from_the_current_projects_files(window, app, tmp_path):
    window.wizard.paths = [str(tmp_path / "from-an-earlier-project.csv")]
    window.wizard.begin()
    assert str(tmp_path / "from-an-earlier-project.csv") not in window.wizard.paths


def test_zoned_axis_labels_follow_daylight_saving(app):
    from dancr.ui.chartview import ZonedDateAxis
    axis = ZonedDateAxis("Europe/Oslo")
    before = datetime(2024, 3, 31, 0, 30, tzinfo=__import__("zoneinfo").ZoneInfo("UTC")).timestamp()   # 01:30 CET
    after = before + 2 * 3600                                                                              # 04:30 CEST
    axis.zoomLevel = type("Z", (), {"tickSpecs": [type("S", (), {"spacing": 1, "format": "%H:%M"})()]})()
    assert axis.tickStrings([before, after], 1, 1) == ["01:30", "04:30"]


def test_report_charts_use_the_columns_zone(tmp_path):
    from dancr.views.render import _wall_times
    x = np.array([datetime(2024, 7, 1, 10, tzinfo=__import__("zoneinfo").ZoneInfo("UTC")).timestamp()])
    assert str(_wall_times(x, "Europe/Oslo")[0]).startswith("2024-07-01T12:00")
    assert str(_wall_times(x)[0]).startswith("2024-07-01T10:00")


def test_copied_headers_have_no_type_marks(app):
    from dancr.ui.grid import Grid
    from dancr.views.table import TablePager
    g = Grid()
    notes = []
    g.notice.connect(notes.append)
    lf = pl.LazyFrame({"v": [1.0]})
    g.model.set_pager(TablePager(lf, rows=1), column_meta={"v": {"label": "Flow", "unit": "L/s"}})
    g.copy_all_visible()
    settle(app, lambda: any(m.startswith("Copied") for m in notes))
    assert QApplication.clipboard().text().split("\n")[0] == "Flow (L/s)"


def test_map_runs_go_through_the_windows_checks(window, app, monkeypatch):
    calls = []
    monkeypatch.setattr(window, "run", lambda targets=None, force=False: calls.append(targets))
    window.scene.runRequested.connect(lambda t: None)
    window.scene.runRequested.emit(["x"])
    pump(app, 20)
    assert calls == [["x"]]


def test_moving_a_connection_is_checked_without_copying_the_project(app, tmp_path, monkeypatch):
    from dancr.ui.document import Document
    from dancr.core import Pipeline
    doc = Document()
    a = doc.add_node("enter_data", 0, 0)
    s1 = doc.add_node("sort", 300, 0); s2 = doc.add_node("sort", 300, 100)
    doc.connect(a, s1)
    monkeypatch.setattr(Pipeline, "from_dict", classmethod(lambda cls, *a, **k: (_ for _ in ()).throw(AssertionError("copied"))))
    doc.move_edge(doc.pipeline.edges[0], s2, "in")
    doc.set_input("rate", 2)
    doc.set_column_meta("v", "Flow", "L/s")
    assert doc.pipeline.inputs_of(s2) == {"in": [a]}
    monkeypatch.undo()
    doc.undo.setClean(); doc.shutdown()
