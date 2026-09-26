"""Runs in the window: results, progress, waiting views, and stopping."""
import time
from datetime import datetime, timedelta

import numpy as np
import polars as pl
from PySide6.QtGui import QKeySequence

from helpers import pump, settle


def _csv(tmp_path, name="a.csv", n=50):
    pl.DataFrame({"t": [datetime(2024, 1, 1) + timedelta(minutes=i) for i in range(n)], "v": np.arange(n, dtype=float)}).write_csv(tmp_path / name)
    return str(tmp_path / name)


def test_run_and_results(opened_window, app):
    opened_window.run()
    import time
    t = time.time()
    while opened_window.doc.running and time.time() - t < 30:
        app.processEvents()
    assert not opened_window.doc.running
    assert opened_window.doc.state("tb").status == "done"
    opened_window.show_node("tb")
    pump(app, 200)
    assert opened_window.table.grid.model.rowCount() == opened_window.doc.state("tb").rows
    opened_window.add_node("chart", None, params={"kind": "line", "series": [{"column": opened_window.doc.state("tb").schema_names[1]}]}, connect_from="tb")
    pump(app, 800)
    assert opened_window.pages.currentWidget() is opened_window.chart
    assert opened_window.chart.panels[0].items, "chart drew nothing"


def test_close_during_run_waits(opened_window, app, probe_dir):
    opened_window.run(force=True)
    assert opened_window.doc.running
    opened_window.doc.undo.setClean()
    from PySide6.QtWidgets import QMessageBox
    orig = QMessageBox.question
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
    try:
        opened_window.close()
    finally:
        QMessageBox.question = orig
    assert not opened_window.doc.running


def test_views_wait_for_run(opened_window, app):
    from dancr.ui.workers import run_gate
    opened_window.run(force=True)
    assert not run_gate.is_set()
    opened_window.show_node("tb"); opened_window.table.tabs.setCurrentIndex(1)
    pump(app, 100)
    if opened_window.doc.running:
        assert "Waiting for the run" in opened_window.table.summary_overlay.text()
    import time
    t = time.time()
    while opened_window.doc.running and time.time() - t < 60:
        app.processEvents()
    assert run_gate.is_set()
    pump(app, 1500)
    assert not opened_window.table.summary_overlay.isVisible() or "Waiting" not in opened_window.table.summary_overlay.text()


def test_progress_bar_advances_within_a_step(window, app):
    import time as _time
    window._on_run_started(); window._tick.stop()
    window._run_progress(1, 4, "Big step")           # the 2nd of 4 steps owns 250..500
    window._run_step_start = _time.monotonic() - 2.0; window._run_est = 2.0
    window._tick_progress(); v1 = window.progress_bar.value()
    window._run_step_start = _time.monotonic() - 6.0
    window._tick_progress(); v2 = window.progress_bar.value()
    assert window.progress_bar.maximum() == 1000
    assert 250 <= v1 < v2 < 500                       # moves inside the step, never past its slot
    window.progress.hide()


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


def test_escape_does_not_stop_a_run(window):
    assert window.a_stop.shortcut() != QKeySequence("Escape")
