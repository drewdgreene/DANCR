"""The window shows the right thing and undo puts things back exactly (review 2026-09-24, phase 5)."""
import sys
import threading
import time

import polars as pl
import pytest

from helpers import pump, settle
from PySide6.QtCore import QCoreApplication, QEvent, QItemSelectionModel
from PySide6.QtWidgets import QApplication

from dancr.core import Pipeline, PipelineError


@pytest.fixture
def doc(app, tmp_path):
    from dancr.ui.document import Document
    d = Document()
    yield d
    d.undo.setClean()
    d.shutdown()


@pytest.fixture
def stacked(doc, tmp_path):
    """Three tables feeding one stack step, in the order 1, 2, 3."""
    for i in (1, 2, 3):
        pl.DataFrame({"v": [float(i)]}).write_csv(tmp_path / f"t{i}.csv")
    ids = [doc.add_node("load_file", 0, i * 100, params={"path": str(tmp_path / f"t{i}.csv")}) for i in (1, 2, 3)]
    st = doc.add_node("stack", 300, 100)
    for s in ids:
        doc.connect(s, st, "tables")
    assert doc.pipeline.inputs_of(st)["tables"] == ids
    return ids, st


# ------------------------------------------------------------------ undo keeps the order of inputs
def test_undoing_a_delete_puts_inputs_back_in_order(doc, stacked):
    ids, st = stacked
    doc.remove_nodes([ids[0]])
    assert doc.pipeline.inputs_of(st)["tables"] == ids[1:]
    doc.undo.undo()
    assert doc.pipeline.inputs_of(st)["tables"] == ids
    doc.undo.redo(); doc.undo.undo()
    assert doc.pipeline.inputs_of(st)["tables"] == ids


def test_undoing_a_multi_delete_puts_inputs_back_in_order(doc, stacked):
    ids, st = stacked
    doc.remove_nodes([ids[2], ids[0]])
    doc.undo.undo()
    assert doc.pipeline.inputs_of(st)["tables"] == ids


def test_undoing_a_disconnect_puts_the_input_back_in_place(doc, stacked):
    ids, st = stacked
    edge = next(e for e in doc.pipeline.edges if e.source == ids[1])
    doc.disconnect(edge)
    doc.undo.undo()
    assert doc.pipeline.inputs_of(st)["tables"] == ids


# ------------------------------------------------------------------ moving a connection
def test_moving_a_connection_onto_a_taken_input_changes_nothing(doc, tmp_path):
    pl.DataFrame({"v": [1.0]}).write_csv(tmp_path / "a.csv")
    a = doc.add_node("load_file", 0, 0, params={"path": str(tmp_path / "a.csv")})
    b = doc.add_node("load_file", 0, 100, params={"path": str(tmp_path / "a.csv")})
    s1 = doc.add_node("sort", 300, 0)
    s2 = doc.add_node("sort", 300, 100)
    doc.connect(a, s1); doc.connect(b, s2)
    edges_before = [e.key() for e in doc.pipeline.edges]
    depth = doc.undo.index()
    edge = next(e for e in doc.pipeline.edges if e.source == a)
    with pytest.raises(PipelineError):
        doc.move_edge(edge, s2, "in")                  # s2.in is already fed by b
    assert [e.key() for e in doc.pipeline.edges] == edges_before
    assert doc.undo.index() == depth                   # nothing to undo either
    doc.disconnect(next(e for e in doc.pipeline.edges if e.source == b))
    doc.move_edge(edge, s2, "in")
    assert doc.pipeline.inputs_of(s2) == {"in": [a]} and doc.pipeline.inputs_of(s1) == {}
    doc.undo.undo()                                    # one step back: the move
    assert doc.pipeline.inputs_of(s1) == {"in": [a]} and doc.pipeline.inputs_of(s2) == {}


# ------------------------------------------------------------------ theme switch
def test_a_theme_switch_leaves_no_listeners_behind(app, tmp_path, monkeypatch):
    from dancr.ui.app import _rebuild_for_theme
    from dancr.ui.mainwindow import MainWindow
    from dancr.ui.tableview import TableView
    import shiboken6
    errors = []
    monkeypatch.setattr(sys, "excepthook", lambda *e: errors.append(e))
    pl.DataFrame({"v": [1.0, 2.0]}).write_csv(tmp_path / "a.csv")
    w = MainWindow(); w.show()
    w._add_load_node(str(tmp_path / "a.csv"), None)
    ref = [w]
    for _ in range(3):
        _rebuild_for_theme(app, ref)
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        pump(app, 30)
    nid = next(iter(ref[0].doc.pipeline.nodes))
    ref[0].doc.rename(nid, "Renamed")
    ref[0].doc.undo.undo(); ref[0].doc.undo.redo()
    pump(app, 100)
    assert errors == []
    import gc
    gc.collect()
    live = [o for o in gc.get_objects() if isinstance(o, TableView) and shiboken6.isValid(o) and o.doc is ref[0].doc]
    assert len(live) == 1
    ref[0].doc.undo.setClean(); ref[0].close()


# ------------------------------------------------------------------ a chart follows its input
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


# ------------------------------------------------------------------ only the latest background result lands
def test_an_older_result_already_queued_is_not_delivered(app):
    from dancr.ui.workers import Serial, view_pool
    got = []
    s = Serial(waits_for_run=False)
    s.submit(lambda: "old", got.append)
    view_pool().waitForDone(5000)                       # "old" is done and its signal is queued
    release = threading.Event()
    s.submit(lambda: (release.wait(5), "new")[1], got.append)
    release.set()
    view_pool().waitForDone(5000)
    pump(app, 100)
    assert got == ["new"]


# ------------------------------------------------------------------ copy reads the table, not the screen
def test_copy_takes_every_row_from_the_table(app, tmp_path):
    from dancr.ui.grid import Grid
    from dancr.views.table import TablePager
    n = 30_000                                          # far more than the pages the grid keeps
    df = pl.DataFrame({"i": range(n), "x": [i / 3 for i in range(n)], "s": [None if i % 7 == 0 else f"r{i}" for i in range(n)]})
    df.write_parquet(tmp_path / "t.parquet")
    lf = pl.scan_parquet(tmp_path / "t.parquet")
    g = Grid()
    notes = []
    g.notice.connect(notes.append)
    g.model.set_pager(TablePager(lf, rows=n))
    g.copy_all_visible()
    settle(app, lambda: any(m.startswith("Copied") for m in notes))
    lines = QApplication.clipboard().text().split("\n")
    assert len(lines) == n + 1 and "·" not in QApplication.clipboard().text()
    assert lines[1].split("\t") == ["0", "0.0", ""]                  # a blank stays blank
    assert lines[2].split("\t")[1] == repr(1 / 3)                    # every digit
    g.table.selectRow(5); g.table.selectionModel().select(g.model.index(20_000, 0), QItemSelectionModel.Select | QItemSelectionModel.Rows)
    notes.clear()
    g.copy_selection()
    settle(app, lambda: any(m.startswith("Copied") for m in notes))
    lines = QApplication.clipboard().text().split("\n")
    assert [ln.split("\t")[0] for ln in lines[1:]] == ["5", "20000"]


# ------------------------------------------------------------------ settings typed a moment ago
def test_a_setting_typed_a_moment_ago_is_not_lost_or_misplaced(app, tmp_path):
    from dancr.ui.mainwindow import MainWindow
    pl.DataFrame({"v": [1.0]}).write_csv(tmp_path / "a.csv")
    w = MainWindow(); w.show()
    doc = w.doc
    nid = doc.add_node("sort", 0, 0)
    w.inspector.set_node(nid)
    w.inspector._pending = {"descending": True}
    w.inspector._timer.start()                          # typed, not yet committed
    path = doc.save(tmp_path / "p.json")
    assert Pipeline.load(path).nodes[nid].params["descending"] is True   # saving took it
    other = Pipeline("other"); other.add_node("sort", id=nid)
    w.inspector._pending = {"descending": True}; w.inspector._timer.start()
    doc.replace_pipeline(other)                          # e.g. File → Open
    assert doc.pipeline.nodes[nid].params["descending"] is False         # never lands on the new project
    doc.undo.setClean(); w.close()


# ------------------------------------------------------------------ the periodic check runs in the background
def test_the_periodic_state_check_runs_off_the_gui_thread_and_drops_stale_answers(doc, tmp_path, app, monkeypatch):
    from dancr.core.executor import Executor
    pl.DataFrame({"v": [1.0]}).write_csv(tmp_path / "a.csv")
    doc.add_node("load_file", 0, 0, params={"path": str(tmp_path / "a.csv")})
    doc.save(tmp_path / "p.json")
    threads = []
    orig = Executor.states
    monkeypatch.setattr(Executor, "states", lambda self: (threads.append(threading.current_thread()), orig(self))[1])
    doc._poll_states()
    settle(app, lambda: doc._poll_task is None)
    assert threads and threads[-1] is not threading.main_thread()
    doc._poll_states()
    doc.refresh_states()                                 # an edit refreshed meanwhile: the poll's answer is old
    before = dict(doc._states_cache)
    settle(app, lambda: doc._poll_task is None)
    assert doc._states_cache == before


# ------------------------------------------------------------------ undo after Save As into another folder
def test_undo_after_save_as_elsewhere_still_points_at_the_same_files(doc, tmp_path):
    a, b = tmp_path / "A", tmp_path / "B"
    a.mkdir(); b.mkdir()
    pl.DataFrame({"v": [1.0]}).write_csv(a / "data.csv")
    pl.DataFrame({"v": [1.0]}).write_csv(a / "other.csv")
    doc.save(a / "p.json")
    keep = doc.add_node("load_file", 0, 0, params={"path": "data.csv"})
    gone = doc.add_node("load_file", 0, 100, params={"path": "other.csv"})
    doc.set_params(keep, {"path": "other.csv"})
    doc.remove_nodes([gone])
    doc.save(b / "p.json")
    assert doc.pipeline.nodes[keep].params["path"] == str(a / "other.csv")
    doc.undo.undo()                                     # the delete
    assert doc.pipeline.nodes[gone].params["path"] == str(a / "other.csv")
    doc.undo.undo()                                     # the setting
    assert doc.pipeline.nodes[keep].params["path"] == str(a / "data.csv")
    doc.undo.redo()
    assert doc.pipeline.nodes[keep].params["path"] == str(a / "other.csv")


def test_save_as_leaves_no_lease_behind(doc, tmp_path):
    from dancr.core.executor import LIVE_DIR
    pl.DataFrame({"v": [1.0]}).write_csv(tmp_path / "a.csv")
    doc.add_node("load_file", 0, 0, params={"path": str(tmp_path / "a.csv")})
    doc.save(tmp_path / "one.json")
    doc.executor.run()
    doc.refresh_states()                                # holds its results
    assert list((doc.executor.cache_dir / LIVE_DIR).glob("*.json"))
    doc.save(tmp_path / "two.json")
    doc.executor.release()
    assert not list((doc.executor.cache_dir / LIVE_DIR).glob("*.json"))
