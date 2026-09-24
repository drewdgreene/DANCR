"""Headless GUI smoke tests (QT_QPA_PLATFORM=offscreen)."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from helpers import pump
from PySide6.QtCore import QPointF, Qt
from PySide6.QtWidgets import QApplication

from dancr.core import Pipeline


@pytest.fixture
def window(app, tmp_path, probe_dir):
    from dancr.ui.mainwindow import MainWindow
    p = Pipeline("gui")
    p.add_node("load_file", "A", {"path": str(probe_dir / "probe_A.csv")}, id="a", x=0, y=0)
    p.add_node("time_buckets", params={"every": "1m"}, id="tb", x=300, y=0)
    p.connect("a", "tb")
    path = p.save(tmp_path / "gui.json")
    w = MainWindow()
    w.open_path(path)
    w.show()
    app.processEvents()
    yield w
    w.doc.undo.setClean()
    w.close()


def test_open_builds_scene(window, app):
    assert set(window.scene.nodes) == {"a", "tb"}
    assert len(window.scene.edges) == 1


def test_add_node_autoconnects_and_undo(window, app):
    window.scene.select_node("tb")
    nid = window.add_node("keep_rows")
    assert window.doc.pipeline.inputs_of(nid) == {"in": ["tb"]}
    assert nid in window.scene.nodes
    window.doc.undo.undo()
    assert nid not in window.doc.pipeline.nodes and nid not in window.scene.nodes
    window.doc.undo.redo()
    assert nid in window.scene.nodes


def test_inspector_edit_commits_params(window, app):
    window.scene.select_node("tb")
    pump(app)
    ins = window.inspector
    assert ins.nid == "tb"
    w = ins.widgets["every"]
    w.edit.setText("5m"); w.edit.textEdited.emit("5m")
    ins._commit()
    assert window.doc.pipeline.nodes["tb"].params["every"] == "5m"
    assert window.doc.dirty
    window.doc.undo.undo()
    assert window.doc.pipeline.nodes["tb"].params["every"] == "1m"
    assert w.value() == "1m"      # widget follows undo


def test_run_and_results(window, app):
    window.run()
    import time
    t = time.time()
    while window.doc.running and time.time() - t < 30:
        app.processEvents()
    assert not window.doc.running
    assert window.doc.state("tb").status == "done"
    window.show_node("tb")
    pump(app, 200)
    assert window.table.grid.model.rowCount() == window.doc.state("tb").rows
    window.add_node("chart", None, params={"kind": "line", "series": [{"column": window.doc.state("tb").schema_names[1]}]}, connect_from="tb")
    pump(app, 800)
    assert window.pages.currentWidget() is window.chart
    assert window.chart.panels[0].items, "chart drew nothing"


def test_file_drop_creates_loader(window, app, probe_dir):
    window._file_dropped(str(probe_dir / "probe_B.csv"), QPointF(10, 300))
    loaders = [n for n in window.doc.pipeline.nodes.values() if n.type == "load_file"]
    assert len(loaders) == 2
    import time
    t = time.time()
    while window.doc.running and time.time() - t < 30:
        app.processEvents()


def test_connect_via_scene_validation(window, app):
    nid = window.add_node("sort", QPointF(600, 0))
    for e in list(window.doc.pipeline.edges):        # new steps attach to the current table; detach for this test
        if e.target == nid:
            window.doc.disconnect(e)
    # connect tb -> sort by simulating the port drag end
    src = window.scene.nodes["tb"].output
    dst = window.scene.nodes[nid].inputs[0]
    window.scene.begin_connection(src)
    window.scene.end_connection(dst.scenePos())
    assert window.doc.pipeline.inputs_of(nid) == {"in": ["tb"]}
    # a loop must be refused with a status message
    msgs = []
    window.scene.status.connect(msgs.append)
    window.scene.begin_connection(window.scene.nodes[nid].output)
    window.scene.end_connection(window.scene.nodes["tb"].inputs[0].scenePos())
    assert msgs and ("loop" in msgs[0].lower() or "already connected" in msgs[0].lower())


def test_external_edit_reloads(window, app, tmp_path):
    p = Pipeline.load(window.doc.path)
    p.add_node("summarize", id="ext", x=0, y=500)
    p.save()
    import time
    t = time.time()
    while "ext" not in window.scene.nodes and time.time() - t < 5:
        app.processEvents()
    assert "ext" in window.scene.nodes


def test_step_picker_adds_connected_node(window, app):
    window.scene.select_node("tb")
    window.open_picker()
    window.picker.search.setText("sort")
    pump(app)
    items = [window.picker.list.item(i) for i in range(window.picker.list.count()) if window.picker.list.item(i).data(Qt.UserRole)]
    assert [i.data(Qt.UserRole) for i in items] == ["sort"]
    window.picker._pick(items[0])
    pump(app)
    new = [n for n in window.doc.pipeline.nodes if n.startswith("sort")]
    assert new and window.doc.pipeline.inputs_of(new[0]) == {"in": ["tb"]}
    assert window.scene.nodes[new[0]].problem is not None      # "Sort by is not set" shows on the node


def test_close_during_run_waits(window, app, probe_dir):
    window.run(force=True)
    assert window.doc.running
    window.doc.undo.setClean()
    from PySide6.QtWidgets import QMessageBox
    import dancr.ui.mainwindow as mw
    orig = QMessageBox.question
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
    try:
        window.close()
    finally:
        QMessageBox.question = orig
    assert not window.doc.running


def test_views_wait_for_run(window, app):
    from dancr.ui.workers import run_gate
    window.run(force=True)
    assert not run_gate.is_set()
    window.show_node("tb"); window.table.tabs.setCurrentIndex(1)
    pump(app, 100)
    if window.doc.running:
        assert "Waiting for the run" in window.table.summary_overlay.text()
    import time
    t = time.time()
    while window.doc.running and time.time() - t < 60:
        app.processEvents()
    assert run_gate.is_set()
    pump(app, 1500)
    assert not window.table.summary_overlay.isVisible() or "Waiting" not in window.table.summary_overlay.text()


def test_async_table_pages_arrive(window, app):
    window.run()
    import time
    t = time.time()
    while window.doc.running and time.time() - t < 30:
        app.processEvents()
    window.show_node("a"); window.table.tabs.setCurrentIndex(0)
    pump(app, 300)
    m = window.table.grid.model
    assert not m.in_memory
    first = m.data(m.index(0, 1), Qt.DisplayRole)
    t = time.time()
    while m.data(m.index(0, 1), Qt.DisplayRole) in ("·", "…") and time.time() - t < 10:
        app.processEvents()
    assert m.data(m.index(0, 1), Qt.DisplayRole) not in ("·", "…", "")
    tip = m.headerData(1, Qt.Horizontal, Qt.ToolTipRole)
    assert "blank:" in tip and "min:" in tip


def test_save_as_keeps_cache(window, app, tmp_path):
    window.run()
    import time
    t = time.time()
    while window.doc.running and time.time() - t < 30:
        app.processEvents()
    assert window.doc.state("tb").status == "done"
    window.doc.save(tmp_path / "renamed.json")
    assert window.doc.state("tb").status == "done"
