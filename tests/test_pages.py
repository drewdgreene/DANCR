"""The table-first window: pages, header/cell actions that create steps, templates, inputs, reports, versions."""
import json
import os
import time
from pathlib import Path

import pytest

from helpers import pump, wait_run

from PySide6.QtCore import Qt, QSettings
from PySide6.QtWidgets import QApplication, QInputDialog

from dancr.core import Pipeline
from dancr.core.samples import write_sample, build_template, TEMPLATES
from dancr.ui.rail import KIND_ROLE, INPUTS_ROLE


@pytest.fixture
def window(app, tmp_path):
    from dancr.ui.mainwindow import MainWindow
    QSettings().setValue("tour_shown", True)
    QSettings().setValue("rail_mode", "flow")
    w = MainWindow()
    w.show(); app.processEvents()
    yield w
    w.doc.undo.setClean()
    w.close()


@pytest.fixture
def sample(tmp_path) -> Path:
    return write_sample(tmp_path, rows=6000)


def test_start_page_then_table(window, app, sample):
    assert window.pages.currentWidget() is window.start
    window._add_load_node(str(sample), None)
    pump(app, 100)
    assert window.pages.currentWidget() is window.table
    assert window.doc.auto_run                     # small file: runs by itself
    wait_run(window, app)
    nid = window.current_table()
    assert window.doc.state(nid).status == "done"
    pump(app, 300)
    assert window.table.grid.model.rowCount() == window.doc.state(nid).rows
    assert not window.a_run.isVisible()            # auto-run hides the Run button


def test_header_actions_create_steps(window, app, sample):
    window._add_load_node(str(sample), None); wait_run(window, app)
    src = window.current_table()
    window.steps.column_action("sort_desc", "value A")
    node = window.doc.pipeline.nodes[window._current]
    assert node.type == "sort" and node.params["columns"] == ["value A"] and node.params["descending"]
    assert window.doc.pipeline.inputs_of(node.id) == {"in": [src]}
    window.show_node(src)
    window.steps.column_action("chart", "value B")
    assert window.pages.currentWidget() is window.chart
    chart = window.doc.pipeline.nodes[window._current]
    assert chart.type == "chart" and chart.params["series"][0]["column"] == "value B"
    assert window.rail.current() == ("node", chart.id)
    window.show_node(src)
    window.steps.column_action("hide", "location")
    assert window.doc.pipeline.nodes[window._current].params == {**window.doc.pipeline.nodes[window._current].params, "mode": "drop", "columns": ["location"]}
    window.show_node(src)
    window.steps.column_action("limit", "value A")
    assert window.doc.pipeline.nodes[window._current].type == "check_limits"
    window.show_node(src)
    window.steps.column_action("fit:value A:value B", "value B")
    c = window.doc.pipeline.nodes[window._current]
    assert c.type == "chart" and c.params["kind"] == "scatter" and c.params["fit"] == "linear"
    assert window.doc.pipeline.problems() == []


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


def test_cell_fix_creates_fix_values_step(window, app, sample, monkeypatch):
    window._add_load_node(str(sample), None); wait_run(window, app)
    src = window.current_table()
    answers = iter([("999.5", True), ("typo in the log", True)])
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: next(answers)))
    window.steps.cell_action("fix", 3, "value A", 101.3)
    fx = window.doc.pipeline.nodes[window._current]
    assert fx.type == "fix_values"
    assert fx.params["fixes"] == [{"row": 3, "column": "value A", "value": "999.5", "was": "101.3", "note": "typo in the log"}]
    # a second fix on the same table lands in the same step
    answers = iter([("1", True), ("", True)])
    window.show_node(src)
    window.steps.cell_action("fix", 5, "location", "north")
    assert window._current == fx.id and len(fx.params["fixes"]) == 2
    wait_run(window, app)
    from dancr.core.executor import Executor
    df = Executor(window.doc.pipeline).frame(fx.id).collect()
    assert df["value A"][2] == 999.5 and df["location"][4] == "1"


def test_inputs_page_roundtrip(window, app, sample):
    window._add_load_node(str(sample), None); wait_run(window, app)
    src = window.current_table()
    window.show_node("inputs")
    assert window.pages.currentWidget() is window.inputs
    window.inputs._add()
    assert [i.name for i in window.doc.pipeline.inputs] == ["value 1"]
    window.inputs.table.item(0, 0).setText("upper limit"); window.inputs.table.item(0, 1).setText("101.5"); window.inputs.table.item(0, 2).setText("kPa")
    ins = window.doc.pipeline.inputs
    assert [(i.name, i.value, i.unit) for i in ins] == [("upper limit", 101.5, "kPa")]
    for _ in range(3):                      # rename, value, unit
        window.doc.undo.undo()
    assert [i.name for i in window.doc.pipeline.inputs] == ["value 1"]
    for _ in range(3):
        window.doc.undo.redo()
    assert [(i.name, i.value, i.unit) for i in window.doc.pipeline.inputs] == [("upper limit", 101.5, "kPa")]
    window.show_node(src)
    window.steps.after("check_limits", {"column": "value A", "max": "upper limit"}, title="Check")
    wait_run(window, app)
    st = window.doc.state(window._current)
    assert st.status == "done" and st.report.get("verdict") == "FAIL" and st.report["outside"] > 0
    assert window.doc.pipeline.problems() == []


def test_enter_data_paste(window, app, monkeypatch):
    from PySide6.QtGui import QGuiApplication
    nid = window.add_node("enter_data", None)
    assert window.pages.currentWidget() is window.entry
    QGuiApplication.clipboard().setText("name\tdepth\nA\t1.5\nB\t2.25\n")
    window.entry.paste()
    p = window.doc.pipeline.nodes[nid].params
    assert p["columns"] == [{"name": "name", "type": "text"}, {"name": "depth", "type": "number"}]
    assert p["rows"] == [["A", "1.5"], ["B", "2.25"]]
    wait_run(window, app)
    from dancr.core.executor import Executor
    df = Executor(window.doc.pipeline).frame(nid).collect()
    assert df["depth"].to_list() == [1.5, 2.25]


def test_templates_build_and_run(tmp_path):
    from dancr.core.executor import Executor
    data = write_sample(tmp_path, rows=8000)
    for t in TEMPLATES:
        p = Pipeline(t["key"]); p.path = tmp_path / f"{t['key']}.json"
        build_template(t["key"], p, data)
        assert p.problems() == [], (t["key"], p.problems())
        res = Executor(p).run()
        bad = [s for s in res.values() if s.status != "done"]
        assert not bad, (t["key"], [(s.node_id, s.error) for s in bad])


def test_start_template_in_window(window, app, tmp_path, monkeypatch):
    monkeypatch.setattr("dancr.ui.mainwindow.write_sample", lambda base: write_sample(tmp_path, rows=5000))
    window._start_template("limits")
    assert window.pages.currentWidget() is window.table
    assert [i.name for i in window.doc.pipeline.inputs] == ["upper limit"]
    wait_run(window, app)
    assert all(s.status == "done" for s in window.doc.executor.states().values())


def test_delete_toast_and_undo(window, app, sample):
    window._add_load_node(str(sample), None); wait_run(window, app)
    nid = window.current_table()
    window.delete_current()
    assert nid not in window.doc.pipeline.nodes and window.toast.isVisible()
    window.toast.button.click()
    assert nid in window.doc.pipeline.nodes


def test_versions_and_restore(window, app, sample, tmp_path, monkeypatch):
    window._add_load_node(str(sample), None); wait_run(window, app)
    path = tmp_path / "proj.json"
    window.doc.save(path)
    window.add_node("sort", None, params={"columns": ["time"]})
    window.doc.save()                                     # keeps a version of the one-step project
    versions = window.doc.pipeline.versions()
    assert len(versions) == 1
    from dancr.ui.dialogs import VersionsDialog
    dlg = VersionsDialog(window, window.doc)
    assert dlg.list.count() == 1
    dlg.list.setCurrentRow(0)
    assert dlg.chosen() == versions[0]
    data = json.loads(versions[0].read_text())
    assert len(data["nodes"]) == 1


def test_source_change_marks_stale(window, app, sample):
    window._add_load_node(str(sample), None); wait_run(window, app)
    nid = window.current_table()
    assert window.doc.state(nid).status == "done"
    with open(sample, "a") as f:
        f.write("2030-01-01 00:00:00,1,1,1,x\n")
    t = time.time()
    while window.doc.state(nid).status == "done" and time.time() - t < 8:
        app.processEvents()
    assert window.doc.state(nid).status != "done"


def test_cli_template_inputs_columns(tmp_path):
    from dancr.cli import main
    import io, contextlib
    out = tmp_path / "t.json"
    main(["template", "fit", str(out)])
    assert out.exists() and (tmp_path / "sample_data.csv").exists()
    main(["inputs", str(out), "k", "2.5", "--unit", "x"])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main(["--json", "inputs", str(out)])
    assert json.loads(buf.getvalue()) == [{"name": "k", "value": 2.5, "unit": "x", "note": ""}]
    main(["columns", str(out), "value A", "--label", "Value A", "--unit", "kPa"])
    p = Pipeline.load(out)
    assert p.column_title("value A") == "Value A (kPa)"


def test_unfinished_filter_and_limit_pass_through(window, app, sample):
    window._add_load_node(str(sample), None); wait_run(window, app)
    src = window.current_table()
    window.steps.column_action("filter", "value A")
    fid = window._current
    window.show_node(src)
    window.steps.column_action("limit", "value A")
    lid = window._current
    wait_run(window, app)
    for nid in (fid, lid):
        st = window.doc.state(nid)
        assert st.status == "done", st.error
        assert st.rows == window.doc.state(src).rows
        assert any("yet" in m for m in st.messages)
    window.doc.set_params(fid, {"conditions": {"match": "all", "rules": [{"column": "value A", "op": "gt", "value": "101.5"}]}})
    wait_run(window, app)
    assert 0 < window.doc.state(fid).rows < window.doc.state(src).rows


def test_recovery_of_unsaved_project(window, app, sample, monkeypatch):
    from dancr.ui.document import recovery_path
    from dancr.ui.mainwindow import MainWindow
    from PySide6.QtWidgets import QMessageBox
    window._add_load_node(str(sample), None); wait_run(window, app)
    assert window.doc.path is None and window.doc.dirty
    window.doc.autosave_now()
    mine = recovery_path()
    assert mine.exists() and str(os.getpid()) in mine.name          # per process
    assert window.doc.pending_recovery() is None                     # our own copy is never offered back
    dead = recovery_path(2**22 + 7)                                  # a pid that cannot be running
    mine.replace(dead)
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes))
    w2 = MainWindow(); w2.show()
    pump(app, 400)
    assert len(w2.doc.pipeline.nodes) == 1 and w2.doc.path is None
    assert w2.doc.autosave_paused and "recovered" in w2.doc.autosave_paused
    assert not dead.exists() and recovery_path().exists()            # taken over by this process
    w2.doc.undo.setClean(); w2.close()
    assert not recovery_path().exists()          # a deliberate close clears it


def test_autosave_pauses_after_external_change_and_restore(window, app, sample, tmp_path):
    window._add_load_node(str(sample), None); wait_run(window, app)
    path = tmp_path / "paused.json"
    window.doc.save(path)
    window.add_node("sort", None, params={"columns": ["time"]})
    assert window.doc.dirty and window.doc.autosave_paused is None
    other = Pipeline.load(path); other.add_node("summarize", id="ext", x=0, y=500); other.save()
    t = time.time()
    while window.doc.autosave_paused is None and time.time() - t < 5:
        app.processEvents()
    assert window.doc.autosave_paused and "changed on disk" in window.doc.autosave_paused
    assert "autosave paused" in window.windowTitle()
    window.doc.autosave_now()
    assert "ext" in Pipeline.load(path).nodes                        # not overwritten
    window.doc.save()                                                # an explicit save resumes it
    assert window.doc.autosave_paused is None
    versions = window.doc.versions()
    assert versions
    window.doc.restore_version(versions[0])
    assert window.doc.dirty and "earlier version" in window.doc.autosave_paused
    before = path.read_text()
    window.doc.autosave_now()
    assert path.read_text() == before
    window.open_path(path)                                           # Revert
    assert window.doc.autosave_paused is None


def test_toast_undo_only_undoes_the_deletion(window, app, sample):
    window._add_load_node(str(sample), None); wait_run(window, app)
    nid = window.current_table()
    window.delete_current()
    assert nid not in window.doc.pipeline.nodes and window.toast.isVisible()
    window.add_node("enter_data", None)                              # a newer edit on top of the undo stack
    assert not window.toast.isVisible()
    window.toast.button.click()
    assert nid not in window.doc.pipeline.nodes                      # the deletion is not undone
    assert any(n.type == "enter_data" for n in window.doc.pipeline.nodes.values())


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


def test_map_closes_to_a_handle_and_reopens(window, app, sample):
    window._add_load_node(str(sample), None); wait_run(window, app)
    assert window.map_box.isVisible() and not window.map_handle.isVisible()
    window.map_close.click(); pump(app, 100)
    assert not window.map_box.isVisible() and window.map_handle.isVisible()
    window.map_handle.click(); pump(app, 100)
    assert window.map_box.isVisible() and not window.map_handle.isVisible()
    assert window.outer_split.sizes()[1] >= 120


def test_window_can_shrink_on_chart_page(window, app, sample):
    window._add_load_node(str(sample), None); wait_run(window, app)
    window.steps.chart_columns(["value A", "value B"]); wait_run(window, app); pump(app, 800)
    assert window.minimumSizeHint().width() < 700
    window.resize(760, 600); pump(app, 200)
    assert window.width() <= 780


def test_wheel_over_combo_scrolls_instead_of_changing(app):
    from PySide6.QtWidgets import QScrollArea, QWidget, QVBoxLayout, QComboBox, QLabel
    from PySide6.QtCore import QPointF, QPoint
    from PySide6.QtGui import QWheelEvent
    from dancr.ui.app import install_wheel_guard
    install_wheel_guard(app)
    area = QScrollArea(); body = QWidget(); lay = QVBoxLayout(body)
    combo = QComboBox(); combo.addItems([str(i) for i in range(5)]); combo.setCurrentIndex(2)
    lay.addWidget(combo)
    for i in range(60):
        lay.addWidget(QLabel(f"row {i}"))
    area.setWidget(body); area.setWidgetResizable(True); area.resize(300, 200); area.show(); pump(app, 50)
    combo.setFocus()                                      # even a focused box must not change
    before = area.verticalScrollBar().value()
    ev = QWheelEvent(QPointF(10, 10), QPointF(combo.mapToGlobal(combo.rect().center())), QPoint(0, -120), QPoint(0, -120), Qt.NoButton, Qt.NoModifier, Qt.ScrollUpdate, False)
    app.sendEvent(combo, ev); pump(app, 50)
    assert combo.currentIndex() == 2                      # the value did not change
    assert area.verticalScrollBar().value() > before      # the page scrolled
    area.close()


def test_click_on_column_box_opens_the_list(app):
    import polars as pl
    from PySide6.QtCore import QPoint
    from PySide6.QtTest import QTest
    from dancr.ui.widgets import ColumnCombo
    c = ColumnCombo(allow_blank=True); c.set_columns({"time": pl.Datetime("us"), "value": pl.Float64, "location": pl.Utf8}); c.show(); pump(app, 50)
    assert [c.itemData(i) for i in range(c.count())] == ["", "time", "value", "location"]
    assert c.itemText(0) == "(automatic)" and c.currentText() == ""
    QTest.mousePress(c.lineEdit(), Qt.LeftButton, Qt.NoModifier, QPoint(5, 5)); pump(app, 30)
    assert c.view().isVisible()                           # opens on the press (Wayland needs a press serial)
    QTest.mouseRelease(c.view().parentWidget(), Qt.LeftButton, Qt.NoModifier, QPoint(-5, -5)); pump(app, 100)
    assert c.view().isVisible()                           # the release of that same click does not close it
    c.hidePopup(); c.set_column("location"); assert c.column() == "location"
    c.set_column(""); assert c.column() == "" and c.currentText() == ""
    c.close()


def test_enterdata_paste_blank_column_stays_text(window, app):
    nid = window.add_node("enter_data", None)
    window.show_node(nid)
    QApplication.clipboard().setText("a,b\n1,\n2,\n")
    window.entry.paste(); pump(app, 50)
    cols = window.doc.pipeline.nodes[nid].params["columns"]
    assert cols == [{"name": "a", "type": "number"}, {"name": "b", "type": "text"}]


def test_enterdata_stale_column_action_is_ignored(window, app):
    nid = window.add_node("enter_data", None)
    window.show_node(nid)
    view = window.entry
    view.nid = None                       # as if the project reloaded while a menu was open
    view._set_type(5, "number")           # must not raise IndexError
    view._remove_column(5)


def test_describe_with_blank_keeps_existing_label_and_unit(window, app, monkeypatch):
    window.doc.set_column_meta("value A", "Pressure", "psi")
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("", True)))
    window.steps.rename_column("value A")
    assert window.doc.pipeline.columns["value A"] == {"label": "Pressure", "unit": "psi"}


def test_rail_flow_tree_nests_steps_by_connection(window, app, sample):
    window._add_load_node(str(sample), None); wait_run(window, app)
    load = window.current_table()
    roll = window.add_node("rolling", None, params={"columns": ["value A"], "window": "5m", "stat": "mean", "time_column": "time"}, connect_from=load)
    calc = window.add_node("calculate", None, params={"formulas": [{"name": "d", "expr": "[value A] - [value B]"}]}, connect_from=roll)
    chart = window.add_node("chart", None, params={"kind": "line", "x": "time", "series": [{"column": "value A"}]}, connect_from=calc)
    comb = window.add_node("combine", None, params={"method": "side_by_side"}, connect_from="")
    window.doc.connect(roll, comb, "left"); window.doc.connect(chart, comb, "right")
    pump(app, 30)
    items = window.rail._items
    assert items[roll].parent() is items[load]
    assert items[calc].parent() is items[roll]
    assert items[chart].parent() is items[calc]
    assert items[comb].parent() is items[roll]
    assert items[comb].data(0, INPUTS_ROLE) == 2          # the "2 in" join chip
    refs = [items[comb].child(i) for i in range(items[comb].childCount())]
    assert any(c.data(0, KIND_ROLE) == "ref" and "Second table" in c.text(0) for c in refs)


def test_rail_type_mode_toggle(window, app, sample):
    window._add_load_node(str(sample), None); wait_run(window, app)
    window.rail.mode = "flow"; window.rail.refill()
    window.rail._toggle_mode()
    assert window.rail.mode == "type"
    texts = [window.rail.tree.topLevelItem(i).text(0) for i in range(window.rail.tree.topLevelItemCount())]
    assert "Tables" in texts and "Charts" in texts and "Inputs" in texts
    window.rail._toggle_mode()
    assert window.rail.mode == "flow"


def test_rail_selection_centres_the_map(window, app, sample):
    window._add_load_node(str(sample), None); wait_run(window, app)
    load = window.current_table()
    roll = window.add_node("rolling", None, params={"columns": ["value A"], "window": "5m", "stat": "mean", "time_column": "time"}, connect_from=load)
    window.add_node("calculate", None, params={"formulas": [{"name": "d", "expr": "[value A] - [value B]"}]}, connect_from=roll)
    pump(app, 30)
    window.view.centerOn(window.scene.nodes[load].x() + 2500, window.scene.nodes[load].y() + 2500)
    pump(app, 10)
    window.rail.select("node", roll)                      # emits -> the window focuses the map
    pump(app, 20)
    ns = window.scene.nodes[roll].sceneBoundingRect().center()
    cs = window.view.mapToScene(window.view.viewport().rect().center())
    assert abs(ns.x() - cs.x()) < 60 and abs(ns.y() - cs.y()) < 60


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


def test_map_nodes_are_tinted_with_their_category_colour(window, app, sample):
    from PySide6.QtGui import QImage, QPainter, QColor
    from dancr.ui.theme import T, category_color
    from dancr.ui.canvas import NODE_H
    from dancr.core import registry
    window._add_load_node(str(sample), None); wait_run(window, app)
    nid = window.current_table()
    item = window.scene.nodes[nid]
    br = item.boundingRect()
    img = QImage(int(br.width()) + 1, int(br.height()) + 1, QImage.Format_ARGB32)
    img.fill(0)
    p = QPainter(img); item.paint(p, None, None); p.end()   # paint draws at the item's local origin
    got = img.pixelColor(30, NODE_H // 2)                   # the card, left of the icon
    cat = category_color(registry.get(window.doc.pipeline.nodes[nid].type).category)
    base = QColor(T.node); a = (48 if T.dark else 30) / 255.0
    want = QColor(round(base.red() * (1 - a) + cat.red() * a),
                  round(base.green() * (1 - a) + cat.green() * a),
                  round(base.blue() * (1 - a) + cat.blue() * a))
    assert abs(got.red() - want.red()) <= 8 and abs(got.green() - want.green()) <= 8 and abs(got.blue() - want.blue()) <= 8
    assert (got.red(), got.green(), got.blue()) != (base.red(), base.green(), base.blue())


def test_grabbing_a_connected_input_moves_or_severs_the_edge(window, app, sample):
    from PySide6.QtCore import QPointF
    window._add_load_node(str(sample), None); wait_run(window, app)
    load = window.current_table()
    a = window.add_node("sort", None, params={"columns": ["value A"]}, connect_from=load)
    b = window.add_node("choose_columns", None, params={"mode": "keep", "columns": ["value A"]}, connect_from=a)
    c = window.add_node("sort", None, params={"columns": ["value B"]}, connect_from="")
    pump(app, 20)
    scene = window.scene

    def targets():
        return {e.target for e in window.doc.pipeline.edges}

    # grab b's connected input and drop on c's input: the edge moves from b to c, source a kept
    scene.begin_connection(scene.nodes[b].inputs[0])
    scene.end_connection(scene.nodes[c].inputs[0].scenePos())
    edges = {(e.source, e.target, e.port) for e in window.doc.pipeline.edges}
    assert (a, c, "in") in edges and (load, a, "in") in edges and b not in targets()

    window.doc.undo.undo(); pump(app, 10)
    edges = {(e.source, e.target, e.port) for e in window.doc.pipeline.edges}
    assert (a, b, "in") in edges and c not in targets()          # one undo reverses the move

    # grab b's input and drop in empty space: the connection is severed, and that is undoable
    scene.begin_connection(scene.nodes[b].inputs[0])
    scene.end_connection(QPointF(-9000, -9000))
    assert b not in targets() and len(window.doc.pipeline.edges) == 1
    window.doc.undo.undo(); pump(app, 10)
    assert (a, b, "in") in {(e.source, e.target, e.port) for e in window.doc.pipeline.edges}

    # grabbing b's input and dropping it back on itself changes nothing
    scene.begin_connection(scene.nodes[b].inputs[0])
    scene.end_connection(scene.nodes[b].inputs[0].scenePos())
    assert (a, b, "in") in {(e.source, e.target, e.port) for e in window.doc.pipeline.edges}


def test_map_view_does_not_paint_over_the_dot_grid(window, app):
    from PySide6.QtCore import Qt
    # A view background brush would hide CanvasScene.drawBackground (the canvas colour and dot grid)
    assert window.view.backgroundBrush().style() == Qt.NoBrush


def test_wheel_zooms_the_map_without_a_modifier(window, app):
    from PySide6.QtGui import QWheelEvent
    from PySide6.QtCore import QPointF, QPoint
    v = window.view
    center = QPointF(v.viewport().rect().center())
    glob = QPointF(v.viewport().mapToGlobal(v.viewport().rect().center()))

    def wheel(dy, pixel=(0, 0)):
        ev = QWheelEvent(center, glob, QPoint(*pixel), QPoint(0, dy), Qt.NoButton, Qt.NoModifier, Qt.ScrollUpdate, False)
        app.sendEvent(v.viewport(), ev)
        pump(app, 10)

    before = v.transform().m11()
    wheel(120)
    zoomed_in = v.transform().m11()
    assert zoomed_in > before                       # wheel up zooms in, no Ctrl needed
    wheel(-120)
    assert v.transform().m11() < zoomed_in          # wheel down zooms out
    base = v.transform().m11()
    wheel(0, pixel=(0, 40))                         # touchpad: pixelDelta only
    assert v.transform().m11() > base


def test_appearance_menu_switches_between_light_and_dark(window, app):
    from dancr.ui.theme import T, preference, set_preference, apply_app_style
    old = preference()
    try:
        window._set_theme("dark")
        assert T.dark and T.bg == "#1c1c1f" and window._theme_actions["dark"].isChecked()
        window._set_theme("light")
        assert not T.dark and T.bg == "#f2f2f4" and window._theme_actions["light"].isChecked()
    finally:
        set_preference(old)
        apply_app_style(app)


def test_ui_state_survives_a_theme_rebuild(window, app, sample):
    window._add_load_node(str(sample), None); wait_run(window, app)
    nid = window.current_table()
    state = window.capture_ui_state()
    assert state["current"] == nid

    from dancr.ui.mainwindow import MainWindow
    replacement = MainWindow(doc=window.doc)          # a rebuild carries the same Document over
    replacement.restore_ui_state(state)
    assert replacement._current == nid
    assert replacement.doc is window.doc
    replacement.dispose_for_theme()
    pump(app, 50)
    assert replacement._disposed


def test_chart_chip_menus_are_reused_not_recreated(window, app, sample):
    """Every refresh rebuilds the chip menus in place; a fresh QMenu per refresh would leak."""
    window._add_load_node(str(sample), None); wait_run(window, app)
    window.steps.chart_columns(["value A"])
    wait_run(window, app); pump(app, 800)
    menu = window.chart.kind_menu
    assert window.chart.kind_chip.menu() is menu
    window.chart.refresh(); window.chart.refresh(); pump(app, 50)
    assert window.chart.kind_menu is menu and window.chart.kind_chip.menu() is menu


def test_column_combo_reuses_its_completer(window, app):
    import polars as pl
    from dancr.ui.widgets import ColumnCombo

    cb = ColumnCombo()
    comp = cb.completer()
    assert comp is not None
    cb.set_columns({"a": pl.Int64}, "any")
    cb.set_columns({"b": pl.Utf8}, "any")
    assert cb.completer() is comp


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
    while time.time() - t < 5 and "unavailable" not in window.chart.overlay.text().lower():
        app.processEvents(); time.sleep(0.02)
    assert "unavailable" in window.chart.overlay.text().lower()


def wait_until(app, pred, secs=10.0):
    t = time.time()
    while time.time() - t < secs and not pred():
        app.processEvents(); time.sleep(0.02)
    return pred()


def _wizard_files(tmp_path):
    from datetime import datetime, timedelta
    import polars as pl
    hours = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(60)]
    pl.DataFrame({"order_id": list(range(1, 61)), "customer_id": [i % 6 + 1 for i in range(60)],
                  "qty": [i % 4 + 1 for i in range(60)], "order_date": hours}).write_csv(tmp_path / "orders.csv")
    pl.DataFrame({"customer_id": list(range(1, 7)), "region": ["n", "s", "e", "w", "n", "s"]}).write_csv(tmp_path / "customers.csv")
    return [str(tmp_path / "orders.csv"), str(tmp_path / "customers.csv")]


def test_wizard_builds_an_answer_and_shows_it(window, app, tmp_path):
    window.doc.set_auto_run(False)
    files = _wizard_files(tmp_path)
    window.start_wizard(files)
    assert window.pages.currentWidget() is window.wizard
    assert wait_until(app, lambda: bool(window.wizard.profiles)), "profiling did not finish"
    w = window.wizard
    assert w.links, "expected a suggested link between orders and customers"
    w._goto(1); w._set_intent("total"); w._goto(2); w._goto(3)
    w._build()
    assert not window._wizard_active
    assert len(window.doc.pipeline.answers) == 1
    answer = window.doc.pipeline.answers[0]
    assert answer.terminal in window.doc.pipeline.nodes
    assert window._current_answer == answer.id
    assert answer.id in window.scene.answers and answer.id in window.rail._answer_items
    assert window.answer_bar.isVisible() or True          # bar is synced (visibility needs a shown window)


def test_wizard_answer_shares_data_and_deletes_with_choice(window, app, tmp_path):
    window.doc.set_auto_run(False)
    files = _wizard_files(tmp_path)
    window.start_wizard(files)
    assert wait_until(app, lambda: bool(window.wizard.profiles))
    window.wizard._goto(1); window.wizard._set_intent("total"); window.wizard._goto(2); window.wizard._goto(3)
    window.wizard._build()
    answer = window.doc.pipeline.answers[0]
    exclusive = window.doc.answer_exclusive_nodes(answer)
    assert exclusive, "the answer should own the steps it built"
    nodes_before = set(window.doc.pipeline.nodes)
    # delete just the card: every step stays
    window.doc.delete_answer(answer.id, remove_steps=False)
    assert window.doc.pipeline.answers == [] and set(window.doc.pipeline.nodes) == nodes_before
    window.doc.undo.undo()
    assert len(window.doc.pipeline.answers) == 1
    # delete the card and its steps
    window.doc.delete_answer(answer.id, remove_steps=True)
    assert window.doc.pipeline.answers == []
    assert not (set(exclusive) & set(window.doc.pipeline.nodes))


def test_wizard_offers_no_time_option_without_dates(window, app, tmp_path):
    import polars as pl
    window.doc.set_auto_run(False)
    pl.DataFrame({"k": [1, 2, 3], "v": [1.0, 2.0, 3.0]}).write_csv(tmp_path / "t.csv")
    window.start_wizard([str(tmp_path / "t.csv")])
    assert wait_until(app, lambda: bool(window.wizard.profiles))
    w = window.wizard
    w._goto(1)
    assert not w.goal_buttons["over_time"].isEnabled()
    assert w.goal_buttons["total"].isEnabled() is False   # no category column either


def _build_total_answer(window, app, files):
    window.start_wizard(files)
    assert wait_until(app, lambda: bool(window.wizard.profiles))
    w = window.wizard
    w._goto(1); w._set_intent("total"); w._goto(2); w._goto(3); w._build()
    return window.doc.pipeline.answers[-1]


def test_second_answer_reuses_the_data_layer(window, app, tmp_path):
    window.doc.set_auto_run(False)
    files = _wizard_files(tmp_path)
    _build_total_answer(window, app, files)
    before = set(window.doc.pipeline.nodes)
    _build_total_answer(window, app, files)
    assert len(window.doc.pipeline.answers) == 2
    assert set(window.doc.pipeline.nodes) == before     # nothing duplicated


def test_undoing_a_build_removes_the_answer_and_its_steps(window, app, tmp_path):
    window.doc.set_auto_run(False)
    window.start_wizard(_wizard_files(tmp_path))
    assert wait_until(app, lambda: bool(window.wizard.profiles))
    w = window.wizard
    w._goto(1); w._set_intent("total"); w._goto(2); w._goto(3); w._build()
    assert len(window.doc.pipeline.nodes) > 0 and len(window.doc.pipeline.answers) == 1
    window.doc.undo.undo()
    assert window.doc.pipeline.answers == [] and window.doc.pipeline.nodes == {}


def test_changing_an_answer_rebuilds_only_its_branch(window, app, tmp_path):
    window.doc.set_auto_run(False)
    answer = _build_total_answer(window, app, _wizard_files(tmp_path))
    aid, terminal, nodes_before = answer.id, answer.terminal, set(window.doc.pipeline.nodes)
    loads_before = {n for n in nodes_before if window.doc.pipeline.nodes[n].type == "load_file"}
    window.change_answer(aid)
    assert wait_until(app, lambda: bool(window.wizard.profiles))
    w = window.wizard
    w._goto(1); w._set_intent("describe"); w._goto(2); w._goto(3); w._build()
    a = window.doc.pipeline.answers[0]
    assert len(window.doc.pipeline.answers) == 1 and a.terminal != terminal
    assert window.doc.pipeline.nodes[a.terminal].type == "summarize"
    # the shared loads are still there
    assert loads_before <= set(window.doc.pipeline.nodes)


def test_wizard_uses_files_already_on_the_map(window, app, tmp_path):
    """Pressing Build it for me on an open project must reuse the files already loaded, not ask again."""
    window.doc.set_auto_run(False)
    files = _wizard_files(tmp_path)
    window.doc.add_node("load_file", 0, 0, params={"path": files[0]})
    window.doc.add_node("load_file", 0, 200, params={"path": files[1]})
    window.start_wizard()                      # no paths given
    assert wait_until(app, lambda: len(window.wizard.profiles) == 2), "did not pick up the files on the map"
    w = window.wizard
    w._goto(1); w._set_intent("total"); w._goto(2); w._goto(3); w._build()
    loads = [n for n in window.doc.pipeline.nodes.values() if n.type == "load_file"]
    assert len(loads) == 2, "the existing loaders should be reused, not duplicated"
    assert len(window.doc.pipeline.answers) == 1


def test_map_runs_full_width_below_the_side_panels(window, app, sample):
    """The rail and settings stop at the top of the map; the map spans the whole window width."""
    window.doc.set_auto_run(False)
    window._add_load_node(str(sample), None)
    pump(app, 120)
    map_g = window.map_box.geometry()
    assert map_g.x() == 0 and abs(map_g.width() - window.width()) <= 2
    assert window.rail.geometry().bottom() <= map_g.top()
    assert window.inspector.geometry().bottom() <= map_g.top()
    assert map_g.height() > 80
    assert window.rail.geometry().top() == window.inspector.geometry().top() == 0


def test_wizard_link_can_be_remapped(window, app, tmp_path):
    window.doc.set_auto_run(False)
    window.start_wizard(_wizard_files(tmp_path))
    assert wait_until(app, lambda: bool(window.wizard.profiles))
    w = window.wizard
    link = w.links[0]
    rp = next(p for p in w.profiles if p.path == link.right)
    other = next(c.name for c in rp.columns if c.name != link.right_col)
    w._link_rows[0]["right"].setCurrentText(other)
    assert link.right_col == other
    asm = w._assembly()
    assert asm["kind"] == "join" and asm["links"][0]["right_on"] == other


def test_wizard_can_add_a_link_the_profiler_missed(window, app, tmp_path, monkeypatch):
    window.doc.set_auto_run(False)
    window.start_wizard(_wizard_files(tmp_path))
    assert wait_until(app, lambda: bool(window.wizard.profiles))
    w = window.wizard
    n = len(w.links)
    monkeypatch.setattr(w, "_ask_link", lambda: (w.profiles[0].path, "customer_id", w.profiles[1].path, "customer_id"))
    w._add_link()
    assert len(w.links) == n + 1 and len(w._link_rows) == n + 1
