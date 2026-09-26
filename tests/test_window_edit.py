"""Editing steps in the window: adding, settings, inputs, deleting with undo, and the settings widgets."""
import polars as pl
from PySide6.QtCore import Qt

from dancr.core import Pipeline
from helpers import pump, wait_run


def test_add_node_autoconnects_and_undo(opened_window, app):
    opened_window.scene.select_node("tb")
    nid = opened_window.add_node("keep_rows")
    assert opened_window.doc.pipeline.inputs_of(nid) == {"in": ["tb"]}
    assert nid in opened_window.scene.nodes
    opened_window.doc.undo.undo()
    assert nid not in opened_window.doc.pipeline.nodes and nid not in opened_window.scene.nodes
    opened_window.doc.undo.redo()
    assert nid in opened_window.scene.nodes


def test_inspector_edit_commits_params(opened_window, app):
    opened_window.scene.select_node("tb")
    pump(app)
    ins = opened_window.inspector
    assert ins.nid == "tb"
    w = ins.widgets["every"]
    w.edit.setText("5m"); w.edit.textEdited.emit("5m")
    ins._commit()
    assert opened_window.doc.pipeline.nodes["tb"].params["every"] == "5m"
    assert opened_window.doc.dirty
    opened_window.doc.undo.undo()
    assert opened_window.doc.pipeline.nodes["tb"].params["every"] == "1m"
    assert w.value() == "1m"      # widget follows undo


def test_step_picker_adds_connected_node(opened_window, app):
    opened_window.scene.select_node("tb")
    opened_window.open_picker()
    opened_window.picker.search.setText("sort")
    pump(app)
    items = [opened_window.picker.list.item(i) for i in range(opened_window.picker.list.count()) if opened_window.picker.list.item(i).data(Qt.UserRole)]
    assert [i.data(Qt.UserRole) for i in items] == ["sort"]
    opened_window.picker._pick(items[0])
    pump(app)
    new = [n for n in opened_window.doc.pipeline.nodes if n.startswith("sort")]
    assert new and opened_window.doc.pipeline.inputs_of(new[0]) == {"in": ["tb"]}
    assert opened_window.scene.nodes[new[0]].problem is not None      # "Sort by is not set" shows on the node


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


def test_delete_toast_and_undo(window, app, sample):
    window._add_load_node(str(sample), None); wait_run(window, app)
    nid = window.current_table()
    window.delete_current()
    assert nid not in window.doc.pipeline.nodes and window.toast.isVisible()
    window.toast.button.click()
    assert nid in window.doc.pipeline.nodes


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


def test_column_combo_reuses_its_completer(window, app):
    import polars as pl
    from dancr.ui.widgets import ColumnCombo

    cb = ColumnCombo()
    comp = cb.completer()
    assert comp is not None
    cb.set_columns({"a": pl.Int64}, "any")
    cb.set_columns({"b": pl.Utf8}, "any")
    assert cb.completer() is comp


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
