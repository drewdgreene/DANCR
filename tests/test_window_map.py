"""The map of steps and the rail that lists them: building, connecting, deleting, zooming and navigating."""
from datetime import datetime, timedelta

import numpy as np
import polars as pl
from PySide6.QtCore import QPointF, Qt

from dancr.ui.rail import INPUTS_ROLE, KIND_ROLE
from helpers import pump, wait_run


def _csv(tmp_path, name="a.csv", n=50):
    pl.DataFrame({"t": [datetime(2024, 1, 1) + timedelta(minutes=i) for i in range(n)], "v": np.arange(n, dtype=float)}).write_csv(tmp_path / name)
    return str(tmp_path / name)


def test_open_builds_scene(opened_window, app):
    assert set(opened_window.scene.nodes) == {"a", "tb"}
    assert len(opened_window.scene.edges) == 1


def test_connect_via_scene_validation(opened_window, app):
    nid = opened_window.add_node("sort", QPointF(600, 0))
    for e in list(opened_window.doc.pipeline.edges):        # new steps attach to the current table; detach for this test
        if e.target == nid:
            opened_window.doc.disconnect(e)
    # connect tb -> sort by simulating the port drag end
    src = opened_window.scene.nodes["tb"].output
    dst = opened_window.scene.nodes[nid].inputs[0]
    opened_window.scene.begin_connection(src)
    opened_window.scene.end_connection(dst.scenePos())
    assert opened_window.doc.pipeline.inputs_of(nid) == {"in": ["tb"]}
    # a loop must be refused with a status message
    msgs = []
    opened_window.scene.status.connect(msgs.append)
    opened_window.scene.begin_connection(opened_window.scene.nodes[nid].output)
    opened_window.scene.end_connection(opened_window.scene.nodes["tb"].inputs[0].scenePos())
    assert msgs and ("loop" in msgs[0].lower() or "already connected" in msgs[0].lower())


def test_result_panel_closes_to_a_handle_and_reopens(window, app, sample):
    window._add_load_node(str(sample), None); wait_run(window, app)
    assert window.result_box.isVisible() and not window.result_handle.isVisible()
    window.a_result.setChecked(False); pump(app, 100)
    assert not window.result_box.isVisible() and window.result_handle.isVisible()
    window.result_handle.click(); pump(app, 100)
    assert window.result_box.isVisible() and not window.result_handle.isVisible()
    assert window.outer_split.sizes()[1] >= 140


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


def test_result_panel_runs_full_width_below_the_canvas(window, app, sample):
    """The canvas is the main stage; the result drawer spans the whole window width below it."""
    window.doc.set_auto_run(False)
    window._add_load_node(str(sample), None)
    pump(app, 120)
    result_g = window.result_box.geometry()
    assert result_g.x() == 0 and abs(result_g.width() - window.width()) <= 2
    assert window.outer_split.widget(0).geometry().bottom() <= result_g.top() + 2
    assert result_g.height() > 80


def test_map_dot_grid_has_zoom_independent_spacing():
    from PySide6.QtCore import QRectF
    from dancr.ui.canvas import dot_grid
    rect = QRectF(-500, -300, 1000, 600)
    for scale in (0.15, 0.4, 1.0, 2.5):
        minor, major = dot_grid(rect, scale)
        assert minor and major
        xs = sorted({round(p.x(), 6) for p in minor})
        gap = min(b - a for a, b in zip(xs, xs[1:]))
        assert 10 <= gap * scale <= 40          # minor dots stay readable on screen at every zoom
        assert all(round(p.x(), 6) in xs for p in major)  # major dots sit on the minor grid
    assert dot_grid(rect, 0.0) == ([], [])


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


def test_map_runs_go_through_the_windows_checks(window, app, monkeypatch):
    calls = []
    monkeypatch.setattr(window, "run", lambda targets=None, force=False: calls.append(targets))
    window.scene.runRequested.connect(lambda t: None)
    window.scene.runRequested.emit(["x"])
    pump(app, 20)
    assert calls == [["x"]]
