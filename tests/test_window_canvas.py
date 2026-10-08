"""Phase-1 canvas-first UI: the empty-canvas invite, the Sources tray, node-card findings, the Insight bar."""
from datetime import datetime, timedelta

import numpy as np
import polars as pl

from helpers import pump, settle, wait_run


def _csv(tmp_path, n=40):
    pl.DataFrame({"time": [datetime(2024, 1, 1) + timedelta(minutes=i) for i in range(n)],
                  "value": np.arange(n, dtype=float)}).write_csv(tmp_path / "a.csv")
    return str(tmp_path / "a.csv")


def test_empty_project_shows_the_drop_invite(window, app):
    window.new_pipeline()                       # a new project: the empty canvas, not the start page
    pump(app, 50)
    assert window.root.currentWidget() is window.workspace
    assert window.view.empty.isVisible()        # "Drop your spreadsheets here"
    assert not window.sources.isVisible()       # no sources yet


def test_sources_tray_lists_and_hides(window, app, tmp_path):
    path = _csv(tmp_path)
    window._add_load_node(path, None)
    wait_run(window, app)
    pump(app, 50)
    assert window.sources.isVisible()
    assert len(window.sources._chips) == 1
    nid = window.current_table()
    window.doc.remove_nodes([nid])
    pump(app, 50)
    assert not window.sources.isVisible() or not window.sources._chips


def test_sources_tray_collapses_until_two_sources(window, app, tmp_path):
    window._add_load_node(_csv(tmp_path), None)
    wait_run(window, app); pump(app, 80)
    assert window.sources.isVisible()
    assert not window.sources.scroll.isVisibleTo(window.sources)      # one source: collapsed by default
    pl.DataFrame({"x": [1, 2, 3]}).write_csv(tmp_path / "b.csv")
    window._add_load_node(str(tmp_path / "b.csv"), None)
    wait_run(window, app); pump(app, 120)
    assert window.sources.scroll.isVisibleTo(window.sources)          # two sources: opened to compare them
    window.sources.toggle.setChecked(False)                           # a manual choice sticks
    pump(app, 20)
    assert not window.sources.scroll.isVisibleTo(window.sources)
    assert not window.sources._auto


def test_node_card_carries_the_finding(window, app, tmp_path):
    path = _csv(tmp_path)
    src = window._add_load_node(path, None)
    wait_run(window, app)
    nid = window.add_node("check_limits", None, params={"column": "value", "min": 0.0}, connect_from=src)
    wait_run(window, app)
    item = window.scene.nodes[nid]
    assert item.status == "done"
    assert item.finding                     # the plain sentence the step produced


def test_insight_bar_shows_the_selected_finding(window, app, tmp_path):
    path = _csv(tmp_path)
    src = window._add_load_node(path, None)
    wait_run(window, app)
    nid = window.add_node("check_limits", None, params={"column": "value", "min": 0.0}, connect_from=src)
    wait_run(window, app)
    window.show_node(nid)
    pump(app, 50)
    assert window.insight.isVisible() and window.insight.label.text()
    assert window.insight._node == nid


def test_ai_steps_are_marked_and_removable(window, app):
    window.new_pipeline(); pump(app, 20)
    nid = window.add_node("enter_data", None, params={"columns": [{"name": "x", "type": "number"}], "rows": [[1]]})
    window.doc.add_ai_steps([nid]); pump(app, 20)
    assert nid in window.doc.ai_steps() and window.scene.nodes[nid].is_ai
    assert window.a_remove_ai.isEnabled()
    window.remove_ai_steps(); pump(app, 20)
    assert nid not in window.doc.pipeline.nodes and not window.doc.ai_steps()
    window.doc.undo.undo(); pump(app, 20)                 # one undo brings it back
    assert nid in window.doc.pipeline.nodes


def test_ghost_proposal_draws_and_clears(window, app):
    window.new_pipeline(); pump(app, 20)
    window.scene.show_ghost([{"type": "keep_rows", "title": "Filter rows"},
                             {"type": "chart", "title": "Chart"}])
    assert window.scene._ghost_items            # dashed, non-committed cards on the canvas
    assert not window.doc.pipeline.nodes        # nothing was added to the project
    window.scene.clear_ghost()
    assert not window.scene._ghost_items


def test_result_panel_expands_and_restores(window, app, tmp_path):
    path = _csv(tmp_path)
    window._add_load_node(path, None); wait_run(window, app); pump(app, 40)
    window.result_expand.setChecked(True); pump(app, 40)
    sizes = window.outer_split.sizes()
    assert sizes[1] > sizes[0]                  # the result takes most of the window
    window.result_expand.setChecked(False); pump(app, 40)
    assert window.outer_split.sizes()[0] > 90   # the canvas is back on top


def test_relations_button_and_build(window, app, tmp_path):
    pl.DataFrame({"id": [1, 2, 3], "name": ["A", "B", "C"]}).write_csv(tmp_path / "customers.csv")
    pl.DataFrame({"customer_id": [1, 1, 2, 3], "amount": [10.0, 20.0, 30.0, 40.0]}).write_csv(tmp_path / "orders.csv")
    a = window._add_load_node(str(tmp_path / "customers.csv"), None)
    b = window._add_load_node(str(tmp_path / "orders.csv"), None)
    wait_run(window, app)
    settle(app, lambda: window.understanding.full and window.understanding.model is not None, 30)
    assert window.sources.relate_btn.isEnabled()
    rel = next((r for r in window.understanding.model.relations if r.kind == "link"), None)
    assert rel is not None
    window.build_relation(rel)
    pump(app, 40)
    combos = [n for n in window.doc.pipeline.nodes.values() if n.type == "combine"]
    assert combos
    ins = window.doc.pipeline.inputs_of(combos[0].id)
    assert set(ins.get("left", [])) | set(ins.get("right", [])) == {a, b}


def test_insight_bar_jumps_to_its_step(window, app, tmp_path):
    path = _csv(tmp_path)
    src = window._add_load_node(path, None)
    wait_run(window, app)
    nid = window.add_node("check_limits", None, params={"column": "value", "min": 0.0}, connect_from=src)
    wait_run(window, app)
    window.insight.set_insight("a finding", nid)
    window.insight.jumped.emit(nid)
    pump(app, 20)
    assert window._current == nid
