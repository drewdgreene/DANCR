"""The table page: pages that arrive in the background, header and cell actions that create steps, copying and
typed-in tables."""
import polars as pl
from PySide6.QtCore import QItemSelectionModel, Qt
from PySide6.QtWidgets import QApplication, QInputDialog

from helpers import pump, settle, wait_run


def test_async_table_pages_arrive(opened_window, app):
    opened_window.run()
    import time
    t = time.time()
    while opened_window.doc.running and time.time() - t < 30:
        app.processEvents()
    opened_window.show_node("a"); opened_window.table.tabs.setCurrentIndex(0)
    pump(app, 300)
    m = opened_window.table.grid.model
    assert not m.in_memory
    t = time.time()
    while m.data(m.index(0, 1), Qt.DisplayRole) in ("·", "…") and time.time() - t < 10:
        app.processEvents()
    assert m.data(m.index(0, 1), Qt.DisplayRole) not in ("·", "…", "")
    tip = m.headerData(1, Qt.Horizontal, Qt.ToolTipRole)
    assert "blank:" in tip and "min:" in tip


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


def test_cell_fix_creates_fix_values_step(window, app, sample, monkeypatch):
    window._add_load_node(str(sample), None); wait_run(window, app)
    src = window.current_table()
    answers = iter([("999.5", True), ("typo in the log", True)])
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: next(answers)))
    from dancr.core.executor import Executor
    was = Executor(window.doc.pipeline).frame(src).collect()["value A"][2]     # the window passes the cell's own value
    window.steps.cell_action("fix", 3, "value A", was)
    fx = window.doc.pipeline.nodes[window._current]
    assert fx.type == "fix_values"
    assert fx.params["fixes"] == [{"row": 3, "column": "value A", "value": "999.5", "was": str(was), "note": "typo in the log"}]
    # a second fix on the same table lands in the same step
    answers = iter([("1", True), ("", True)])
    window.show_node(src)
    window.steps.cell_action("fix", 5, "location", "north")
    assert window._current == fx.id and len(fx.params["fixes"]) == 2
    wait_run(window, app)
    df = Executor(window.doc.pipeline).frame(fx.id).collect()
    assert df["value A"][2] == 999.5 and df["location"][4] == "1"


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


def test_fixing_a_value_needs_the_real_rows(app):
    from dancr.ui.grid import Grid
    from dancr.views.table import TablePager
    g = Grid()
    lf = pl.LazyFrame({"v": [1.0, 2.0]})
    g.model.set_pager(TablePager(lf, rows=2), in_memory=True)       # a preview
    assert not g.can_fix_values()
    g.model.set_pager(TablePager(lf, rows=2), in_memory=False)      # the step's computed rows
    assert g.can_fix_values()


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
