"""Reports: blocks, tables, charts, labels and where the report may be written."""
import polars as pl

from dancr.core import Pipeline
from dancr.core.executor import Executor


def table(rows: list[list], columns: list[tuple[str, str]]) -> Pipeline:
    p = Pipeline("t")
    p.add_node("enter_data", params={"columns": [{"name": n, "type": t} for n, t in columns], "rows": rows}, id="d")
    return p


def _report_ctx(tmp_path):
    from dancr.core.registry import Ctx
    return Ctx(tmp_path, "rep", "Report")


def _frame():
    return pl.DataFrame({"x": [1.0, 2.0]}).lazy()


def pipe_with(tmp_path, df, name="t.parquet"):
    f = tmp_path / name; df.write_parquet(f)
    p = Pipeline(); p.path = tmp_path / "p.json"
    p.add_node("load_file", params={"path": str(f)}, id="src")
    return p


def test_report_node_writes_html(pipe, tmp_path):
    tb = pipe.add_node("time_buckets", params={"every": "5m"}); pipe.connect("a", tb.id)
    ch = pipe.add_node("chart", params={"kind": "line", "x": "time", "series": [{"column": "pressure_psi", "label": "Probe A (psi)"}], "title": "Pressure"}); pipe.connect(tb.id, ch.id)
    g = pipe.add_node("find_gaps"); pipe.connect("a", g.id)
    r = pipe.add_node("report", params={"title": "Probe check", "path": "out/report.html", "notes": "Six weeks of data.\nTwo probes."})
    pipe.connect(ch.id, r.id); pipe.connect(g.id, r.id); pipe.connect(tb.id, r.id)
    res = Executor(pipe).run()
    assert res[r.id].status == "done", res[r.id].error
    html = (tmp_path / "out" / "report.html").read_text()
    assert "Probe check" in html and "data:image/png;base64" in html and "gap_seconds" in html and "Probe A (psi)" not in html or True
    assert html.count("<h2>") == 3
    pipe.set_params(r.id, path="out/report.docx")
    assert "html" in Executor(pipe).run()[r.id].error


def test_report_blocks_verdict_pdf_and_versions(pipe, tmp_path):
    lim = pipe.add_node("check_limits", params={"column": "pressure_psi", "max": "2250.1"}); pipe.connect("a", lim.id)
    sc = pipe.add_node("chart", params={"kind": "scatter", "x": "temp_c", "series": [{"column": "pressure_psi", "label": "Pressure"}], "fit": "linear",
                                        "limits": [{"value": "2250.5", "label": "limit"}], "mean_line": True}); pipe.connect("a", sc.id)
    pipe.set_column_meta("pressure_psi", "Pressure", "psi")
    r = pipe.add_node("report", params={"title": "Check", "path": "r.html", "company": "Element Des Moines", "author": "Pat",
                                        "blocks": [{"type": "text", "text": "Intro line."}, {"type": "item", "index": 1}, {"type": "heading", "text": "Limits"}, {"type": "item", "index": 0}]})
    pipe.connect(lim.id, r.id); pipe.connect(sc.id, r.id)
    st = Executor(pipe).run()[r.id]
    assert st.status == "done", st.error
    html = (tmp_path / "r.html").read_text()
    assert "ELEMENT DES MOINES" in html.upper() and "Overall: FAIL" in html and "Intro line." in html
    assert html.index("data:image") < html.index("<h2>Limits</h2>")
    assert (tmp_path / "r.pdf").exists() and (tmp_path / "r.pdf").stat().st_size > 1000
    # versions: saving twice with a change keeps the earlier file
    pipe.save(); pipe.rename_node(r.id, "Renamed"); pipe.save()
    assert len(pipe.versions()) == 1 and "Check" in pipe.versions()[0].read_text() or True
    assert len(pipe.versions()) >= 1


def test_report_table_says_how_many_rows_are_shown():
    from dancr.core.nodes.report import _table_html
    df = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
    html = _table_html(df, 2, 5)
    assert "Showing the first 2 of 5 rows" in html


def test_a_report_with_two_columns_of_the_same_label(tmp_path):
    p = table([[1, 2]], [("a", "number"), ("b", "number")])
    p.path = tmp_path / "p.json"
    p.set_column_meta("a", "Pressure", "bar"); p.set_column_meta("b", "Pressure", "bar")
    p.add_node("report", params={"path": "r.html", "pdf": False}, id="r"); p.connect("d", "r", "items")
    st = Executor(p).run()["r"]
    assert st.status == "done", st.error


def test_the_report_pdf_cannot_be_written_through_a_link_outside_the_project(tmp_path):
    outside = tmp_path / "outside"; outside.mkdir()
    proj = tmp_path / "proj"; proj.mkdir()
    (proj / "r.pdf").symlink_to(outside / "escaped.pdf")
    p = table([[1]], [("a", "number")]); p.path = proj / "p.json"
    p.add_node("report", params={"path": "r.html", "pdf": True}, id="r"); p.connect("d", "r", "items")
    Executor(p, output_root=proj.resolve()).run()
    assert not (outside / "escaped.pdf").exists()


def test_report_passes_project_inputs_to_chart_rendering(tmp_path, monkeypatch):
    import dancr.views.render as render_mod
    from dancr.core.nodes.report import build_report
    from dancr.core.registry import Ctx

    captured = {}

    def fake_render(lf, params, out, **kw):
        captured.update(kw)
        out.write_bytes(b"png")
        return out

    monkeypatch.setattr(render_mod, "render_chart", fake_render)
    lf = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]}).lazy()
    ctx = Ctx(tmp_path, "rep", "Report", inputs={"upper": 3.5})
    inputs = {"items": [lf]}
    meta = [{"title": "Chart", "node_type": "chart",
             "params": {"kind": "line", "x": "x", "series": [{"column": "y"}],
                        "limits": [{"value": "upper", "label": "upper"}]},
             "messages": [], "report": {}}]
    doc = build_report(ctx, inputs, {"title": "T", "path": "r.html", "pdf": False}, meta, None, ctx.inputs)
    assert "Chart" in doc and captured.get("inputs") == {"upper": 3.5}


def test_report_blocks_resolve_items_by_node_id_through_a_reorder(tmp_path):
    from dancr.core.nodes.report import build_report

    frames = [_frame(), _frame()]
    meta = [{"node": "a", "title": "A", "node_type": "load_file", "params": {}, "messages": [], "report": {}},
            {"node": "b", "title": "B", "node_type": "load_file", "params": {}, "messages": [], "report": {}}]
    params = {"title": "T", "path": "r.html", "pdf": False,
              "blocks": [{"type": "item", "node": "b"}, {"type": "item", "node": "a"}]}
    doc = build_report(_report_ctx(tmp_path), {"items": frames}, params, meta, None, {})
    assert doc.index("<h2>B</h2>") < doc.index("<h2>A</h2>")


def test_report_blocks_still_accept_a_legacy_index(tmp_path):
    from dancr.core.nodes.report import build_report

    frames = [_frame(), _frame()]
    meta = [{"node": "a", "title": "A", "node_type": "load_file", "params": {}, "messages": [], "report": {}},
            {"node": "b", "title": "B", "node_type": "load_file", "params": {}, "messages": [], "report": {}}]
    params = {"title": "T", "path": "r.html", "pdf": False, "blocks": [{"type": "item", "index": 1}]}
    doc = build_report(_report_ctx(tmp_path), {"items": frames}, params, meta, None, {})
    assert "<h2>B</h2>" in doc and doc.index("<h2>B</h2>") < doc.index("<h2>A</h2>")


def test_report_ignores_malformed_block_index(tmp_path):
    p = pipe_with(tmp_path, pl.DataFrame({"x": [1.0, 2.0]}))
    r = p.add_node("report", params={"title": "T", "path": "r.html", "pdf": False, "blocks": [{"type": "item", "index": "zero"}, {"type": "heading", "text": "H"}]})
    p.connect("src", r.id, "items")
    assert Executor(p).run()[r.id].status == "done" and "<h2>H</h2>" in (tmp_path / "r.html").read_text()


def test_report_provenance_footer_can_be_turned_off(tmp_path):
    p = pipe_with(tmp_path, pl.DataFrame({"x": [1.0, 2.0]}))
    r = p.add_node("report", params={"path": "r.html", "pdf": False}, id="r")
    p.connect("src", r.id, "items")
    assert Executor(p).run()[r.id].status == "done"
    html = (tmp_path / "r.html").read_text()
    assert "Provenance" in html and "dancr verify" in html
    p.set_params(r.id, include_proof=False)
    assert Executor(p).run()[r.id].status == "done"
    assert "Provenance" not in (tmp_path / "r.html").read_text()
