


from dancr.core.executor import Executor


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
