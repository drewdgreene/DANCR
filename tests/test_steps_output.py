"""Output steps: charts, exports and workbooks."""
from datetime import datetime
from pathlib import Path

import polars as pl

from conftest import run_one
from dancr.core import Pipeline
from dancr.core.executor import Executor


def pipe_with(tmp_path, df, name="t.parquet"):
    f = tmp_path / name; df.write_parquet(f)
    p = Pipeline(); p.path = tmp_path / "p.json"
    p.add_node("load_file", params={"path": str(f)}, id="src")
    return p


def failed(p, node_id):
    st = Executor(p).run(targets=[node_id])[node_id]
    assert st.status == "failed", "expected a failure"
    return st.error


def test_export_and_chart_pass_through(pipe, tmp_path):
    e = pipe.add_node("export", params={"path": "out/a.csv"}); pipe.connect("a", e.id)
    c = pipe.add_node("chart", params={"kind": "line", "x": "time", "series": [{"column": "pressure_psi"}]}); pipe.connect(e.id, c.id)
    res = Executor(pipe).run()
    assert (tmp_path / "out" / "a.csv").exists()
    assert res[c.id].status == "done" and res[c.id].rows == res["a"].rows
    pipe.set_params(c.id, series=[{"column": "nope"}])
    assert Executor(pipe).run()[c.id].status == "failed"


def test_workbook_and_summarise_around(pipe, tmp_path):
    tb = pipe.add_node("time_buckets", params={"every": "5m"}); pipe.connect("a", tb.id)
    g = pipe.add_node("find_gaps"); pipe.connect("a", g.id)
    w = pipe.add_node("workbook", params={"path": "out/book.xlsx"}); pipe.connect(tb.id, w.id); pipe.connect(g.id, w.id)
    st = Executor(pipe).run()[w.id]
    assert st.status == "done", st.error
    import fastexcel
    names = fastexcel.read_excel(str(tmp_path / "out" / "book.xlsx")).sheet_names
    assert names == ["Average over time", "Find gaps"]
    # samples every 10 minutes; summarise the 10 minutes before each
    samples = pl.DataFrame({"when": [datetime(2024, 6, 1, 0, 10), datetime(2024, 6, 1, 0, 20)], "lab_value": [1.0, 2.0]})
    samples.write_parquet(tmp_path / "s.parquet")
    pipe.add_node("load_file", params={"path": "s.parquet"}, id="s")
    ar = pipe.add_node("summarise_around", params={"window": "10m", "columns": ["pressure_psi"], "stats": ["mean", "count"]})
    pipe.connect("s", ar.id, "samples"); pipe.connect("a", ar.id, "log")
    out = run_one(pipe, ar.id)
    assert out.columns == ["when", "lab_value", "pressure_psi_mean", "pressure_psi_count"]
    a = pl.read_parquet(Executor(pipe).state("a").output)
    expected = a.filter((pl.col("time") > datetime(2024, 6, 1, 0, 0)) & (pl.col("time") <= datetime(2024, 6, 1, 0, 10))).height
    assert out["pressure_psi_count"][0] == expected and expected > 10000


def test_temporary_files_are_never_shared():
    from dancr.core.nodes._common import private_temp
    out = Path("/x/report.html")
    assert private_temp(out) != private_temp(out) and private_temp(out).suffix == ".html"


def test_chart_validates_every_named_column_and_export_accepts_txt(tmp_path):
    p = pipe_with(tmp_path, pl.DataFrame({"x": [1.0, 2.0], "g": ["a", "b"]}))
    c = p.add_node("chart", params={"kind": "bar", "category": "nope", "value": "x"}); p.connect("src", c.id)
    assert "nope" in failed(p, c.id)
    p.set_params(c.id, kind="line", x="x", series=[{"column": "x"}], color_by="missing")
    assert "missing" in failed(p, c.id)
    p.set_params(c.id, kind="histogram", column="x", split_by="g")
    assert Executor(p).run(targets=[c.id])[c.id].status == "done"
    e = p.add_node("export", params={"path": "out.txt"}); p.connect("src", e.id)
    run_one(p, "src"); assert Executor(p).run(targets=[e.id])[e.id].status == "done" and (tmp_path / "out.txt").read_text().startswith("x,g")
    p.set_params(e.id, path="out.json")
    assert ".tsv, .txt" in failed(p, e.id)


def test_export_is_atomic_and_strips_tz(tmp_path):
    df = pl.DataFrame({"t": [datetime(2024, 1, 1)], "v": [1.0]}).with_columns(pl.col("t").dt.replace_time_zone("UTC"))
    p = pipe_with(tmp_path, df)
    e = p.add_node("export", params={"path": "out.xlsx"}); p.connect("src", e.id)
    assert Executor(p).run()[e.id].status == "done" and (tmp_path / "out.xlsx").exists()
    target = tmp_path / "keep.csv"; target.write_text("precious\n")
    bad = p.add_node("calculate", params={"formulas": [{"name": "z", "expr": "nope"}]}); p.connect("src", bad.id)
    e2 = p.add_node("export", params={"path": "keep.csv"}); p.connect(bad.id, e2.id)
    Executor(p).run()
    assert target.read_text() == "precious\n"
    assert not list(tmp_path.glob(".keep.csv.*"))
