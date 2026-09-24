"""Second audit (2026-09-24): wrong results, raw errors and lost work found after the first round of fixes."""
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from dancr.core import Pipeline
from dancr.core.executor import Executor


def table(rows: list[list], columns: list[tuple[str, str]]) -> Pipeline:
    p = Pipeline("t")
    p.add_node("enter_data", params={"columns": [{"name": n, "type": t} for n, t in columns], "rows": rows}, id="d")
    return p


def after(p: Pipeline, type_key: str, params: dict, src: str = "d", nid: str = "s") -> pl.DataFrame:
    p.add_node(type_key, params=params, id=nid)
    p.connect(src, nid)
    return Executor(p).preview(nid)[0]


def calc(p: Pipeline, expr: str) -> list:
    if "c" in p.nodes:
        p.remove_node("c")
    return after(p, "calculate", {"formulas": [{"name": "r", "expr": expr}]}, nid="c")["r"].to_list()


# ------------------------------------------------------------------ engine
def test_an_input_named_on_its_own_line_of_a_formula_reruns_the_step(tmp_path):
    p = table([[1], [2]], [("x", "number")])
    p.path = tmp_path / "p.json"
    p.set_input("k", 2)
    p.add_node("calculate", params={"formulas": [{"name": "y", "expr": "[x] *\nk"}]}, id="c"); p.connect("d", "c")
    before = Executor(p).plan_hash("c")
    p.set_input("k", 10)
    assert Executor(p).plan_hash("c") != before


@pytest.mark.parametrize("text,out", [("1.001s", "1001ms"), ("2.01s", "2010ms"), ("0.563ms", "563us"), ("1.5h", "90m")])
def test_fractional_durations_are_exact(text, out):
    from dancr.core.timeutil import parse_duration
    assert parse_duration(text)[0] == out


def test_versions_list_newest_first_whatever_the_file_clock(tmp_path):
    p = Pipeline("v"); path = tmp_path / "v.json"
    for i in range(5):
        p.meta["i"] = i
        p.save(path)
        os.utime(path, (1_700_000_000, 1_700_000_000))       # a drive that keeps coarse times
    assert [json.loads(v.read_text())["meta"]["i"] for v in p.versions()] == [3, 2, 1, 0]


def test_a_sheet_named_like_a_number_is_that_sheet(tmp_path):
    import xlsxwriter
    wb = xlsxwriter.Workbook(tmp_path / "b.xlsx")
    for name in ("2023", "2024"):
        ws = wb.add_worksheet(name); ws.write_row(0, 0, ["year"]); ws.write_row(1, 0, [int(name)])
    wb.close()
    p = Pipeline("t"); p.path = tmp_path / "t.json"
    p.add_node("load_file", params={"path": "b.xlsx", "sheet": "2024"}, id="l")
    assert Executor(p).preview("l")[0]["year"].to_list() == [2024]


# ------------------------------------------------------------------ steps and formulas
def _with_nan():
    p = table([[float(i), 1.0] for i in range(199)] + [[0.0, 0.0]], [("a", "number"), ("b", "number")])
    p.add_node("calculate", params={"formulas": [{"name": "r", "expr": "[a] / [b] + 0 * [a]"}]}, id="c0"); p.connect("d", "c0")
    return p


@pytest.mark.parametrize("method", ["zscore", "iqr"])
def test_one_blank_does_not_make_every_row_an_outlier(method):
    p = table([[float(i % 10)] for i in range(200)] + [[None]], [("v", "number")])
    p.add_node("calculate", params={"formulas": [{"name": "r", "expr": "SQRT([v] - 20)"}]}, id="c0"); p.connect("d", "c0")   # NaN
    df = after(p, "remove_outliers", {"columns": ["r"], "method": method}, src="c0")
    assert df.height == 201


def test_clipping_whole_numbers_to_fractional_limits():
    p = table([[1], [2], [3], [4]], [("n", "number")])
    p.add_node("change_type", params={"columns": ["n"], "to": "integer"}, id="i"); p.connect("d", "i")
    df = after(p, "remove_outliers", {"columns": ["n"], "method": "range", "min": "1.5", "max": "3.5", "action": "clip"}, src="i")
    assert df["n"].to_list() == [1.5, 2.0, 3.0, 3.5]


def test_long_ids_written_as_text_become_exact_integers():
    p = table([["12345678901234567"], ["1,000"]], [("id", "text")])
    df = after(p, "change_type", {"columns": ["id"], "to": "integer"})
    assert df["id"].to_list() == [12345678901234567, 1000]


def test_formulas_behave_like_excel():
    p = table([["01/02/2024", 1, 0], ["12/25/2024", 2, 2], ["2024-03-04 10:00", 4, 4], [None, 3, 1]],
              [("s", "text"), ("a", "number"), ("b", "number")])
    assert calc(p, "DATE([s])") == [datetime(2024, 1, 2), datetime(2024, 12, 25), datetime(2024, 3, 4, 10), None]
    assert calc(p, "[a] / [b]") == [None, 1.0, 1.0, 3.0]            # #DIV/0! is a blank, not infinity
    assert calc(p, "RANK([a])") == [4, 3, 1, 2]                      # the largest is 1
    assert calc(p, "RANK([a], 1)") == [1, 2, 4, 3]


def test_negative_text_lengths_are_refused_plainly():
    p = table([["abc"]], [("s", "text")])
    with pytest.raises(Exception, match="cannot be negative"):
        calc(p, "LEFT([s], -1)")


def test_nan_is_a_blank_in_filters_and_statistics():
    p = table([[1.0], [2.0], [3.0], [100.0], [-1.0]], [("v", "number")])
    p.add_node("calculate", params={"formulas": [{"name": "w", "expr": "IF([v] < 0, SQRT([v]), [v])"}]}, id="c0"); p.connect("d", "c0")
    kept = after(p, "keep_rows", {"conditions": {"match": "all", "rules": [{"column": "w", "op": "gt", "value": 50}]}}, src="c0", nid="k")
    assert kept["w"].to_list() == [100.0]
    g = after(p, "group_summary", {"columns": ["w"], "default_stats": ["median"]}, src="c0", nid="g")
    assert g["w"].to_list() == [2.5]


def test_a_date_matched_to_a_zoned_time_uses_the_local_clock():
    left = pl.LazyFrame({"day": [datetime(2024, 6, 1), datetime(2024, 6, 2)]}).with_columns(pl.col("day").cast(pl.Date))
    right = (pl.LazyFrame({"t": [datetime(2024, 6, 1, 0, 20), datetime(2024, 6, 2, 0, 20)], "v": [10, 20]})
             .with_columns(pl.col("t").dt.replace_time_zone("Europe/Oslo")))
    from dancr.core.nodes.combine import _combine
    from dancr.core.registry import Ctx
    out = _combine(Ctx(Path("."), "c", "c"), {"left": [left], "right": [right]},
                   {"method": "nearest_time", "left_time": "day", "right_time": "t", "tolerance": "30m"}).frame.collect()
    assert out["v"].to_list() == [10, 20]


def test_a_time_inside_a_clock_change_gap_is_blank_not_an_error():
    from dancr.core.dtypes import align_time_column
    df = pl.DataFrame({"t": [datetime(2024, 3, 31, 2, 30), datetime(2024, 3, 31, 4)]})
    out = df.select(align_time_column(pl.col("t"), pl.Datetime("us"), pl.Datetime("us", "Europe/Oslo")))
    assert out["t"].to_list()[0] is None and out["t"].to_list()[1] is not None


def test_tables_in_different_time_zones_stack():
    from dancr.core.nodes.basic import _stack
    from dancr.core.registry import Ctx
    a = pl.LazyFrame({"t": [datetime(2024, 1, 1, 12)]}).with_columns(pl.col("t").dt.replace_time_zone("Europe/Oslo"))
    b = pl.LazyFrame({"t": [datetime(2024, 1, 1, 12)]})
    res = _stack(Ctx(Path("."), "s", "s"), {"tables": [a, b]}, {})
    assert res.frame.collect()["t"].to_list()[0].hour == 11 and res.messages


def test_two_tables_on_the_same_grid_share_their_ticks():
    p = Pipeline("t")
    for nid, off in (("a", 0.12), ("b", 0.45)):
        p.add_node("enter_data", params={"columns": [{"name": "time", "type": "datetime"}, {"name": "v", "type": "number"}],
                                         "rows": [[f"2024-01-01 00:00:{off + i:06.3f}", i] for i in range(5)]}, id=nid)
        p.add_node("regular_grid", params={"every": "1s"}, id="g" + nid); p.connect(nid, "g" + nid)
    p.add_node("combine", params={"method": "match", "on": ["time"]}, id="c")
    p.connect("ga", "c", "left"); p.connect("gb", "c", "right")
    assert Executor(p).preview("c")[0].height == 4


def test_an_infinite_x_does_not_squash_a_chart():
    from dancr.views.lod import x_bounds
    lf = pl.LazyFrame({"x": [0.0, 1.0, 2.0, math.inf]})
    lo, hi, _ = x_bounds(lf, "x")
    assert (lo, hi) == (0.0, 2.0)


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


def test_temporary_files_are_never_shared():
    from dancr.core.nodes._common import private_temp
    out = Path("/x/report.html")
    assert private_temp(out) != private_temp(out) and private_temp(out).suffix == ".html"


# ------------------------------------------------------------------ command line and window
def test_a_step_that_fails_when_read_exits_1(tmp_path):
    pj = tmp_path / "p.json"
    p = Pipeline("p"); p.add_node("load_file", params={"path": "missing.csv"}, id="l"); p.save(pj)
    r = subprocess.run([sys.executable, "-m", "dancr.cli", "sample", str(pj), "l", "--run"], capture_output=True, text=True)
    assert r.returncode == 1


def test_saving_never_silently_overwrites_another_programs_change(app, tmp_path):
    from dancr.ui.document import Document, ChangedOnDisk
    pj = tmp_path / "p.json"
    Pipeline("p").save(pj)
    doc = Document(); doc.load(pj)
    doc.add_node("enter_data", 0, 0)                   # unsaved edit in the window
    other = Pipeline.load(pj); other.add_node("enter_data", id="agents"); other.save()     # an agent's edit
    with pytest.raises(ChangedOnDisk):
        doc.save()
    assert "agents" in Pipeline.load(pj).nodes
    doc.save(overwrite=True)
    assert "agents" not in Pipeline.load(pj).nodes and doc.versions()      # theirs is kept as an earlier version
    doc.shutdown()


def test_colour_by_counts_rows_once_and_says_what_it_leaves_out():
    from dancr.views.chartquery import query_one
    n = 2000
    lf = pl.LazyFrame({"x": [float(i) for i in range(n)], "y": [1.0] * n,
                       "g": [None if i % 50 == 0 else f"g{i % 15}" for i in range(n)]})
    cd = query_one(lf, dict(lf.collect_schema()), {"kind": "line", "x": "x", "series": [{"column": "y"}], "color_by": "g"})
    labels = [g for g, _ in cd.groups]
    assert len(labels) == 12 and f"of {n:,} rows" in cd.summary()
    assert "not shown" in cd.note


def test_a_pdf_written_from_a_worker_thread_without_a_window(tmp_path):
    import threading
    from dancr.views.pdf import html_to_pdf
    out, got = tmp_path / "r.pdf", []
    t = threading.Thread(target=lambda: got.append(html_to_pdf("<h1>Report</h1>", out)))
    t.start(); t.join()
    assert got and out.stat().st_size > 500
