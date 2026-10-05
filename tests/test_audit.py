"""Regressions from an audit of the whole engine: each test is a defect that was found, reproduced and fixed."""
import csv
import json
import os
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from dancr.core import Pipeline, answers as A
from dancr.core.ask import ask
from dancr.core.conditions import rule_mask
from dancr.core.executor import Executor
from dancr.core.expr import compile_formula
from dancr.core.recipes import plan
from dancr.core.registry import registry, Ctx
from dancr.core.understand import understand, deepen

from sheets import write_rows


def apply(key, lf, **params):
    return registry.get(key).apply(Ctx(Path("."), "x", "x"), {"in": [lf]}, params)


def load(tmp_path, name, **params):
    return registry.get("load_file").apply(Ctx(tmp_path, "x", "x"), {}, {"path": name, **params})


@pytest.fixture
def shop(tmp_path):
    from test_corpus import shop as make
    files = make(tmp_path)
    p = Pipeline("shop"); p.path = tmp_path / "shop.json"
    for f in files:
        p.add_node("load_file", title=Path(f).stem, params={"path": f})
    ex = Executor(p)
    return p, deepen(p, ex, understand(p, ex))


# ------------------------------------------------------------------ reading questions
@pytest.mark.parametrize("question,check", [
    ("total price since 2024", lambda s: s["filters"][0]["column"][1] == "ordered_at"),       # a year, not price ≥ 2024
    ("total price before 2025", lambda s: s["filters"][0]["column"][1] == "ordered_at"),
    ("price from 10 to 20", lambda s: s["filters"][0] == {"column": ["load_file_1", "price"], "op": "between", "value": 10, "value2": 20}),
    ("orders where price below .5", lambda s: s["filters"][0]["value"] == 0.5),                 # not 5
    ("top5 customers by quantity", lambda s: s["n"] == 5),
    ("which region had the lowest quantity", lambda s: (s["recipe"], s["n"], s.get("bottom")) == ("top", 1, True)),
    ("how many products", lambda s: (s["recipe"], s["table"]) == ("single", "load_file_3")),
    ("quantity this month vs last month", lambda s: (s["recipe"], s["every"]) == ("change", "1mo")),
])
def test_questions_read_as_meant(shop, question, check):
    a = ask(shop[1], question)
    assert a.ok, a.message
    assert check(a.spec), a.spec


def test_questions_refused_rather_than_guessed(shop):
    _, m = shop
    assert not ask(m, "total quantity by region region").ok
    assert not ask(m, "total quantity by season").ok               # "season" is not corrected into "reason"
    a = ask(m, "orders where ordered_at above 5")
    assert not a.ok and "dates" in a.message


def test_study_groups_by_the_column_whose_values_are_named(tmp_path):
    rng = np.random.default_rng(3)
    rows = [[t, s, round(float(rng.normal(10 + i, 1)), 2)] for i, t in enumerate(("Control", "Drug A", "Drug B"))
            for s in ("F", "M") for _ in range(8)]
    pl.DataFrame(rows, schema=["treatment", "sex", "score"], orient="row").write_csv(tmp_path / "trial.csv")
    p = Pipeline("t"); p.path = tmp_path / "t.json"
    p.add_node("load_file", title="trial", params={"path": "trial.csv"}, id="trial")
    ex = Executor(p)
    m = deepen(p, ex, understand(p, ex))
    a = ask(m, "compare Control and Drug A for F")
    assert a.spec["by"] == ["trial", "treatment"] and {"column": ["trial", "sex"], "op": "eq", "value": "F"} in a.spec["filters"]
    b = ask(m, "is score higher in Drug A for M")
    assert b.spec["by"] == ["trial", "treatment"]


def test_a_spread_is_not_offered_where_it_cannot_be_built(shop):
    _, m = shop
    from dancr.core.recipes import PlanError, chips
    with pytest.raises(PlanError):
        plan(m, {"recipe": "change", "table": "load_file_1", "measure": ["load_file_1", "price"], "stat": "std"})
    ch = chips(m, {"recipe": "change", "table": "load_file_1", "measure": ["load_file_1", "price"]})
    assert "std" not in [c["value"] for x in ch if x["key"] == "stat" for c in x["choices"]]


def test_changing_then_removing_answers_leaves_no_step_behind(shop):
    p, m = shop
    a1, _ = A.build(p, m, ask(m, "total quantity by region").spec)
    a2, _ = A.build(p, m, ask(m, "average price by segment").spec)
    A.change(p, m, a1.id, "by", ["load_file_3", "category"])
    A.remove(p, a1.id, True)
    A.remove(p, a2.id, True)
    assert set(p.nodes) == {"load_file_1", "load_file_2", "load_file_3"}


def test_a_forecast_of_orders_projects_period_totals(shop):
    p, m = shop
    pl_ = plan(m, ask(m, "where is quantity heading").spec)
    assert [s.type for s in pl_.steps if s.type != "@"] == ["time_buckets", "forecast"]


# ------------------------------------------------------------------ steps and formulas
def test_the_latest_period_is_compared_like_for_like():
    d = [date(2024, 3, 1) + timedelta(days=i) for i in range(41)]           # to 10 April
    r = apply("compare_periods", pl.DataFrame({"t": d, "v": [100.0] * 41}).lazy(), time_column="t", every="1mo", measure="v")
    assert r.report["change"] == 0 and "so far" in r.report["finding"]["statement"]


def test_forecast_names_and_date_steps():
    d = [date(2024, 1, 1) + timedelta(days=i // 4) for i in range(40)]
    r = apply("forecast", pl.DataFrame({"t": d, "v": [float(i) for i in range(40)]}).lazy(), time_column="t", column="v", horizon=3)
    out = r.frame.collect()
    assert "v_lower" in out.columns and out["t"].to_list() == [date(2024, 1, 11), date(2024, 1, 12), date(2024, 1, 13)]


def test_whole_numbers_compare_exactly():
    df = pl.DataFrame({"id": [1234567890123456789, 1234567890123456790], "flag": [True, False]})
    s = dict(df.schema)
    assert df.select(rule_mask(s, {"column": "id", "op": "eq", "value": 1234567890123456789})).to_series().to_list() == [True, False]
    assert df.select(rule_mask(s, {"column": "id", "op": "in", "value": "1234567890123456789"})).to_series().to_list() == [True, False]
    for f, want in (("[id] = 1234567890123456789", [True, False]), ("[flag] = 1", [True, False])):
        e, _, _ = compile_formula(f, s)
        assert df.select(e).to_series().to_list() == want


@pytest.mark.parametrize("formula,want", [
    ("LEN([n])", [2, 3]), ("RIGHT([n], 1)", ["2", "3"]), ("AVERAGE([v])", [2.0]), ("SQRT(-1)", [None]),
    ("RANK([v])", [2, None, 1, None]), ('DATE("01/02/24")', [None]),
])
def test_formulas_as_excel_gives_them(formula, want):
    df = pl.DataFrame({"n": [12.0, 0.1 + 0.2, 1.0, 2.0], "v": [1.0, float("nan"), 3.0, None]})
    e, _, _ = compile_formula(formula, dict(df.schema))
    got = df.select(e.alias("r"))["r"].to_list()
    if formula.startswith("DATE"):
        assert got[0].year == 2024
    else:
        assert got[:len(want)] == want if len(got) > 1 else got == want


def test_group_summary_and_contribution_do_not_share_out_averages():
    lf = pl.DataFrame({"g": ["a", "a", "b", "b"], "x": [10.0, 20.0, 30.0, 50.0]}).lazy()
    r = apply("group_summary", lf, by=["g"], default_stats=["min"])
    assert "of the total" not in r.report["finding"]["statement"]
    c = apply("contribution", lf, by=["g"], measure="x", stat="mean")
    assert "share_percent" not in c.frame.collect().columns


def test_per_group_fits_read_the_table_once():
    from dancr.core.fits import fit_frame
    rng = np.random.default_rng(0)
    g = rng.integers(0, 5, 2000)
    lf = pl.DataFrame({"g": g.astype(str), "x": rng.uniform(1, 10, 2000), "y": g * 1.0}).lazy()
    lf = lf.with_columns((pl.col("x") * 2 + pl.col("y")).alias("y"))
    fits = fit_frame(lf, "x", "y", "linear", group="g")
    assert [f.group for f in fits] == ["0", "1", "2", "3", "4"] and all(abs(f.params[0] - 2) < 1e-9 for f in fits)


# ------------------------------------------------------------------ comparing groups
def test_compare_groups_edges():
    df = pl.DataFrame({"treated": [True] * 6 + [False] * 6, "pid": list(range(6)) * 2, "y": [float(i) for i in range(12)]})
    r = apply("compare_groups", df.lazy(), by="treated", test="rank")
    assert r.frame.collect()["p value"][0] is not None                      # True/False groups match their rows
    r = apply("compare_groups", df.lazy(), by="treated", pair_by="pid")
    assert r.frame.collect()["measure"].to_list() == ["y"] and r.frame.collect()["test"][0] == "paired t-test"
    inf = pl.DataFrame({"g": ["a"] * 4 + ["b"] * 4, "y": [1.0, 2, 3, float("inf"), 1, 2, 3, 4]})
    assert "inf" not in apply("compare_groups", inf.lazy(), by="g").report["finding"]["statement"]
    rng = np.random.default_rng(1); n = 20000
    x = np.r_[rng.uniform(10, 20, n), rng.uniform(30, 40, n)]
    gap = np.r_[x[:n] * 0.3, x[n:] * 0.2] + rng.normal(0, 0.3, 2 * n)
    r = apply("compare_groups", pl.DataFrame({"g": ["s"] * n + ["h"] * n, "size": x, "gap": gap}).lazy(), by="g")
    assert r.report["relative"] and r.report["relative"][0]["turned"]         # p = 0 is the clearest, not unknown


# ------------------------------------------------------------------ reading sheets
@pytest.mark.parametrize("text,rows,columns", [
    ("name,score,age\nAnn,1,20\nBob,,\nCat,3,22\nDan,,\nEve,5,30\n", 5, ["name", "score", "age"]),     # not sections
    ("name,score,age\nAnn,1,20\nBob,2,3\nCat,3,4\nDan,4,5\nEve,,\n", 5, ["name", "score", "age"]),     # not a title
    ("name,city,country,age\nAnn,Paris,FR,30\nDee,Lima,PE,31\n\nBob,Rome,IT,\nCy,Oslo,NO,50\n", 4, ["name", "city", "country", "age"]),
    (",a,b\n0,x,y\n1,z,w\n2,q,r\n", 3, ["", "a", "b"]),                                              # pandas' index column
    ("stat,group_a,group_b\nmean,10,20\nmedian,9,19\nsd,1,2\nn,15,15\n", 4, ["stat", "group_a", "group_b"]),  # a table of statistics
    ("Sales by year\nRegion,2019,2020,2021\nNorth,1,2,3\nSouth,4,5,6\nEast,7,8,9\n", 3, ["Region", "2019", "2020", "2021"]),
    ("sample,flag,value\ns1,outlier,1\ns2,,2\ns3,,3\ns4,recheck,4\ns5,,5\ns6,,6\n", 6, ["sample", "flag", "value"]),
])
def test_plain_sheets_are_not_read_as_layouts(tmp_path, text, rows, columns):
    (tmp_path / "t.csv").write_text(text)
    df = load(tmp_path, "t.csv").frame.collect()
    assert df.height == rows and df.columns == columns
    if "flag" in columns:
        assert df["flag"].to_list()[1] is None                                 # a notes column is not filled down


def test_a_sheet_that_starts_below_and_right_of_a1(tmp_path):
    write_rows(tmp_path / "off.xlsx", [(2, 1, ["Sample report"]), (4, 1, ["a", "b"])] + [(5 + i, 1, [i, i * 2]) for i in range(3)])
    df = load(tmp_path, "off.xlsx").frame.collect()
    assert df.columns == ["a", "b"] and df["b"].to_list() == [0, 2, 4]
    write_rows(tmp_path / "sec.xlsx", [(1, 1, ["Item", "Qty"]), (2, 1, ["Fruit"]), (3, 1, ["F0", 1]), (4, 1, ["F1", 2]),
                                       (5, 1, ["Veg"]), (6, 1, ["V0", 3]), (7, 1, ["V1", 4])])
    df = load(tmp_path, "sec.xlsx").frame.collect()
    assert df["group"].to_list() == ["Fruit", "Fruit", "Veg", "Veg"] and df["Qty"].to_list() == [1, 2, 3, 4]


def test_a_byte_order_mark_and_an_empty_sheet(tmp_path):
    (tmp_path / "bom.csv").write_bytes("﻿site,a,b\nX\n1,2,3\n4,5,6\nY\n7,8,9\n1,1,1\n".encode())
    assert "site" in load(tmp_path, "bom.csv").frame.collect().columns
    write_rows(tmp_path / "empty.xlsx", [])
    with pytest.raises(ValueError, match="has nothing in it"):
        load(tmp_path, "empty.xlsx")


def test_a_big_csv_with_a_quoted_line_break_and_totals(tmp_path):
    with open(tmp_path / "q.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "note", "v"])
        w.writerows([[i, "multi\nline" if i == 5 else "x", i] for i in range(25_000)]); w.writerow(["Total", "", 999])
    df = load(tmp_path, "q.csv").frame.collect()
    assert df.height == 25_000 and df["id"].dtype == pl.Int64


def test_a_total_row_and_an_empty_row_together(tmp_path):
    (tmp_path / "t.csv").write_text("region,sales,units\nnorth,10,1\nNA,NA,NA\nsouth,20,2\neast,30,3\nwest,40,4\nAll,100,10\n")
    p = Pipeline("t"); p.path = tmp_path / "t.json"
    p.add_node("load_file", params={"path": "t.csv"}, id="t")
    assert not understand(p, Executor(p)).skipped


# ------------------------------------------------------------------ the cache, the CLI and MCP
def test_a_relabelled_column_reruns_the_steps_that_say_it(tmp_path):
    pl.DataFrame({"g": ["a", "b"] * 5, "m": [float(i) for i in range(10)]}).write_csv(tmp_path / "g.csv")
    p = Pipeline("t"); p.path = tmp_path / "t.json"
    p.add_node("load_file", params={"path": "g.csv"}, id="l")
    p.add_node("compare_groups", params={"by": "g", "columns": ["m"]}, id="c"); p.connect("l", "c")
    ex = Executor(p); ex.run()
    h = ex.plan_hash("c"); hl = ex.plan_hash("l")
    p.set_column_label("m", "Sample mass", "g") if hasattr(p, "set_column_label") else p.columns.update({"m": {"label": "Sample mass"}})
    ex2 = Executor(p)
    assert ex2.plan_hash("c") != h and ex2.plan_hash("l") == hl                 # the loader does not rerun


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="named pipes")
def test_a_named_pipe_source_does_not_hang(tmp_path):
    os.mkfifo(tmp_path / "fifo")
    p = Pipeline("t"); p.path = tmp_path / "t.json"
    p.add_node("load_file", params={"path": "fifo"}, id="l")
    assert Executor(p).plan_hash("l")                                         # returns, without reading the pipe


def test_cli_params_must_be_an_object(tmp_path):
    subprocess.run([sys.executable, "-m", "dancr", "new", str(tmp_path / "p.json")], check=True, capture_output=True)
    r = subprocess.run([sys.executable, "-m", "dancr", "--json", "add", str(tmp_path / "p.json"), "load_file", "--params", "[1]"],
                       capture_output=True, text=True)
    assert r.returncode == 2 and "JSON object" in r.stdout + r.stderr


def test_project_files_are_strict_json(tmp_path):
    p = Pipeline("t"); p.path = tmp_path / "t.json"
    p.add_node("keep_rows", params={"conditions": {"match": "all", "rules": [{"column": "x", "op": "gt", "value": float("nan")}]}}, id="k")
    json.loads(p.dumps(), parse_constant=lambda c: pytest.fail(f"{c} in the project file"))


def test_step_ids_differing_only_in_capitals(tmp_path):
    from dancr.core.model import PipelineError
    p = Pipeline("t")
    p.add_node("load_file", id="A")
    with pytest.raises(PipelineError, match="capitals"):
        p.add_node("load_file", id="a")


def test_mcp_runs_no_unsafe_step_when_given_an_empty_list(tmp_path, monkeypatch):
    from dancr import mcp_server as srv
    monkeypatch.setattr(srv, "ROOT", tmp_path); monkeypatch.setattr(srv, "ROOT_REFUSED", False)
    (tmp_path / "src.csv").write_text("x,y\n1,2\n3,4\n")
    pj = str(tmp_path / "p.json")
    srv.create_pipeline(pj)
    srv.add_node(pj, "load_file", node_id="L", params={"path": "src.csv"})
    p = Pipeline.load(pj)
    p.add_node("export", params={"path": "src.csv"}, id="E"); p.connect("L", "E"); p.save()
    srv.run_pipeline(pj, node_ids=[])
    assert (tmp_path / "src.csv").read_text() == "x,y\n1,2\n3,4\n"


# ------------------------------------------------------------------ charts
def test_chart_edges():
    from dancr.views.chartquery import query_one
    from dancr.views.lod import bar_data, group_values_info
    empty = pl.DataFrame({"t": pl.Series([], dtype=pl.Float64), "v": pl.Series([], dtype=pl.Float64), "g": pl.Series([], dtype=pl.Utf8)})
    cd = query_one(empty.lazy(), dict(empty.schema), {"kind": "line", "x": "t", "series": [{"column": "v"}], "color_by": "g"})
    assert cd.line is not None                                                 # no rows to colour: a plain (empty) line
    df = pl.DataFrame({"c": [f"k{i}" for i in range(100)], "v": [None] * 70 + [float(i) for i in range(30)]})
    b = bar_data(df.lazy(), "c", "v", "mean")
    assert not np.isnan(b.values[0]) and b.total == 100
    assert group_values_info(pl.DataFrame({"n": [1, 1, 2]}).lazy(), "n")[0] == [1, 2]
