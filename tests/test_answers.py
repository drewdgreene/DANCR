"""Answers: understanding tables, recipes, reading questions, planning, and changing answers in place."""
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from dancr.core import Pipeline
from dancr.core import answers as A
from dancr.core.ask import ask
from dancr.core.executor import Executor
from dancr.core.planner import apply_plan, PipelineEdits, instantiate
from dancr.core.recipes import suggest, plan, chips, apply_choice, PlanError
from dancr.core.understand import understand, deepen, bucket_for, distinct_labels, SERIES, LOOKUP, EVENTS


# ------------------------------------------------------------------- data
@pytest.fixture(scope="module")
def shop(tmp_path_factory) -> Path:
    """Orders (events) with two lookups: customers (in Excel) and products."""
    d = tmp_path_factory.mktemp("shop")
    hours = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(100)]
    pl.DataFrame({"order_id": list(range(1, 101)), "customer_id": [i % 12 + 1 for i in range(100)],
                  "product_id": [i % 5 + 1 for i in range(100)], "qty": [i % 4 + 1 for i in range(100)],
                  "order_date": hours}).write_csv(d / "orders.csv")
    pl.DataFrame({"customer_id": list(range(1, 13)), "name": [f"C{i}" for i in range(1, 13)],
                  "region": ["north", "south", "east", "west"] * 3}).write_excel(d / "customers.xlsx")
    pl.DataFrame({"product_id": list(range(1, 6)), "name": list("abcde"),
                  "unit_cost": [1.5, 2.0, 3.5, 4.0, 5.25]}).write_csv(d / "products.csv")
    return d


def project(folder: Path, *files: str, name: str = "p") -> Pipeline:
    p = Pipeline(name); p.path = folder / f"{name}.json"
    for f in files:
        p.add_node("load_file", title=Path(f).stem, params={"path": f}, id=Path(f).stem)
    return p


def full_model(p: Pipeline):
    ex = Executor(p)
    return deepen(p, ex, understand(p, ex))


@pytest.fixture
def shop_project(shop, tmp_path):
    p = project(shop, "orders.csv", "customers.xlsx", "products.csv", name=f"p_{tmp_path.name}")
    return p, full_model(p)


# ------------------------------------------------------------------- understanding
def test_columns_and_tables_are_understood(shop_project):
    _, m = shop_project
    orders, customers, products = m.tables["orders"], m.tables["customers"], m.tables["products"]
    roles = {c.name: c.role for c in orders.columns}
    assert roles == {"order_id": "id", "customer_id": "id", "product_id": "id", "qty": "measure", "order_date": "time"}
    assert orders.shape == EVENTS and customers.shape == LOOKUP and products.shape == LOOKUP
    assert customers.column("region").values == ["east", "north", "south", "west"]     # most frequent first, then text
    assert orders.rows == 100 and orders.rows_exact and orders.time == "order_date"


def test_links_point_from_the_many_side_to_the_lookup(shop_project):
    _, m = shop_project
    links = {(r.tables[0], r.left_on, r.tables[1], r.right_on, r.cardinality) for r in m.relations if r.kind == "link"}
    assert ("orders", "customer_id", "customers", "customer_id", "many-to-one") in links
    assert ("orders", "product_id", "products", "product_id", "many-to-one") in links
    assert all(r.match_pct == 100.0 and r.exact for r in m.relations if r.kind == "link")


def test_a_key_that_repeats_is_never_linked_without_asking(tmp_path):
    pl.DataFrame({"patient_id": [1, 1, 2, 2, 3, 3] * 3, "cost": [1.0] * 18}).write_csv(tmp_path / "visits.csv")
    pl.DataFrame({"patient_id": [1, 1, 2, 3, 3, 3], "drug": list("abcdef")}).write_csv(tmp_path / "drugs.csv")
    p = project(tmp_path, "visits.csv", "drugs.csv")
    m = full_model(p)
    link = next(r for r in m.relations if r.kind == "link")
    assert link.cardinality == "many-to-many" and "repeat" in link.why
    assert not [s for s in suggest(m) if s.recipe == "linked"]
    with pytest.raises(PlanError, match="not linked"):
        plan(m, {"recipe": "breakdown", "table": "visits", "by": ["drugs", "drug"], "measure": ["visits", "cost"]})


def test_reading_every_row_corrects_what_a_sample_suggested(tmp_path, monkeypatch):
    """A key unique in the first rows but repeated later is a many-to-many link once every row is read."""
    import dancr.core.understand as u
    monkeypatch.setattr(u, "SAMPLE_ROWS", 20)
    pl.DataFrame({"code": [f"K{i}" for i in range(50)], "v": list(range(50))}).write_csv(tmp_path / "a.csv")
    pl.DataFrame({"code": [f"K{i}" for i in range(30)] + ["K1"] * 20, "w": [1.0] * 50}).write_csv(tmp_path / "b.csv")
    p = project(tmp_path, "a.csv", "b.csv")
    ex = Executor(p)
    quick = understand(p, ex)
    assert quick.tables["b"].column("code").unique and not quick.tables["b"].complete
    deep = deepen(p, ex, understand(p, ex))
    assert deep.tables["b"].rows == 50 and not deep.tables["b"].column("code").unique
    link = next(r for r in deep.relations if r.kind == "link")
    assert link.tables == ["b", "a"] and link.cardinality == "many-to-one"       # re-pointed at the true lookup


def test_the_loaders_own_settings_are_what_is_read(tmp_path):
    """The model reads a step's output, so a loader's sheet (or header row) is respected."""
    import xlsxwriter
    wb = xlsxwriter.Workbook(tmp_path / "book.xlsx")
    for name, region in (("2023", "north"), ("2024", "south")):
        ws = wb.add_worksheet(name)
        ws.write_row(0, 0, ["region", "amount"])
        for i in range(6):
            ws.write_row(i + 1, 0, [region if i % 2 else "east", i + 1])
    wb.close()
    p = Pipeline("t"); p.path = tmp_path / "t.json"
    p.add_node("load_file", "book", {"path": "book.xlsx", "sheet": "2"}, id="book")
    m = full_model(p)
    assert set(m.tables["book"].column("region").values) == {"east", "south"}


def test_stacked_files_are_labelled_by_what_differs_in_their_names():
    assert distinct_labels(["probe_MJ03E.csv", "probe_MJ03F.csv"], ["a", "b"]) == ["MJ03E", "MJ03F"]
    assert distinct_labels(["sales_2024-01.csv", "sales_2024-02.csv"], ["a", "b"]) == ["2024-01", "2024-02"]
    assert distinct_labels(["same.csv", "same.csv"], ["x", "y"]) == ["x", "y"]


def test_series_stack_and_line_up(probe_dir, tmp_path):
    p = project(probe_dir, "probe_A.csv", "probe_B.csv", name=f"s_{tmp_path.name}")
    m = full_model(p)
    assert m.tables["probe_A"].shape == SERIES
    kinds = {r.kind for r in m.relations}
    assert {"stack", "align"} <= kinds
    align = next(r for r in m.relations if r.kind == "align")
    assert align.tolerance == "50ms" and "pressure_psi" in align.shared


@pytest.mark.parametrize("span,cadence,target,every", [
    (4.1 * 86400, 3600, 60, "1h"),          # hourly orders over four days: never finer than hourly
    (42 * 86400, 0.05, 400, "1h"),          # six weeks of 20 Hz readings: hourly
    (86400, 1, 400, "5m"),
    (365 * 86400, 86400, 60, "1w"),
    (3 * 365 * 86400, 86400, 60, "1mo"),
])
def test_time_buckets_fit_the_span_and_the_rows(span, cadence, target, every):
    assert bucket_for(span, cadence, target) == every


# ------------------------------------------------------------------- suggestions and plans
def test_suggestions_are_varied_ranked_and_all_run(shop_project, tmp_path):
    p, m = shop_project
    sugs = suggest(m)
    recipes = [s.recipe for s in sugs]
    assert recipes[:2] == ["trend", "breakdown"] and len(set(recipes)) >= 5
    assert sugs[1].title == "Total qty by region"
    for s in sugs:
        q = Pipeline.from_dict(p.to_dict(), p.path)
        pl_ = plan(m, s.spec)
        res = instantiate(q, pl_)
        st = Executor(q).run(targets=[res[pl_.terminal]])[res[pl_.terminal]]
        assert st.status == "done", (s.title, st.error)


def test_plans_are_deterministic(shop_project):
    _, m = shop_project
    spec = {"recipe": "top", "table": "orders", "by": ["customers", "name"], "measure": ["orders", "qty"], "n": 3}
    assert plan(m, spec).to_dict() == plan(m, dict(spec)).to_dict()
    assert [s.to_dict() for s in suggest(m)] == [s.to_dict() for s in suggest(m)]


def test_a_breakdown_brings_in_the_lookup_and_totals_exactly(shop_project, tmp_path):
    p, m = shop_project
    pl_ = plan(m, {"recipe": "breakdown", "table": "orders", "by": ["customers", "region"], "measure": ["orders", "qty"], "stat": "sum"})
    assert [s.type for s in pl_.new_steps] == ["combine", "group_summary", "sort", "chart"]
    assert any(a["id"].startswith("link:") for a in pl_.assumptions)
    res = instantiate(p, pl_)
    ex = Executor(p)
    ex.run(targets=[res[pl_.terminal]])
    df = pl.read_parquet(ex.state(res["order"]).output)
    assert df["qty"].sum() == 250 and df["qty"].to_list() == sorted(df["qty"].to_list(), reverse=True)


def test_monthly_totals_follow_the_calendar(tmp_path):
    days = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(91)]
    pl.DataFrame({"day": days, "amount": [1.0] * 91}).write_csv(tmp_path / "sales.csv")
    p = project(tmp_path, "sales.csv")
    m = full_model(p)
    pl_ = plan(m, {"recipe": "trend", "table": "sales", "measures": [["sales", "amount"]], "every": "1mo", "stat": "sum"})
    res = instantiate(p, pl_)
    ex = Executor(p); ex.run()
    assert pl.read_parquet(ex.state(res["buckets"]).output)["amount"].to_list() == [31.0, 29.0, 31.0]


def test_comparing_two_series_pairs_fits_and_charts_the_difference(probe_dir, tmp_path):
    p = project(probe_dir, "probe_A.csv", "probe_B.csv", name=f"c_{tmp_path.name}")
    m = full_model(p)
    s = next(s for s in suggest(m) if s.recipe == "compare")
    pl_ = plan(m, s.spec)
    res = instantiate(p, pl_)
    ex = Executor(p)
    st = ex.run(targets=[res[pl_.terminal]])
    assert all(x.status == "done" for x in st.values())
    fit = ex.state(res["fit"]).report["fits"][0]
    assert fit["r2"] > 0.5 and fit["n"] > 1000


def test_chips_list_the_choices_and_apply_them(shop_project):
    _, m = shop_project
    spec = {"recipe": "breakdown", "table": "orders", "by": ["customers", "region"], "measure": ["orders", "qty"]}
    cs = {c["key"]: c for c in chips(m, spec)}
    assert {"stat", "measure", "by"} <= set(cs)
    assert ["products", "name"] in [ch["value"] for ch in cs["by"]["choices"]]
    new = apply_choice(spec, "by", ["products", "name"])
    assert plan(m, new).title == "Total qty by products"
    f = apply_choice({**spec, "filters": [{"column": ["orders", "qty"], "op": "gt", "value": 2}]}, "filters.0", None)
    assert "filters" not in f


# ------------------------------------------------------------------- asking
@pytest.mark.parametrize("question,title", [
    ("total qty by region", "Total qty by region"),
    ("average qty per day", "Average qty per day"),
    ("top 3 customers by qty", "Top 3 customers by total qty"),
    ("bottom 2 products by qty", "Bottom 2 products by total qty"),
    ("how many orders by region", "Rows by region"),
    ("monthly qty", "Total qty per month"),
    ("orders where qty above 2", "orders where qty above 2"),
    ("gaps in orders", "Gaps in orders"),
    ("average unit_cost", "Average unit_cost"),
    ("describe products", "What is in products"),
    ("qty between 2 and 3 by region", "Rows by region where qty between 2 and 3"),
])
def test_questions_are_read_into_answers(shop_project, question, title):
    _, m = shop_project
    a = ask(m, question)
    assert a.ok, a.message
    assert a.title == title


def test_a_filter_on_a_category_value(shop_project):
    _, m = shop_project
    a = ask(m, "total qty by product name for north")
    assert a.ok
    assert a.spec["filters"] == [{"column": ["customers", "region"], "op": "eq", "value": "north"}]
    assert a.spec["by"] == ["products", "name"]


def test_unknown_words_are_reported_not_guessed(shop_project):
    _, m = shop_project
    a = ask(m, "total qty by colour")
    assert not a.ok and a.unknown == ["colour"] and "colour" in a.message
    b = ask(m, "total qyt by region")
    assert not b.ok and "qty" in b.hints


def test_a_word_meaning_two_columns_is_noted(shop_project):
    _, m = shop_project
    a = ask(m, "qty by name")
    assert a.ok and a.spec["by"] == ["customers", "name"]
    assert a.ambiguous and a.ambiguous[0]["choices"][0]["value"] == ["products", "name"]


def test_questions_about_series(probe_dir, tmp_path):
    p = project(probe_dir, "probe_A.csv", "probe_B.csv", name=f"q_{tmp_path.name}")
    m = full_model(p)
    a = ask(m, "compare probe_A and probe_B")
    assert a.ok and a.spec["recipe"] == "compare"
    b = ask(m, "average pressure per minute for probe_B")
    assert b.ok and b.spec["table"] == "probe_B" and b.spec["every"] == "1m" and not b.spec.get("together")
    c = ask(m, "pressure over time")
    assert c.ok and c.spec["recipe"] == "trend" and c.spec.get("together")


# ------------------------------------------------------------------- building and changing answers
def test_answers_share_steps_and_change_in_place(shop_project):
    p, m = shop_project
    a1, _ = A.build(p, m, ask(m, "total qty by region").spec)
    base = set(p.nodes)
    a2, _ = A.build(p, m, ask(m, "average qty by region").spec)
    link = a1.steps[next(k for k in a1.steps if k.startswith("link:"))]["node"]
    assert not [k for k in a2.steps if k.startswith("link:")]                # a1's link is used, not claimed
    assert link in p.upstream_closure(a2.terminal)
    assert link in p.nodes and len(set(p.nodes) - base) == 3                 # the link is shared, not copied
    groups = a1.steps["groups"]["node"]
    A.change(p, m, a1.id, "by", ["products", "name"])
    assert a1.steps["groups"]["node"] == groups                              # the same step, new settings
    assert p.nodes[groups].params["by"] == ["name"] and a1.title == "Total qty by products"
    assert p.nodes[a2.steps["groups"]["node"]].params["by"] == ["region"]    # the other answer is untouched
    assert link in p.nodes                                                   # still needed by the other answer


def test_hand_edits_survive_a_change(shop_project):
    p, m = shop_project
    a, _ = A.build(p, m, ask(m, "total qty by region").spec)
    order = a.steps["order"]["node"]
    p.nodes[order].params["descending"] = False          # the person's own tweak to a step the change does not touch
    p.nodes[order].title = "My order"
    A.change(p, m, a.id, "stat", "mean")
    assert p.nodes[order].params["descending"] is False and p.nodes[order].title == "My order"
    assert a.steps["order"]["node"] == order and A.kept_by_hand(p, a) == ["My order"]


def test_an_edited_step_the_change_needs_is_set_aside_not_overwritten(shop_project):
    p, m = shop_project
    a, _ = A.build(p, m, ask(m, "total qty by region").spec)
    g = a.steps["groups"]["node"]
    p.nodes[g].params["default_stats"] = ["max"]; p.nodes[g].title = "My groups"
    _, pl_ = A.change(p, m, a.id, "measure", None)       # count rows instead: the grouping step must change
    assert g in p.nodes and p.nodes[g].params["default_stats"] == ["max"]      # left exactly as the person made it
    assert a.steps["groups"]["node"] != g and pl_.set_aside == ["My groups"]
    st = Executor(p).run(targets=[a.terminal])[a.terminal]
    assert st.status == "done", st.error


def test_an_answer_never_takes_over_a_step_someone_built(shop_project):
    p, m = shop_project
    spec = ask(m, "total qty by region").spec
    mine = instantiate(p, plan(m, spec))                  # the same steps, built by hand (no answer)
    a, _ = A.build(p, m, spec)
    assert not set(a.nodes) & set(mine.values()) - {"orders", "customers"}
    A.change(p, m, a.id, "stat", "mean")
    assert p.nodes[mine["groups"]].params["default_stats"] == ["sum"]
    A.remove(p, a.id, remove_steps=True)
    assert all(n in p.nodes for n in mine.values())


def test_a_step_someone_added_after_an_answer_is_kept(shop_project):
    p, m = shop_project
    a, _ = A.build(p, m, ask(m, "total qty by region").spec)
    order = a.steps["order"]["node"]
    p.add_node("export", params={"path": "out.csv"}, id="mine"); p.connect(order, "mine")
    A.change(p, m, a.id, "set", {"recipe": "single", "measure": ["orders", "qty"], "stat": "sum"})
    assert order in p.nodes and "mine" in p.nodes


def test_removing_an_answer_keeps_what_others_need(shop_project):
    p, m = shop_project
    a1, _ = A.build(p, m, ask(m, "total qty by region").spec)
    a2, _ = A.build(p, m, ask(m, "average qty by region").spec)
    gone = A.remove(p, a1.id, remove_steps=True)
    assert a2.terminal in p.nodes and all(n in p.nodes for n in p.upstream_closure(a2.terminal))
    assert gone and not set(gone) & set(p.nodes)
    assert {"orders", "customers", "products"} <= set(p.nodes)               # tables are never an answer's to remove


def test_answers_survive_a_save_and_load(shop_project):
    p, m = shop_project
    a, _ = A.build(p, m, ask(m, "total qty by region").spec)
    p.save()
    q = Pipeline.load(p.path)
    b = q.answer(a.id)
    assert b.spec == a.spec and b.steps == a.steps and b.assumptions == a.assumptions and b.rules == a.rules


def test_the_runner_ignores_answers(shop_project):
    p, m = shop_project
    A.build(p, m, ask(m, "total qty by region").spec)
    before = Executor(p).plan_hash("orders")
    p.answers.clear()
    assert Executor(p).plan_hash("orders") == before


# ------------------------------------------------------------------- the command line and MCP
def _cli(*args):
    import subprocess, sys
    r = subprocess.run([sys.executable, "-m", "dancr.cli", *args], capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


def test_cli_suggest_ask_and_change(shop, tmp_path):
    import json, shutil
    for f in ("orders.csv", "customers.xlsx", "products.csv"):
        shutil.copy(shop / f, tmp_path / f)
    pj = str(tmp_path / "p.json")
    assert _cli("new", pj)[0] == 0
    code, out, err = _cli("--json", "suggest", pj, "--file", str(tmp_path / "orders.csv"), "--file", str(tmp_path / "customers.xlsx"))
    assert code == 0, err
    sugs = json.loads(out)
    assert any(s["title"] == "Total qty by region" for s in sugs)
    code, out, err = _cli("--json", "ask", pj, "total qty by region")
    assert code == 0, err
    ans = json.loads(out)["answer"]
    assert ans["title"] == "Total qty by region" and ans["assumptions"]
    code, out, _ = _cli("--json", "answer", pj, ans["id"], "--set", "stat=mean")
    assert code == 0 and json.loads(out)["title"] == "Average qty by region"
    code, out, _ = _cli("--json", "run", pj)
    assert code == 0 and json.loads(out)["ok"]
    code, out, _ = _cli("ask", pj, "qty by colour")
    assert code == 2
    code, out, _ = _cli("--json", "understand", pj, "--quick")
    assert code == 0 and {t["title"] for t in json.loads(out)["tables"]} == {"orders", "customers"}
    code, out, _ = _cli("--json", "answer", pj, ans["id"], "--remove", "--steps")
    assert code == 0 and json.loads(out)["steps_removed"]


def test_mcp_answer_tools(shop, tmp_path, monkeypatch):
    import json, shutil
    import dancr.mcp_server as srv
    from mcp.server.mcpserver.exceptions import ToolError
    monkeypatch.setattr(srv, "ROOT", tmp_path.resolve())
    for f in ("orders.csv", "customers.xlsx"):
        shutil.copy(shop / f, tmp_path / f)
    srv.create_pipeline("p.json")
    sugs = json.loads(srv.suggest_answers("p.json", files=["orders.csv", "customers.xlsx"]))["suggestions"]
    idx = next(s["index"] for s in sugs if s["title"] == "Total qty by region")
    built = json.loads(srv.suggest_answers("p.json", build=idx))["answer"]
    assert built["title"] == "Total qty by region"
    asked = json.loads(srv.ask("p.json", "top 3 customers by qty"))
    assert asked["answer"]["title"] == "Top 3 customers by total qty"
    changed = json.loads(srv.change_answer("p.json", built["id"], key="stat", value="mean"))["answer"]
    assert changed["title"] == "Average qty by region"
    with pytest.raises(ToolError, match="colour"):
        srv.ask("p.json", "qty by colour")
    model = json.loads(srv.understand_data("p.json"))
    assert {r["kind"] for r in model["relations"]} == {"link"}
    assert json.loads(srv.remove_answer("p.json", built["id"], remove_steps=True))["ok"]
    assert len(Pipeline.load(tmp_path / "p.json").answers) == 1


def test_row_spacing_is_measured_on_consecutive_rows_of_a_stored_result(probe_dir, tmp_path, monkeypatch):
    """Once a table has run, samples are spread over the whole result; the pairing tolerance must still come from
    the true spacing between readings, not the spacing between sampled rows."""
    import dancr.core.understand as u
    monkeypatch.setattr(u, "SAMPLE_ROWS", 500)
    p = project(probe_dir, "probe_A.csv", "probe_B.csv", name=f"r_{tmp_path.name}")
    Executor(p).run()
    m = full_model(p)
    assert m.tables["probe_A"].column("time").cadence == pytest.approx(0.05)
    assert next(r for r in m.relations if r.kind == "align").tolerance == "50ms"


# ------------------------------------------------------------------- review 2026-09-24 (answer engine)
def _frames(tmp_path, **frames):
    p = Pipeline("r"); p.path = tmp_path / "r.json"
    for name, df in frames.items():
        df.write_csv(tmp_path / f"{name}.csv")
        p.add_node("load_file", title=name, params={"path": f"{name}.csv"}, id=name)
    return p, full_model(p)


def test_a_bare_id_links_to_the_table_it_is_named_after(tmp_path):
    p, m = _frames(tmp_path,
                   orders=pl.DataFrame({"id": list(range(1, 201)), "customer_id": [i % 20 + 1 for i in range(200)],
                                        "amount": [float(i % 7) for i in range(200)]}),
                   customers=pl.DataFrame({"id": list(range(1, 21)), "name": [f"c{i}" for i in range(20)],
                                           "region": ["N", "S", "E", "W"] * 5}))
    links = [(r.left_on, r.right_on, r.cardinality) for r in m.relations if r.kind == "link"]
    assert links == [("customer_id", "id", "many-to-one")]
    a = ask(m, "total amount by region")
    res = instantiate(p, plan(m, a.spec))
    ex = Executor(p); ex.run()
    df = pl.read_parquet(ex.state(res["order"]).output)
    assert df["amount"].sum() == pytest.approx(sum(float(i % 7) for i in range(200))) and None not in df["region"].to_list()


@pytest.fixture
def sales_model(tmp_path):
    t0 = datetime(2024, 1, 1)
    n = 400
    return _frames(tmp_path, sales=pl.DataFrame({
        "date": [t0 + timedelta(hours=6 * i) for i in range(n)], "region": (["North", "South", "Total", "West"] * 100),
        "month": [(t0 + timedelta(hours=6 * i)).month for i in range(n)], "day": [(t0 + timedelta(hours=6 * i)).day for i in range(n)],
        "sales": [float(i % 13) for i in range(n)]}))


@pytest.mark.parametrize("question,expect", [
    ("average sales per 15 minutes", {"recipe": "trend", "every": "15m"}),           # 15 is a number, not day 15
    ("sales per 12 hours", {"recipe": "trend", "every": "12h"}),
    ("sales between 10 and 20", {"recipe": "rows", "filters": [{"column": ["sales", "sales"], "op": "between", "value": 10, "value2": 20}]}),
    ("sales where day is 3", {"filters": [{"column": ["sales", "day"], "op": "eq", "value": 3}]}),
    ("total sales by region", {"recipe": "breakdown", "stat": "sum", "by": ["sales", "region"]}),   # a region called Total
    ("sales where region is Total", {"filters": [{"column": ["sales", "region"], "op": "eq", "value": "Total"}]}),
    ("average sales per day by region", {"recipe": "trend", "every": "1d", "by": ["sales", "region"]}),
])
def test_questions_read_by_context(sales_model, question, expect):
    _, m = sales_model
    a = ask(m, question)
    assert a.ok, a.message
    for k, v in expect.items():
        assert a.spec.get(k) == v, (question, a.spec)


def test_a_trend_split_by_a_group_runs(sales_model):
    p, m = sales_model
    pl_ = plan(m, ask(m, "average sales per day by region").spec)
    res = instantiate(p, pl_)
    ex = Executor(p); ex.run()
    df = pl.read_parquet(ex.state(res["buckets"]).output)
    assert set(df.columns) == {"date", "region", "sales"} and df["region"].n_unique() == 4
    assert p.nodes[res["chart"]].params["color_by"] == "region"


def test_rows_are_counted_blank_or_not(tmp_path):
    p, m = _frames(tmp_path, staff=pl.DataFrame({"note": [None, "x"] * 50, "dept": (["HR", "IT", "Ops", None] * 25),
                                                 "pay": [float(i) for i in range(100)]}))
    res = instantiate(p, plan(m, ask(m, "how many rows by dept").spec))
    ex = Executor(p); ex.run()
    df = pl.read_parquet(ex.state(res["order"]).output)
    assert df["rows"].to_list() == [25, 25, 25, 25]


def test_two_logs_matched_by_unit_are_compared(tmp_path):
    t0 = datetime(2024, 1, 1)
    p, m = _frames(tmp_path,
                   logA=pl.DataFrame({"time": [t0 + timedelta(seconds=i) for i in range(300)], "Pressure (bar)": [1.0 + i / 1000 for i in range(300)]}),
                   logB=pl.DataFrame({"time": [t0 + timedelta(seconds=i, milliseconds=200) for i in range(300)], "P (bar)": [1.1 + i / 1000 for i in range(300)]}))
    s = next(s for s in suggest(m) if s.recipe == "compare")
    res = instantiate(p, plan(m, s.spec))
    assert all(v.status == "done" for v in Executor(p).run().values())


def test_readings_are_averaged_even_in_a_table_with_ids(tmp_path):
    t0 = datetime(2024, 1, 1)
    _, m = _frames(tmp_path, sensors=pl.DataFrame({"time": [t0 + timedelta(minutes=i) for i in range(300)],
                                                   "sensor_id": [f"S{i % 3}" for i in range(300)],
                                                   "temperature (C)": [20 + (i % 5) / 10 for i in range(300)]}))
    assert suggest(m)[0].spec["stat"] == "mean"
    a = ask(m, "average temperature per sensor")
    assert a.ok and a.spec["by"] == ["sensors", "sensor_id"]
