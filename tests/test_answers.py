"""Guided build: profiler, planner, the Answer entity, and the shared-data reuse."""
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from dancr.core import Pipeline
from dancr.core.executor import Executor
from dancr.core.profile import profile_paths, suggest_links, suggest_stacks
from dancr.core.planner import plan, instantiate


@pytest.fixture(scope="module")
def demo_dir(tmp_path_factory) -> Path:
    """The origin-story shape: an orders event table and two lookup tables."""
    d = tmp_path_factory.mktemp("wizard")
    hours = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(100)]
    pl.DataFrame({
        "order_id": list(range(1, 101)),
        "customer_id": [i % 12 + 1 for i in range(100)],
        "product_id": [i % 5 + 1 for i in range(100)],
        "qty": [i % 4 + 1 for i in range(100)],
        "order_date": hours,
    }).write_csv(d / "orders.csv")
    pl.DataFrame({"customer_id": list(range(1, 13)), "name": [f"C{i}" for i in range(1, 13)],
                  "region": ["north", "south", "east", "west"] * 3}).write_excel(d / "customers.xlsx")
    pl.DataFrame({"product_id": list(range(1, 6)), "name": list("abcde"),
                  "unit_cost": [1.5, 2.0, 3.5, 4.0, 5.25]}).write_csv(d / "products.csv")
    pl.DataFrame({"month": ["jan"], "sales": [1.0]}).write_csv(d / "m1.csv")
    pl.DataFrame({"month": ["feb"], "sales": [2.0]}).write_csv(d / "m2.csv")
    return d


# ------------------------------------------------------------------- profiler
def test_profiler_describes_tables(demo_dir):
    profs = profile_paths([demo_dir / "orders.csv", demo_dir / "customers.xlsx", demo_dir / "products.csv"])
    orders, customers, products = profs
    assert orders.rows == 100 and orders.time_column == "order_date"
    assert "order_id" in orders.key_columns and "qty" in orders.measures
    assert "customer_id" not in orders.measures          # an id-like column is not a measure
    assert customers.category_columns == ["region"]
    assert "unit_cost" in products.measures


def test_profiler_suggests_the_right_links_and_stacks(demo_dir):
    profs = profile_paths([demo_dir / "orders.csv", demo_dir / "customers.xlsx", demo_dir / "products.csv",
                           demo_dir / "m1.csv", demo_dir / "m2.csv"])
    links = {(Path(l.left).name, l.left_col, Path(l.right).name, l.right_col) for l in suggest_links(profs)}
    assert ("orders.csv", "customer_id", "customers.xlsx", "customer_id") in links
    assert ("orders.csv", "product_id", "products.csv", "product_id") in links
    stacks = suggest_stacks(profs)
    assert any({Path(p).name for p in s.paths} == {"m1.csv", "m2.csv"} for s in stacks)


def test_profile_files_skips_unreadable_files(tmp_path):
    from dancr.core.profile import profile_files

    good = tmp_path / "good.csv"
    pl.DataFrame({"a": [1, 2, 3]}).write_csv(good)
    profs, skipped = profile_files([good, tmp_path / "missing.csv", tmp_path])
    assert len(profs) == 1 and profs[0].rows == 3
    assert len(skipped) == 2 and all(reason for _, reason in skipped)


def test_containment_percent_is_the_smaller_side_found_on_the_other():
    from dancr.core.profile import containment_percent

    assert containment_percent(["a", "b", "c"], ["a", "b"]) == 100.0
    assert containment_percent([1, 2, 3, 4], [1, 2]) == 100.0
    assert containment_percent([1, 2, 3], [3, 4, 5]) == pytest.approx(33.333, rel=1e-3)
    assert containment_percent([], [1]) == 0.0


# ------------------------------------------------------------------- planner
def _config_total(demo_dir):
    return {
        "intent": "total", "measure": "qty", "group": "region", "title": "Which region ordered the most",
        "assembly": {"kind": "join", "primary": str(demo_dir / "orders.csv"),
                     "links": [{"right": str(demo_dir / "customers.xlsx"), "left_on": "customer_id", "right_on": "customer_id"}]},
    }


def test_plan_builds_a_runnable_pipeline(demo_dir):
    profs = profile_paths([demo_dir / "orders.csv", demo_dir / "customers.xlsx"])
    p = plan(profs, _config_total(demo_dir))
    assert [s.type for s in p.steps] == ["load_file", "load_file", "combine", "group_summary", "chart"]
    assert "link on customer_id" in p.sentence() and "chart" in p.sentence()

    pipe = Pipeline("t"); pipe.path = demo_dir / "p.json"
    resolved = instantiate(pipe, p)
    st = Executor(pipe).run(targets=[resolved[p.terminal]])[resolved[p.terminal]]
    assert st.status == "done", st.error
    df = pl.read_parquet(st.output)
    assert df.height == 4 and set(df["region"].to_list()) == {"north", "south", "east", "west"}
    assert df["qty"].sum() == 250                       # 100 rows, qty cycles 1..4


def test_plan_over_time_buckets_and_charts(demo_dir):
    profs = profile_paths([demo_dir / "orders.csv"])
    cfg = {"intent": "over_time", "measure": "qty", "time_column": "order_date", "every": "1d",
           "assembly": {"kind": "single", "path": str(demo_dir / "orders.csv")}}
    p = plan(profs, cfg)
    assert [s.type for s in p.steps] == ["load_file", "time_buckets", "chart"]
    pipe = Pipeline("t"); pipe.path = demo_dir / "p2.json"
    resolved = instantiate(pipe, p)
    st = Executor(pipe).run(targets=[resolved[p.terminal]])[resolved[p.terminal]]
    assert st.status == "done", st.error


def test_plan_refuses_impossible_requests(demo_dir):
    profs = profile_paths([demo_dir / "orders.csv"])
    with pytest.raises(ValueError, match="date or time"):
        plan(profs, {"intent": "over_time", "assembly": {"kind": "single", "path": profs[0].path}})
    with pytest.raises(ValueError, match="category"):
        plan(profs, {"intent": "total", "assembly": {"kind": "single", "path": profs[0].path}})


def test_plan_describe_needs_no_measure_or_group(demo_dir):
    profs = profile_paths([demo_dir / "orders.csv"])
    p = plan(profs, {"intent": "describe", "assembly": {"kind": "single", "path": profs[0].path}})
    assert [s.type for s in p.steps] == ["load_file", "summarize"] and p.view == "table"
    pipe = Pipeline("t"); pipe.path = demo_dir / "p_desc.json"
    resolved = instantiate(pipe, p)
    st = Executor(pipe).run(targets=[resolved[p.terminal]])[resolved[p.terminal]]
    assert st.status == "done" and st.rows == len(profs[0].columns)


def test_shared_steps_are_reused_across_answers(demo_dir):
    """Two questions on the same files must share one data layer, not duplicate it."""
    profs = profile_paths([demo_dir / "orders.csv", demo_dir / "customers.xlsx"])
    pipe = Pipeline("t"); pipe.path = demo_dir / "p3.json"
    first = instantiate(pipe, plan(profs, _config_total(demo_dir)))
    total_steps = len(pipe.nodes)
    cfg2 = dict(_config_total(demo_dir)); cfg2["group"] = "region"; cfg2["measure"] = "qty"
    cfg2["assembly"] = {"kind": "join", "primary": str(demo_dir / "orders.csv"),
                        "links": [{"right": str(demo_dir / "customers.xlsx"), "left_on": "customer_id", "right_on": "customer_id"}]}
    second = instantiate(pipe, plan(profs, cfg2))
    assert len(pipe.nodes) == total_steps                 # identical plan: everything reused
    assert second["load:" + str(demo_dir / "orders.csv")] == first["load:" + str(demo_dir / "orders.csv")]


# ------------------------------------------------------------------- Answer entity
def test_answer_survives_a_save_and_load(tmp_path):
    p = Pipeline("t"); p.path = tmp_path / "p.json"
    p.add_node("enter_data", params={"columns": [{"name": "x", "type": "number"}], "rows": [["1"]]}, id="src")
    p.add_answer("My question", "src", 10, 20, "table", {"intent": "total"})
    p.save()
    q = Pipeline.load(tmp_path / "p.json")
    assert len(q.answers) == 1
    a = q.answers[0]
    assert (a.title, a.terminal, a.view, a.config) == ("My question", "src", "table", {"intent": "total"})
    assert q.answer(a.id).x == 10


def test_answer_needs_no_execution_and_is_ignored_by_the_runner(tmp_path):
    p = Pipeline("t"); p.path = tmp_path / "p.json"
    p.add_node("load_file", params={"path": "missing.csv"}, id="src")
    p.add_answer("Q", "src")
    # the executor has no concept of an Answer; states cover only the nodes
    st = Executor(p).states()
    assert set(st) == {"src"}


def test_a_project_without_answers_still_loads(tmp_path):
    """Older project files have no "answers" key; they must load unchanged."""
    p = Pipeline("t"); p.path = tmp_path / "p.json"
    p.add_node("enter_data", params={"columns": [{"name": "x", "type": "number"}], "rows": [["1"]]}, id="s")
    data = p.to_dict()
    data.pop("answers", None)
    q = Pipeline.from_dict(data, tmp_path / "p.json")
    assert q.answers == [] and "s" in q.nodes


def test_cli_show_lists_answers(tmp_path):
    import json
    import subprocess
    import sys

    p = tmp_path / "p.json"
    pipe = Pipeline("t"); pipe.path = p
    pipe.add_node("enter_data", params={"columns": [{"name": "x", "type": "number"}], "rows": [["1"]]}, id="s")
    pipe.add_answer("My question", "s")
    pipe.save()
    r = subprocess.run([sys.executable, "-m", "dancr.cli", "--json", "show", str(p)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["answers"][0]["title"] == "My question"
