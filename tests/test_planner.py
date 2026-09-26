"""Changing an answer keeps what the person did by hand: a step they put between the answer's steps, and a step
they edited that the change no longer needs."""
import polars as pl
import pytest

from dancr.core import Pipeline
from dancr.core import answers as A
from dancr.core.ask import ask
from dancr.core.executor import Executor
from dancr.core.understand import understand, deepen


@pytest.fixture
def shop(tmp_path):
    pl.DataFrame({"order_id": list(range(1, 61)), "region": ["N", "S", "E"] * 20, "qty": [i % 5 + 1 for i in range(60)],
                  "price": [float(i % 7) for i in range(60)]}).write_csv(tmp_path / "orders.csv")
    p = Pipeline("p"); p.path = tmp_path / "p.json"
    p.add_node("load_file", title="orders", params={"path": "orders.csv"}, id="orders")
    ex = Executor(p)
    return p, deepen(p, ex, understand(p, ex))


def test_a_step_put_between_an_answers_steps_stays_there(shop):
    p, m = shop
    a, _ = A.build(p, m, ask(m, "total qty by region").spec)
    groups, order = a.steps["groups"]["node"], a.steps["order"]["node"]
    p.add_node("keep_rows", title="Mine", params={"mode": "keep", "conditions": {"match": "all", "rules": [
        {"column": "qty", "op": "gt", "value": 5}]}}, id="mine")
    p.disconnect(groups, order, "in"); p.connect(groups, "mine"); p.connect("mine", order)
    A.change(p, m, a.id, "stat", "mean")
    assert p.inputs_of(order)["in"] == ["mine"] and p.inputs_of("mine")["in"] == [groups]
    assert a.steps["order"]["node"] == order and a.steps["order"].get("hand")
    assert p.nodes[groups].params["default_stats"] == ["mean"]
    st = Executor(p).run(targets=[a.terminal])[a.terminal]
    assert st.status == "done", st.error


def test_an_edited_step_the_change_no_longer_needs_is_kept(shop):
    p, m = shop
    a, _ = A.build(p, m, ask(m, "total qty by region where price above 2").spec)
    f = a.steps["filter"]["node"]
    p.nodes[f].params["conditions"]["rules"][0]["value"] = 3        # the person's own threshold
    _, pl_ = A.change(p, m, a.id, "filters.0", None)                # the answer no longer filters
    assert f in p.nodes and p.nodes[f].params["conditions"]["rules"][0]["value"] == 3
    assert "filter" not in a.steps and pl_.set_aside == [p.nodes[f].title]


def test_a_step_nobody_touched_that_is_no_longer_needed_is_removed(shop):
    p, m = shop
    a, _ = A.build(p, m, ask(m, "total qty by region where price above 2").spec)
    f = a.steps["filter"]["node"]
    A.change(p, m, a.id, "filters.0", None)
    assert f not in p.nodes


def test_a_step_put_in_front_of_an_answers_step_stays_in_its_path_when_the_input_changes(shop):
    p, m = shop
    a, _ = A.build(p, m, ask(m, "total qty by region").spec)
    groups = a.steps["groups"]["node"]
    src = p.inputs_of(groups)["in"][0]
    p.add_node("keep_rows", title="Mine", params={"mode": "keep", "conditions": {"match": "all", "rules": [
        {"column": "qty", "op": "gt", "value": 3}]}}, id="mine")
    p.disconnect(src, groups, "in"); p.connect(src, "mine"); p.connect("mine", groups)
    a, pl_ = A.build(p, m, ask(m, "total qty by region where price above 2").spec, answer_id=a.id)
    f = a.steps["filter"]["node"]
    assert a.steps["groups"]["node"] == groups and pl_.set_aside == []
    assert p.inputs_of(groups)["in"] == ["mine"] and p.inputs_of("mine")["in"] == [f] and p.inputs_of(f)["in"] == [src]
    ex = Executor(p)
    ex.run(targets=[groups])
    got = dict(pl.read_parquet(ex.state(groups).output).select("region", "qty").iter_rows())
    rows = pl.read_csv(p.directory / "orders.csv").filter((pl.col("qty") > 3) & (pl.col("price") > 2))
    assert got == dict(rows.group_by("region").agg(pl.col("qty").sum()).iter_rows())
    A.change(p, m, a.id, "filters.0", None)                         # and back: the filter goes, "Mine" stays
    assert f not in p.nodes and p.inputs_of("mine")["in"] == [src] and p.inputs_of(groups)["in"] == ["mine"]


def test_steps_of_a_removed_answer_that_another_uses_become_that_answers(shop):
    p, m = shop
    a1, _ = A.build(p, m, ask(m, "total qty by region").spec)
    a2, _ = A.build(p, m, ask(m, "total qty by region").spec)       # the same steps, used not owned
    old = set(a1.nodes)
    assert A.remove(p, a1.id, True) == [] and old <= set(p.nodes)
    A.change(p, m, a2.id, "stat", "mean")                           # changed in place, nothing left behind
    assert set(a2.nodes) == old and set(p.nodes) == old | {"orders"}
    assert A.remove(p, a2.id, True) == sorted(old) and set(p.nodes) == {"orders"}
