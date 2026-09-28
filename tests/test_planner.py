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


def _chain():
    """a, b -> mine -> step: a step of the person's ("mine") in front of an answer's step."""
    p = Pipeline("p")
    for n in ("a", "b", "c"):
        p.add_node("enter_data", title=n, id=n)
    p.add_node("keep_rows", title="Mine", id="mine")
    p.add_node("sort", title="Step", id="step")
    p.connect("a", "mine"); p.connect("mine", "step")
    return p


def _edges(p):
    return sorted((s, t, port) for t in p.nodes for port, srcs in p.inputs_of(t).items() for s in srcs)


@pytest.mark.parametrize("case", ["ports differ", "wants two inputs", "fixed", "not a chain", "gone"])
def test_steps_put_in_front_are_not_moved_unless_they_are_a_simple_chain_of_the_persons_own(case):
    from dancr.core.planner import PipelineEdits, _move_inserted
    p = _chain()
    made, want, fixed = {"in": ["a"]}, {"in": ["b"]}, set()
    if case == "ports differ":
        want = {"in": ["b"], "other": ["c"]}
    elif case == "wants two inputs":
        want = {"in": ["b", "c"]}
    elif case == "fixed":
        fixed = {"mine"}                                    # another answer's step: never moved
    elif case == "not a chain":
        p.add_node("combine", title="Joined", id="join")    # two inputs in front: no single path back
        p.disconnect("mine", "step", "in"); p.connect("mine", "join", "left"); p.connect("c", "join", "right")
        p.connect("join", "step")
    elif case == "gone":
        made = {"in": ["x"]}                                # what it read before is no longer upstream
        p.disconnect("a", "mine", "in")
    before = _edges(p)
    assert _move_inserted(p, PipelineEdits(p), "step", made, want, fixed) is False
    assert _edges(p) == before


def test_steps_put_in_front_move_onto_the_new_input():
    from dancr.core.planner import PipelineEdits, _move_inserted
    p = _chain()
    assert _move_inserted(p, PipelineEdits(p), "step", {"in": ["a"]}, {"in": ["b"]}, set()) is True
    assert p.inputs_of("mine")["in"] == ["b"] and p.inputs_of("step")["in"] == ["mine"]


def test_rewiring_a_step_changes_its_ports_in_name_order():
    from dancr.core.planner import Edits, _rewire
    p = Pipeline("p")
    for n in ("a", "b", "c", "d"):
        p.add_node("enter_data", title=n, id=n)
    p.add_node("combine", title="Joined", id="join")
    p.connect("a", "join", "right"); p.connect("b", "join", "left")
    calls = []

    class Log(Edits):
        def connect(self, s, t, port): calls.append(("+", s, port))
        def disconnect(self, s, t, port): calls.append(("-", s, port))
    _rewire(p, Log(p), "join", {"right": ["c"], "left": ["d"]})
    assert calls == [("-", "b", "left"), ("+", "d", "left"), ("-", "a", "right"), ("+", "c", "right")]


def test_a_removed_answers_step_goes_to_the_first_other_answer_that_uses_it(shop):
    p, m = shop
    a1, _ = A.build(p, m, ask(m, "total qty by region").spec)
    a2, _ = A.build(p, m, ask(m, "total qty by region").spec)
    a3, _ = A.build(p, m, ask(m, "total qty by region").spec)
    A.remove(p, a1.id, True)
    assert set(a2.steps) == set(a1.steps) and a3.steps == {}      # handed over once, to one answer
    assert A.remove(p, a3.id, True) == [] and set(a2.nodes) <= set(p.nodes)
