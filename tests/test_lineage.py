"""Lineage and proof cards: where a step came from, what depends on it, and how to re-check it."""
import json

import polars as pl
import pytest

from dancr.core import Pipeline
from dancr.core.executor import Executor
from dancr.core.lineage import build_lineage, proof_card


def make_project(tmp_path):
    (tmp_path / "data").mkdir()
    pl.DataFrame({"region": ["N", "S", "N", "S"], "amount": [1.0, 2.0, 3.0, 4.0]}).write_csv(tmp_path / "data" / "m.csv")
    p = Pipeline("m")
    p.path = tmp_path / "m.json"
    p.add_node("load_file", params={"path": "data/m.csv"}, id="src")
    p.add_node("group_summary", params={"by": ["region"]}, id="g")
    p.connect("src", "g")
    p.add_node("report", params={"title": "R", "path": "r.html", "pdf": False}, id="rep")
    p.connect("g", "rep")
    p.add_answer("Totals", "g", view="table", spec={"recipe": "breakdown", "table": "src", "by": ["src", "region"]})
    p.save()
    return p


def test_up_lineage_lists_the_chain(tmp_path):
    p = make_project(tmp_path)
    lin = build_lineage(p, "g", direction="up")
    assert lin["up"]["nodes"] == ["src"]
    assert ["src", "g"] in lin["up"]["edges"]
    assert "down" not in lin


def test_down_lineage_lists_steps_answers_files_and_turns(tmp_path):
    p = make_project(tmp_path)
    ex = Executor(p)
    ex.run()                                   # so the report's file exists
    from dancr.core.assistant.store import Thread, Turn, save_thread
    th = Thread()
    th.add(Turn(role="user", text="show totals"))
    th.add(Turn(role="assistant", kind="answer", node="g", finding="N is higher"))
    save_thread(p, th)

    lin = build_lineage(p, "src", direction="down", executor=Executor(p))
    down = lin["down"]
    assert set(down["nodes"]) == {"g", "rep"}
    assert any(a["id"] == "answer_1" for a in down["answers"])
    assert any(a["file"].endswith("r.html") for a in down["artifacts"])
    assert down["agent_turns"] and down["agent_turns"][0]["node"] == "g"


def test_unknown_node_names_itself(tmp_path):
    p = make_project(tmp_path)
    with pytest.raises(ValueError) as e:
        build_lineage(p, "nope")
    assert "nope" in str(e.value)


def test_proof_card_has_hashes_sources_and_a_verify_line(tmp_path):
    p = make_project(tmp_path)
    Executor(p).run()
    card = proof_card(p, Executor(p), "g")
    assert card["kind"] == "dancr.proof" and card["node"] == "g"
    ids = [s["id"] for s in card["steps"]]
    assert "src" in ids and "g" in ids
    assert all(s["plan_hash"] for s in card["steps"])
    assert any(s.get("path", "").endswith("m.csv") for s in card["sources"])
    assert card["output_hash"]
    assert card["verify"].startswith("dancr verify m.json")


def test_mcp_lineage_and_proof(tmp_path, mcp_root):
    import dancr.mcp_server as srv
    (tmp_path / "data").mkdir()
    pl.DataFrame({"a": [1, 2, 3]}).write_csv(tmp_path / "data" / "s.csv")
    pj = tmp_path / "p.json"
    srv.create_pipeline(str(pj))
    srv.add_node(str(pj), "load_file", {"path": "data/s.csv"}, node_id="src")
    srv.add_node(str(pj), "group_summary", {"by": ["a"]}, node_id="g", after="src")

    lin = json.loads(srv.lineage(str(pj), "g", direction="up"))
    assert lin["up"]["nodes"] == ["src"]
    card = json.loads(srv.get_proof(str(pj), "g"))
    assert card["node"] == "g" and card["steps"]
