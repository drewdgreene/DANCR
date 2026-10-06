"""Agent trust: scoring questions against a project (the engine and the Assistant) and the saved conversation trace."""
import json

import polars as pl

from dancr.core import Pipeline
from dancr.headless import assistant_trace, assistant_turn, load_eval_set, run_eval


def make(tmp_path):
    p = Pipeline("e"); p.path = tmp_path / "e.json"
    p.add_node("enter_data", params={
        "columns": [{"name": "region", "type": "text"}, {"name": "amount", "type": "number"}],
        "rows": [["North", 1.0], ["South", 2.0], ["North", 3.0]]}, id="d")
    p.save()
    return p


def test_deterministic_eval_scores_and_refuses(tmp_path):
    p = make(tmp_path)
    cases = [
        {"id": "ok", "question": "total amount by region", "expect_recipe": "breakdown"},
        {"id": "clean", "question": "check the data", "expect_recipe": "quality"},
        {"id": "junk", "question": "zzzz qqqq", "expect_refuse": True},
        {"id": "wrong", "question": "total amount by region", "expect_recipe": "trend"},
    ]
    res = run_eval(p, cases)
    by = {c["id"]: c for c in res["cases"]}
    assert by["ok"]["ok"] and by["clean"]["ok"] and by["junk"]["ok"]
    assert not by["wrong"]["ok"] and res["ok"] is False
    assert res["summary"] == {"total": 4, "passed": 3, "failed": 1}


def test_eval_set_parsing():
    assert load_eval_set('[{"question": "a"}]')[0]["question"] == "a"
    assert load_eval_set('{"cases": [{"question": "b"}]}')[0]["question"] == "b"
    assert load_eval_set('{"question": "c"}\n{"question": "d"}\n')[1]["question"] == "d"


def test_model_eval_and_trace_with_the_fake(tmp_path, monkeypatch):
    monkeypatch.setenv("DANCR_ASSISTANT_FAKE", "1")
    p = make(tmp_path)
    res = run_eval(p, [{"id": "m", "question": "hello", "expect_no_text": "ZZZ-not-here"}], model=True)
    assert res["model"] is True and res["cases"][0]["ok"] is True
    assert res["cases"][0]["unverified"] == []

    assistant_turn(p, "hello")
    tr = assistant_trace(p)
    assert tr["kind"] == "dancr.trace" and tr["count"] >= 1
    assert tr["turns"][0]["role"] in ("user", "assistant")


def test_mcp_get_trace_and_run_eval(tmp_path, mcp_root):
    import dancr.mcp_server as srv
    pj = tmp_path / "p.json"
    srv.create_pipeline(str(pj))
    srv.add_node(str(pj), "enter_data", {"columns": [{"name": "x", "type": "number"}], "rows": [[1], [2]]}, node_id="d")
    tr = json.loads(srv.get_trace(str(pj)))
    assert tr["count"] == 0 and tr["turns"] == []
    out = json.loads(srv.run_eval(str(pj), cases=[{"id": "c", "question": "check the data", "expect_recipe": "quality"}]))
    assert out["ok"] is True and out["summary"]["passed"] == 1
