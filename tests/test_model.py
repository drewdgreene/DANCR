import json
import pytest

from dancr.core import Pipeline, PipelineError, registry


def test_add_and_ids():
    p = Pipeline()
    a = p.add_node("load_file")
    b = p.add_node("load_file")
    assert a.id == "load_file_1" and b.id == "load_file_2"
    assert a.params["has_header"] is True  # defaults filled
    with pytest.raises(PipelineError):
        p.add_node("load_file", id="load_file_1")
    with pytest.raises(KeyError):
        p.add_node("no_such_type")


def test_connect_rules():
    p = Pipeline()
    a = p.add_node("load_file", id="a")
    f = p.add_node("keep_rows", id="f")
    g = p.add_node("keep_rows", id="g")
    p.connect("a", "f")
    with pytest.raises(PipelineError, match="already connected"):
        p.connect("g", "f")
    s = p.add_node("stack", id="s")
    p.connect("f", "s")
    p.connect("s", "g")
    with pytest.raises(PipelineError, match="loop"):
        p.connect("g", "s")
    with pytest.raises(PipelineError, match="does not take inputs"):
        p.connect("f", "a")
    with pytest.raises(PipelineError, match="itself"):
        p.connect("s", "s")
    assert p.topological_order() == ["a", "f", "s", "g"]
    assert p.upstream_closure("g") == {"a", "f", "s"}
    assert p.downstream_closure("a") == {"f", "s", "g"}


def test_combine_ports_and_stack_multiple():
    p = Pipeline()
    p.add_node("load_file", id="a"); p.add_node("load_file", id="b"); p.add_node("load_file", id="c")
    c = p.add_node("combine", id="j")
    p.connect("a", "j")             # auto -> left
    p.connect("b", "j")             # auto -> right
    assert p.inputs_of("j") == {"left": ["a"], "right": ["b"]}
    with pytest.raises(PipelineError):
        p.connect("c", "j")
    s = p.add_node("stack", id="s")
    p.connect("a", "s"); p.connect("b", "s"); p.connect("c", "s")
    assert p.inputs_of("s") == {"tables": ["a", "b", "c"]}


def test_roundtrip(tmp_path):
    p = Pipeline("demo")
    a = p.add_node("load_file", "Data", {"path": "x.csv"}, x=10, y=20)
    k = p.add_node("keep_rows", params={"conditions": {"match": "any", "rules": [{"column": "v", "op": "gt", "value": "3"}]}})
    p.connect(a.id, k.id)
    p.add_note("hello", 5, 5)
    path = p.save(tmp_path / "p.json")
    q = Pipeline.load(path)
    assert q.to_dict() == p.to_dict()
    assert q.nodes[k.id].params["conditions"]["rules"][0]["op"] == "gt"
    assert q.notes[0].text == "hello"


def test_set_params_validation():
    p = Pipeline()
    n = p.add_node("time_buckets")
    with pytest.raises(ValueError):
        p.set_params(n.id, every="banana")
    with pytest.raises(PipelineError):
        p.set_params(n.id, nonsense=1)
    p.set_params(n.id, every="5 min")
    assert p.nodes[n.id].params["every"] == "5 min"


def test_problems_reports_missing_inputs_and_settings():
    p = Pipeline()
    k = p.add_node("keep_rows")
    s = p.add_node("sort")
    probs = p.problems()
    assert any("nothing is connected" in x for x in probs)
    assert any("Sort by is not set" in x for x in probs)


def test_load_rejects_wrong_format(tmp_path):
    f = tmp_path / "old.json"
    f.write_text(json.dumps({"nodes": [], "connections": []}))
    with pytest.raises(PipelineError, match="DANCR 2"):
        Pipeline.load(f)


def test_registry_json_is_serializable():
    data = registry.to_json()
    json.dumps(data)
    assert {t["key"] for t in data} >= {"load_file", "keep_rows", "calculate", "combine", "time_buckets", "chart"}
