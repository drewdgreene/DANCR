"""The project model: steps and connections, settings, inputs, saving, loading and versions."""
import json
import os
import sys
from pathlib import Path

import polars as pl
import pytest

from conftest import run_one
from dancr.core import Pipeline, PipelineError, registry
from dancr.core.dtypes import text_to_bool, typed_value
from dancr.core.executor import Executor
from dancr.core.params import Param


def pipe_with(tmp_path, df, name="t.parquet"):
    f = tmp_path / name; df.write_parquet(f)
    p = Pipeline(); p.path = tmp_path / "p.json"
    p.add_node("load_file", params={"path": str(f)}, id="src")
    return p


def saved_table(tmp_path: Path, name: str = "p.json") -> Pipeline:
    pl.DataFrame({"x": [1.0, 2.0, 3.0], "g": ["a", "b", "c"]}).write_csv(tmp_path / "in.csv")
    p = Pipeline("t")
    p.add_node("load_file", "Load", {"path": "in.csv"}, id="src")
    p.path = tmp_path / name
    p.save()
    return p


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


@pytest.mark.parametrize("typed,stored", [("007", "007"), ("0042", "0042"), ("7", 7), ("0.5", 0.5), ("0", 0), ("1,5", 1.5)])
def test_inputs_with_leading_zeros_stay_text(typed, stored):
    p = Pipeline()
    assert p.set_input("code", typed).value == stored


def test_inputs_in_formulas_and_filters(tmp_path):
    df = pl.DataFrame({"flow": [100.0, 200.0]})
    p = pipe_with(tmp_path, df)
    p.set_input("belt area", 12.5, "m²"); p.set_input("cutoff", 150)
    c = p.add_node("calculate", params={"formulas": [{"name": "per_area", "expr": "flow / [belt area]"}]}); p.connect("src", c.id)
    assert run_one(p, c.id)["per_area"].to_list() == [8.0, 16.0]
    k = p.add_node("keep_rows", params={"conditions": {"match": "all", "rules": [{"column": "flow", "op": "gt", "value": "cutoff"}]}}); p.connect("src", k.id)
    assert run_one(p, k.id)["flow"].to_list() == [200.0]
    ex = Executor(p)
    assert set(ex.inputs_used(c.id)) == {"belt area"} and set(ex.inputs_used(k.id)) == {"cutoff"} and ex.inputs_used("src") == {}
    p.save(); q = Pipeline.load(p.path)
    assert q.input_values() == {"belt area": 12.5, "cutoff": 150}
    with pytest.raises(PipelineError):
        p.set_input("1bad", 3)


def test_saves_in_the_same_second_each_keep_a_version(tmp_path):
    p = Pipeline("v")
    path = tmp_path / "v.json"
    for i in range(4):
        p.meta["i"] = i
        p.save(path)
        os.utime(path, (1_700_000_000, 1_700_000_000))     # the same second every time
    kept = {json.loads(v.read_text())["meta"]["i"] for v in p.versions()}
    assert kept == {0, 1, 2}


def test_topological_order_is_stable_and_linear(tmp_path):
    p = Pipeline("t")
    p.add_node("enter_data", id="a")
    prev = "a"
    for i in range(300):
        p.add_node("sort", id=f"s{i}"); p.connect(prev, f"s{i}"); prev = f"s{i}"
    order = p.topological_order()
    assert order[0] == "a" and order[-1] == "s299" and len(order) == 301
    assert p.upstream_closure("s299") == set(order[:-1])


def test_inputs_keep_their_value_exactly():
    p = Pipeline("t")
    assert p.set_input("id", "12345678901234567890").value == 12345678901234567890
    assert p.set_input("rate", "1,5").value == 1.5
    assert p.set_input("label", "nan").value == "nan"            # not a number: kept as text
    with pytest.raises(PipelineError):
        p.set_input("bad", float("nan"))
    with pytest.raises(PipelineError):
        p.set_input("bad", {"a": 1})
    assert typed_value("1_000") is None


def test_numeric_settings_are_checked_not_truncated():
    p = Pipeline("t")
    p.add_node("take_sample", id="s")
    with pytest.raises(ValueError, match="whole number"):
        p.set_params("s", rows=2.9)
    with pytest.raises(ValueError):
        p.set_params("s", rows=1e400)
    p.set_params("s", rows="1,000")
    assert p.nodes["s"].params["rows"] == 1000


def test_versions_list_newest_first_whatever_the_file_clock(tmp_path):
    p = Pipeline("v"); path = tmp_path / "v.json"
    for i in range(5):
        p.meta["i"] = i
        p.save(path)
        os.utime(path, (1_700_000_000, 1_700_000_000))       # a drive that keeps coarse times
    assert [json.loads(v.read_text())["meta"]["i"] for v in p.versions()] == [3, 2, 1, 0]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX directory permissions do not deny a write on Windows")
def test_a_failed_save_as_leaves_the_project_as_it_was(tmp_path):
    a = tmp_path / "A"; a.mkdir()
    p = saved_table(a)
    locked = tmp_path / "locked"; locked.mkdir(); locked.chmod(0o500)
    try:
        with pytest.raises(OSError):
            p.save(locked / "p.json", auto=True)
    finally:
        locked.chmod(0o700)
    assert p.path == a / "p.json"
    assert p.nodes["src"].params["path"] == "in.csv"                    # still the same file
    assert "autosaved" not in p.meta


@pytest.mark.parametrize("value", [10 ** 9, 1, -5])
def test_param_bounds_are_enforced_outside_the_gui(value):
    with pytest.raises(ValueError):
        Param("bins", "Bins", "int", default=50, min=2, max=2000).coerce(value)


def test_param_bounds_do_not_break_loading_a_bad_stored_value():
    from dancr.core.registry import NodeType

    node = NodeType("x", "X", "cat", "d", lambda *a: None,
                    params=[Param("bins", "Bins", "int", default=50, min=2, max=2000)])
    assert node.normalize_params({"bins": 10 ** 9}, strict=False)["bins"] == 10 ** 9
    with pytest.raises(ValueError):
        node.normalize_params({"bins": 10 ** 9}, strict=True)


def test_boolean_text_vocabulary_is_consistent():
    for word in ("true", "1", "yes", "y", "t", "on", "ON", " True "):
        assert Param("flag", "Flag", "bool").coerce(word) is True
        assert text_to_bool(word) is True
    for word in ("false", "0", "no", "off", ""):
        assert Param("flag", "Flag", "bool").coerce(word) is False
        assert text_to_bool(word) is False


def test_save_leaves_no_temp_file(tmp_path):
    p = Pipeline("x")
    target = tmp_path / "p.json"
    p.save(target)
    p.save(target)
    leftovers = [f.name for f in tmp_path.iterdir() if ".tmp" in f.name]
    assert leftovers == []


def test_malformed_pipeline_files_raise_pipeline_error(tmp_path):
    for bad in [[], {"dancr": 2, "nodes": [{"id": "a"}]}, {"dancr": 2, "nodes": [{"id": "a", "type": "nope"}]},
                {"dancr": 2, "nodes": [{"id": "a", "type": "sort", "params": ["x"]}]},
                {"dancr": 2, "nodes": None, "edges": [{"source": "a"}]}, {"dancr": 2, "edges": [{"source": "a", "target": "b"}]}]:
        f = tmp_path / "bad.json"; f.write_text(json.dumps(bad))
        with pytest.raises(PipelineError):
            Pipeline.load(f)
    f = tmp_path / "nullx.json"
    f.write_text(json.dumps({"dancr": 2, "nodes": [{"id": "a", "type": "sort", "x": None, "y": "zz"}]}))
    p = Pipeline.load(f)
    p.save()   # must not raise


def test_save_keeps_old_path_on_failure(tmp_path):
    p = Pipeline(); p.save(tmp_path / "ok.json")
    with pytest.raises(PipelineError):
        p.save(tmp_path)          # a directory
    assert p.path == tmp_path / "ok.json"


def test_none_params_use_defaults_and_shapes_validated():
    p = Pipeline()
    n = p.add_node("sort", params={"columns": ["x"], "descending": None})
    assert n.params["descending"] is False
    with pytest.raises(ValueError):
        p.add_node("keep_rows", params={"conditions": "x"})
    with pytest.raises(ValueError):
        p.add_node("keep_rows", params={"conditions": {"rules": "n"}})
    k = p.add_node("keep_rows")
    k.params["conditions"]["rules"].append({"column": "x"})
    assert p.add_node("keep_rows").params["conditions"]["rules"] == []      # defaults are not shared


def test_set_params_one_key_when_another_is_invalid():
    p = Pipeline()
    n = p.add_node("time_buckets", params={"every": "banana"}, strict=False)
    p.set_params(n.id, count_column="n")
    assert p.nodes[n.id].params["count_column"] == "n"


def test_autosave_versions_never_push_out_saved_versions(tmp_path):
    p = Pipeline("v")
    path = tmp_path / "v.json"
    t = 1_700_000_000

    def save(auto: bool, i: int):
        p.meta["i"] = i                                    # a change, so a version is kept
        p.save(path, auto=auto)
        os.utime(path, (t + i, t + i))                     # versions are named by the time of the file they keep

    save(False, 0)
    for i in range(1, 6):
        save(False, i)
    for i in range(6, 6 + Pipeline.MAX_AUTO_VERSIONS + 40):
        save(True, i)
    names = [v.name for v in p.versions()]
    saved = [n for n in names if not n.endswith(".auto.json")]
    autos = [n for n in names if n.endswith(".auto.json")]
    assert len(saved) == 6                                 # every saved version is still there
    assert len(autos) == Pipeline.MAX_AUTO_VERSIONS
