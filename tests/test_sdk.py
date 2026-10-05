"""The public Python API (`import dancr`): lazily imported names, a light import, and the documented surface."""
import importlib
import json
import subprocess
import sys

import polars as pl

import dancr


def test_the_public_surface_is_present_and_documented():
    for name in ("Pipeline", "Executor", "NodeState", "read_project", "editing", "project_lock",
                 "write_text_atomic",
                 "add_step", "build_template", "ask_question", "build_answer", "data_model",
                 "build_context", "context_jsonl", "context_text", "project_profile", "table_card"):
        assert name in dancr.__all__, name
        assert getattr(dancr, name) is not None, name
    assert "__version__" in dancr.__all__


def test_unknown_attribute_raises_attributeerror():
    try:
        dancr.definitely_not_a_real_name
    except AttributeError as e:
        assert "definitely_not_a_real_name" in str(e)
    else:  # pragma: no cover
        raise AssertionError("expected AttributeError")


def test_import_dancr_does_not_pull_in_the_window_or_the_engine():
    """A light import: nothing heavy is imported until a name is used (so a script pays only for what it uses)."""
    code = ("import sys, dancr; "
            "assert 'dancr.ui.mainwindow' not in sys.modules; "
            "assert dancr.__version__ == '2.1.0'; "
            "print('ok')")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr


def test_the_api_builds_and_runs_a_project(tmp_path):
    pl.DataFrame({"region": ["N", "S", "N"], "amount": [1.0, 2.0, 3.0]}).write_csv(tmp_path / "o.csv")
    p = dancr.Pipeline("shop")
    p.path = tmp_path / "shop.json"
    dancr.add_step(p, "load_file", {"path": "o.csv"}, node_id="orders")
    p.save()
    project, _text = dancr.read_project(tmp_path / "shop.json")
    with dancr.editing(tmp_path / "shop.json") as e:
        dancr.add_step(e, "group_summary", {"by": ["region"]}, node_id="by_region", after="orders")
    loaded, _ = dancr.read_project(tmp_path / "shop.json")
    res = dancr.Executor(loaded).run()
    assert res["by_region"].status == "done"
    ctx = dancr.build_context(project, stats=False)
    assert ctx["tables"][0]["node"] == "orders"
