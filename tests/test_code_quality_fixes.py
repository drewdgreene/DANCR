"""Regression tests for the code-quality pass: correctness fixes found by review."""
import json
from datetime import datetime

import polars as pl
import pytest

from dancr.core import Pipeline
from dancr.core.executor import Executor
from dancr.core.params import Param
from dancr.core.dtypes import text_to_bool


def pipe_with(tmp_path, df, name="t.parquet"):
    f = tmp_path / name
    df.write_parquet(f)
    p = Pipeline()
    p.path = tmp_path / "p.json"
    p.add_node("load_file", params={"path": str(f)}, id="src")
    return p


def run_one(p, node_id):
    st = Executor(p).run(targets=[node_id])[node_id]
    assert st.status == "done", st.error
    return pl.read_parquet(st.output)


# --------------------------------------------------------------- executor.frame
def test_schema_and_preview_handle_a_diamond_without_a_false_loop(tmp_path):
    """Two branches sharing one not-yet-run ancestor must not be mistaken for a loop."""
    p = pipe_with(tmp_path, pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]}))
    p.add_node("calculate", params={"formulas": [{"name": "u", "expr": "x + 1"}]}, id="c1")
    p.connect("src", "c1")
    p.add_node("calculate", params={"formulas": [{"name": "v", "expr": "y + 1"}]}, id="c2")
    p.connect("src", "c2")
    p.add_node("combine", params={"method": "match", "on": ["x"], "right_on": ["x"], "how": "inner"}, id="cb")
    p.connect("c1", "cb", "left")
    p.connect("c2", "cb", "right")
    ex = Executor(p)
    schema = ex.schema("cb")
    assert schema is not None and {"x", "y", "u", "v"} <= set(schema)
    df, _res, _kind = ex.preview("cb")
    assert df.height == 3


# --------------------------------------------------------------- regular_grid
def test_regular_grid_keeps_a_zoned_instant(tmp_path):
    t0 = datetime(2024, 1, 1, 0, 0)
    zoned = (pl.DataFrame({"t": [t0, t0.replace(hour=1)], "v": [1.0, 2.0]})
             .with_columns(pl.col("t").dt.replace_time_zone("Europe/Oslo")))
    zoned.write_parquet(tmp_path / "z.parquet")
    p = Pipeline()
    p.path = tmp_path / "p.json"
    p.add_node("load_file", params={"path": "z.parquet"}, id="z")
    p.add_node("regular_grid", params={"every": "1h", "method": "nearest"}, id="g")
    p.connect("z", "g")
    out = run_one(p, "g")
    assert [str(x) for x in out["t"].to_list()] == ["2024-01-01 00:00:00+01:00", "2024-01-01 01:00:00+01:00"]
    assert out["v"].to_list() == [1.0, 2.0]


# --------------------------------------------------------------- fix_missing
def test_fix_missing_value_keeps_an_integer_column_integer(tmp_path):
    p = pipe_with(tmp_path, pl.DataFrame({"n": [1, None, 3]}))
    p.add_node("fix_missing", params={"method": "value", "value": "5"}, id="fm")
    p.connect("src", "fm")
    out = run_one(p, "fm")
    assert out.schema["n"] == pl.Int64 and out["n"].to_list() == [1, 5, 3]


def test_fix_missing_value_refuses_decimals_for_an_integer_column(tmp_path):
    p = pipe_with(tmp_path, pl.DataFrame({"n": [1, None, 3]}))
    p.add_node("fix_missing", params={"method": "value", "value": "5.5"}, id="fm")
    p.connect("src", "fm")
    st = Executor(p).run(targets=["fm"])["fm"]
    assert st.status == "failed" and "decimals" in st.error


# --------------------------------------------------------------- Param bounds
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


# --------------------------------------------------------------- boolean text
def test_boolean_text_vocabulary_is_consistent():
    for word in ("true", "1", "yes", "y", "t", "on", "ON", " True "):
        assert Param("flag", "Flag", "bool").coerce(word) is True
        assert text_to_bool(word) is True
    for word in ("false", "0", "no", "off", ""):
        assert Param("flag", "Flag", "bool").coerce(word) is False
        assert text_to_bool(word) is False


# --------------------------------------------------------------- rate_of_change
def test_rate_of_change_returns_a_node_result(tmp_path):
    t0 = datetime(2024, 1, 1)
    df = pl.DataFrame({"t": [t0.replace(second=s) for s in range(5)], "v": [1.0, 2.0, 4.0, 8.0, 16.0]})
    p = pipe_with(tmp_path, df)
    p.add_node("rate_of_change", params={"columns": ["v"], "per": "s"}, id="r")
    p.connect("src", "r")
    out = run_one(p, "r")
    assert "v_per_second" in out.columns


# --------------------------------------------------------------- headless limits
def test_report_passes_project_inputs_to_chart_rendering(tmp_path, monkeypatch):
    import dancr.views.render as render_mod
    from dancr.core.nodes.report import build_report
    from dancr.core.registry import Ctx

    captured = {}

    def fake_render(lf, params, out, **kw):
        captured.update(kw)
        out.write_bytes(b"png")
        return out

    monkeypatch.setattr(render_mod, "render_chart", fake_render)
    lf = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]}).lazy()
    ctx = Ctx(tmp_path, "rep", "Report", inputs={"upper": 3.5})
    inputs = {"items": [lf]}
    meta = [{"title": "Chart", "node_type": "chart",
             "params": {"kind": "line", "x": "x", "series": [{"column": "y"}],
                        "limits": [{"value": "upper", "label": "upper"}]},
             "messages": [], "report": {}}]
    doc = build_report(ctx, inputs, {"title": "T", "path": "r.html", "pdf": False}, meta, None, ctx.inputs)
    assert "Chart" in doc and captured.get("inputs") == {"upper": 3.5}


def test_headless_render_resolves_an_input_named_limit(tmp_path):
    from dancr.views.render import render_chart

    lf = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [3.0, 4.0, 5.0]}).lazy()
    out = tmp_path / "c.png"
    render_chart(lf, {"kind": "line", "x": "x", "series": [{"column": "y"}],
                      "limits": [{"value": "upper", "label": "upper"}]},
                 out, inputs={"upper": 4.0})
    assert out.exists() and out.stat().st_size > 0


# --------------------------------------------------------------- preview failures
def test_preview_failure_is_typed_and_friendly(tmp_path):
    """A node that cannot run on a sample must raise PreviewUnavailable, never a raw traceback."""
    from dancr.core.executor import PreviewUnavailable

    p = pipe_with(tmp_path, pl.DataFrame({"x": [1.0], "y": [2.0]}))
    p.add_node("fit_curve", params={"x": "x", "y": "y", "kind": "linear"}, id="fit")
    p.connect("src", "fit")
    with pytest.raises(PreviewUnavailable) as ei:
        Executor(p).preview("fit")
    assert "at least two points" in str(ei.value)


# --------------------------------------------------------------- save hygiene
def test_save_leaves_no_temp_file(tmp_path):
    p = Pipeline("x")
    target = tmp_path / "p.json"
    p.save(target)
    p.save(target)
    leftovers = [f.name for f in tmp_path.iterdir() if ".tmp" in f.name]
    assert leftovers == []


# --------------------------------------------------------------- DATE / conditions
def test_date_with_a_format_on_a_time_column_is_rejected():
    from dancr.core.expr import check_formula

    msg = check_formula('DATE(t, "%Y-%m-%d")', {"t": pl.Datetime("us")})
    assert msg is not None and "already" in msg
    assert check_formula('DATE("01/02/2024", "%d/%m/%Y")', {}) is None


def test_conditions_in_is_case_insensitive_like_eq():
    from dancr.core.conditions import rule_mask

    df = pl.DataFrame({"s": ["OK", "ok", "no"]})
    m = lambda rule: df.select(rule_mask(df.schema, rule).alias("m"))["m"].to_list()
    assert m({"column": "s", "op": "eq", "value": "ok"}) == [True, True, False]
    assert m({"column": "s", "op": "in", "value": "ok"}) == [True, True, False]
    assert m({"column": "s", "op": "in", "value": "ok", "case_sensitive": True}) == [False, True, False]


# --------------------------------------------------------------- report blocks
def _report_ctx(tmp_path):
    from dancr.core.registry import Ctx
    return Ctx(tmp_path, "rep", "Report")


def _table():
    return pl.DataFrame({"x": [1.0, 2.0]}).lazy()


def test_report_blocks_resolve_items_by_node_id_through_a_reorder(tmp_path):
    from dancr.core.nodes.report import build_report

    frames = [_table(), _table()]
    meta = [{"node": "a", "title": "A", "node_type": "load_file", "params": {}, "messages": [], "report": {}},
            {"node": "b", "title": "B", "node_type": "load_file", "params": {}, "messages": [], "report": {}}]
    params = {"title": "T", "path": "r.html", "pdf": False,
              "blocks": [{"type": "item", "node": "b"}, {"type": "item", "node": "a"}]}
    doc = build_report(_report_ctx(tmp_path), {"items": frames}, params, meta, None, {})
    assert doc.index("<h2>B</h2>") < doc.index("<h2>A</h2>")


def test_report_blocks_still_accept_a_legacy_index(tmp_path):
    from dancr.core.nodes.report import build_report

    frames = [_table(), _table()]
    meta = [{"node": "a", "title": "A", "node_type": "load_file", "params": {}, "messages": [], "report": {}},
            {"node": "b", "title": "B", "node_type": "load_file", "params": {}, "messages": [], "report": {}}]
    params = {"title": "T", "path": "r.html", "pdf": False, "blocks": [{"type": "item", "index": 1}]}
    doc = build_report(_report_ctx(tmp_path), {"items": frames}, params, meta, None, {})
    assert "<h2>B</h2>" in doc and doc.index("<h2>B</h2>") < doc.index("<h2>A</h2>")


# --------------------------------------------------------------- shared formatting / palette
def test_series_palette_is_shared_between_the_view_and_the_renderer():
    from dancr.views import palette, render

    assert render.PALETTE == palette.SERIES_COLORS
    assert palette.series_color(0) == palette.SERIES_COLORS[0]
    assert palette.series_color(3, {"color": "#000000"}) == "#000000"
    assert palette.series_color(len(palette.SERIES_COLORS)) == palette.SERIES_COLORS[0]


def test_column_title_is_formatted_in_one_place():
    from dancr.views.table import column_title

    assert column_title("x", {"x": {"label": "Pressure", "unit": "psi"}}) == "Pressure (psi)"
    assert column_title("x", {"x": {"label": "Pressure"}}) == "Pressure"
    assert column_title("x", None) == "x"
    assert column_title("", None) == ""


# --------------------------------------------------------------- CLI / MCP alignment
def test_mcp_run_and_status_share_cli_field_names(probe_dir, tmp_path, mcp_root):
    from dancr.mcp_server import create_pipeline, add_node, run_pipeline, node_status

    pj = tmp_path / "p.json"
    create_pipeline(str(pj))
    add_node(str(pj), "load_file", {"path": str(probe_dir / "probe_A.csv")}, node_id="a")
    data = json.loads(run_pipeline(str(pj)))
    assert "elapsed" in data and "cache_dir" in data and isinstance(data["failed"], list)
    assert data["nodes"]["a"]["node_id"] == "a" and "elapsed" in data["nodes"]["a"]
    st = json.loads(node_status(str(pj), "a"))
    assert st["node_id"] == "a" and st["id"] == "a" and "elapsed" in st


def test_cli_run_json_reports_failed_nodes(probe_dir, tmp_path):
    import subprocess
    import sys

    pj = tmp_path / "p.json"
    subprocess.run([sys.executable, "-m", "dancr.cli", "new", str(pj)], capture_output=True, text=True)
    subprocess.run([sys.executable, "-m", "dancr.cli", "add", str(pj), "load_file", "--id", "a",
                    "--set", f"path={probe_dir / 'probe_A.csv'}"], capture_output=True, text=True)
    r = subprocess.run([sys.executable, "-m", "dancr.cli", "--json", "run", str(pj)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["failed"] == []

