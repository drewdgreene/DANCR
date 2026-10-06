"""Trust invariants: DANCR is never silently wrong.

Three promises, each a regression guard:
  1. a forgiving column match in a formula is announced (and refused in strict mode);
  2. a file that is not a table is refused with a plain reason, never misread as CSV;
  3. a column that looks like a key but repeats a value is named by 'Check the data'.
"""
from pathlib import Path

import polars as pl
import pytest

from dancr.core import Pipeline
from dancr.core.executor import Executor
from dancr.core.nodes.load import NON_TABLE_EXT, H5_EXT, NC_EXT, _refuse_non_table


def build(tmp_path: Path, *nodes) -> Pipeline:
    p = Pipeline("t")
    p.path = tmp_path / "t.json"
    for type_key, params, nid, after in nodes:
        p.add_node(type_key, params=params, id=nid)
        if after:
            p.connect(after, nid)
    return p


def status(p, nid):
    return Executor(p).run(targets=[nid])[nid]


# ---------------------------------------------------------------- 1. formulas
def _sheet(tmp_path) -> Pipeline:
    return build(tmp_path, ("enter_data",
                            {"columns": [{"name": "plant_height_cm", "type": "number"}], "rows": [[10], [20]]},
                            "d", None))


def test_forgiving_column_match_is_announced(tmp_path):
    p = _sheet(tmp_path)
    p.add_node("calculate", params={"formulas": [{"name": "t", "expr": "[plantheightcm] * 1"}]}, id="c")
    p.connect("d", "c")
    st = status(p, "c")
    assert st.status == "done"
    assert any("plantheightcm" in m and "plant_height_cm" in m for m in st.messages)


def test_exact_column_match_is_silent(tmp_path):
    p = _sheet(tmp_path)
    p.add_node("calculate", params={"formulas": [{"name": "t", "expr": "[plant_height_cm] * 1"}]}, id="c")
    p.connect("d", "c")
    st = status(p, "c")
    assert st.status == "done"
    assert not any("matched" in m for m in st.messages)


def test_strict_columns_refuses_a_forgiving_match(tmp_path):
    p = _sheet(tmp_path)
    p.add_node("calculate", params={"formulas": [{"name": "t", "expr": "[plantheightcm] * 1"}],
                                    "strict_columns": True}, id="c")
    p.connect("d", "c")
    st = status(p, "c")
    assert st.status == "failed"
    assert "exact column name" in (st.error or "")


def test_keep_rows_formula_announces_too(tmp_path):
    p = _sheet(tmp_path)
    p.add_node("keep_rows", params={"formula": "[plantheightcm] > 5"}, id="k")
    p.connect("d", "k")
    st = status(p, "k")
    assert st.status == "done"
    assert any("plant_height_cm" in m for m in st.messages)


# ---------------------------------------------------------------- 2. loaders
@pytest.mark.parametrize("ext", sorted(NON_TABLE_EXT))
def test_non_table_extension_is_refused(ext, tmp_path):
    f = tmp_path / f"x{ext}"
    f.write_bytes(b"whatever")
    p = build(tmp_path, ("load_file", {"path": str(f)}, "l", None))
    st = status(p, "l")
    assert st.status == "failed"
    assert "not a table" in (st.error or "")


def test_hdf5_points_at_load_hdf5(tmp_path):
    f = tmp_path / "x.h5"
    f.write_bytes(b"\x89HDF\r\n")
    p = build(tmp_path, ("load_file", {"path": str(f)}, "l", None))
    st = status(p, "l")
    assert st.status == "failed"
    assert "Load HDF5" in (st.error or "")


def test_netcdf_points_at_load_netcdf(tmp_path):
    f = tmp_path / "x.nc"
    f.write_bytes(b"CDF")
    p = build(tmp_path, ("load_file", {"path": str(f)}, "l", None))
    st = status(p, "l")
    assert st.status == "failed"
    assert "Load NetCDF" in (st.error or "")


def test_plain_json_is_refused_as_not_a_table(tmp_path):
    f = tmp_path / "x.json"
    f.write_text('{"a": 1, "b": 2}')
    p = build(tmp_path, ("load_file", {"path": str(f)}, "l", None))
    st = status(p, "l")
    assert st.status == "failed"
    assert "not a table" in (st.error or "").lower()


def test_refuse_non_table_direct():
    with pytest.raises(ValueError):
        _refuse_non_table(Path("a.png"), ".png")
    # a readable extension is left alone
    _refuse_non_table(Path("a.csv"), ".csv")


# ---------------------------------------------------------------- 3. check data
def test_check_data_names_a_repeated_key(tmp_path):
    f = tmp_path / "reg.csv"
    f.write_text("canonical_name,registry_id,lims_id\n"
                 "A,VYLR-L0007,PIO-1\nB,VYLR-L0007,PIO-2\nC,VYLR-L0008,PIO-3\n")
    p = build(tmp_path, ("load_file", {"path": "reg.csv"}, "src", None),
              ("check_data", {}, "cd", "src"))
    st = status(p, "cd")
    assert st.status == "done"
    assert st.report.get("duplicate_keys") == [{"column": "registry_id", "duplicates": 1}]
    assert "registry_id" in st.report["finding"]["statement"]


def test_check_data_clean_table_has_no_key_dups(tmp_path):
    f = tmp_path / "reg.csv"
    f.write_text("registry_id,lims_id\nA,1\nB,2\nC,3\n")
    p = build(tmp_path, ("load_file", {"path": "reg.csv"}, "src", None),
              ("check_data", {}, "cd", "src"))
    st = status(p, "cd")
    assert st.status == "done"
    assert st.report.get("duplicate_keys") == []


# ---------------------------------------------------------------- 4. agent edits
import dancr  # noqa: E402
from dancr.core.assistant.client import ChatResult, FakeProvider, Message, ToolCall  # noqa: E402
try:
    from dancr import assistant_turn
except AttributeError:  # pragma: no cover
    from dancr.headless import assistant_turn


def _proposal_edit_reply(edits):
    return ChatResult(Message("assistant", "", [ToolCall("c1", "propose_edits",
                                                         {"reply": "done", "edits": edits})]))


def test_headless_assistant_applies_edits(tmp_path):
    (tmp_path / "d.csv").write_text("a\n1\n2\n")
    p = Pipeline("t"); p.path = tmp_path / "p.json"
    p.add_node("load_file", params={"path": "d.csv"}, id="src")
    p.save(p.path)
    prov = FakeProvider([_proposal_edit_reply([{"op": "rename", "node": "src", "title": "Source data"}])])
    with dancr.editing(str(p.path)) as proj:
        out = assistant_turn(proj, "rename the step", provider=prov, build=True)
        assert out.get("applied_edits") == ["renamed src to “Source data”"]
        assert proj.nodes["src"].title == "Source data"


def test_headless_assistant_edits_need_build(tmp_path):
    (tmp_path / "d.csv").write_text("a\n1\n2\n")
    p = Pipeline("t"); p.path = tmp_path / "p.json"
    p.add_node("load_file", params={"path": "d.csv"}, id="src")
    p.save(p.path)
    prov = FakeProvider([_proposal_edit_reply([{"op": "rename", "node": "src", "title": "Nope"}])])
    with dancr.editing(str(p.path)) as proj:
        out = assistant_turn(proj, "rename", provider=prov, build=False)
        assert "applied_edits" not in out
        assert proj.nodes["src"].title != "Nope"


def test_apply_edits_rejects_an_unknown_op():
    from dancr.core.assistant.edits import apply_edits
    with pytest.raises(ValueError):
        apply_edits(Pipeline("t"), [{"op": "explode", "node": "x"}])
