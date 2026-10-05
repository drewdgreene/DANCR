"""UCUM unit codes behind units.py, so FAIR descriptors are machine-interoperable."""
import polars as pl
import pytest

from dancr.core.units import to_ucum
from dancr.core import fair
from dancr import headless as hl
from dancr.core import Pipeline
from dancr.core.executor import Executor


@pytest.mark.parametrize("written, code", [
    ("g", "g"), ("kg", "kg"), ("cm2", "cm2"), ("g/m2", "g/m2"), ("°C", "Cel"), ("%", "%"),
    ("ha", "har"), ("km/h", "km/h"), ("mbar", "mbar"), ("mg/L", "mg/L"), ("us", ""),
])
def test_known_and_unknown_units(written, code):
    assert to_ucum(written) == code


def test_units_are_not_guessed():
    assert to_ucum("") == "" and to_ucum("widgets") == ""


def test_fair_descriptors_carry_ucum(tmp_path):
    pl.DataFrame({"mass": [1.0, 2.0], "area": [3.0, 4.0]}).write_csv(tmp_path / "d.csv")
    p = Pipeline("p"); p.path = tmp_path / "p.json"
    p.add_node("load_file", title="Data", params={"path": "d.csv"}, id="d")
    p.set_column_meta("mass", unit="g")
    p.set_column_meta("area", unit="cm2")
    p.save()
    ctx = hl.build_context(p, Executor(p), stats=False)
    doc = fair.dataset_jsonld(ctx)
    got = {v["name"]: v.get("unitText") for v in doc["variableMeasured"]}
    assert got["mass"] == "g" and got["area"] == "cm2"
    pkg = fair.datapackage(ctx)
    fields = {f["name"]: f.get("unit") for f in pkg["resources"][0]["schema"]["fields"]}
    assert fields["mass"] == "g" and fields["area"] == "cm2"
