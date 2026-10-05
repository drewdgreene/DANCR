"""FAIR descriptors: schema.org/Dataset, Frictionless Data Package, the run manifest, and RO-Crate — through
the Python API (`dancr.export_fair`), the CLI (`dancr fair`, `dancr dataset`) and MCP (`export_fair`)."""
import json
import subprocess
import sys

import polars as pl
import pytest

from dancr import headless as hl
from dancr.core import Pipeline, fair
from dancr.core.executor import Executor


def run(*args, cwd=None):
    r = subprocess.run([sys.executable, "-m", "dancr.cli", *args], capture_output=True, text=True, cwd=cwd)
    return r.returncode, r.stdout, r.stderr


@pytest.fixture()
def shop(tmp_path):
    pl.DataFrame({"id": [1, 2, 3], "name": ["A", "B", "C"]}).write_csv(tmp_path / "customers.csv")
    pl.DataFrame({"customer_id": [1, 1, 2, 3], "amount": [10.0, 20.0, 30.0, 40.0],
                  "when": ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"]}).write_csv(tmp_path / "orders.csv")
    p = Pipeline("shop"); p.path = tmp_path / "shop.json"
    p.add_node("load_file", title="Customers", params={"path": "customers.csv"}, id="customers")
    p.add_node("load_file", title="Orders", params={"path": "orders.csv"}, id="orders")
    p.set_dataset_meta(license="CC-BY-4.0", creator={"name": "Jane Doe", "email": "jane@example.org"},
                       description="Field trial data", keywords=["maize", "yield"], version="1.0")
    p.save()
    return p


# ---------------------------------------------------------------- metadata round-trip
def test_dataset_meta_round_trips_and_survives_meta(tmp_path):
    p = Pipeline("p"); p.path = tmp_path / "p.json"
    p.set_dataset_meta(description="x", license="MIT", keywords="a, b")
    p.save()
    again = Pipeline.load(p.path)
    assert again.dataset_meta()["license"] == "MIT"
    assert again.dataset_meta()["keywords"] == ["a", "b"]
    # a project file written by an older DANCR drops unknown *top-level* keys, but keeps meta wholesale
    data = json.loads(p.path.read_text())
    data["something_new"] = 1
    back = Pipeline.from_dict(data, p.path)
    assert back.dataset_meta()["description"] == "x"


def test_unknown_dataset_field_is_refused(shop):
    with pytest.raises(Exception):
        shop.set_dataset_meta(nonsense="x")


# ---------------------------------------------------------------- schema.org
def test_schema_org_dataset_shape(shop):
    ex = Executor(shop); ex.run(["customers", "orders"])
    ctx = hl.build_context(shop, ex, stats=True)
    doc = fair.dataset_jsonld(ctx)
    assert doc["@context"] == "https://schema.org/" and doc["@type"] == "Dataset"
    assert doc["license"].endswith("CC-BY-4.0.html")
    assert any(p.get("name") == "Jane Doe" for p in doc["creator"])
    names = {v["name"] for v in doc["variableMeasured"]}
    assert {"id", "customer_id", "amount", "when"} <= names
    assert doc["temporalCoverage"] and doc["temporalCoverage"].startswith("2024")
    assert {d["name"] for d in doc["distribution"]} == {"Customers", "Orders"}


# ---------------------------------------------------------------- Frictionless
def test_frictionless_datapackage(shop):
    ex = Executor(shop); ex.run(["customers", "orders"])
    ctx = hl.build_context(shop, ex, stats=True)
    pkg = fair.datapackage(ctx)
    assert pkg["profile"] == "tabular-data-package"
    assert [r["name"] for r in pkg["resources"]] == ["customers", "orders"]
    fields = {f["name"]: f for f in pkg["resources"][1]["schema"]["fields"]}
    assert fields["customer_id"]["type"] == "integer"
    assert fields["amount"]["type"] == "number"
    assert fields["when"]["type"] == "datetime"
    assert pkg["licenses"][0]["name"] == "CC-BY-4.0"


def test_frictionless_types_are_deterministic(shop):
    ex = Executor(shop); ex.run(["customers", "orders"])
    ctx = hl.build_context(shop, ex)
    a = json.dumps(fair.datapackage(ctx), sort_keys=True)
    b = json.dumps(fair.datapackage(ctx), sort_keys=True)
    assert a == b


# ---------------------------------------------------------------- run manifest
def test_run_manifest_records_engine_inputs_and_steps(shop):
    ex = Executor(shop); ex.run(["customers", "orders"])
    man = fair.run_manifest(shop, ex)
    assert man["kind"] == "dancr.manifest" and man["engine"]["name"] == "DANCR"
    assert man["engine"]["fingerprint"] and man["engine"]["version"]
    assert man["libraries"]["polars"]
    assert {s["node"] for s in man["sources"]} == {"customers", "orders"}
    assert all(s["files"] for s in man["sources"])
    assert {n["id"] for n in man["nodes"]} == {"customers", "orders"}
    assert all(n["hash"] for n in man["nodes"])
    assert next(n for n in man["nodes"] if n["id"] == "orders")["rows"] == 4


def test_manifest_and_rocrate_build(shop):
    ex = Executor(shop); ex.run(["customers", "orders"])
    ctx = hl.build_context(shop, ex, stats=True)
    man = fair.run_manifest(shop, ex)
    crate = fair.ro_crate(ctx, man, pipeline_file="shop.json")
    assert crate["@context"].startswith("https://w3id.org/ro/crate")
    root = crate["@graph"][1]
    assert root["@type"] == "Dataset" and any(p["@id"] == "dancr-manifest.json" for p in root["hasPart"])


# ---------------------------------------------------------------- export_fair / CLI / MCP
def test_export_fair_writes_and_returns(shop, tmp_path):
    doc = hl.export_fair(shop, Executor(shop), fmt="schema.org", out=tmp_path / "dataset.jsonld")
    assert json.loads((tmp_path / "dataset.jsonld").read_text())["@type"] == "Dataset"
    assert doc["@type"] == "Dataset"


def test_cli_dataset_and_fair(shop, tmp_path):
    code, out, err = run("--json", "dataset", str(shop.path))
    assert code == 0 and json.loads(out)["license"] == "CC-BY-4.0"
    code, out, err = run("dataset", str(shop.path), "--set", "version=2.0")
    assert code == 0, err
    code, out, err = run("--json", "fair", str(shop.path), "--format", "frictionless")
    assert code == 0 and json.loads(out)["profile"] == "tabular-data-package"
    target = tmp_path / "man.json"
    code, out, err = run("fair", str(shop.path), "--format", "manifest", "--out", str(target))
    assert code == 0 and json.loads(target.read_text())["kind"] == "dancr.manifest"


def test_package_rocrate_directory_and_zip(shop):
    import zipfile
    ex = Executor(shop); ex.run(["customers", "orders"])
    rec = hl.package_rocrate(shop, ex, out="crate", copy="metadata")
    assert rec["format"] == "directory" and rec["copy"] == "metadata"
    d = shop.directory / "crate"
    graph = json.loads((d / "ro-crate-metadata.json").read_text())
    assert graph["@graph"][1]["@type"] == "Dataset"
    assert (d / "dancr-pipeline.json").exists() and (d / "dancr-manifest.json").exists()

    hl.package_rocrate(shop, ex, out="crate.zip", copy="data", zip=True)
    z = zipfile.ZipFile(shop.directory / "crate.zip")
    names = z.namelist()
    assert "ro-crate-metadata.json" in names and "dancr-manifest.json" in names
    assert any(n.startswith("data/") for n in names)
    crate = json.loads(z.read("ro-crate-metadata.json"))
    ids = {e["@id"] for e in crate["@graph"] if e.get("@type") == "File"}
    assert "dancr-pipeline.json" in ids


def test_package_rocrate_refuses_outside_and_existing(shop):
    ex = Executor(shop)
    with pytest.raises(ValueError):
        hl.package_rocrate(shop, ex, out="/tmp/elsewhere-crate.zip", zip=True)
    (shop.directory / "taken.zip").write_text("x")
    with pytest.raises(ValueError):
        hl.package_rocrate(shop, ex, out="taken.zip", zip=True)


def test_cli_package(shop, tmp_path):
    code, out, err = run("package", str(shop.path), "--out", "cli.rocrate.zip")
    assert code == 0, err
    assert (shop.directory / "cli.rocrate.zip").exists()


def test_dataset_dialog_collects_fields(app):
    from dancr.ui.dialogs import DatasetDialog
    dlg = DatasetDialog(None, {"creator": {"name": "A Person", "email": "a@example.org"}})
    dlg.edits["license"].setText("MIT")
    dlg.edits["keywords"].setText("maize, yield")
    dlg.edits["creator"].setText("Jane Doe <jane@example.org>")
    dlg.description.setPlainText("A study")
    f = dlg.fields()
    assert f["license"] == "MIT" and f["keywords"] == ["maize", "yield"]
    assert f["creator"] == {"name": "Jane Doe", "email": "jane@example.org"}
    assert f["description"] == "A study"


def test_mcp_export_fair_and_meta(tmp_path, mcp_root):
    import dancr.mcp_server as srv
    pl.DataFrame({"x": [1, 2]}).write_csv(tmp_path / "d.csv")
    pj = tmp_path / "p.json"
    srv.create_pipeline(str(pj))
    srv.add_node(str(pj), "load_file", {"path": "d.csv"}, node_id="src")
    srv.set_dataset_meta(str(pj), {"license": "MIT", "creator": "A Person"})
    assert json.loads(srv.get_dataset_meta(str(pj)))["license"] == "MIT"
    out = json.loads(srv.export_fair(str(pj), format="frictionless"))
    assert out["resources"][0]["name"] == "src"
