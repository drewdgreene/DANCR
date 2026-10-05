"""A catalog of every DANCR project under a folder, through the Python API (`dancr.build_catalog`), the CLI
(`dancr catalog`) and MCP (`catalog`)."""
import json
import subprocess
import sys

import polars as pl
import pytest

from dancr import headless as hl
from dancr.core import Pipeline


def run(*args, cwd=None):
    r = subprocess.run([sys.executable, "-m", "dancr.cli", *args], capture_output=True, text=True, cwd=cwd)
    return r.returncode, r.stdout, r.stderr


@pytest.fixture()
def root(tmp_path):
    """Two projects and one non-DANCR JSON that must be ignored."""
    (tmp_path / "a").mkdir(); (tmp_path / "b").mkdir()
    for sub, name in (("a", "shop"), ("b", "trials")):
        pl.DataFrame({"x": [1, 2, 3], "y": [1.0, 2.0, 3.0]}).write_csv(tmp_path / sub / "d.csv")
        p = Pipeline(name); p.path = tmp_path / sub / "p.json"
        p.add_node("load_file", title="Data", params={"path": "d.csv"}, id="src")
        p.save()
    (tmp_path / "notdancr.json").write_text('{"hello": 1}')
    (tmp_path / ".dancr").mkdir()
    (tmp_path / ".dancr" / "skip.json").write_text('{"dancr": 2, "nodes": []}')
    return tmp_path


def test_find_pipelines_ignores_non_dancr_and_dancr_folder(root):
    found = {p.parent.name for p in hl.find_pipelines(root)}
    assert found == {"a", "b"}


def test_build_catalog_lists_datasets(root):
    cat = hl.build_catalog(root, recursive=True)
    assert cat["kind"] == "dancr.catalog" and cat["count"] == 2
    names = {p["name"] for p in cat["projects"]}
    assert names == {"shop", "trials"}
    ds = cat["projects"][0]["datasets"][0]
    assert ds["id"] == "src" and ds["content_hash"] and ds["text"]


def test_catalog_jsonl_and_by_project(root):
    cat = hl.build_catalog(root)
    per_ds = [json.loads(x) for x in hl.catalog_jsonl(cat).splitlines() if x]
    assert len(per_ds) == 2 and all("project" in x for x in per_ds)
    per_proj = [json.loads(x) for x in hl.catalog_jsonl(cat, by_project=True).splitlines() if x]
    assert len(per_proj) == 2 and all("datasets" in x for x in per_proj)


def test_catalog_changes(root):
    before = hl.build_catalog(root)
    unchanged = hl.catalog_changes(hl.build_catalog(root), before)
    assert unchanged["count"] == 0
    (root / "a" / "d.csv").write_text("x,y\n9,9.0\n")
    delta = hl.catalog_changes(hl.build_catalog(root), before)
    assert delta["count"] == 1 and delta["projects"][0]["name"] == "shop"


def test_cli_catalog(root, tmp_path):
    code, out, err = run("--json", "catalog", str(root))
    assert code == 0, err
    assert json.loads(out)["count"] == 2
    target = tmp_path / "index.jsonl"
    code, out, err = run("catalog", str(root), "--jsonl", "--output", str(target))
    assert code == 0 and len(target.read_text().splitlines()) == 2


def test_catalog_dialog_lists_projects(app, root):
    from dancr.ui.dialogs import CatalogDialog
    dlg = CatalogDialog(None)
    dlg._show(hl.build_catalog(root))
    assert dlg.tree.topLevelItemCount() == 2
    dlg._apply_filter("zzz-nothing")
    assert all(dlg.tree.topLevelItem(i).isHidden() for i in range(dlg.tree.topLevelItemCount()))


def test_mcp_catalog(tmp_path, mcp_root):
    import dancr.mcp_server as srv
    (tmp_path / "p").mkdir()
    pl.DataFrame({"x": [1]}).write_csv(tmp_path / "p" / "d.csv")
    pj = tmp_path / "p" / "p.json"
    srv.create_pipeline(str(pj))
    srv.add_node(str(pj), "load_file", {"path": "d.csv"}, node_id="src")
    out = json.loads(srv.catalog("."))
    assert out["count"] == 1 and out["projects"][0]["datasets"][0]["id"] == "src"
