"""The cross-project graph folded into the context/KB export: block, JSONL, --changed, FAIR, MCP, CLI."""
import json
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

from dancr.core import Pipeline
from dancr.headless import (build_context, build_graph, context_changes, context_jsonl, export_fair,
                            load_graph)


def make_repo(root: Path, *, b_sensitivity: str | None = None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "a").mkdir(exist_ok=True)
    (root / "b").mkdir(exist_ok=True)
    pl.DataFrame({"id": [1, 2, 3]}).write_csv(root / "a" / "customers.csv")
    pl.DataFrame({"order_id": [1, 2, 3], "customer_id": [1, 2, 1]}).write_csv(root / "a" / "orders.csv")
    pl.DataFrame({"customer_id": [1, 2, 3]}).write_csv(root / "b" / "contacts.csv")
    pl.DataFrame({"sale_id": [1, 2, 3], "customer_id": [1, 2, 3]}).write_csv(root / "b" / "sales.csv")
    for folder, name, nodes in (("a", "a.json", [("customers", "customers.csv"), ("orders", "orders.csv")]),
                                ("b", "b.json", [("contacts", "contacts.csv"), ("sales", "sales.csv")])):
        p = Pipeline(name[:-5])
        p.path = root / folder / name
        for nid, f in nodes:
            p.add_node("load_file", nid.title(), {"path": f}, id=nid)
        if folder == "b" and b_sensitivity:
            p.meta["sensitivity"] = b_sensitivity
        p.save()


def test_context_carries_incident_edges_neighbours_and_per_dataset(tmp_path):
    make_repo(tmp_path)
    build_graph(tmp_path)
    ctx = build_context(Pipeline.load(tmp_path / "a" / "a.json"))
    g = ctx["graph"]
    assert g["project"] == "a/a.json" and g["content_hash"]
    edges = [(e["left"], e["right"]) for e in g["edges"]]
    assert ("a/a.json#orders", "b/b.json#contacts") in edges
    assert ("a/a.json#orders", "a/a.json#customers") in edges          # an intra-project edge is incident too
    assert {n["id"] for n in g["neighbours"]} >= {"a/a.json#customers", "b/b.json#contacts"}
    assert sorted(g["per_dataset"]) == ["customers", "orders"]
    assert g["per_dataset"]["orders"]["edges"]
    # the per-dataset slice also reaches the document and its JSONL line
    doc = next(d for d in ctx["documents"] if d["node"] == "orders")
    assert doc["graph"]["edges"]
    line = json.loads(next(l for l in context_jsonl(ctx).splitlines() if '"node": "orders"' in l))
    assert line["graph"]["edges"]


def test_context_is_deterministic(tmp_path):
    make_repo(tmp_path)
    build_graph(tmp_path)
    p = Pipeline.load(tmp_path / "a" / "a.json")
    a = build_context(p)["graph"]
    b = build_context(p)["graph"]
    assert a == b and a["content_hash"] == b["content_hash"]


def test_no_graph_means_no_block(tmp_path):
    make_repo(tmp_path)                              # no graph built here
    ctx = build_context(Pipeline.load(tmp_path / "a" / "a.json"))
    assert "graph" not in ctx
    assert all("graph" not in d for d in ctx["documents"])


def test_explicit_root_and_a_project_outside_the_graph(tmp_path):
    repo = tmp_path / "repo"
    make_repo(repo)
    build_graph(repo)
    ctx = build_context(Pipeline.load(repo / "a" / "a.json"), root=repo)
    assert ctx["graph"]["root"] == str(repo.resolve())
    # a project with no datasets in the graph gets no block, even with a root
    other = Pipeline("solo")
    other.path = tmp_path / "solo.json"
    other.add_node("load_file", params={"path": "solo.csv"}, id="s")
    (tmp_path / "solo.csv").write_text("x\n1\n")
    assert "graph" not in build_context(other, root=repo)


def test_changed_reports_whether_the_graph_moved(tmp_path):
    make_repo(tmp_path)
    build_graph(tmp_path)
    p = Pipeline.load(tmp_path / "a" / "a.json")
    first = build_context(p)
    same = context_changes(build_context(p), first)
    assert same["changes"]["graph_changed"] is False
    # change B's shared key and rebuild: the graph changes, and --changed says so
    pl.DataFrame({"buyer_id": [1, 2, 3]}).write_csv(tmp_path / "b" / "contacts.csv")
    pl.DataFrame({"sale_id": [1, 2, 3], "buyer_id": [1, 2, 3]}).write_csv(tmp_path / "b" / "sales.csv")
    build_graph(tmp_path)
    moved = context_changes(build_context(p), first)
    assert moved["changes"]["graph_changed"] is True
    assert "graph" in moved                            # the new graph block is carried


def test_restricted_datasets_are_withheld_unless_allowed(tmp_path):
    make_repo(tmp_path, b_sensitivity="confidential")
    build_graph(tmp_path)
    p = Pipeline.load(tmp_path / "a" / "a.json")
    ctx = build_context(p)
    assert all("b/b.json" not in e["left"] and "b/b.json" not in e["right"] for e in ctx["graph"]["edges"])
    assert all("b/b.json" not in n["id"] for n in ctx["graph"]["neighbours"])
    allowed = build_context(p, allow_restricted=True)
    assert any("b/b.json" in e["left"] or "b/b.json" in e["right"] for e in allowed["graph"]["edges"])


def test_restricted_project_identity_is_withheld_from_the_slice(tmp_path):
    from dancr.headless import graph_slice
    make_repo(tmp_path, b_sensitivity="confidential")
    build_graph(tmp_path)
    sl = graph_slice(tmp_path)
    assert all("b/b.json" not in p["id"] for p in sl["projects"])
    full = graph_slice(tmp_path, allow_restricted=True)
    assert any("b/b.json" in p["id"] for p in full["projects"])


def test_fair_descriptors_carry_the_graph(tmp_path):
    make_repo(tmp_path)
    build_graph(tmp_path)
    p = Pipeline.load(tmp_path / "a" / "a.json")
    assert export_fair(p, fmt="schema.org").get("dancrGraph")
    assert export_fair(p, fmt="frictionless").get("dancrGraph")
    assert export_fair(p, fmt="manifest").get("graph")
    crate = export_fair(p, fmt="rocrate")
    assert any(e.get("dancrGraph") for e in crate["@graph"] if isinstance(e, dict))


def test_cli_context_carries_the_graph(tmp_path):
    make_repo(tmp_path)
    build_graph(tmp_path)
    r = subprocess.run([sys.executable, "-m", "dancr.cli", "--json", "context", "a/a.json"],
                       capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["graph"]["edges"]
    human = subprocess.run([sys.executable, "-m", "dancr.cli", "context", "a/a.json"],
                           capture_output=True, text=True, cwd=tmp_path)
    assert "Cross-project relations" in human.stdout


def test_mcp_profile_carries_the_graph(tmp_path, mcp_root):
    import dancr.mcp_server as srv
    make_repo(tmp_path)
    build_graph(tmp_path)
    out = json.loads(srv.profile("a/a.json"))
    assert out["graph"]["edges"]
