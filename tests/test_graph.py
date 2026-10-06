"""The cross-project entity graph: build, query, relations, sensitivity and determinism."""
import json
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

from dancr.core import Pipeline
from dancr.core.graph import Column, Dataset, Edge, Graph, Project, graph_fingerprint
from dancr.headless import (build_graph, graph_neighbors, graph_path, graph_query, graph_shared_keys,
                            graph_summary)


def _pure_graph() -> Graph:
    g = Graph()
    g.add_project(Project("p/a.json", "A", "a.json"))
    g.add_project(Project("q/b.json", "B", "b.json"))
    for dsid, proj, node in (("p/a.json#one", "p/a.json", "one"), ("p/a.json#two", "p/a.json", "two"),
                             ("q/b.json#three", "q/b.json", "three")):
        g.add_dataset(Dataset(dsid, proj, node, node.title(),
                              columns=[Column(f"{dsid}:column:id", dsid, "id", "id", "number", "", True)]))
    g.add_edge(Edge("e1", "link", "p/a.json#one", "p/a.json#two", "id", "id", "many-to-one", 90.0, 0.9, "keys match"))
    g.add_edge(Edge("e2", "link", "p/a.json#two", "q/b.json#three", "id", "id", "many-to-one", 80.0, 0.8, "cross"))
    return g


def test_pure_graph_query_neighbors_path_and_shared_keys(tmp_path):
    g = _pure_graph()
    assert len(g.query(kind="link")["edges"]) == 2
    n = g.neighbors("p/a.json#two")
    assert n["count"] == 2
    p = g.path("p/a.json#one", "q/b.json#three")
    assert p["found"] and p["datasets"] == ["p/a.json#one", "p/a.json#two", "q/b.json#three"]
    shared = g.shared_keys()
    assert len(shared) == 1 and shared[0]["left_on"] == "id"


def test_graph_sqlite_round_trip_is_deterministic(tmp_path):
    g = _pure_graph()
    db = tmp_path / "graph.db"
    g.write(db)
    back = Graph.from_sqlite(db)
    assert graph_fingerprint(g) == graph_fingerprint(back)
    assert back.summary()["datasets"] == 3 and back.summary()["edges"] == 2
    # writing twice gives byte-identical databases (deterministic serialisation)
    db2 = tmp_path / "graph2.db"
    back.write(db2)
    assert db.read_bytes() == db2.read_bytes()


def test_restricted_datasets_are_withheld_unless_allowed():
    g = _pure_graph()
    g.datasets["p/a.json#two"].sensitivity = "confidential"
    assert all(d["id"] != "p/a.json#two" for d in g.query()["datasets"])
    assert g.query(allow_restricted=True)["count"] == 3
    assert g.edges_of("p/a.json#one") == []            # the edge to the restricted dataset is withheld too
    assert len(g.neighbors("p/a.json#one", allow_restricted=True)["neighbors"]) == 1


def test_neighbors_refuses_a_restricted_dataset():
    g = _pure_graph()
    g.datasets["p/a.json#two"].sensitivity = "restricted"
    with pytest.raises(ValueError, match="restricted"):
        g.neighbors("p/a.json#two")
    assert g.neighbors("p/a.json#two", allow_restricted=True)["count"] == 2


def make_repo(root: Path) -> None:
    (root / "a").mkdir()
    (root / "b").mkdir()
    pl.DataFrame({"id": [1, 2, 3], "region": ["N", "S", "N"]}).write_csv(root / "a" / "customers.csv")
    pl.DataFrame({"order_id": [1, 2, 3], "customer_id": [1, 2, 1]}).write_csv(root / "a" / "orders.csv")
    pl.DataFrame({"customer_id": [1, 2, 3], "email": ["a@x", "b@x", "c@x"]}).write_csv(root / "b" / "contacts.csv")
    pl.DataFrame({"sale_id": [1, 2, 3], "customer_id": [1, 2, 3]}).write_csv(root / "b" / "sales.csv")
    for folder, name, nodes in (("a", "a.json", [("customers", "customers.csv"), ("orders", "orders.csv")]),
                                ("b", "b.json", [("contacts", "contacts.csv"), ("sales", "sales.csv")])):
        p = Pipeline(name.rsplit(".", 1)[0])
        p.path = root / folder / name
        for nid, f in nodes:
            p.add_node("load_file", nid.title(), {"path": f}, id=nid)
        p.save()


def test_build_graph_is_incremental_and_deterministic(tmp_path):
    make_repo(tmp_path)
    first = build_graph(tmp_path)
    assert first["projects"] == 2 and first["datasets"] == 4 and first["cross_project_edges"] >= 1
    assert sorted(first["rebuilt"]) == ["a/a.json", "b/b.json"] and first["reused"] == []

    from dancr.headless import load_graph
    g1 = load_graph(tmp_path)
    fp1 = graph_fingerprint(g1)

    second = build_graph(tmp_path)
    assert second["rebuilt"] == [] and sorted(second["reused"]) == ["a/a.json", "b/b.json"]
    g2 = load_graph(tmp_path)
    assert graph_fingerprint(g2) == fp1               # unchanged content, identical graph

    # a changed source file makes only its own project rebuild and moves the graph
    pl.DataFrame({"id": [1, 2, 3, 4], "region": ["N", "S", "N", "S"]}).write_csv(tmp_path / "a" / "customers.csv")
    third = build_graph(tmp_path)
    assert third["rebuilt"] == ["a/a.json"] and third["reused"] == ["b/b.json"]
    assert graph_fingerprint(load_graph(tmp_path)) != fp1


def test_shared_keys_and_path_across_projects(tmp_path):
    make_repo(tmp_path)
    build_graph(tmp_path)
    keys = graph_shared_keys(tmp_path)
    assert keys["count"] >= 1
    assert any(k["left_on"] == "customer_id" and k["right_on"] == "customer_id" for k in keys["keys"])
    path = graph_path(tmp_path, "b/b.json#contacts", "a/a.json#orders")
    assert path["found"] and path["hops"] == 1


def test_query_text_finds_a_dataset(tmp_path):
    make_repo(tmp_path)
    build_graph(tmp_path)
    out = graph_query(tmp_path, text="orders")
    assert any(d["id"].endswith("#orders") for d in out["datasets"])
    assert graph_summary(tmp_path)["projects"] == 2


def test_project_level_sensitivity_is_respected(tmp_path):
    make_repo(tmp_path)
    p = Pipeline.load(tmp_path / "a" / "a.json")
    p.meta["sensitivity"] = "confidential"
    p.save()
    build_graph(tmp_path, force=True)
    assert all("a/a.json" not in d["id"] for d in graph_query(tmp_path)["datasets"])
    assert any("a/a.json" in d["id"] for d in graph_query(tmp_path, allow_restricted=True)["datasets"])


def test_cli_graph_build_and_query(tmp_path):
    make_repo(tmp_path)
    r = subprocess.run([sys.executable, "-m", "dancr.cli", "graph", "build", str(tmp_path)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    r2 = subprocess.run([sys.executable, "-m", "dancr.cli", "graph", "summary", str(tmp_path)],
                        capture_output=True, text=True)
    assert r2.returncode == 0 and "project(s)" in r2.stdout


def test_mcp_graph_tools(tmp_path, mcp_root):
    import dancr.mcp_server as srv
    pj = tmp_path / "shop.json"
    srv.create_pipeline(str(pj))
    srv.add_node(str(pj), "enter_data", {"columns": [{"name": "c", "type": "number"}], "rows": [[1], [2]]}, node_id="d")
    out = json.loads(srv.graph_build())
    assert out["datasets"] >= 1
    q = json.loads(srv.graph_query(text="d"))
    assert q["count"] >= 1
    assert json.loads(srv.graph_shared_keys())["count"] == 0
