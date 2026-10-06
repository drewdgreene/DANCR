"""Cross-project structural QA (C2): deterministic questions over the graph, citing edge evidence."""
import json
import subprocess
import sys

import polars as pl

from dancr.core import Pipeline
from dancr.core.crossask import ask, classify, resolve, suggest
from dancr.core.graph import Column, Dataset, Edge, Graph, Project
from dancr.headless import build_graph, cross_ask, cross_suggest


def _graph():
    g = Graph()
    g.add_project(Project("a/a.json", "A", "a.json"))
    g.add_project(Project("b/b.json", "B", "b.json"))
    for dsid, proj, node, title in (("a/a.json#orders", "a/a.json", "orders", "Orders"),
                                    ("a/a.json#customers", "a/a.json", "customers", "Customers"),
                                    ("b/b.json#contacts", "b/b.json", "contacts", "Contacts")):
        g.add_dataset(Dataset(dsid, proj, node, title,
                              columns=[Column(f"{dsid}:column:customer_id", dsid, "customer_id", "id", "number", "", True)]))
    g.add_edge(Edge("e1", "link", "a/a.json#orders", "a/a.json#customers", "customer_id", "id", "many-to-one", 95.0, 0.9, "keys match"))
    g.add_edge(Edge("e2", "link", "a/a.json#orders", "b/b.json#contacts", "customer_id", "customer_id", "", 0.0, 0.8, "same key by name"))
    return g


def test_classify():
    assert classify("which keys link projects?") == "shared_keys"
    assert classify("what joins a and b?") == "path"
    assert classify("where does orders come from?") == "upstream"
    assert classify("what feeds orders?") == "downstream"
    assert classify("what relates to orders?") == "neighbors"
    assert classify("find orders") == "find"


def test_resolve_by_id_title_node_and_ambiguity():
    g = _graph()
    assert resolve(g, "a/a.json#orders")[0].id == "a/a.json#orders"
    assert resolve(g, "Orders")[0].id == "a/a.json#orders"
    assert resolve(g, "contacts")[0].id == "b/b.json#contacts"
    d, candidates = resolve(g, "customer_id")            # a column name shared by every dataset
    assert d is None and len(candidates) == 3


def test_ask_intents_cite_evidence():
    g = _graph()
    shared = ask(g, "which keys link projects?")
    assert shared["ok"] and len(shared["keys"]) == 1 and shared["keys"][0]["left"] == "a/a.json#orders"
    path = ask(g, "what joins a/a.json#orders and b/b.json#contacts?")
    assert path["ok"] and path["hops"] == 1 and path["edges"]
    n = ask(g, "what relates to a/a.json#orders?")
    assert n["ok"] and len(n["edges"]) == 2
    up = ask(g, "where does a/a.json#orders come from?")
    assert up["ok"] and up["edges"]
    find = ask(g, "find orders")
    assert find["ok"] and find["datasets"][0]["id"] == "a/a.json#orders"


def test_ask_ambiguity_is_reported_not_guessed():
    g = _graph()
    out = ask(g, "what relates to customer_id?")
    assert out["ok"] is False and out["candidates"]


def test_suggest_grounded_in_the_graph():
    g = _graph()
    intents = [s["intent"] for s in suggest(g)]
    assert "shared_keys" in intents and "path" in intents and "neighbors" in intents


def make_repo(root):
    (root / "a").mkdir()
    (root / "b").mkdir()
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
        p.save()


def test_headless_cross_ask_and_suggest(tmp_path):
    make_repo(tmp_path)
    build_graph(tmp_path)
    assert cross_ask(tmp_path, "which keys link projects?")["intent"] == "shared_keys"
    assert cross_ask(tmp_path, "what joins b/b.json#contacts and a/a.json#orders?")["ok"] is True
    assert cross_suggest(tmp_path)["suggestions"]


def test_cli_graph_ask(tmp_path):
    make_repo(tmp_path)
    build_graph(tmp_path)
    r = subprocess.run([sys.executable, "-m", "dancr.cli", "graph", "ask", str(tmp_path),
                        "which keys link projects?"], capture_output=True, text=True)
    assert r.returncode == 0 and "key(s)" in r.stdout, r.stderr
    r2 = subprocess.run([sys.executable, "-m", "dancr.cli", "graph", "suggest", str(tmp_path)],
                        capture_output=True, text=True)
    assert r2.returncode == 0


def test_mcp_graph_ask_and_suggest(tmp_path, mcp_root):
    import dancr.mcp_server as srv
    pj = tmp_path / "shop.json"
    srv.create_pipeline(str(pj))
    srv.add_node(str(pj), "enter_data", {"columns": [{"name": "x", "type": "number"}], "rows": [[1]]}, node_id="d")
    srv.graph_build()
    out = json.loads(srv.graph_ask(question="find d"))
    assert out["intent"] == "find"
    assert json.loads(srv.graph_suggest())["suggestions"]
