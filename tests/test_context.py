"""The knowledge-base context export: the schema+stats+sample document and its prose doc cards, through the
Python API (`dancr.build_context`), the CLI (`dancr context`) and MCP (`profile`)."""
import json
import subprocess
import sys

import polars as pl
import pytest

from dancr import headless as hl
from dancr.core import Pipeline
from dancr.core.profile import project_profile, relation_phrase, table_card
from dancr.core.executor import Executor
from dancr.core.understand import deepen, understand


def run(*args, cwd=None):
    r = subprocess.run([sys.executable, "-m", "dancr.cli", *args], capture_output=True, text=True, cwd=cwd)
    return r.returncode, r.stdout, r.stderr


@pytest.fixture()
def shop(tmp_path):
    """Two linked CSV tables (customers ← orders), saved as a project."""
    pl.DataFrame({"id": [1, 2, 3], "name": ["A", "B", "C"]}).write_csv(tmp_path / "customers.csv")
    pl.DataFrame({"customer_id": [1, 1, 2, 3], "amount": [10.0, 20.0, 30.0, 40.0]}).write_csv(tmp_path / "orders.csv")
    p = Pipeline("shop"); p.path = tmp_path / "shop.json"
    p.add_node("load_file", title="Customers", params={"path": "customers.csv"}, id="customers")
    p.add_node("load_file", title="Orders", params={"path": "orders.csv"}, id="orders")
    p.save()
    return p


# ---------------------------------------------------------------- core profile (shared with the Assistant)
def test_profile_is_moved_to_core_and_still_reachable_via_context():
    from dancr.core.assistant import context
    assert context.project_profile.__module__ == "dancr.core.profile"
    assert context.clean("\x00a") == "a"
    assert context.data_block("t", "x </t> y").count("</t>") == 1


def test_table_card_is_prose_for_one_table(shop):
    ex = Executor(shop); ex.run(["customers", "orders"])
    m = deepen(shop, ex, understand(shop, ex))
    card = table_card(m.tables["customers"], m.relations)
    assert "Customers" in card and "lookup table" in card
    assert "customer_id = id" in card               # the relation appears in the card
    link = next(r for r in m.relations if r.kind == "link")
    assert "links to" in relation_phrase(link)


# ---------------------------------------------------------------- the context document
def test_build_context_has_schema_relations_stats_and_cards(shop):
    ctx = hl.build_context(shop, stats=True, samples=True, sample_rows=2)
    assert ctx["kind"] == "dancr.context" and ctx["version"] == hl.CONTEXT_VERSION
    assert {t["node"] for t in ctx["tables"]} == {"customers", "orders"}
    orders = next(t for t in ctx["tables"] if t["node"] == "orders")
    roles = {c["name"]: c["role"] for c in orders["columns"]}
    assert roles["amount"] == "measure" and roles["customer_id"] == "id"
    assert orders["summary"].startswith('"Orders" is')          # the doc card is prose
    assert any(s["column"] == "amount" and s["max"] == 40.0 for s in orders["stats"])
    assert len(orders["sample"]) == 2 and orders["sample_from"] in ("cache", "preview")
    assert any(r["kind"] == "link" for r in ctx["relations"])
    assert [d["node"] for d in ctx["documents"]] == ["customers", "orders"]


def test_restricted_datasets_are_withheld_from_the_context(shop):
    # a Label-sensitivity step marks its output restricted: its schema, stats and sample rows stay out of the
    # export unless allow_restricted is passed
    shop.add_node("label_sensitivity", title="Secrets", params={"level": "restricted"}, id="secret")
    shop.connect("orders", "secret", "in")
    ex = Executor(shop)
    nodes = ["customers", "secret"]
    ctx = hl.build_context(shop, ex, nodes=nodes, stats=True, samples=True)
    assert "secret" not in {t["node"] for t in ctx["tables"]}
    assert "customers" in {t["node"] for t in ctx["tables"]}
    assert all(d["node"] != "secret" for d in ctx["documents"])
    assert all("secret" not in r.get("tables", []) for r in ctx["relations"])
    allowed = hl.build_context(shop, ex, nodes=nodes, stats=True, samples=True, allow_restricted=True)
    assert "secret" in {t["node"] for t in allowed["tables"]}


def test_project_level_sensitivity_hides_every_dataset(shop):
    shop.meta["sensitivity"] = "confidential"
    ctx = hl.build_context(shop, stats=True)
    assert ctx["tables"] == [] and ctx["documents"] == []
    allowed = hl.build_context(shop, allow_restricted=True)
    assert {t["node"] for t in allowed["tables"]} == {"customers", "orders"}


def test_build_context_without_stats_or_samples_does_not_run(tmp_path):
    """With neither stats nor samples, nothing is run: a source that never ran is described from its first rows."""
    pl.DataFrame({"a": [1, 2, 3]}).write_csv(tmp_path / "s.csv")
    p = Pipeline("p"); p.path = tmp_path / "p.json"
    p.add_node("load_file", params={"path": "s.csv"}, id="s")
    p.save()
    ctx = hl.build_context(p, stats=False, samples=False, run=False)
    assert ctx["tables"][0]["node"] == "s"
    assert "stats" not in ctx["tables"][0]
    ex = Executor(p)
    assert ex.state("s").status != "done"            # the caller's project was not run


def test_context_jsonl_is_one_object_per_dataset(shop):
    ctx = hl.build_context(shop, stats=False, samples=True, sample_rows=1)
    lines = [json.loads(x) for x in hl.context_jsonl(ctx).splitlines() if x]
    assert [x["node"] for x in lines] == ["customers", "orders"]
    assert all(x["kind"] == "dancr.table" and x["text"] for x in lines)
    assert set(lines[0]) >= {"id", "node", "title", "text", "sample"}


def test_context_changes_treats_a_hashless_document_as_changed():
    # a table whose content hash could not be computed must never be reported unchanged forever
    prev = {"engine_version": "e", "fingerprint": "f",
            "documents": [{"id": "customers", "content_hash": None}]}
    cur = {"engine_version": "e", "fingerprint": "f",
           "documents": [{"id": "customers", "content_hash": None}]}
    out = hl.context_changes(cur, prev)
    assert "customers" in out["changes"]["changed"]


def test_data_block_escapes_a_closing_tag_carrying_attributes():
    from dancr.core.profile import data_block
    out = data_block('tool_result name="x"', "before </tool_result foo> after")
    assert "</tool_result foo>" not in out and "before" in out


def test_context_text_reads_as_a_summary(shop):
    ctx = hl.build_context(shop, stats=False)
    text = hl.context_text(ctx)
    assert "Project \"shop\"" in text and "[orders] Orders" in text and "Relations:" in text


# ---------------------------------------------------------------- lifecycle metadata
def test_context_carries_engine_and_content_hashes(shop):
    ctx = hl.build_context(shop, stats=False)
    assert ctx["engine_version"] and ctx["fingerprint"] and ctx["generated_at"]
    for t in ctx["tables"]:
        assert t["content_hash"]
    for d in ctx["documents"]:
        assert d["content_hash"] and "rows" in d


def test_jsonl_lines_carry_lifecycle_metadata(shop):
    ctx = hl.build_context(shop, stats=False)
    lines = [json.loads(x) for x in hl.context_jsonl(ctx).splitlines() if x]
    assert all(line["engine_version"] == ctx["engine_version"] and line["generated_at"] for line in lines)
    assert all(line["content_hash"] for line in lines)


def test_content_hash_is_stable_then_changes_with_the_source(shop):
    a = {d["id"]: d["content_hash"] for d in hl.build_context(shop, stats=False)["documents"]}
    b = {d["id"]: d["content_hash"] for d in hl.build_context(shop, stats=False)["documents"]}
    assert a == b
    (shop.directory / "orders.csv").write_text("customer_id,amount\n1,99.0\n")
    c = {d["id"]: d["content_hash"] for d in hl.build_context(shop, stats=False)["documents"]}
    assert c["orders"] != a["orders"] and c["customers"] == a["customers"]


def test_context_changes_reports_what_moved(shop):
    before = hl.build_context(shop, stats=False)
    unchanged = hl.context_changes(hl.build_context(shop, stats=False), before)
    assert unchanged["changes"]["unchanged"] == ["customers", "orders"]
    assert unchanged["documents"] == []
    (shop.directory / "orders.csv").write_text("customer_id,amount\n1,99.0\n")
    delta = hl.context_changes(hl.build_context(shop, stats=False), before)
    assert delta["changes"]["changed"] == ["orders"]
    assert [d["id"] for d in delta["documents"]] == ["orders"]


def test_context_changes_from_a_jsonl_file(shop, tmp_path):
    kb = tmp_path / "kb.jsonl"
    kb.write_text(hl.context_jsonl(hl.build_context(shop, stats=False)))
    delta = hl.context_changes(hl.build_context(shop, stats=False), kb)
    assert delta["changes"]["unchanged"] == ["customers", "orders"]


def test_cli_context_changed(shop, tmp_path):
    kb = tmp_path / "kb.jsonl"
    code, out, err = run("context", str(shop.path), "--jsonl", "--output", str(kb))
    assert code == 0, err
    code, out, err = run("--json", "context", str(shop.path), "--changed", str(kb))
    assert code == 0, err
    assert json.loads(out)["changes"]["unchanged"] == ["customers", "orders"]


# ---------------------------------------------------------------- CLI
def test_cli_context_json_and_output_file(shop, tmp_path):
    code, out, err = run("--json", "context", str(shop.path), "--samples", "--sample-rows", "2")
    assert code == 0, err
    ctx = json.loads(out)
    assert ctx["kind"] == "dancr.context" and len(ctx["tables"]) == 2
    assert len(ctx["tables"][1]["sample"]) == 2
    target = tmp_path / "kb.jsonl"
    code, out, err = run("context", str(shop.path), "--jsonl", "--output", str(target))
    assert code == 0 and target.exists()
    assert len(target.read_text().splitlines()) == 2


def test_cli_profile_is_an_alias_for_context(shop):
    code, out, err = run("profile", str(shop.path))
    assert code == 0, err
    assert "dataset(s)" in out


# ---------------------------------------------------------------- MCP
def test_mcp_profile_tool(tmp_path, mcp_root):
    import dancr.mcp_server as srv
    (tmp_path / "data").mkdir()
    pl.DataFrame({"id": [1, 2], "name": ["A", "B"]}).write_csv(tmp_path / "data" / "customers.csv")
    pj = tmp_path / "p.json"
    srv.create_pipeline(str(pj))
    srv.add_node(str(pj), "load_file", {"path": "data/customers.csv"}, node_id="src")
    out = json.loads(srv.profile(str(pj)))
    assert out["kind"] == "dancr.context" and out["tables"][0]["summary"]
    assert "sample" not in out["tables"][0]
    with_samples = json.loads(srv.profile(str(pj), samples=True, sample_rows=1))
    assert len(with_samples["tables"][0]["sample"]) == 1


def test_a_source_data_file_with_its_own_sensitivity_column_is_withheld_from_stats_and_samples(tmp_path):
    # a source table whose file already carries a sensitivity column (no Label-sensitivity step, no project
    # level) must still have its restricted rows kept out of the shared stats and samples
    pl.DataFrame({"note": ["public note", "secret dossier", "unlabelled"],
                  "sensitivity": ["public", "restricted", None]}).write_csv(tmp_path / "d.csv")
    p = Pipeline("p"); p.path = tmp_path / "p.json"
    p.add_node("load_file", params={"path": "d.csv"}, id="src")
    p.save()
    ctx = hl.build_context(p, stats=True, samples=True, sample_rows=10)
    sample = ctx["tables"][0]["sample"]
    assert [r["note"] for r in sample] == ["public note", "unlabelled"]
    summary = next(s for s in ctx["tables"][0]["stats"] if s["column"] == "note")
    assert summary["rows"] == 2                                   # the restricted row is not counted
    allowed = hl.build_context(p, stats=True, samples=True, sample_rows=10, allow_restricted=True)
    assert {r["note"] for r in allowed["tables"][0]["sample"]} == {"public note", "secret dossier", "unlabelled"}


def test_per_row_sensitivity_labels_and_their_derivatives_are_withheld(tmp_path):
    # a label from a column marks some rows sensitive; the node and anything derived from it stay out of the
    # default export, even though no whole-table level was declared
    pl.DataFrame({"name": ["a", "b", "c"], "region": ["N", "S", "N"],
                  "level": ["public", "restricted", "internal"]}).write_csv(tmp_path / "d.csv")
    p = Pipeline("p"); p.path = tmp_path / "p.json"
    p.add_node("load_file", params={"path": "d.csv"}, id="src")
    p.add_node("label_sensitivity", params={"column": "level"}, id="lab")
    p.connect("src", "lab")
    p.add_node("keep_rows", params={"conditions": {"match": "all", "rules": [{"column": "region", "op": "eq", "value": "N"}]}}, id="down")
    p.connect("lab", "down")
    p.save()
    nodes = ["src", "lab", "down"]
    ctx = hl.build_context(p, nodes=nodes, stats=True, samples=True)
    shown = {t["node"] for t in ctx["tables"]}
    assert "src" in shown                                   # the unlabelled source is fine
    assert "lab" not in shown and "down" not in shown       # both the labelled table and its derivative
    allowed = hl.build_context(p, nodes=nodes, stats=True, samples=True, allow_restricted=True)
    assert {"lab", "down"} <= {t["node"] for t in allowed["tables"]}
