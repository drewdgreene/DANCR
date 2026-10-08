"""Governance core (B1): policy decisions, quotas, approvals, audit — no network transport."""
import json
import subprocess
import sys

import pytest

from dancr.core.gateway import Policy, category_for, evaluate
from dancr.headless import (audit_records, decide_approval, enforce, list_approvals, load_policy, policy_check,
                            request_approval, save_policy)

POLICY = {
    "version": 1,
    "principals": {"alice": {"roles": ["analyst"]}, "bob": {}},
    "rules": [
        {"principal": "bob", "category": "export", "action": "deny", "reason": "bob may not export"},
        {"role": "analyst", "category": "export", "action": "allow"},
    ],
    "default": {"read": "allow", "run": "allow", "write_inside": "allow", "write_outside": "approve",
                "export": "approve", "enable_samples": "approve", "graph_write": "approve",
                "gateway_admin": "deny"},
    "quotas": {"alice": {"run": 2}},
}


def test_category_classification():
    assert category_for("inspect_file") == "read"
    assert category_for("run_pipeline") == "run"
    assert category_for("export_node") == "export"
    assert category_for("render_chart") == "write_outside"
    assert category_for("graph_build") == "graph_write"
    assert category_for("something_new") == "write_outside"     # cautious default


def test_no_policy_allows_everything(tmp_path):
    assert load_policy(tmp_path) is None
    assert policy_check(tmp_path, "anyone", "export_node")["verdict"] == "allow"
    assert enforce(tmp_path, "anyone", "export_node")["verdict"] == "allow"


def test_policy_denies_unknown_principal(tmp_path):
    save_policy(tmp_path, POLICY)
    out = policy_check(tmp_path, "carol", "run_pipeline")
    assert out["verdict"] == "deny" and "unknown principal" in out["reason"]


def test_rules_override_the_default(tmp_path):
    save_policy(tmp_path, POLICY)
    assert policy_check(tmp_path, "bob", "export_node")["verdict"] == "deny"
    assert policy_check(tmp_path, "alice", "export_node")["verdict"] == "allow"   # analyst rule
    assert policy_check(tmp_path, "alice", "decide_approval")["verdict"] == "deny"  # default admin


def test_quota_denies_after_the_limit(tmp_path):
    save_policy(tmp_path, POLICY)
    assert enforce(tmp_path, "alice", "run_pipeline")["verdict"] == "allow"
    assert enforce(tmp_path, "alice", "run_pipeline")["verdict"] == "allow"
    third = policy_check(tmp_path, "alice", "run_pipeline")
    assert third["verdict"] == "deny" and "quota reached" in third["reason"]


def test_quota_is_enforced_under_concurrency(tmp_path):
    save_policy(tmp_path, {"principals": {"alice": {}}, "quotas": {"alice": {"run": 1}},
                           "default": dict(POLICY["default"])})
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as pool:
        verdicts = list(pool.map(lambda _: enforce(tmp_path, "alice", "run_pipeline")["verdict"], range(4)))
    assert verdicts.count("allow") == 1          # the read-decide-count sequence must not interleave


def test_an_approved_action_counts_against_the_quota(tmp_path):
    save_policy(tmp_path, {"principals": {"alice": {}}, "quotas": {"alice": {"export": 1}},
                           "default": dict(POLICY["default"])})
    from dancr.headless import authorize
    rec = request_approval(tmp_path, "alice", "export_node", project="a.json")
    decide_approval(tmp_path, rec["id"], True, by="drew")
    first = authorize(tmp_path, "alice", "export_node", project="a.json")
    assert first["verdict"] == "allow" and first["reason"] == "approved"
    rec2 = request_approval(tmp_path, "alice", "export_node", project="a.json")
    decide_approval(tmp_path, rec2["id"], True, by="drew")
    second = authorize(tmp_path, "alice", "export_node", project="a.json")
    assert second["verdict"] == "deny" and "quota" in second["reason"]


def test_an_approval_cannot_be_decided_twice(tmp_path):
    save_policy(tmp_path, POLICY)
    rec = request_approval(tmp_path, "alice", "export_node", project="a.json")
    decide_approval(tmp_path, rec["id"], True, by="drew")
    with pytest.raises(ValueError, match="already"):
        decide_approval(tmp_path, rec["id"], False, by="drew")


def test_audit_records_are_redacted(tmp_path):
    save_policy(tmp_path, POLICY)
    enforce(tmp_path, "alice", "run_pipeline", args={"connection": "postgres://u:secret@h/db"})
    rec = audit_records(tmp_path)["records"][-1]
    assert rec["verdict"] == "allow" and "secret" not in json.dumps(rec)


def test_approval_queue_request_and_decide(tmp_path):
    save_policy(tmp_path, POLICY)
    rec = request_approval(tmp_path, "alice", "export_node", project="shop.json", args={"out": "r.csv"})
    assert rec["status"] == "pending" and rec["id"].startswith("ap_")
    assert list_approvals(tmp_path, pending_only=True)["count"] == 1
    out = decide_approval(tmp_path, rec["id"], True, by="drew")
    assert out["status"] == "approved"
    assert list_approvals(tmp_path, pending_only=True)["count"] == 0
    with pytest.raises(ValueError):
        decide_approval(tmp_path, "ap_999", True)


def test_save_policy_validates(tmp_path):
    with pytest.raises(ValueError):
        save_policy(tmp_path, {"principals": "not a mapping"})
    with pytest.raises(ValueError, match="quotas"):
        save_policy(tmp_path, {"principals": {"a": {}}, "quotas": {"a": "not a mapping"}})


def test_an_approval_is_scoped_to_its_project(tmp_path):
    save_policy(tmp_path, POLICY)
    rec = request_approval(tmp_path, "alice", "export_node", project="a.json")
    decide_approval(tmp_path, rec["id"], True, by="drew")
    from dancr.headless import consume_approval
    assert consume_approval(tmp_path, "alice", "export_node", project="b.json") is False   # wrong project
    assert consume_approval(tmp_path, "alice", "export_node", project="a.json") is True
    assert consume_approval(tmp_path, "alice", "export_node", project="a.json") is False  # used once


def test_cli_policy_approvals_audit(tmp_path):
    save_policy(tmp_path, POLICY)
    r = subprocess.run([sys.executable, "-m", "dancr.cli", "policy", "check", str(tmp_path), "bob", "export_node"],
                       capture_output=True, text=True)
    assert r.returncode == 0 and "DENY" in r.stdout
    r2 = subprocess.run([sys.executable, "-m", "dancr.cli", "audit", str(tmp_path)], capture_output=True, text=True)
    assert r2.returncode == 0
    r3 = subprocess.run([sys.executable, "-m", "dancr.cli", "approvals", str(tmp_path)], capture_output=True, text=True)
    assert r3.returncode == 0 and "approval(s)" in r3.stdout


def test_mcp_policy_check(tmp_path, mcp_root):
    import dancr.mcp_server as srv
    save_policy(tmp_path, POLICY)
    out = json.loads(srv.policy_check(principal="bob", tool="export_node"))
    assert out["verdict"] == "deny"


def test_evaluate_is_pure_and_deterministic():
    pol = Policy.from_dict(POLICY)
    a = evaluate(pol, "alice", "export_node").to_dict()
    b = evaluate(pol, "alice", "export_node").to_dict()
    assert a == b and a["verdict"] == "allow"
