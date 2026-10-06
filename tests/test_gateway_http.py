"""The gateway transport (B1): loopback-only, token-authenticated, policy-gated, audited."""
import contextlib
import socket
import threading
import time

import httpx
import pytest
import uvicorn

from dancr.gateway import build_app, is_loopback
from dancr.gateway import main as gateway_main
from dancr.headless import audit_records, save_policy

POLICY = {
    "principals": {"alice": {"tokens": ["alice-token"]}, "bob": {"tokens": ["bob-token"]}},
    "rules": [{"principal": "bob", "category": "read", "action": "deny"}],
    "default": {"read": "allow", "run": "allow", "write_inside": "allow", "write_outside": "approve",
                "export": "approve", "enable_samples": "approve", "graph_write": "approve", "gateway_admin": "deny"},
}


@pytest.fixture(autouse=True)
def _restore_mcp_root():
    """configure_root mutates mcp_server's globals; put them back after each gateway test."""
    import dancr.mcp_server as srv
    root, refused = srv.ROOT, srv.ROOT_REFUSED
    yield
    srv.ROOT, srv.ROOT_REFUSED = root, refused


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@contextlib.contextmanager
def running(app, host: str, port: int):
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning", lifespan="on"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    for _ in range(200):
        if server.started:
            break
        time.sleep(0.05)
    assert server.started, "gateway did not start"
    try:
        yield f"http://{host}:{port}"
    finally:
        server.should_exit = True
        t.join(10)


def call(base: str, token: str | None = None, tool: str = "inspect_file", args: dict | None = None) -> httpx.Response:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": tool, "arguments": args or {}}}
    with httpx.Client(trust_env=False, timeout=10) as c:
        return c.post(base + "/mcp", json=body, headers=headers)


def test_is_loopback():
    assert is_loopback("127.0.0.1") and is_loopback("localhost") and is_loopback("::1")
    assert not is_loopback("0.0.0.0") and not is_loopback("10.0.0.1")


def test_gateway_fails_closed_without_a_policy(tmp_path):
    with pytest.raises(ValueError, match="policy"):
        build_app(tmp_path)


def test_gateway_fails_closed_without_a_token(tmp_path):
    save_policy(tmp_path, {"principals": {"alice": {"roles": ["analyst"]}}})
    with pytest.raises(ValueError, match="token"):
        build_app(tmp_path)


def test_gateway_refuses_a_non_loopback_bind(tmp_path):
    save_policy(tmp_path, POLICY)
    with pytest.raises(ValueError, match="loopback"):
        gateway_main(tmp_path, host="0.0.0.0", port=free_port())


def test_gateway_requires_a_valid_token(tmp_path):
    save_policy(tmp_path, POLICY)
    port = free_port()
    with running(build_app(tmp_path, port=port), "127.0.0.1", port) as base:
        assert call(base).status_code == 401
        assert call(base, token="wrong").status_code == 401


def test_gateway_policy_denies_and_allows_and_audits(tmp_path):
    save_policy(tmp_path, POLICY)
    port = free_port()
    with running(build_app(tmp_path, port=port), "127.0.0.1", port) as base:
        denied = call(base, token="bob-token")
        assert denied.status_code == 403 and "denied" in denied.text
        allowed = call(base, token="alice-token")
        assert allowed.status_code not in (401, 403)          # passed the gate; the MCP layer answers next
    verdicts = [r["verdict"] for r in audit_records(tmp_path)["records"]]
    assert "deny" in verdicts and "allow" in verdicts
    assert {r["principal"] for r in audit_records(tmp_path)["records"]} == {"alice", "bob"}


def test_gateway_queues_an_approval_for_a_consequential_action(tmp_path):
    save_policy(tmp_path, POLICY)
    port = free_port()
    with running(build_app(tmp_path, port=port), "127.0.0.1", port) as base:
        # export_node is in the 'approve' default: queued, not run
        out = call(base, token="alice-token", tool="export_node", args={"path": "p.json", "node_id": "n", "out_path": "r.csv"})
        assert out.status_code == 403 and "approval_required" in out.text
    from dancr.headless import list_approvals
    pending = list_approvals(tmp_path, pending_only=True)
    assert pending["count"] == 1 and pending["approvals"][0]["tool"] == "export_node"
