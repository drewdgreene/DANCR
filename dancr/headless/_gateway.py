"""Policy enforcement, approvals, audit and quotas for a repository (roadmap B1, core only).

A repository opts into governance by writing ``<root>/.dancr/gateway/policy.json``.
When it is absent, :func:`enforce` and :func:`policy_check` allow every action and
say so — so the existing CLI, MCP server and window are unchanged until a policy is
configured. No listening socket is opened here (docs/adr/0006-gateway-transport.md).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.gateway import Decision, Policy, category_for, evaluate, principal_for_token, redact_args
from ..core.repo import Repo, append_jsonl, read_json, read_jsonl, write_json_atomic


def gateway_dir(root: Path | str) -> Path:
    return Repo(root).path("gateway")


def policy_path(root: Path | str) -> Path:
    return gateway_dir(root) / "policy.json"


def load_policy(root: Path | str) -> Policy | None:
    """The repository's policy, or None when none is configured (nothing is enforced)."""
    p = policy_path(root)
    data = read_json(p)
    return Policy.from_dict(data, source=str(p)) if data else None


def save_policy(root: Path | str, data: dict[str, Any]) -> Path:
    """Validate and write a policy atomically (used by tests and by `dancr policy`)."""
    Policy.from_dict(data)                       # raises on a malformed policy
    return write_json_atomic(policy_path(root), data)


# ----------------------------------------------------------------- audit
def audit(root: Path | str, record: dict[str, Any]) -> None:
    append_jsonl(Repo(root).ensure().audit_log(), {"kind": "dancr.audit", "version": 1, **record})


def audit_records(root: Path | str, *, limit: int | None = None) -> dict[str, Any]:
    records = list(read_jsonl(Repo(root).audit_log()))
    if limit is not None and limit >= 0:
        records = records[-limit:] if limit > 0 else []
    return {"kind": "dancr.audit", "root": str(Path(root).expanduser().resolve()),
            "count": len(records), "records": records}


# ----------------------------------------------------------------- quotas
def _quota_file(root: Path | str) -> Path:
    return gateway_dir(root) / "quotas.json"


def quota_used(root: Path | str, principal: str, kind: str) -> int:
    data = read_json(_quota_file(root)) or {}
    return int((data.get(f"{principal}\x00{kind}") if isinstance(data, dict) else 0) or 0)


def _quota_bump(root: Path | str, principal: str, kind: str) -> None:
    from . import repo_lock
    with repo_lock(root):
        data = read_json(_quota_file(root)) or {}
        if not isinstance(data, dict):
            data = {}
        key = f"{principal}\x00{kind}"
        data[key] = int(data.get(key, 0)) + 1
        write_json_atomic(_quota_file(root), data)


def _quota_over(policy: Policy, used: int, principal: str, kind: str) -> str | None:
    limit = (policy.quotas.get(principal) or {}).get(kind)
    if limit is not None and used >= int(limit):
        return f"quota reached: {principal} has used {used}/{limit} {kind} action(s)"
    return None


# ----------------------------------------------------------------- enforcement
def policy_check(root: Path | str, principal: str, tool: str, *, category: str | None = None,
                 project: str | None = None) -> dict[str, Any]:
    """What the policy says about an action, without doing it or recording anything."""
    policy = load_policy(root)
    if policy is None:
        d = Decision("allow", category or category_for(tool), principal, tool, "no policy configured")
        return {"kind": "dancr.policy", "configured": False, **d.to_dict()}
    d = evaluate(policy, principal, tool, category=category, project=project)
    over = _quota_over(policy, quota_used(root, principal, d.category), principal, d.category)
    if over and d.verdict == "allow":
        d = Decision("deny", d.category, principal, tool, over)
    return {"kind": "dancr.policy", "configured": True, **d.to_dict()}


def enforce(root: Path | str, principal: str, tool: str, *, category: str | None = None,
            project: str | None = None, args: dict[str, Any] | None = None,
            result_hash: str | None = None) -> dict[str, Any]:
    """Decide an action and record it in the audit log. ``allow`` also counts against the principal's quota;
    ``approve`` means the caller must obtain an approval first (see :func:`request_approval`)."""
    from . import repo_lock
    # Read the quota, decide, count and audit under one lock: otherwise two calls both read used=limit-1, both
    # pass the check, and both count, overspending the quota.
    with repo_lock(root):
        policy = load_policy(root)
        if policy is None:
            dec = Decision("allow", category or category_for(tool), principal, tool, "no policy configured")
        else:
            dec = evaluate(policy, principal, tool, category=category, project=project)
            # Quota applies to every verdict, so an action needing approval cannot be an unlimited bypass either.
            over = _quota_over(policy, quota_used(root, principal, dec.category), principal, dec.category)
            if over:
                dec = Decision("deny", dec.category, principal, tool, over)
        audit(root, {"principal": principal, "tool": tool, "category": dec.category, "verdict": dec.verdict,
                     "project": project, "args": redact_args(args), "result_hash": result_hash})
        if dec.verdict == "allow":
            _quota_bump(root, principal, dec.category)
    return dec.to_dict()


# ----------------------------------------------------------------- approvals
def _approvals_file(root: Path | str) -> Path:
    return gateway_dir(root) / "approvals.jsonl"


def _next_approval_id(root: Path | str) -> str:
    seen = {str(r.get("id")) for r in read_jsonl(_approvals_file(root))}
    n = 1
    while f"ap_{n}" in seen:
        n += 1
    return f"ap_{n}"


def request_approval(root: Path | str, principal: str, tool: str, *, category: str | None = None,
                     project: str | None = None, args: dict[str, Any] | None = None) -> dict[str, Any]:
    """Queue a consequential action for a person to approve. Returns the pending record."""
    from . import repo_lock
    with repo_lock(root):
        rec = {"id": _next_approval_id(root), "event": "requested", "status": "pending", "principal": principal,
               "tool": tool, "category": category or category_for(tool), "project": project,
               "args": redact_args(args)}
        append_jsonl(_approvals_file(root), rec)
    audit(root, {"principal": principal, "tool": tool, "category": rec["category"], "verdict": "approve-requested",
                 "project": project, "args": redact_args(args), "approval": rec["id"]})
    return rec


def list_approvals(root: Path | str, *, pending_only: bool = False) -> dict[str, Any]:
    """Every approval request with its current status, oldest first. A later 'decided' record sets the status."""
    items: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for r in read_jsonl(_approvals_file(root)):
        aid = str(r.get("id"))
        if aid not in items:
            items[aid] = dict(r)
            order.append(aid)
        if r.get("event") == "decided":
            items[aid]["status"] = r.get("status", "pending")
            items[aid]["decided_by"] = r.get("by", "")
    out = [items[a] for a in order if not pending_only or items[a].get("status") == "pending"]
    return {"kind": "dancr.approvals", "root": str(Path(root).expanduser().resolve()),
            "count": len(out), "approvals": out}


def decide_approval(root: Path | str, approval_id: str, approve: bool, *, by: str = "") -> dict[str, Any]:
    """Approve or deny a queued action. A person's decision, recorded and audited."""
    from . import repo_lock
    with repo_lock(root):
        current = list_approvals(root)["approvals"]
        item = next((a for a in current if a["id"] == approval_id), None)
        if item is None:
            raise ValueError(f"No approval {approval_id!r}")
        if item.get("status") != "pending":
            raise ValueError(f"Approval {approval_id!r} was already {item.get('status')}")
        rec = {"id": approval_id, "event": "decided", "status": "approved" if approve else "denied", "by": by}
        append_jsonl(_approvals_file(root), rec)
    audit(root, {"principal": by or "approver", "tool": "decide_approval", "category": "gateway_admin",
                 "verdict": "approved" if approve else "denied", "approval": approval_id})
    return {"kind": "dancr.approvals", "id": approval_id, "status": rec["status"], "by": by}


def principal_for_request(root: Path | str, token: str) -> tuple[str | None, Policy | None]:
    """The principal a bearer token authenticates (constant-time), and the policy, for the gateway."""
    policy = load_policy(root)
    if policy is None:
        return None, None
    return principal_for_token(policy, token), policy


def consume_approval(root: Path | str, principal: str, tool: str, *, project: str | None = None) -> bool:
    """Take a matching approved-but-unused approval for (principal, tool, project), marking it consumed. True if
    one was found. Matching the project too means an approval for one project never authorises another; the read
    and the consume happen under the repository lock, so two callers cannot spend the same approval."""
    from . import repo_lock
    with repo_lock(root):
        records = list(read_jsonl(_approvals_file(root)))
        consumed = {r.get("id") for r in records if r.get("event") == "consumed"}
        items: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for r in records:
            aid = str(r.get("id"))
            if aid not in items:
                items[aid] = dict(r)
                order.append(aid)
            if r.get("event") == "decided":
                items[aid]["status"] = r.get("status", "pending")
        match = next((items[a] for a in order if items[a].get("status") == "approved" and a not in consumed
                      and items[a].get("principal") == principal and items[a].get("tool") == tool
                      and items[a].get("project") == project), None)
        if match is None:
            return False
        append_jsonl(_approvals_file(root), {"id": match["id"], "event": "consumed"})
    return True


def authorize(root: Path | str, principal: str, tool: str, *, category: str | None = None,
              project: str | None = None, args: dict[str, Any] | None = None) -> dict[str, Any]:
    """The gateway's per-call decision: enforce the policy (audited, quota-counted); an ``approve`` verdict is
    satisfied by a person's earlier approval, else queued and reported as ``approve``."""
    policy = load_policy(root)
    if policy is None:
        return {"verdict": "allow", "category": category or category_for(tool), "principal": principal,
                "tool": tool, "reason": "no policy configured", "rule": None}
    dec = enforce(root, principal, tool, category=category, project=project, args=args)
    if dec["verdict"] != "approve":
        return dec
    if consume_approval(root, principal, tool, project=project):
        # an approved action still counts against the principal's quota, or approvals would be an unlimited bypass
        _quota_bump(root, principal, dec["category"])
        audit(root, {"principal": principal, "tool": tool, "category": dec["category"],
                     "verdict": "allowed-after-approval", "project": project})
        return {**dec, "verdict": "allow", "reason": "approved"}
    rec = request_approval(root, principal, tool, category=dec["category"], project=project, args=args)
    return {**dec, "verdict": "approve", "approval": rec["id"], "reason": f"queued for approval ({rec['id']})"}
