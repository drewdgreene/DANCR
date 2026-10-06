"""The governed-agent core (roadmap B1): principals, policy, approvals, audit and quotas.

This is the *decision* layer, not a network service. It answers one question —
*may this principal take this action on this project?* — with ``allow``,
``approve`` or ``deny``, and records every decision in an append-only audit log.
It has no dependency beyond the standard library and opens no socket: the
transport is a separate, explicit decision (``docs/adr/0006-gateway-transport.md``).

Without a policy file nothing is enforced, so the existing CLI/MCP/window
behaviour is unchanged. A repository opts in by writing
``<root>/.dancr/gateway/policy.json``.
"""
from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from typing import Any

GATEWAY_VERSION = 1
ACTIONS = ("allow", "approve", "deny")
CATEGORIES = ("read", "run", "write_inside", "write_outside", "export", "enable_samples",
              "graph_write", "gateway_admin")

# Which category each known tool/action belongs to. An unknown tool is treated as the most cautious
# category (write_outside), so a new tool is safe by default. Deterministic and explicit.
_READ = {"inspect_file", "get_schema", "get_sample", "get_stats", "describe_pipeline", "node_status",
         "list_node_types", "formula_reference", "search_knowledge", "get_events", "graph_query",
         "graph_neighbors", "graph_path", "graph_shared_keys", "get_trace", "catalog", "understand_data",
         "profile", "suggest_answers", "connections", "policy_check", "list_approvals", "get_audit",
         "read_document", "lineage", "get_proof", "formula_reference"}
_RUN = {"run_pipeline", "run_batch", "ask", "assistant", "run_eval", "verify_pipeline", "record_attestation"}
_WRITE_INSIDE = {"create_pipeline", "add_node", "set_params", "connect_nodes", "disconnect_nodes",
                 "remove_node", "rename_node", "set_input", "remove_input", "set_column_label",
                 "set_dataset_meta", "apply_edits", "build_template"}
_WRITE_OUTSIDE = {"render_chart", "render_map", "open_in_gui"}
_EXPORT = {"export_node", "export_fair", "package_project"}
_GRAPH_WRITE = {"graph_build"}
_ADMIN = {"decide_approval", "gateway_admin"}


def category_for(tool: str) -> str:
    """The action category a tool belongs to. Unknown tools are ``write_outside`` (the cautious default)."""
    t = str(tool or "")
    for names, cat in ((_READ, "read"), (_RUN, "run"), (_WRITE_INSIDE, "write_inside"), (_EXPORT, "export"),
                       (_WRITE_OUTSIDE, "write_outside"), (_GRAPH_WRITE, "graph_write"), (_ADMIN, "gateway_admin")):
        if t in names:
            return cat
    return "write_outside"


@dataclass
class Principal:
    id: str
    roles: tuple[str, ...] = ()
    projects: tuple[str, ...] = ()          # fnmatch patterns this principal may touch ("*" = any)

    def may_touch(self, project: str | None) -> bool:
        if not self.projects:
            return True
        if project is None:
            return True
        return any(fnmatch.fnmatch(project, pat) for pat in self.projects)


@dataclass
class Decision:
    verdict: str                            # allow | approve | deny
    category: str
    principal: str
    tool: str
    reason: str
    rule: int | None = None                 # index of the matched rule, or None for the default

    @property
    def allowed(self) -> bool:
        return self.verdict == "allow"

    @property
    def needs_approval(self) -> bool:
        return self.verdict == "approve"

    def to_dict(self) -> dict[str, Any]:
        return {"verdict": self.verdict, "category": self.category, "principal": self.principal,
                "tool": self.tool, "reason": self.reason, "rule": self.rule}


DEFAULT_POLICY = {"read": "allow", "run": "allow", "write_inside": "allow", "write_outside": "approve",
                  "export": "approve", "enable_samples": "approve", "graph_write": "approve",
                  "gateway_admin": "deny"}


@dataclass
class Policy:
    principals: dict[str, Principal] = field(default_factory=dict)
    rules: list[dict[str, Any]] = field(default_factory=list)
    default: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_POLICY))
    quotas: dict[str, dict[str, int]] = field(default_factory=dict)
    source: str = ""

    @classmethod
    def from_dict(cls, data: Any, source: str = "") -> "Policy":
        if not isinstance(data, dict):
            raise ValueError("A policy is a JSON object")
        raw_principals = data.get("principals") or {}
        raw_default = data.get("default") or {}
        raw_quotas = data.get("quotas") or {}
        raw_rules = data.get("rules") or []
        if not isinstance(raw_principals, dict):
            raise ValueError("policy 'principals' must be an object of principal -> settings")
        if not isinstance(raw_default, dict):
            raise ValueError("policy 'default' must be an object of category -> action")
        if not isinstance(raw_quotas, dict):
            raise ValueError("policy 'quotas' must be an object of principal -> limits")
        if not isinstance(raw_rules, list):
            raise ValueError("policy 'rules' must be a list")
        principals: dict[str, Principal] = {}
        for pid, spec in raw_principals.items():
            spec = spec if isinstance(spec, dict) else {}
            principals[str(pid)] = Principal(str(pid), tuple(str(r) for r in (spec.get("roles") or [])),
                                             tuple(str(p) for p in (spec.get("projects") or [])))
        default = dict(DEFAULT_POLICY)
        for k, v in raw_default.items():
            if k in CATEGORIES and v in ACTIONS:
                default[k] = v
        quotas = {str(k): {str(a): int(n) for a, n in (v or {}).items()} for k, v in raw_quotas.items()}
        rules = [r for r in raw_rules if isinstance(r, dict)]
        return cls(principals=principals, rules=rules, default=default, quotas=quotas, source=source)

    def resolve(self, principal: str) -> Principal | None:
        """The principal entry for an id, or a role-less wildcard entry, or None when unknown."""
        if principal in self.principals:
            return self.principals[principal]
        if "*" in self.principals:
            return self.principals["*"]
        return None


def _rule_matches(rule: dict[str, Any], principal: str, roles: tuple[str, ...], category: str, tool: str) -> bool:
    """A rule matches when its principal (id, role or ``*``) and its category or tool match. A missing field is
    a wildcard."""
    who = rule.get("principal")
    if who not in (None, "*", principal) and who not in roles:
        return False
    role = rule.get("role")
    if role not in (None, "*") and role not in roles:
        return False
    if (cat := rule.get("category")) is not None and cat != category:
        return False
    if (t := rule.get("tool")) is not None and t != tool:
        return False
    return True


def evaluate(policy: Policy, principal: str, tool: str, *, category: str | None = None,
             project: str | None = None) -> Decision:
    """Decide whether ``principal`` may use ``tool`` (of ``category``, on ``project``). The first matching rule
    wins; otherwise the category's default applies. An unknown principal is denied unless the policy has a
    wildcard entry."""
    cat = category or category_for(tool)
    who = policy.resolve(principal)
    if who is None:
        return Decision("deny", cat, principal, tool, f"unknown principal {principal!r}", None)
    if not who.may_touch(project):
        return Decision("deny", cat, principal, tool, f"{principal!r} may not touch {project}", None)
    for i, rule in enumerate(policy.rules):
        if _rule_matches(rule, principal, who.roles, cat, tool):
            action = str(rule.get("action") or "deny")
            if action not in ACTIONS:
                return Decision("deny", cat, principal, tool, f"rule {i} has an unknown action {action!r}", i)
            return Decision(action, cat, principal, tool, str(rule.get("reason") or f"matched rule {i}"), i)
    return Decision(policy.default.get(cat, "deny"), cat, principal, tool, f"default for {cat}", None)


def redact_args(args: dict[str, Any] | None) -> dict[str, Any]:
    """Settings safe to log: anything that looks like a credential is blanked (reuses the project's own
    redaction so a token never reaches the audit log)."""
    from ..secrets import _redact_value
    return {str(k): _redact_value(v) for k, v in (args or {}).items()}
