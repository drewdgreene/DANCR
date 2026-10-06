# Governance (B1): policy, approvals, audit, quotas

This is the **decision layer** for agents: principals, a policy file, an approval
queue, an append-only audit log and quotas. It answers one question — *may this
principal take this action on this project?* — with `allow`, `approve` or `deny`.

It opens **no network socket**. The multi-client HTTP transport is a separate,
deliberately deferred decision with production security consequences; see
`docs/adr/0006-gateway-transport.md` for the four questions that must be answered
first. Until then, the policy/audit machinery can gate the existing local surfaces
(CLI, MCP over stdio, the SDK) and be used by an operator.

## Opt-in, and off by default

Nothing is enforced until a repository writes
`<root>/.dancr/gateway/policy.json`. Without it, `policy_check` and `enforce`
return `allow` with the reason *"no policy configured"*, so existing behaviour is
unchanged.

## Categories

Every tool maps to one **action category** (`dancr.core.gateway.category_for`):

| category | examples |
|---|---|
| `read` | `inspect_file`, `get_sample`, `graph_query`, `search_knowledge`, `get_events`, `policy_check` |
| `run` | `run_pipeline`, `run_batch`, `ask`, `assistant`, `verify_pipeline` |
| `write_inside` | `create_pipeline`, `add_node`, `set_params`, `set_input` |
| `export` | `export_node`, `export_fair`, `package_project` |
| `write_outside` | `render_chart`, `render_map`, `open_in_gui` |
| `enable_samples` | (sending sample rows to a model) |
| `graph_write` | `graph_build` |
| `gateway_admin` | `decide_approval` |

An unknown tool is classified `write_outside` — the cautious default — so a new
tool is safe until it is classified.

## The policy file

```json
{
  "version": 1,
  "principals": {
    "alice": {"roles": ["analyst"], "projects": ["data/*"]},
    "bob": {}
  },
  "rules": [
    {"principal": "bob", "category": "export", "action": "deny", "reason": "bob may not export"},
    {"role": "analyst", "category": "export", "action": "allow"}
  ],
  "default": {"read": "allow", "run": "allow", "write_inside": "allow",
              "export": "approve", "write_outside": "approve",
              "enable_samples": "approve", "graph_write": "approve", "gateway_admin": "deny"},
  "quotas": {"alice": {"run": 100}}
}
```

- A principal may be named by **id**, by **role**, or `*`. An unknown principal is
  denied unless there is a `*` entry.
- `projects` are fnmatch patterns a principal may touch; a missing list means any.
- The **first matching rule** wins; a rule may match on `principal`, `role`,
  `category` and/or `tool` (missing fields are wildcards). Otherwise the
  category's **default** applies.
- A quota counts **allowed** actions per principal and category; once reached, the
  action is denied with the used/limit in the reason.

## Using it

```bash
dancr policy show .                          # is a policy configured?
dancr policy check . alice export_node       # allow / approve / deny, with the reason
dancr approvals .                            # queued consequential actions
dancr approvals . --approve ap_1 --by drew
dancr audit . --limit 20                     # every enforced action, with its verdict
```

The MCP server exposes one read-only tool, `policy_check`. Enforcement
(`dancr.enforce`) is available to the SDK and to any surface that opts in; the
existing MCP tool contracts are unchanged.

- `enforce(root, principal, tool, …)` decides **and** records the decision in
  `<root>/.dancr/audit/audit.jsonl` (params redacted with `core.secrets`), and
  counts an `allow` against the quota.
- `request_approval` / `list_approvals` / `decide_approval` manage the queue in
  `<root>/.dancr/gateway/approvals.jsonl`; a person's decision is itself audited.
- Quotas live in `<root>/.dancr/gateway/quotas.json`.

## Audit and approvals format

The audit log is one JSON object per line: `principal`, `tool`, `category`,
`verdict`, `project`, `args` (redacted) and an optional `result_hash`. The
approvals log is append-only: a `requested` record, then a `decided` record with
the same id; `list_approvals` folds them.

## Honest limits

- **No transport.** There is no listening socket; a single-client stdio MCP server
  and the CLI remain the surfaces. The gateway transport awaits the ADR-0006
  decision.
- Enforcement is **opt-in and manual**: existing tools are not wrapped by default,
  so a caller must call `enforce`/`policy_check` to be governed. Wiring it into
  every tool is the job of the transport phase.
- The policy language is deliberately tiny (principals × categories/rules ×
  approvals × quotas). Anything larger becomes an enterprise product; extend only
  on demand.
