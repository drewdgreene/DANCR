# Governance (B1): policy, approvals, audit, quotas

This is the **decision layer** for agents: principals, a policy file, an approval
queue, an append-only audit log and quotas. It answers one question — *may this
principal take this action on this project?* — with `allow`, `approve` or `deny`.

It also has one **network transport**, `dancr gateway`, and it is opt-in and
fail-closed: loopback only by default, bearer-token authenticated, policy-gated
and audited. See the transport section below and
`docs/adr/0006-gateway-transport.md`.

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
    "alice": {"roles": ["analyst"], "projects": ["data/*"], "tokens": ["alice-secret-token"]},
    "bob": {"tokens": ["bob-secret-token"]}
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

## The transport (`dancr gateway`)

```bash
# policy.json: each principal needs a token, or the gateway refuses to start
dancr gateway .                 # http://127.0.0.1:8765/mcp  (loopback, token-gated)
dancr gateway . --port 9000 --host 127.0.0.1
dancr gateway . --allow-remote  # bind a network address (put TLS in front)
```

It serves the same MCP tools over the SDK's **streamable-HTTP** transport
(Starlette/uvicorn ship with `mcp`; no new dependency), so an MCP client can point
at `http://127.0.0.1:8765/mcp` with `Authorization: Bearer <token>`.

Security posture (fail-closed):

- **Loopback only by default.** A non-loopback `--host` is refused unless
  `--allow-remote`; there is no built-in TLS, so remote use belongs behind a
  reverse proxy.
- **Bearer tokens from the policy file.** A principal lists `tokens`; the token is
  compared in constant time and resolved to that principal. With no policy, or no
  principal token, `build_app`/`dancr gateway` raises — it can never come up open.
- **Every tool call is policy-gated and audited.** The decision is `allow`, `deny`
  or `approve`; a `deny` returns a JSON-RPC `denied` error, an `approve` is
  satisfied by a person's earlier decision (`dancr approvals --approve ap_N`) or
  queued and returned as `approval_required`. All are written to the audit log and
  counted against quotas.
- **DNS-rebinding protection** is on, with the bind host/port as the only allowed
  hosts and origins.

SDK: `dancr.gateway_app(root, host=…, port=…)` builds the ASGI app (for tests or a
custom server); `dancr.is_loopback(host)`.

## Honest limits

- **TLS is not built in.** The default is loopback, which needs none; remote use
  requires a reverse proxy.
- Enforcement on the **stdio** server and CLI is still opt-in/manual (call
  `enforce`/`policy_check`); only the `dancr gateway` transport enforces per call
  automatically. Wiring the stdio server through the same gate is a possible
  follow-up.
- The policy language is deliberately tiny (principals × categories/rules ×
  approvals × quotas). Anything larger becomes an enterprise product; extend only
  on demand.
- One gateway serves one repository root; multi-root is out of scope.
