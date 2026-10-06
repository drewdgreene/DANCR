# ADR 0006 — Governance and the gateway transport (B1)

Status: **accepted and implemented** (Phase 2 transport)

## Context

The roadmap (B1) turns the single-client stdio MCP server into a multi-client,
policy-governed control plane: principals, policy, approvals, audit and quotas.
That introduces a network surface with production security consequences.

## Decision

The policy/approval/audit/quota machinery is pure, dependency-free and useful on
its own; it was built first and gates the existing local surfaces. The network
transport is now implemented with **fail-closed, local-first** defaults, using the
MCP SDK's streamable-HTTP transport (Starlette/uvicorn ship with `mcp`; no new
dependency):

1. **Bind address — loopback only by default.** `dancr gateway` binds `127.0.0.1`.
   A non-loopback host is refused unless `--allow-remote` is passed.
2. **Authentication — bearer tokens from the policy file.** Each principal lists
   `tokens`; a request must carry `Authorization: Bearer <token>`, compared in
   constant time (`hmac.compare_digest`) and resolved to a principal. With no
   policy, or no principal token, the gateway **refuses to start** — it can never
   come up open.
3. **TLS — not built in.** Loopback needs none; for remote use, terminate TLS at a
   reverse proxy in front of `--allow-remote`. This keeps the dependency surface
   minimal and the certificate lifecycle out of the app.
4. **Threat model — assume the network is hostile.** Least privilege (only the
   existing typed MCP tools, no shell/network/arbitrary writes), per-call policy
   (allow/deny/approve) with an audit record for every call, quotas, DNS-rebinding
   protection (allowed hosts/origins are the bind host and port), and approval
   gates for consequential actions. MCP tool contracts are unchanged.

## Consequences

- The offline-first promise holds: the socket exists only when a person runs
  `dancr gateway`, and it is loopback-only and token-gated.
- A repository opts in by writing `<root>/.dancr/gateway/policy.json` with tokens;
  until then the command refuses to run.
- Every gateway call is auditable (`dancr audit`); an `approve` verdict is
  satisfied by a person's decision (`dancr approvals --approve ID`) or queued.
- No new dependency, no TLS certificates to manage in-app, no change to any
  existing MCP tool.
