# ADR 0006 — Governance and the gateway transport (B1)

Status: proposed — **awaiting a decision before any listening socket is built**

## Context

The roadmap (B1) turns the single-client stdio MCP server into a multi-client,
policy-governed control plane: principals, policy, approvals, audit and quotas.
That introduces a network surface with production security consequences, which
the roadmap itself lists under STOP-AND-ASK.

## Decision (proposed; core built, transport deferred)

The **policy, approval, audit, quota and principal machinery** is pure,
dependency-free and useful on its own (it can gate the existing stdio tools and
the CLI). It is implemented without opening any socket. The pieces:

- A dependency-light **policy file** (JSON) evaluated by a tiny matcher:
  principals × tools × action categories × approvals × quotas.
- An **approval queue** for consequential actions (writes outside the project
  folder, exports, enabling samples, graph writes).
- An append-only **audit log** (`<root>/.dancr/audit/audit.jsonl`) with redacted
  params (`core.secrets.redact_params`) and result hashes.
- **Quotas** per principal/project (run count, wall time, model tokens).

The **network transport** (streamable HTTP MCP) is deliberately **not** started
by default and is left for an explicit follow-up decision. Open questions that
must be answered before it ships:

1. Bind address (loopback only vs. a configurable interface) and default.
2. Authentication (bearer tokens issued from the policy file vs. OS identity).
3. TLS termination (built-in vs. reverse proxy) and certificate handling.
4. Threat model review and rate-limit defaults.

## Consequences

- Phase 2 can gate and audit the existing local surfaces immediately with no new
  security exposure.
- No listening socket exists until the four questions above are answered, so the
  offline-first guarantee is preserved.
