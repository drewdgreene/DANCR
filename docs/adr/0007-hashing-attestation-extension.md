# ADR 0007 — Hashing and attestation extension point

Status: accepted (Phase 0)

## Context

`build_attestation` (`dancr/core/verify.py`) records a deterministic,
checkable summary of a project. New features (the graph, retrieval
configuration, scenario sets) need to fold their own evidence into a record
without breaking old attestations or old readers.

## Decision

Add a single, generic, **optional** `extra` block:

- `build_attestation(..., extra: dict | None = None)` stores it under `"extra"`
  and includes it in `attestation_hash` (which already excludes only
  `attestation_hash` and `generated_at`).
- `verify_pipeline` compares `extra` **only when the reference carries it**
  (exactly as it treats `output_hash` for a plain run manifest): an older
  attestation without `extra` still verifies unchanged.
- The top-level `"version"` (`ATTESTATION_VERSION`) is **not** bumped: the record
  stays readable by an older DANCR, which ignores the unknown key.

Scenario sets (F2) and hybrid-retrieval configuration (C1) will place their
per-item hashes under `extra`, so `verify` can report an aggregate verdict while
old attestations remain valid.

## Consequences

- Backward compatible both ways (old attestations load; new ones load in an old
  build).
- One extension point, used once, instead of a schema bump per feature.
- `attestation_hash` changes when `extra` is present, which is correct: the
  evidence genuinely differs.
