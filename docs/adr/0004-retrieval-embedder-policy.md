# ADR 0004 — Retrieval embedder policy (C1)

Status: accepted (Phase 0, implemented in Phase 1)

## Context

`dancr/core/rag.py` is a deterministic, offline lexical embedder with an
incremental persistent index. The roadmap (C1) wants pluggable, hybrid retrieval
without breaking determinism or the offline-first promise, and without a vector
database.

## Decision

- **Deterministic by default.** The default retriever is the existing lexical
  embedder; its identity string is `dancr-lexical-v1`. With the default, results
  are bit-reproducible and the existing contract (a `vector` column, cosine
  scores, cited source rows) is unchanged.
- **Hybrid is opt-in and still deterministic.** A BM25 retriever (standard
  library, computed at query time from the stored passages) can be fused with the
  lexical one by a deterministic weighted/reciprocal-rank fusion. No randomness,
  no clock, no model.
- **A future neural embedder is recorded, not assumed.** The `Retriever`
  interface carries an embedder id; the persistent index stores its identity and
  dimension in a small sidecar file next to the index (`<index>.meta.json`), so a
  run can say how it was produced. No neural embedder ships now, so no new
  dependency and no nondeterminism is introduced. When one is added it must be
  pinned by id/version and its use recorded for auditability.
- **No vector database.** The index stays a parquet table read into memory for
  scoring; sensitivity filtering (`sensitivity` column, `allow_restricted`)
  applies on every path.

## Consequences

- Default behaviour is byte-identical to before; existing indexes keep working
  (a missing sidecar is read as the lexical embedder at the stored dimension).
- Hybrid retrieval is available without the network or a new dependency.
- The provenance block on a search result states the retrievers, their scores
  and the embedder, so a hybrid (or, later, neural) result is auditable.
