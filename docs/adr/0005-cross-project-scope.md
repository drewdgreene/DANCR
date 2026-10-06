# ADR 0005 — Cross-project question answering scope (C2)

Status: accepted (Phase 0, implemented in Phase 3)

## Context

The answer engine (`ask`, `recipes`) answers questions inside one project's
`DataModel`. The roadmap (C2) wants relational questions across a repository.

## Decision

- **Structural answers first; no cross-project execution.** A cross-project
  answer is a fact about the graph — a shared key, a join path, where a metric
  comes from, what feeds a report — and cites the edges (and their evidence) that
  support it. It does **not** build or run a pipeline spanning two projects.
- When an answer naturally calls for execution, DANCR offers to **materialise the
  join into one project** as an ordinary follow-up the person approves; that
  reuses the existing answer engine inside a single project.
- Answers remain deterministic grammar/plan output over the graph, and honour
  sensitivity: a restricted dataset's contents are never surfaced across a
  project boundary, matching `search_knowledge`'s `allow_restricted`.

## Consequences

- No change to the executor's LazyFrame model is required.
- The first cut is provably offline and deterministic.
- A later "run it across projects" feature is additive: it would build a plan in
  one project rather than change the answer engine's contract.
