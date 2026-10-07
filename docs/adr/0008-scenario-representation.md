# ADR 0008 — Scenario representation (F2)

Status: accepted

## Context

The roadmap's F2 asked for scenario analysis and left the representation open:
"Decide whether the node is a control node handled by the executor or a headless
primitive surfaced by a node; the former is cleaner but touches the core execution
model." It also sketched registry node types (`scenario`, `sweep`, `monte_carlo`,
`sensitivity`) with an aggregate step.

## Decision

Ship scenarios as a **headless primitive**, not registry node types:

- `dancr/core/scenarios.py` — deterministic scenario-set builders: `named`,
  `sweep`, `monte_carlo`, `sensitivity`, and `from_spec`. These are functions, not
  nodes.
- `dancr/headless/_scenarios.py` — `run_scenarios`, which runs a target once per
  scenario on a private clone of the project and returns a record with per-scenario
  plan/output hashes (the attestation evidence block).
- Surfaces: CLI `dancr scenarios`, MCP `run_scenarios`, SDK `run_scenarios` /
  `scenario_set`. Proof: `dancr verify … --scenarios spec.json`.

## Rationale

A scenario overrides a project **Input**, and Inputs are global to a run: the
executor folds the values a step names into that step's plan hash
(`Executor.inputs_used`). `NodeType.apply(ctx, inputs, params)` receives only its
input LazyFrames — it cannot clone the pipeline and re-run the upstream subgraph
with different Input values. A scenario node therefore cannot be an ordinary
transform. Making it a *control* node would mean teaching `Executor.run` to fan out
over Input combinations, fan back in, and cache per combination — a large change to
the most hardened, invariant-carrying part of the engine (one writer at a time,
cache leases, streaming materialisation), for which the roadmap itself allows the
primitive instead.

The primitive already delivers the product and keeps every invariant:

- **Deterministic** — the spec is ordered and seeded; each scenario's hashes are
  recorded and reproducible.
- **Incremental** — Input values are part of the plan hash, so changing one scenario
  recomputes only the steps that use it.
- **Confined** — writes stay inside the project folder, never over a source file
  (`unsafe_write`), like `run_batch`.
- **One writer at a time** — the project file is never changed (each scenario runs on
  a clone); only the shared cache is touched, which is safe for the existing leases.
- **Proven** — per-scenario plan and output hashes fold into an attestation
  (`docs/adr/0007`).

## Consequences

- No canvas node, so scenarios are not yet visible as a step in the GUI; they are a
  command/tool/SDK call. A future "run scenarios" node could wrap `run_scenarios` if
  the executor ever gains control nodes, without changing the primitive.
- `scenario` / `sweep` / `monte_carlo` / `sensitivity` are spec kinds and builder
  functions, not separate node types; the wording in the roadmap and `SCENARIOS.md`
  reflects that.
- The decision is reversible: the primitive is the implementation behind any future
  node, so adding one later is additive.
