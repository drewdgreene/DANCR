# Roadmap: a context graph and a governed agent gateway

Status: **implemented.** The six features below have since landed; `docs/ROADMAP-PROGRESS.md` records
what was built and how it was verified. This document is kept as the design rationale for how DANCR
grows from a per-project analysis-and-evidence tool into a general **context-and-evidence layer**
with a **governed agent control plane**, while keeping it a general product and keeping every
existing promise (determinism, provenance, offline-first, one install, one writer at a time).

It is generalized on purpose. The domain-specific counterpart it grew out of, and the one it would be
pointed at, live outside this repo; this file is the domain-neutral engineering plan.

The six features:

| id | feature |
|----|---------|
| A1 | cross-project entity graph |
| A2 | repository-level change detection and an event log |
| B1 | a governed `dancr gateway` service |
| C1 | pluggable, hybrid retrieval with provenance |
| C2 | cross-project question answering |
| F2 | scenario and sweep nodes |

---

## 1. Design invariants

These are the properties that make DANCR worth building on. Every feature below must preserve them.

1. **Determinism.** Same inputs produce the same plan hashes, output hashes and attestation hashes
   (`dancr/core/engine_hash.py`, `plan_hash` in `executor.py`, `dancr/core/verify.py`). Any new
   computation that can affect a result must fold its configuration into a hash, and any
   nondeterministic ingredient (a neural embedder, a sampler) must be pinned or recorded.
2. **Offline-first.** Today only three things reach the network: `load_url`, `load_sql`, and the
   Assistant client (`dancr/core/assistant/client.py`); `load_document` (MinerU) is local unless
   `allow_remote`. New features default to local. A gateway is the one place network access is
   expected, and it must be explicit and optional.
3. **No-extra-dependencies ethos.** Geo, database, NetCDF/HDF5 and plotting are core dependencies so
   "nothing asks the person to add an extra" (`pyproject.toml`). New storage for the graph, events
   and audit should prefer the standard library (SQLite) over a new database dependency.
4. **One writer at a time.** OS file locks per project (`project_lock`/`_os_lock` in
   `dancr/headless/__init__.py`), 30 s wait, plus executor cache leases
   (`Executor.hold/_write_lease`, `.live/`). Cross-project work multiplies this: define lock ordering.
5. **Format compatibility.** A project is `{"dancr": 2, …}` (`FORMAT_VERSION`,
   `dancr/core/model.py`). New project data goes in `meta`, never a new top-level key, because
   `_from_dict` drops unknown top-level keys but preserves unknown `meta` keys. A version bump is
   allowed only if old files still load.
6. **Confinement.** MCP writes are confined to the project folder and never into `.dancr`
   (`mcp_server._in_root`, `_check_outputs`, `headless/_safety.unsafe_write`). Every new writing
   surface must run the same checks.
7. **Pure core / Qt split.** `dancr/core/`, `dancr/headless/` and `dancr/views/` carry no Qt;
   `dancr/ui/` is the only Qt layer. New core logic stays Qt-free; new views live in `ui/`.

---

## 2. Current architecture (the recon, condensed)

What matters for this roadmap, with the seams each feature will hook into.

- **Project model.** `dancr/core/model.py`: `Node`, `Edge`, `Note`, `Answer`, `Input`,
  `Pipeline`, `portable_path`, `rebase_params`, `resolve_path`. `Pipeline.save` writes atomically and
  archives versions under `<dir>/.dancr/versions/`. Locking in `headless/__init__.py`
  (`project_lock`, `editing`, `editing_deferred`).
- **Execution.** `dancr/core/executor.py`: `Executor.run` walks `Pipeline.topological_order`, runs
  each `NodeType.apply(ctx, inputs, params)`, and materializes to
  `<project>/.dancr/cache/<file>/<node>/<hash>.parquet`. `plan_hash` folds in `CODE_FINGERPRINT`,
  params, upstream hashes, a source fingerprint (`_source_fingerprint`, `_content_sample`) and used
  `Input` values (`inputs_used`/`params.inputs_named`). `run_batch` (`headless/__init__.py`) clones a
  pipeline per file and rebinds one path param — the existing "many runs, one project" primitive.
- **Hashing / evidence.** `dancr/core/engine_hash.py` (`IMPL_VERSION`, `CODE_FINGERPRINT`,
  `engine_files` AST discovery), `dancr/core/verify.py` (`build_attestation`, `verify_pipeline` with
  verdicts `verified|mismatch|incomplete|engine-changed`, `output_hash`, `flatten_report`),
  `dancr/core/lineage.py` (`build_lineage` up/down, `proof_card`).
- **Node registry.** `dancr/core/registry.py` (`NodeType`, `InputSpec`, `Ctx`, `NodeResult`,
  `registry.register`) and `dancr/core/params.py` (`Param`, `KINDS`). A new step is a module in
  `dancr/core/nodes/` imported by `nodes/__init__.py`. `kind` is `source|transform|sink`;
  `materialize=False` for visual sinks; `uses_labels` when output depends on column labels;
  `source_files`/`source_digest` for invalidation.
- **Understand / connection map.** `dancr/core/understand/__init__.py` builds a `DataModel` of
  `Column`/`Table`/`Relation` (kinds `link|stack|align|near|containment`, with `cardinality`,
  `match_pct`, `score`, `why`). The persisted connection map is `headless.connection_map`, stored in
  `pipeline.meta["assistant"]["connections"]` via `assistant.store`. `deepen()` reads every row for
  exact counts.
- **Ask / answers.** `dancr/core/ask/` (`ask()` a deterministic grammar), `planner.py`
  (`apply_plan`, `Edits`), `answers.py` (`build`, `change`, `remove`), `recipes/_plan.py`
  (`PLANNERS`, `plan`, `suggest`), `bank.py`, `lexicon.py`, `memory.py` (`meta["ask"]`).
  `RULES_VERSION` gates a rebuild when the same spec would produce different steps.
- **Retrieval.** `dancr/core/rag.py`: a **deterministic, lexical, offline** bag-of-words/bigram/
  trigram hashing embedder (`embed`, `chunks`, `matrix`, `top_k`), an incremental persistent parquet
  index (`merge_index`, `doc_content_hash`, `load_index`/`save_index`), and `search_knowledge`
  (sensitivity-aware). Nodes in `dancr/core/nodes/rag.py` (`build_index`, `retrieve`). The module
  docstring explicitly states the **retrieval contract** (a `vector` column, cosine scores, cited
  source rows) "is exactly what a model embedding would fill later without changing anything else."
- **Context / FAIR.** `dancr/headless/_context.py` (`build_context`, `CONTEXT_VERSION=2`,
  `context_changes`, `find_pipelines`, `build_catalog`, `catalog_jsonl`, `catalog_changes`,
  `package_rocrate`) and `dancr/core/profile.py`, `dancr/core/fair/` (schema.org, Frictionless,
  manifest, RO-Crate). `content_hash` is the plan hash; the catalog keys datasets as `file#node`
  (`_catalog_key`).
- **Assistant.** `dancr/core/assistant/` — `Provider`/`OpenAIProvider`/`provider_for`, `session.py`
  turn loop, `tools.py` `ToolRunner`, `store.py` (`meta["assistant"]`), and the **trust contract**
  (`_flags_for`, unverified-figure check).
- **Interfaces.** `dancr/mcp_server.py` (`@mcp.tool`, root confinement), `dancr/cli.py`
  (`cmd_*`, `--json`), `dancr/__init__.py` (`_PUBLIC` SDK names), `dancr/headless/`, and
  `dancr/core/secrets.py` (`expand_env`, `redact`, `redact_params`).
- **UI.** `dancr/ui/`: `Document` (pipeline + executor + undo stack + signals), `canvas/`,
  `workers.py` (`RunThread`, `ViewPool`), `document.py`, and stacked result pages in `mainwindow.py`.

---

## 3. The six features

Each section states the goal, what exists to build on, the concrete work, the new surface, the effect
on determinism/provenance, and the hard problems.

### A1. Cross-project entity graph

**Goal.** A persistent graph over every project in a repository: datasets, sources, columns and
resolved entities as vertices; inferred joins, stacks, time alignments and derivation as edges, each
carrying its evidence and confidence. This is the substrate for A2, C2 and the KB.

**Build on.** `find_pipelines`/`build_catalog`/`catalog_changes` (the read-only walk, with
`file#node` keys and content hashes), `understand.find_relations`/`deepen` (the relation inference),
`connection_map` (per-project, persisted in `meta["assistant"]["connections"]`).

**Work.**

1. New pure module `dancr/core/graph.py`: schema, build, upsert, query. No Qt.
2. Storage: a repo-level `graph.db` (SQLite, standard library) under a `.dancr/graph/` root, written
   with the existing atomic-write helper. Tables: `project`, `dataset`, `source`, `column`, `entity`,
   `edge`. An `edge` row records `kind`, `left_key`, `right_key`, `cardinality`, `match_pct`,
   `confidence`, `evidence`, both endpoints' content hashes, and `updated_at`.
3. **Identity scheme (the crux).** A dataset vertex's identity is `(project fingerprint, node id)`;
   its *version* is the plan hash. An entity vertex's identity is a resolved real-world key
   (a declared key or a value fingerprint). Decide and document how identity survives renames and code
   changes before writing storage, because migration is expensive to undo.
4. Population: `build_graph(root)` walks the catalog, and for each changed project (via
   `catalog_changes`) reads its `DataModel` and its connection map and upserts vertices and edges.
5. Entity resolution, deterministic first: declared keys, contract-declared keys
   (`check_contract`), and exact value overlap on a seeded sample (`understand` already samples and
   scores). Fuzzy matching is opt-in, confidence-scored, and explainable.
6. Query API: neighbors, shortest join path between two datasets, shared keys, upstream/downstream
   across projects, find by entity. Deterministic ordering throughout.
7. Provenance: every edge keeps the evidence that produced it and can be re-verified; the graph
   records the `CODE_FINGERPRINT` and library versions it was built with.
8. Sensitivity: carry `label_sensitivity` labels onto vertices and edges; withhold
   confidential/restricted nodes and their edges in any export unless explicitly allowed, matching
   `search_knowledge`'s `allow_restricted`.

**New surface.** CLI `dancr graph build|query|neighbors|path`; MCP `graph_build`, `graph_query`,
`graph_neighbors`, `graph_path`, `graph_shared_keys`; SDK `build_graph`, `graph_query`. Include the
graph slice in `context`/FAIR exports.

**Effect on determinism/provenance.** The graph is a projection of hashed project state, so it is
rebuildable and diffable. Its own verification: `graph verify` re-derives edges for one dataset and
compares. Because it is derived, it never becomes a second source of truth.

**Hard problems.** Identity vs content-addressing; entity resolution quality without a model;
cross-project privacy; keeping the build affordable on large catalogs.

### A2. Repository-level change detection and an event log

**Goal.** Know what changed anywhere in a repository, ripple the consequence, and publish a
subscribable change feed — the "living" property.

**Build on.** `headless.watch` and `_watch_state`/`_changed_files` (the per-project polling loop with
content sampling), `catalog_changes`, and the A1 graph (which supplies cross-project dependencies).

**Work.**

1. New `dancr/core/events.py`: a typed event schema and an append-only log (atomic JSONL, or a
   SQLite table) with monotonic sequence numbers. Events: `source_changed`, `dataset_invalidated`,
   `project_recomputed`, `index_stale`, `graph_updated`, `answer_stale`, `gateway_action`.
2. `watch_repo(root)`: a loop over `find_pipelines(root)` that reuses `_watch_state` for cheap
   change detection. Keep `watch` (single project) unchanged; the repo watcher is a supervisor.
3. Dependency ripple: a reverse-dependency index built from each project's DAG plus the A1 cross-project
   edges. When a source changes, mark dependent datasets stale, optionally re-run them, re-index RAG,
   and update the graph.
4. **Policy for what auto-runs.** Honor the existing rule that the window never auto-runs a step that
   writes outside the project folder or over the project's data; route those to B1's approval queue.
5. Subscriptions: CLI `dancr events [--follow]`; when B1 is present, a stream over the gateway.
6. Concurrency: use `project_lock` and executor leases; define lock ordering (project locks acquired in
   a stable order) and a backoff so a burst of changes does not thrash.

**New surface.** CLI `dancr events`, `dancr watch --repo`; MCP `get_events` (read-only); SDK
`watch_repo`, `read_events`.

**Effect on determinism/provenance.** A re-run triggered by the watcher produces identical hashes to
a manual run (the engine is deterministic); the event log records *why* a re-run happened, which
feeds lineage.

**Hard problems.** Safe auto-recompute at scale; event ordering and idempotency; cost control; no OS
file notifications exist today (polling only) — add notifications behind a fallback.

### B1. A governed `dancr gateway` service

**Goal.** Turn the single-client stdio MCP server into a multi-client, policy-governed control plane
for agents, without turning DANCR into an enterprise appliance.

**Build on.** `mcp_server.py` (tools, `_in_root`, `_check_outputs`, `_safety.unsafe_write`),
`headless.project_lock`/`editing_deferred`, `core/secrets.py`, and the Assistant's provider
abstraction (`provider_for`) for brokering model calls.

**Work.**

1. A long-running `dancr gateway` mode of the same binary, adding an HTTP/streamable MCP transport so
   several clients can connect. Keep stdio MCP working and its tool contracts stable.
2. Principals and roles with API tokens; optional OS-identity mapping. Everything a tool does is
   attributed to a principal.
3. **Policy as code.** A dependency-light policy file (JSON/YAML plus a tiny evaluator, not a full
   rules engine) covering principals, projects, tools, data classes, action categories, approvals,
   quotas and time windows. Every denial explains itself.
4. Per-tool permissions: generalize the existing confinement (`_in_root`, `unsafe_write`) into policy
   instances rather than special cases. A write tool is `allow` when policy permits, `approve` when it
   is consequential, `deny` otherwise.
5. An **approval queue** for consequential actions (writes outside the project folder, exports,
   enabling sample rows, graph writes), with a CLI/UI/MCP surface to approve, and an audit record.
6. **Audit log**: every gateway call appended with principal, tool, redacted params
   (`secrets.redact_params`), verdict, timestamps and result hashes.
7. Quotas and cost: per principal/project limits on run count, wall time, and model tokens (usage is
   already reported by `assistant.client`). Route model calls through the gateway so `choice` and
   `cost` are enforceable.
8. Security: TLS, secrets handling, rate limits, strict input validation, and an explicit threat
   model for the network surface.
9. Config: a `gateway` config file; document deployment (foreground service, or a mode of the app).

**New surface.** `dancr gateway`, `dancr policy check`, `dancr approvals`, `dancr audit`; MCP stays
the same tool set, plus gateway-specific admin tools guarded by policy.

**Effect on determinism/provenance.** The gateway changes *who may act*, not *what a computation
produces*; runs remain reproducible. The audit log is itself append-only and, later, signable.

**Hard problems.** Policy DSL scope (too small is useless, too large becomes an enterprise product);
network security; multi-user concurrency; resisting dependency bloat.

### C1. Pluggable, hybrid retrieval with provenance

**Goal.** Keep deterministic retrieval as the reproducible default, but allow a stronger retriever
(semantic embeddings, keyword, graph traversal, rerank) without breaking what makes DANCR trustworthy.

**Build on.** `dancr/core/rag.py` — the module already declares the contract ("a `vector` column,
cosine scores, cited source rows") that a model embedder fills. `merge_index`/`doc_content_hash`
give incremental, hashed reuse. `nodes/rag.py` and `search_knowledge` enforce sensitivity.

**Work.**

1. Introduce a `Retriever` interface with fusion (reciprocal rank fusion or weighted) over: lexical
   (BM25), vector (the current deterministic embedder plus an optional neural one), graph traversal
   (via A1), and rerank.
2. Embedder abstraction: keep `embed()` as the default; add a neural embedder behind a provider. The
   embedder's identity, dimension and version are stored in the index metadata and in retrieval
   provenance so a result can say how it was produced. Embeddings are cached by content hash (the
   incremental machinery already reuses unchanged chunks).
3. BM25 in the standard library (no dependency), or a light optional one.
4. Rerank: a local cross-encoder (optional) or an LLM reranker whose model is recorded. Never the
   default.
5. Retrieval provenance: each result states which retrievers matched, their scores, and any graph path
   used. The index records the retriever set and chunker params.
6. Index format versioning and migration; sensitivity and `allow_restricted` preserved on every path.

**New surface.** `build_index`/`retrieve` params (`retriever`, `embedder`, `rerank`); `search_knowledge`
gains a provenance block; new MCP/SDK options.

**Effect on determinism/provenance.** Determinism is preserved when the default deterministic
retriever is used; when a neural embedder or LLM reranker is used, its identity is *recorded*, so a
run is auditable even though it is not bit-reproducible. The policy is "reproducible by default,
documented when not."

**Hard problems.** Determinism once a semantic model is in the loop; fusion tuning; latency;
dependency creep.

### C2. Cross-project question answering

**Goal.** Let the deterministic `ask` engine answer relational questions across the repository, not
just within one project.

**Build on.** `ask`/`vocab`/`recipes` operate on one `DataModel`; `understand` builds that model;
`Bank`/`lexicon`/`memory` support it; `build_catalog`/`context_changes` supply cross-project schema
and hashes; A1 supplies cross-project relations.

**Work.**

1. Synthesize a combined, namespaced `DataModel` across projects, keying nodes as `file#node` (the
   `_catalog_key` convention) so column references `[node, column]` stay unambiguous. A new
   `find_relations` pass discovers links between projects, seeded by the A1 graph.
2. Extend the grammar and vocabulary (`ask/vocab.py`, `ask/__init__.py`) with relational intents:
   "what joins to this", "which datasets share the customer key", "where does this metric come from",
   "what changed downstream", "join A and B", "what feeds the report".
3. New recipes in `recipes/_plan.py`: `relations`, `shared_key`, `join_path`, `lineage`, `provenance`.
   Register them in `RECIPES`/`PLANNERS`/`WEIGHT`/`chips()`.
4. **Structural answers.** Some answers are facts about the graph, not a data pipeline (e.g., "these
   two datasets share a key"). Extend the answer representation so a read-only, graph-sourced answer
   can carry edge evidence, while `apply_plan`/`answers.build` continue to handle dataflow answers
   into a single project. Cross-project *execution* is out of scope for the first cut; offer "build
   this join" as a follow-up that materializes into one project.
5. Sensitivity across projects: an answer must never surface restricted data from another project.
6. Extend `bank`/`suggest`/`memory` to catalog scope and add repo-scale golden question sets for
   `run_eval`.

**New surface.** `dancr ask --root DIR "…"`, `dancr suggest --root DIR`; MCP `ask`/`suggest_answers`
gain a scope argument; SDK `ask_question(..., scope=)`.

**Effect on determinism/provenance.** Answers remain deterministic grammar output; every answer cites
the graph edges and their evidence, so a reader can trace the claim.

**Hard problems.** Cross-project execution semantics if you go past structural answers; keeping the
combined model deterministic and affordable; scope and permissions (a question must not leak another
project's restricted data).

### F2. Scenarios and sweeps

Status: **implemented** as a headless primitive — see
`docs/adr/0008-scenario-representation.md` and `docs/SCENARIOS.md`. The wording
below is kept for the design record; items 1–2 were resolved to the primitive
rather than registry node types.

**Goal.** Make "run this analysis across many assumptions" a first-class, provable thing.

**Build on.** `Pipeline.inputs` (values already enter `plan_hash` via `inputs_used`), `run_batch`
(clones a pipeline per file, shares the cache, threads with `jobs`), the executor and lease
machinery, and `findings`.

**Work.**

1. Scenario-set builders in `dancr/core/scenarios.py`: `named` (named input sets), `sweep` (grid over
   input ranges), `monte_carlo` (sampled distributions, seeded), `sensitivity` (one-at-a-time). They
   build deterministic `{id, inputs}` lists; they are not node types (a scenario overrides global
   Inputs, which a `NodeType.apply` cannot do — it sees only its input frames).
2. **Execution model.** A headless `run_scenarios()` (in `dancr/headless/_scenarios.py`) enumerates the
   scenarios, runs the target on a private clone per scenario with the existing
   cache/lease/confinement, and aggregates (one output each plus a combined table). It is surfaced by
   the CLI (`dancr scenarios`), the MCP tool `run_scenarios` and the SDK (`run_scenarios`,
   `scenario_set`), not by a control node, so the executor's single-run model is untouched.
3. Per-scenario caching: key a scenario's cached result by the base plan hash plus the scenario's
   input values (already part of `plan_hash`), so changing one scenario recomputes only that one.
4. Inputs gain ranges, distributions and sets with units; the random seed is explicit and stored in
   the project.
5. **Attestation extension.** The current attestation assumes a single deterministic run. A scenario
   set adds a dimension: record the scenario definition, the seed, and per-scenario plan and output
   hashes, and let `verify` report an aggregate verdict. Extend `verify.build_attestation`/
   `flatten_report` and `proof_card`.
6. Views and reports: tornado, fan, and scenario-comparison charts (a new `views/` renderer plus a
   result page in `ui/`).
7. Performance: sampling, parallelism (reuse `jobs`), streaming aggregation, and disk management for
   many runs.

**New surface.** Headless `run_scenarios` (CLI `dancr scenarios`, MCP `run_scenarios`, SDK
`run_scenarios`/`scenario_set`); the attestation/verify extension (`dancr verify … --scenarios`). No
registry node type (see the Status note above).

**Effect on determinism/provenance.** Determinism holds because inputs are already hashed; the seed
and scenario definition are recorded, so a scenario set is reproducible and provable.

**Hard problems.** Fitting fan-out/fan-in into a LazyFrame executor; the attestation schema for N
runs; compute cost.

---

## 4. Cross-cutting workstreams

These are shared by all six features and should be treated as first-class, not tail work.

- **Project format v3.** Add graph, scenario and index metadata, all under `meta`, with
  backward-compatible reading of `{"dancr": 2}`. Only bump `FORMAT_VERSION` if old files still load.
- **Hashing and attestation.** Extend `engine_hash`/`plan_hash`/`verify` covers graph edges,
  retrieval configuration and scenario sets. Extend `run_manifest` once (`fair/manifest.py`) so
  verify and FAIR both gain. Keep old attestations verifiable.
- **Storage layout.** A standard repo-level `.dancr/` home for `graph/`, `events/`, `audit/` and
  `index/`. Prefer SQLite (stdlib) and atomic writes (`write_text_atomic`) over new dependencies.
- **Concurrency.** Generalize per-project locks into a daemon-safe lock manager with defined ordering
  and backoff; keep `editing_deferred` semantics for long compute.
- **Interfaces.** New CLI commands (`cmd_*` + `build_parser`), MCP tools (`@mcp.tool` with
  `_load`/`_editing`/`_check_outputs`), and SDK names (`_PUBLIC`). Update `AGENTS.md`, `help/`,
  `doctor`, `formats`, `nodes -v`. Keep MCP tool contracts stable and versioned.
- **Security.** Threat model for the gateway's network surface; secrets via `core/secrets.py`; TLS;
  rate limits; input validation; sandboxing of writes.
- **Performance.** Benchmarks and budgets for graph builds over large catalogs, retrieval latency,
  repo watching, and scenario fan-out. Incremental everywhere.
- **Testing.** New modules in the existing style: `test_graph.py`, `test_events.py`,
  `test_gateway.py`, `test_policy.py`, `test_retrieval_hybrid.py`, `test_ask_graph.py`,
  `test_scenarios.py`. Determinism tests (same inputs, same hashes), planner property tests, policy
  bypass and path-escape security tests, repo-scale golden question sets. Keep CI green.
- **Migration.** Reading old projects, indexes and attestations; rebuild rules when the graph or index
  schema changes; `doctor` checks for each new capability.
- **Docs.** New `GRAPH.md`, `GATEWAY.md`, `RETRIEVAL.md`, `SCENARIOS.md`; honest limits in `OPEN.md`.
- **UX.** A graph explorer, a change-feed panel, an approval queue, and scenario/sensitivity result
  pages. New views slot into `MainWindow.pages` + `window_pages._show_page`, a new rail section, or a
  side dock; every project mutation stays an undo command.
- **Packaging.** Gateway and the repo watcher are modes of the same binary. This is where the open
  code-signing item starts to matter (see `OPEN.md`).

---

## 5. Phasing and sequencing

Dependencies: A2 needs A1 and B1; C2 needs A1; F2 is independent but gains from A1 and A2; C1 is
largely independent but benefits from A1's graph traversal.

**Phase 0 — foundations.** Identity scheme, repo storage layout, format v3 (meta-only), hashing and
attestation extensions, the lock manager. No user-visible feature.
*Exit:* new stores read and write deterministically; old projects, indexes and attestations still
load; CI green.

**Phase 1 — A1 + C1.** The graph and hybrid retrieval. Mostly local, high user value, and they
reinforce each other.
*Exit:* `dancr graph build/query` works across a repository and is incremental; retrieval supports
lexical plus an optional neural retriever with recorded provenance.

**Phase 2 — B1 + A2.** Gateway core (config, principals, policy, approvals, audit) and the repo
watcher/event log, with the gateway as the daemon behind the watcher.
*Exit:* a policy-governed gateway serves multiple clients with an audit log; the watcher emits events
and consumes them; consequential actions require approval.

**Phase 3 — C2 + F2.** Graph-aware question answering and scenario/sweep nodes with proofs.
*Exit:* cross-project questions answer with cited graph evidence; scenario runs reproduce and verify.

---

## 6. Risk register

| risk | where | mitigation |
|------|-------|-----------|
| Graph identity becomes unrecoverable | A1 | Decide identity vs version in Phase 0; derive, never make it a source of truth |
| Entity resolution without a model is weak | A1 | Deterministic keys and value overlap first; fuzzy is opt-in, scored, explainable |
| Auto-recompute runs away | A2 | Policy-gated auto-run, backoff, quotas (B1) |
| Policy DSL either useless or enormous | B1 | Start with principals × tools × actions × approvals; grow only on demand |
| Network surface introduces vulnerabilities | B1 | TLS, tokens, rate limits, strict validation, explicit threat model |
| Neural retrieval breaks determinism | C1 | Reproducible by default; record the embedder; cache by content hash |
| Cross-project execution is a big jump | C2 | Structural answers first; "build this join" as a follow-up |
| Fan-out fights the LazyFrame executor | F2 | Enumerate via a headless runner first; touch the executor only if needed |
| One-install ethos erodes | A1/B1/C1 | Prefer stdlib (SQLite); make every new dependency optional and justified |
| Cache/attestation invalidation surprises | all | Bump `IMPL_VERSION` deliberately; extend `run_manifest` once; test determinism |

---

## 7. Decisions needed before Phase 0

1. **Identity scheme** for graph vertices (project fingerprint + node id + stable dataset key), and
   how it survives renames and `CODE_FINGERPRINT` changes.
2. **Graph storage**: SQLite `graph.db` versus a file-based JSONL plus index (dependency footprint vs
   query power).
3. **Gateway transport and auth**: streamable HTTP MCP with tokens; how far to go before it stops
   being "local-first".
4. **Embedder policy**: deterministic-only, or allow a neural embedder whose identity is recorded.
5. **Cross-project execution scope** for C2: structural answers only in the first cut, or real
   cross-project pipelines.
6. **Policy DSL scope** for B1.

---

## 8. Appendix: file-level change map

Where each feature attaches, by file.

- **A1** — new `dancr/core/graph.py`; new `dancr/headless/_graph.py` (build/query, atomic writes);
  read `understand/__init__.py`, `headless/_context.py`, `headless/__init__.py:connection_map`;
  surface in `cli.py`, `mcp_server.py`, `__init__.py`.
- **A2** — new `dancr/core/events.py`; extend `headless/__init__.py:watch` or add `watch_repo`; reuse
  `_watch_state`, `_changed_files`, `catalog_changes`; surface in `cli.py` and (via B1) the gateway.
- **B1** — new `dancr/gateway.py` and `dancr/core/gateway/{policy,audit,approvals}.py`; reuse
  `mcp_server.py` tools, `headless` locking, `core/secrets.py`, `assistant/client.py` usage; config
  file; CLI `gateway/policy/approvals/audit`.
- **C1** — refactor `dancr/core/rag.py` (retriever interface, fusion, embedder abstraction, BM25,
  rerank, provenance); update `dancr/core/nodes/rag.py` params; `search_knowledge` provenance; index
  versioning.
- **C2** — extend `ask/vocab.py`, `ask/__init__.py`, `recipes/_plan.py`, `answers.py`, `planner.py`,
  `bank.py`, `memory.py`; new cross-project model builder over `_context.py`; `headless`/CLI/MCP
  scope argument.
- **F2** — new `dancr/core/nodes/scenarios.py`; new `dancr/headless` `run_scenarios`; extend
  `verify.py`/`fair/manifest.py` for scenario sets; new `dancr/views/` renderer and `ui/` result page.
- **Cross-cutting** — `core/model.py` (meta keys), `core/engine_hash.py` (fingerprint inputs),
  `headless/_safety.py` (confinement), `headless/_context.py` (exports), `AGENTS.md`, `docs/`.
