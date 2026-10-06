# Roadmap progress

Companion to `docs/ROADMAP-context-and-gateway.md` (the spec). Updated as phases
land. Statuses: done · in progress · not started.

## Baseline

- `pytest -q` before any change: **1410 passed, 2 warnings in 269.64 s**
  (2026-10-06). Warnings are a numpy/xarray binary-size notice in
  `tests/test_connectors.py::test_load_netcdf`; pre-existing.

## Decisions (ADRs)

| # | decision | status |
|---|---|---|
| [0001](adr/0001-graph-identity.md) | graph vertex identity and versioning | accepted |
| [0002](adr/0002-repository-storage.md) | repository `.dancr/` storage layout | accepted |
| [0003](adr/0003-lock-ordering.md) | lock ordering for cross-project work | accepted |
| [0004](adr/0004-retrieval-embedder-policy.md) | retrieval embedder policy (C1) | accepted |
| [0005](adr/0005-cross-project-scope.md) | cross-project QA scope (C2) | accepted |
| [0006](adr/0006-gateway-transport.md) | gateway transport (B1) | accepted — implemented |
| [0007](adr/0007-hashing-attestation-extension.md) | hashing/attestation extension | accepted |

## Phase 0 — foundations

Status: **done**

- [x] ADRs for the six pre-Phase-0 decisions.
- [x] `dancr/core/identity.py` — deterministic identity/versioning helpers.
- [x] `dancr/core/repo.py` — repo storage layout + atomic SQLite/JSONL writes.
- [x] repo lock + sorted project-lock acquisition (`headless.repo_lock`, `project_locks`).
- [x] `build_attestation(extra=…)` extension point; old attestations still verify.
- [x] tests: `tests/test_identity.py`, `tests/test_repo.py`, `tests/test_backcompat.py`.

Verification (Phase 0):

- `ruff check dancr tests` → **All checks passed**
- `pytest -q` → **1432 passed, 2 warnings in 281.97 s** (baseline 1410 + 22 new)
- mypy is advisory and not in CI. The configured target is 3.11 (`pyproject.toml`)
  but the dev venv runs 3.14 and the installed numpy stubs use 3.12+ syntax, so
  `uv run mypy dancr` aborts in a stub before checking any DANCR code
  (pre-existing environment mismatch). Run with `--python-version 3.12`: no
  findings in `core/identity.py`, `core/repo.py` or `core/verify.py`.

Demo:

```bash
python - <<'PY'
from pathlib import Path
import tempfile
from dancr.core.identity import project_id, dataset_id
from dancr.core.repo import Repo, append_jsonl, read_jsonl
root = Path(tempfile.mkdtemp()); (root/"data").mkdir(); pj = root/"data"/"shop.json"; pj.write_text("{}")
print(dataset_id(project_id(root, pj), "orders"))         # data/shop.json#orders
repo = Repo(root).ensure()
append_jsonl(repo.events_log(), {"seq": 1, "event": "source_changed"})
print(list(read_jsonl(repo.events_log())))
PY
```

## Phase 1 — A1 graph + C1 retrieval

Status: **done**

- [x] `dancr/core/graph.py` — graph schema, in-memory model, deterministic queries, SQLite IO.
- [x] `dancr/headless/_graph.py` — build/load/query; incremental carry-over; cross-project key edges.
- [x] CLI `dancr graph build|summary|query|neighbors|path|shared-keys` (+ `search --retriever`).
- [x] MCP tools `graph_build`, `graph_query`, `graph_neighbors`, `graph_path`, `graph_shared_keys`;
      `search_knowledge` gains `retriever`.
- [x] SDK names (`Graph`, `build_graph`, `load_graph`, `graph_summary`, `graph_query`, `graph_neighbors`,
      `graph_path`, `graph_shared_keys`, `graph_slice`).
- [x] Edges carry evidence + confidence; sensitivity withheld by default.
- [x] C1 hybrid retrieval: `lexical` (default, unchanged) / `bm25` / `hybrid` fusion, embedder sidecar,
      provenance on every result.
- [x] tests: `tests/test_graph.py` (9), `tests/test_retrieval.py` (9).

Verification (Phase 1):

- `ruff check dancr tests` → **All checks passed**
- `pytest -q` → **1450 passed, 2 warnings in 348.86 s** (Phase 0's 1432 + 18 new)

Demo:

```bash
dancr graph build /path/to/repo
dancr graph shared-keys /path/to/repo
dancr graph neighbors /path/to/repo data/shop.json#orders
dancr graph path /path/to/repo a/a.json#orders b/b.json#contacts
dancr search shop.json "drought tolerance" --retriever hybrid
```

## Phase 2 — B1 gateway + A2 events

Status: **done**

A2 (events + repository watcher):

- [x] `dancr/core/events.py` — typed, closed event vocabulary; append-only `EventLog` with monotonic `seq`.
- [x] `dancr/headless/_events.py` — `read_events`, `append_event`, `invalidate` (ripple across projects via
      the graph), `watch_repo` (polling; opt-in safe rerun).
- [x] CLI `dancr events` and `dancr watch --repo [--rerun]`; MCP `get_events`; SDK `read_events`,
      `append_event`, `invalidate`, `watch_repo`.
- [x] tests: `tests/test_events.py` (7).

B1 core (policy, approvals, audit, quotas — no socket):

- [x] `dancr/core/gateway/__init__.py` — principals, categories, tiny policy, `evaluate` → allow/approve/deny.
- [x] `dancr/headless/_gateway.py` — `load_policy`/`save_policy`, `policy_check`, `enforce` (audited),
      approval queue, quota store, `audit_records`.
- [x] CLI `dancr policy check|show`, `dancr approvals`, `dancr audit`; MCP `policy_check`; SDK names.
- [x] Off by default: without `<root>/.dancr/gateway/policy.json` nothing is enforced.
- [x] tests: `tests/test_gateway.py` (11).
- [x] **Network transport (`dancr gateway`) — implemented**, fail-closed: loopback-only by default (remote needs
      `--allow-remote`), bearer tokens from the policy file compared in constant time, per-call policy
      allow/deny/approve with audit + quotas, DNS-rebinding protection, and refusal to start without a policy and
      a token. Uses the MCP SDK's streamable-HTTP transport (Starlette/uvicorn ship with `mcp`; no new
      dependency). CLI `dancr gateway`; SDK `gateway_app`, `is_loopback`. Decision recorded in
      `docs/adr/0006-gateway-transport.md`; tests `tests/test_gateway_http.py` (7).

Verification (Phase 2): `ruff check dancr tests` → **All checks passed**;
`pytest -q` → **1475 passed** (Phase 1's 1450 + 25 new: events 7, governance 11, transport 7).
Full suite after Phase 3 (which adds 18) → **1493 passed, 2 warnings in 264.23 s**.

Demo:

```bash
dancr watch /path/to/repo --repo &            # records source_changed / dataset_invalidated events
dancr events /path/to/repo
dancr policy check /path/to/repo alice export_node
dancr approvals /path/to/repo
dancr audit /path/to/repo
```

## Phase 3 — C2 cross-project QA + F2 scenarios

Status: **done**

C2 (structural cross-project QA, no execution):

- [x] `dancr/core/crossask.py` — a deterministic grammar over the graph: `shared_keys`, `path`, `neighbors`,
      `upstream`, `downstream`, `find`; every answer cites edge evidence; ambiguity returns candidates.
- [x] `headless.cross_ask` / `cross_suggest`; CLI `dancr graph ask|suggest`; MCP `graph_ask`/`graph_suggest`;
      SDK `cross_ask`, `cross_suggest`.
- [x] tests: `tests/test_crossask.py` (9).

F2 (scenario sets + proofs):

- [x] `dancr/core/scenarios.py` — deterministic `named`/`sweep`/`monte_carlo`/`sensitivity` sets.
- [x] `dancr/headless/_scenarios.py` — `run_scenarios` (private clone per scenario, confined writes, one output
      each + a combined table, per-scenario plan/output hashes).
- [x] Attestation/verify extended across scenarios through the `extra` evidence block
      (`dancr verify --record|--manifest … --scenarios spec.json`).
- [x] CLI `dancr scenarios`; MCP `run_scenarios`; SDK `run_scenarios`, `scenario_set`.
- [x] tests: `tests/test_scenarios.py` (9).

Verification (Phase 3): `ruff check dancr tests` → **All checks passed**;
`pytest -q` → **1486 passed, 2 warnings in 305.00 s** (Phase 2's 1468 + 18 new).

Demo:

```bash
dancr graph ask /path/to/repo "which keys link projects?"
dancr scenarios shop.json --set cases.json --out-dir results
dancr verify shop.json --record att.json --scenarios cases.json
dancr verify shop.json --manifest att.json --scenarios cases.json
```

## Cleanup pass (after all phases)

A review pass fixed several real bugs and rough edges, each locked with a test:

- **Sensitivity leak:** `Graph.neighbors` returned a restricted dataset's full profile (and its edges) even
  without `allow_restricted`; it now refuses, like `path` does.
- **Crash:** `crossask.suggest` indexed an empty visible-dataset list when every dataset was restricted; guarded.
- **Unbounded work:** `scenarios.sweep` materialised the whole grid; it is now generated lazily and refuses a
  sweep past `MAX_SWEEP` unless a `limit` is given.
- **Bind hole:** `is_loopback("")`/`None` returned True (an empty bind could reach every interface); it is now
  not loopback.
- **Approval scoping:** `consume_approval` matched only principal+tool, so an approval for one project could
  authorise another; it now matches the project too, and reads/consumes under the repository lock (no
  double-spend).
- **Stale cross-project edge:** a *reused* project carried over its old `xlink:` edges, which could survive after
  another project's key changed; cross-project edges are no longer carried (they are regenerated every build).
- **Unsafe scenario filename:** a scenario id was used directly as an output file name, so an id with a path
  separator caused confusing refusals; names are now sanitised and de-duplicated (the logical id is kept in the
  record and the combined column).
- **Edge case:** `limit=0` in `events.read`/`audit_records` returned everything (`[-0:]`); now empty.
- **Robustness:** `load_graph` rebuilds on a damaged or differently-shaped database (catches `KeyError` etc.);
  `Policy.from_dict` raises a clear `ValueError` for a non-object `quotas`; the MCP `run_scenarios` accepts an
  explicit list spec; the gateway runs its policy/audit file I/O off the event loop; `dancr watch --repo --once`
  is refused (a repo watch has no single-shot mode) rather than silently doing nothing.
- Mechanical cleanups (unused imports/locals, `collections.abc`, `zip(strict=)`), verified with a broader
  advisory ruff selection; the repo's configured `ruff check dancr tests` stays clean.

Final suite: **1499 passed, 2 warnings in 269.91 s**.
