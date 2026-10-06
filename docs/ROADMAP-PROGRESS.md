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
| [0006](adr/0006-gateway-transport.md) | gateway transport (B1) | proposed — transport deferred |
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

Status: not started

## Phase 3 — C2 cross-project QA + F2 scenarios

Status: not started
