# The cross-project entity graph (A1)

A repository — the folder holding your DANCR projects — can be indexed once into a
**graph** of everything in it: projects, datasets, columns and sources as
vertices, and every relation the engine can infer as an edge, each carrying its
evidence and confidence. The graph answers questions a single project cannot:
*what relates to what across the repository*, *what joins these two datasets*, and
*which keys link different projects*.

It is derived state, always rebuildable, never a source of truth. See
`docs/adr/0001-graph-identity.md` for the identity rules.

## Commands

```bash
dancr graph build .                      # or any repository folder
dancr --json graph build . --force       # rebuild every project, not only changed ones
dancr graph summary .
dancr graph query . --text orders --kind link
dancr graph neighbors . shop.json#orders
dancr graph path . a.json#orders b.json#customers
dancr graph shared-keys .
```

The same are MCP tools (`graph_build`, `graph_query`, `graph_neighbors`,
`graph_path`, `graph_shared_keys`) and SDK functions (`dancr.build_graph`,
`dancr.graph_query`, `dancr.graph_neighbors`, `dancr.graph_path`,
`dancr.graph_shared_keys`, `dancr.graph_summary`, `dancr.graph_slice`,
`dancr.load_graph`, and the `dancr.Graph` model).

## Storage

Everything lives under `<root>/.dancr/graph/` (see
`docs/adr/0002-repository-storage.md`):

- `graph.db` — a SQLite database written atomically (temp file, then one
  `os.replace`), with tables `projects`, `datasets`, `columns`, `sources`, `edges`
  and `meta`. Rows are inserted in id order, so the same repository writes the
  same file.
- `graph.json` — a small sidecar with the identity/graph versions, the code
  fingerprint, when it was built and which projects were rebuilt or reused.

No new dependency: SQLite is in the standard library.

## Identity and versioning

- A **dataset** is `"<project file relative to root>#<step id>"` (the catalog key
  convention). Moving a project file inside the repository is a new identity.
- A **column** is scoped to its dataset: `"<dataset id>:column:<normalized name>"`.
- A dataset's **version** is its step's plan hash — it changes when the settings,
  the code or the source content change, and it is recorded on every edge endpoint.
- `IDENTITY_VERSION` / `GRAPH_VERSION` are stored in the database; a mismatch
  triggers a full rebuild.

## What the edges mean

| kind | meaning |
|---|---|
| `link` | two datasets joined on a key (`left_on = right_on`), with cardinality and match % |
| `stack` | tables with the same columns that can be row-stacked |
| `align` | two time series that can be aligned reading by reading (with a tolerance) |
| `near` | two sets of coordinates: each row of one can be matched to its nearest place |
| `containment` | points matched to the polygon they fall inside |

Intra-project edges come straight from `dancr.core.understand` (the answer
engine's relation pass). **Cross-project** edges are added deterministically and
**measured**: two datasets in *different* projects whose resolved (or key-like)
columns share a normalized name have their sampled key values compared, and get a
`link` edge with the measured match percentage, the cardinality and a confidence
derived from the overlap; a same-named key with no shared values is not a link. A
capped sample of each key column is stored with the graph, so a project carried
over from a previous build is compared **without being read again**. Still no
model and no silent join.

## Incremental build

A project is carried over from the stored graph without being read again when its
file bytes **and** every source file's stamp (size, modification time, content
sample) are unchanged since the last build, and the code fingerprint matches.
Otherwise it is re-derived. The result is the same whether you build in one pass
or many, and `graph_fingerprint` (projects + datasets + edges) is stable when
nothing moved — the test suite pins both.

## Sensitivity

Confidential and restricted datasets are marked from a project's
`meta["sensitivity"]` or from a `Label sensitivity` step's level. `graph_query`,
`graph_neighbors` and `graph_slice` withhold restricted datasets **and the edges
touching them** unless `allow_restricted` is passed — the same rule
`search_knowledge` uses.

## In the knowledge-base export and FAIR records

`dancr context` (and MCP `profile`, and `dancr.build_context`) folds the graph into
the knowledge-base document: a **`graph` block** with the edges incident to this
project's datasets (each with its evidence), the neighbouring datasets
(`{id, title, shape, project}`), and a per-dataset slice on each `documents[]`
entry, so a `--jsonl` line carries its own relations. The repository is inferred
from the nearest ancestor folder with a built graph (`.dancr/graph/graph.db`); pass
`--root DIR` (CLI) or `root=…` (SDK/MCP) to name it. Ordering is deterministic (by
id). Restricted datasets and the edges touching them are withheld unless
`--allow-restricted`. `--changed` keeps the block and reports `graph_changed` in its
`changes` block. When there is no graph, the block is absent — the export is
unchanged.

The same block reaches the FAIR descriptors (`dancr fair`, `export_fair`):
`dancrGraph` in the schema.org and Frictionless documents, `graph` in the run
manifest, and `dancrGraph` on the RO-Crate root dataset. So a catalogue, a
repository index or a downstream consumer sees how a dataset connects to the rest
of the repository, not just its own schema.

## Locking

The build takes the repository lock (`<root>/.dancr/locks/repo.lock`) while it
writes `graph.db`, so there is one writer per repository. It never takes a project
lock: it only reads project files (a project save is atomic, so a reader sees the
old or the new file, never a partial one) and writes its own cache results under
each project's `.dancr/cache/` through the executor's existing leases. See
`docs/adr/0003-lock-ordering.md`.

## Honest limits

See the graph and retrieval entries in `docs/OPEN.md`. In short: identity is
path-based (a move is a new identity); relations come from a sample; cross-project
overlap is measured from a capped sample of the key values, not over every row, so
it is an estimate and the cardinality is inferred from whether each key is unique
in its own table.
