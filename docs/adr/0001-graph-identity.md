# ADR 0001 — Graph vertex identity and versioning

Status: accepted (Phase 0)

## Context

The cross-project entity graph (roadmap A1) needs stable names for projects,
datasets, columns and edges so that a rebuild is reproducible and a diff is
meaningful. The roadmap calls this "the crux" and requires the decision before
storage is written, because migration is expensive to undo.

## Decision

Identity is **derived from what a person sees and controls deliberately**, never
from clock time, memory addresses or a hash of volatile content. All helpers live
in `dancr/core/identity.py`.

- **Project id** — the project file's path relative to the repository root,
  POSIX-spelled (`data/shop.json`). It survives content edits and DANCR upgrades;
  it changes only if the file is moved or renamed inside the repository, which is
  treated as a new identity (documented, testable).
- **Dataset id** — `"<project id>#<node id>"`, matching the existing catalog key
  convention (`headless/_context._catalog_key`). A rename of the step *title*
  does not change it; changing the step *id* does.
- **Column id** — `"column:<normalized name>"`, where normalisation is
  lower-case with runs of spaces/underscores collapsed (the same rule the answer
  engine uses for forgiving column matches).
- **Entity id** — `"value:<normalized column>=<value>"` for a resolved key
  value, deterministic and model-free.
- **Version** — a dataset's version is its **plan hash** from
  `Executor.plan_hash` (folds in the code fingerprint, settings, inputs and
  source content). The graph records the version but identity never depends on
  it.

`IDENTITY_VERSION` is bumped if the naming rule changes; a stored graph records
it and is rebuilt when it differs.

## Consequences

- The graph is a *projection* of hashed project state: always rebuildable,
  never a second source of truth.
- Moving a project inside the repo produces a new identity for its datasets.
  This is the least-invasive, most predictable rule; a stable uuid stored in
  `meta` is a possible future upgrade, but it is not needed to ship A1.
- Because ids are readable, `dancr graph` output is usable without a lookup.
