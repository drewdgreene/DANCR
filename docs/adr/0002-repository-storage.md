# ADR 0002 — Repository-level storage layout

Status: accepted (Phase 0)

## Context

The roadmap's cross-cutting workstreams ask for a standard repository-level
`.dancr/` home for the graph, the event log, the audit log and (future) indexes,
preferring the standard library and atomic writes.

## Decision

A repository root gets a `.dancr/` folder holding:

```
<root>/.dancr/
  graph/graph.db        SQLite (stdlib sqlite3), rebuilt atomically
  events/events.jsonl   append-only, one JSON object per line
  audit/audit.jsonl     append-only, one JSON object per line
  index/                reserved for future repo-level indexes
  locks/<name>.lock     OS file locks (repo coordination)
```

`dancr/core/repo.py` defines the layout (`Repo`) and the two write primitives
(`write_sqlite_atomic`, `append_jsonl`). Writes are atomic: a SQLite build goes
to a unique temp file and is `os.replace`d into place, so a reader never sees a
half-written database; JSONL appends are single `os.write` calls of one line
under the repo lock.

Only DANCR writes inside a `.dancr` folder — the existing confinement rule
(`in_dancr_folder`) applies to the repo home as it does to a project's own
`.dancr` folder. A project's own `.dancr/` (cache, versions, locks) is unchanged.

## Consequences

- No new dependency (SQLite is stdlib). `uv sync` without extras is unaffected.
- The repo home is separate from each project's `.dancr/`, so the existing
  "never delete a project's results" rules are untouched.
- The store is disposable: deleting `<root>/.dancr/` loses only derived state.
