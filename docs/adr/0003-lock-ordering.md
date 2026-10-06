# ADR 0003 — Lock ordering for cross-project work

Status: accepted (Phase 0)

## Context

DANCR already serialises writers per project with an OS file lock
(`headless.project_lock`, reentrant per thread). Cross-project features (the
graph, the repo watcher, and in future the gateway) touch many projects and the
repo home at once, so they need a stated order to avoid deadlock.

## Decision

Two lock kinds, acquired in this order:

1. **Repo lock** — `<root>/.dancr/locks/repo.lock`, held while the repo home
   (graph build, event append) is written. At most one writer per repository.
2. **Project locks** — the existing per-project locks, acquired *after* the repo
   lock and, when more than one is needed, in **sorted canonical path order**.

Rules:

- Never take the repo lock while holding a project lock.
- Hold a project lock only for a read-modify-write of that file; a long
  computation runs with the lock released (`editing_deferred`).
- The graph build, the catalog and the watcher are **readers** of projects: they
  compute plan hashes and cache results but do not write project files, so they
  take only the repo lock (for the graph/event write) and no project lock. A
  project save is atomic, so a concurrent reader sees the old or the new file,
  never a partial one.
- `headless.project_locks(paths)` is the helper for the sorted acquisition.

Both `repo_lock` and `project_lock` are reentrant within a thread, so nesting is
safe.

## Consequences

- Deadlock is impossible: the order is total and no cycle can form.
- A repository-wide operation and a per-project edit can still run at the same
  time; they contend only on the repo lock, briefly.
