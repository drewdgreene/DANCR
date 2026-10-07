"""Read, write and ripple the repository event log (roadmap A2).

The pure log is :mod:`dancr.core.events`; this module is where the repository is
watched. A change to a project file or any of its source files produces
``source_changed`` and ``dataset_invalidated`` events; when a graph exists, the
datasets that depend on a changed one (across projects too) are invalidated as
well. Re-running is **opt-in** and never touches a step that writes outside its
project folder or over its data — the same rule the window follows.
"""
from __future__ import annotations

import logging
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..core import Pipeline
from ..core.events import EventLog, make_event
from ..core.executor import Executor
from ..core.graph import load_graph
from ..core.identity import project_id
from ..core.repo import Repo
from ._context import find_pipelines
from ._safety import unsafe_outputs

log = logging.getLogger("dancr.events")


def _event_log(root: Path | str) -> EventLog:
    return EventLog(Repo(root).ensure().events_log())


def read_events(root: Path | str, *, since: int = 0, type: str | None = None,
                limit: int | None = None) -> dict[str, Any]:
    """The repository's events with ``seq > since``, in order. ``type`` filters to one kind; ``limit`` keeps the
    most recent. Read-only."""
    root = Path(root).expanduser().resolve()
    log = _event_log(root)
    events = log.read(since=since, type=type, limit=limit)
    return {"kind": "dancr.events", "root": str(root), "count": len(events),
            "last_seq": (events[-1]["seq"] if events else log.last_seq()), "events": events}


def append_event(root: Path | str, kind: str, **fields: Any) -> dict[str, Any]:
    """Append one event, holding the repository lock so sequence numbers never collide."""
    from . import repo_lock
    root = Path(root).expanduser().resolve()
    with repo_lock(root):
        return _event_log(root).append(kind, **fields)


def invalidate(root: Path | str, projects: list[str], *, files: Iterable[str] = (),
               graph: Any = None) -> list[dict[str, Any]]:
    """The events a change to ``projects`` (absolute project file paths) produces, without writing them.

    Emits ``source_changed`` per project and ``dataset_invalidated`` per affected dataset; with a graph, a
    dataset whose edge endpoint was invalidated is invalidated too, so a change ripples across projects.
    Deterministic order (projects sorted, datasets by id)."""
    root = Path(root).expanduser().resolve()
    g = graph if graph is not None else load_graph(root)
    events: list[dict[str, Any]] = []
    changed: set[str] = set()
    for f in sorted({str(Path(p).expanduser().resolve()) for p in projects}):
        pid = project_id(root, Path(f))
        events.append(make_event("source_changed", project=pid, files=sorted(str(x) for x in files)[:200]))
        if g is not None:
            for d in sorted((x for x in g.datasets.values() if x.project == pid), key=lambda x: x.id):
                changed.add(d.id)
                events.append(make_event("dataset_invalidated", dataset=d.id, reason="its source changed"))
    if g is not None:
        for e in sorted(g.edges.values(), key=lambda x: x.id):
            if e.left in changed and e.right not in changed:
                events.append(make_event("dataset_invalidated", dataset=e.right, reason=f"upstream {e.left} changed"))
            elif e.right in changed and e.left not in changed:
                events.append(make_event("dataset_invalidated", dataset=e.left, reason=f"upstream {e.right} changed"))
    return events


def _rerun_blocker(pipe: Pipeline) -> str | None:
    """Why a project may not be re-run automatically: a step that would save outside the project folder or over
    a data file it reads (the same confinement the window and MCP enforce)."""
    folder = pipe.directory.resolve()
    for nid in pipe.topological_order():
        bad = unsafe_outputs(pipe, nid, folder)
        if bad:
            return f"{pipe.nodes[nid].title}: {bad[0]}"
    return None


def _index_nodes(pipe: Pipeline) -> list[str]:
    return [nid for nid, n in pipe.nodes.items() if n.type == "build_index"]


def watch_repo(root: Path | str, *, interval: float = 2.0, once: bool = False, rerun: bool = False,
               on_event: Any = None, stop: threading.Event | None = None) -> dict[str, Any]:
    """Watch every project in a repository and its data, and record what changed.

    On a quiet poll after a change: append ``source_changed`` / ``dataset_invalidated`` events (with ripple
    across projects when a graph exists); with ``rerun`` also recompute each changed project that is safe to
    recompute (never one that writes outside its folder or over its data) and append ``project_recomputed``.
    Returns a small record. Long-running, so it has no MCP tool; the CLI is `dancr watch --repo`."""
    from . import repo_lock
    root = Path(root).expanduser().resolve()
    Repo(root).ensure()
    emit = on_event or (lambda e: None)
    stop = stop or threading.Event()
    record: dict[str, Any] = {"kind": "dancr.watch.repo", "root": str(root), "interval": float(interval),
                              "cycles": 0, "changed": [], "ok": True}

    def snapshot() -> dict[str, Any]:
        state: dict[str, Any] = {}
        for f in find_pipelines(root):
            try:
                state[str(f)] = _watch_state(Pipeline.load(f))
            except Exception:  # noqa: BLE001 - an unreadable project is simply not watched
                continue
        return state

    def act(prev: dict[str, Any], snap: dict[str, Any]) -> list[str]:
        changed = sorted({f for f in set(prev) | set(snap) if prev.get(f) != snap.get(f)})
        files: set[str] = set()
        for f in changed:
            a, b = prev.get(f) or {}, snap.get(f) or {}
            files |= {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
        events = invalidate(root, changed, files=files)
        if rerun:
            for f in changed:
                try:
                    pipe = Pipeline.load(f)
                except Exception as e:  # noqa: BLE001
                    events.append(make_event("project_recomputed", project=project_id(root, Path(f)), ok=False, error=str(e)))
                    continue
                blocker = _rerun_blocker(pipe)
                if blocker:
                    events.append(make_event("dataset_invalidated", dataset=project_id(root, Path(f)),
                                             reason=f"not re-run automatically: {blocker}"))
                    continue
                for nid in _index_nodes(pipe):
                    events.append(make_event("index_stale", project=project_id(root, Path(f)), node=nid))
                try:
                    res = Executor(pipe).run()
                    events.append(make_event("project_recomputed", project=project_id(root, Path(f)),
                                             ok=all(s.status == "done" for s in res.values())))
                except Exception as e:  # noqa: BLE001
                    events.append(make_event("project_recomputed", project=project_id(root, Path(f)), ok=False, error=str(e)))
        with repo_lock(root):
            event_log = _event_log(root)
            for ev in events:
                event_log.append(ev["type"], **{k: v for k, v in ev.items() if k not in ("kind", "version", "type")})
        record["cycles"] += 1
        record["changed"] = changed
        for ev in events:
            emit(ev)
        return changed

    emit({"type": "watch_started", "root": str(root)})
    previous = snapshot()
    if once:
        return record
    while not stop.wait(max(0.1, float(interval))):
        snap = snapshot()
        if snap != previous:
            act(previous, snap)
            previous = snap
    record["stopped"] = True
    return record


def _watch_state(pipe: Pipeline) -> dict[str, Any]:
    from . import _watch_state as _ws
    return _ws(pipe)
