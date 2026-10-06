"""The repository-level ``.dancr/`` home: the graph, the event log and the audit log.

A *repository* is any folder a person points DANCR at (usually the folder above
their projects). Its derived state lives in ``<root>/.dancr/`` and nothing else
writes there. See ``docs/adr/0002-repository-storage.md``.

This module owns the layout and the two write primitives every store shares:

- :func:`write_sqlite_atomic` — build a SQLite database in a temp file, then one
  ``os.replace``, so a reader never sees a half-written database.
- :func:`append_jsonl` / :func:`read_jsonl` — an append-only log of one JSON
  object per line.
- :func:`write_json_atomic` / :func:`read_json` — a small sidecar record.

Pure standard library. No Qt, no network, no new dependency.
"""
from __future__ import annotations

import json
import os
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Callable, Iterator

STORE_DIR = ".dancr"
GRAPH_DIR = "graph"
EVENTS_DIR = "events"
AUDIT_DIR = "audit"
INDEX_DIR = "index"
LOCKS_DIR = "locks"


def _safe_name(name: str) -> str:
    """A single path segment: no separators, no ``.`` or ``..``, so a caller cannot escape the store."""
    s = str(name)
    if not s or s in (".", "..") or "/" in s or "\\" in s or "\x00" in s:
        raise ValueError(f"Not a valid store name: {name!r}")
    return s


class Repo:
    """The derived-state home of one repository. Creating an instance changes nothing; :meth:`ensure` makes
    the folders."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser().resolve()

    @property
    def home(self) -> Path:
        return self.root / STORE_DIR

    def dir(self, name: str) -> Path:
        return self.home / _safe_name(name)

    def path(self, *parts: str) -> Path:
        out = self.home
        for part in parts:
            out = out / _safe_name(part)
        return out

    # ---- known locations
    def graph_db(self) -> Path:
        return self.path(GRAPH_DIR, "graph.db")

    def graph_meta(self) -> Path:
        return self.path(GRAPH_DIR, "graph.json")

    def events_log(self) -> Path:
        return self.path(EVENTS_DIR, "events.jsonl")

    def audit_log(self) -> Path:
        return self.path(AUDIT_DIR, "audit.jsonl")

    def index_dir(self) -> Path:
        return self.path(INDEX_DIR)

    def lock_file(self, name: str) -> Path:
        return self.path(LOCKS_DIR, f"{_safe_name(name)}.lock")

    def ensure(self) -> "Repo":
        for d in (self.home, self.dir(GRAPH_DIR), self.dir(EVENTS_DIR), self.dir(AUDIT_DIR), self.dir(LOCKS_DIR)):
            d.mkdir(parents=True, exist_ok=True)
        return self


def write_sqlite_atomic(path: Path | str, build: Callable[[sqlite3.Connection], None]) -> Path:
    """Build a SQLite database at ``path`` atomically: ``build(conn)`` fills a unique temp file, which is
    then replaced into place. A crash leaves the previous database intact. Returns the path written."""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(tmp)
        conn.execute("PRAGMA journal_mode=DELETE")
        build(conn)
        conn.commit()
        conn.close()
        conn = None
        _fsync_file(tmp)
        os.replace(tmp, target)
        _fsync_dir(target.parent)
    finally:
        if conn is not None:
            conn.close()
        tmp.unlink(missing_ok=True)
    return target


def append_jsonl(path: Path | str, record: dict[str, Any]) -> None:
    """Append one JSON object as a line. A single ``O_APPEND`` write, so concurrent appenders do not
    interleave inside a line on a local filesystem."""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, allow_nan=False, default=str, separators=(",", ":")) + "\n"
    data = line.encode("utf-8")
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)


def read_jsonl(path: Path | str) -> Iterator[dict[str, Any]]:
    """Every readable JSON object in an append-only log, in order. A damaged or blank line is skipped."""
    target = Path(path).expanduser()
    if not target.is_file():
        return
    with open(target, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if isinstance(obj, dict):
                yield obj


def write_json_atomic(path: Path | str, obj: Any) -> Path:
    """Write a small JSON sidecar atomically (temp file, then one replace)."""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        tmp.write_text(json.dumps(obj, ensure_ascii=False, allow_nan=False, default=str, indent=1) + "\n",
                       encoding="utf-8")
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)
    return target


def read_json(path: Path | str) -> Any:
    """The JSON at ``path``, or ``None`` when it is absent or unreadable."""
    target = Path(path).expanduser()
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _fsync_file(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _fsync_dir(path: Path) -> None:
    if os.name == "nt":
        return
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)
