"""Deterministic identity and versioning for cross-project work.

Nothing here depends on clock time, process ids or memory addresses: every name is
derived from things a person sees and changes deliberately, so the same repository
always produces the same ids and a stored graph can be diffed and rebuilt. See
``docs/adr/0001-graph-identity.md``.

The stable surface:

- :func:`project_id` — a project file's path relative to the repository root.
- :func:`dataset_id` — ``"<project id>#<node id>"`` (the catalog key convention).
- :func:`column_id`, :func:`entity_id` — a column, or a resolved key value.
- :func:`key_norm` — the one normalisation used for column names in identities.
- :func:`digest`, :func:`file_digest` — a short, stable hash of a JSON value or a file.

Pure standard library (plus :mod:`dancr.core.dtypes` for JSON safety). No Qt.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .dtypes import json_safe

IDENTITY_VERSION = 1        # bump if a naming rule below changes; a stored graph is rebuilt when it differs
DIVIDER = "#"
_SEP = re.compile(r"[\s_]+")


def key_norm(name: Any) -> str:
    """The normalisation used inside an identity: lower-case, with runs of spaces and underscores collapsed
    to one underscore (the same forgiving match the answer engine uses for column names)."""
    return _SEP.sub("_", str(name or "").strip().lower()).strip("_")


def canonical(obj: Any) -> str:
    """A stable JSON text for hashing: keys sorted, no incidental whitespace, non-JSON values stringified."""
    return json.dumps(json_safe(obj), sort_keys=True, default=str, ensure_ascii=False, separators=(",", ":"))


def digest(obj: Any, n: int = 12) -> str:
    """A short, stable hex digest of any JSON-able value (used for versions and evidence ids)."""
    import hashlib
    return hashlib.sha1(canonical(obj).encode("utf-8")).hexdigest()[:n]


def file_digest(path: Path | str, n: int = 12) -> str:
    """A short content digest of a file, or ``""`` when it is missing or unreadable. Reads in blocks, so a
    large project file costs one pass and constant memory."""
    import hashlib
    p = Path(path)
    h = hashlib.sha1()
    try:
        with open(p, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
    except OSError:
        return ""
    return h.hexdigest()[:n]


def project_id(root: Path | str, project_path: Path | str) -> str:
    """A project's identity: its path relative to the repository root, POSIX-spelled. A project outside the
    root keeps its absolute path, so an id is still unique and readable. Moving or renaming the file inside
    the repository is, by decision, a new identity."""
    root = Path(root).expanduser().resolve()
    path = Path(project_path).expanduser().resolve()
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def dataset_id(project: str, node: str) -> str:
    """A dataset's identity: ``"<project id>#<node id>"`` (matches ``_catalog_key``)."""
    return f"{project}{DIVIDER}{node}"


def split_dataset_id(value: str) -> tuple[str, str]:
    """A dataset id split back into ``(project, node)``; a string with no divider is treated as a bare node."""
    project, sep, node = str(value).partition(DIVIDER)
    return (project, node) if sep else ("", project)


def column_id(name: Any) -> str:
    """A column's identity: ``"column:<normalized name>"``."""
    return f"column:{key_norm(name)}"


def entity_id(column: Any, value: Any) -> str:
    """A resolved key value's identity: ``"value:<normalized column>=<value>"``. Deterministic and model-free."""
    return f"value:{key_norm(column)}={value}"


def source_id(project: str, node: str, path: Any) -> str:
    """A source file's identity within a project: the project, the source step, and the file's name."""
    name = Path(str(path)).name if path else ""
    return f"source:{project}{DIVIDER}{node}{DIVIDER}{name}"


def edge_id(kind: str, left: str, right: str, detail: str = "") -> str:
    """An edge's identity: kind, both endpoints and an optional detail (a key pair), so two different links
    between the same pair of datasets stay distinct."""
    tail = f":{detail}" if detail else ""
    return f"{kind}:{left}>{right}{tail}"
