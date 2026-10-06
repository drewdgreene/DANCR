"""Where a pipeline may write: inside the project folder, never over a file it reads."""
from __future__ import annotations

import logging
import os
from pathlib import Path

from ..core import Pipeline, registry
from ..core.registry import in_dancr_folder


def _path_settings(p: Pipeline, node_id: str) -> list[Path]:
    """The files named by a step's path settings, resolved (symlinks and `..` included)."""
    node = p.nodes[node_id]
    nt = registry.get(node.type)
    out = []
    for prm in nt.params:
        v = node.params.get(prm.name)
        if prm.kind in ("path", "dir") and isinstance(v, str) and v.strip():
            target = Path(v).expanduser()
            out.append((target if target.is_absolute() else p.directory / target).resolve())
    if node.type == "report" and node.params.get("pdf", True):
        out += [t.with_suffix(".pdf") for t in out]           # the report's PDF sits next to it
    return out


def source_files(p: Pipeline) -> set[Path]:
    """Every file a source step of the project reads, including the members of a folder source (so an export or
    a batch never overwrites a file the project reads from)."""
    out: set[Path] = set()
    for nid, n in p.nodes.items():
        nt = registry.get(n.type)
        if nt.kind != "source":
            continue
        out |= set(_path_settings(p, nid))
        if nt.source_files is not None:
            try:
                out |= {Path(f).resolve() for f in nt.source_files(p.directory, n.params)}
            except Exception as e:  # noqa: BLE001 - an unreadable folder contributes nothing; log the cause
                logging.getLogger("dancr.headless").debug("could not list the files of %s: %s", nid, e)
    return out


def output_files(p: Pipeline, node_id: str) -> list[Path]:
    """The files a step saves (nothing for steps that do not save files)."""
    return _path_settings(p, node_id) if registry.get(p.nodes[node_id].type).kind == "sink" else []


def _same_file(a: Path, b: Path) -> bool:
    """One file under two spellings too: A.CSV and a.csv on a Mac or Windows disk, or a hard link."""
    if a == b:
        return True
    try:
        return os.path.samefile(a, b)
    except OSError:                        # either does not exist yet: then they are not the same file
        return False


def unsafe_write(p: Pipeline, target: Path, folder: Path) -> str | None:
    """Why writing `target` is refused (outside `folder`, or over a data file the project reads), else None."""
    target = target.resolve()
    if not target.is_relative_to(folder):
        return f"Can only save inside the project folder {folder}, not {target}"
    if in_dancr_folder(target, folder):
        return f"Won't save into DANCR's own .dancr folder: {target}"
    if any(_same_file(target, src) for src in source_files(p)):
        return f"Won't save over {target}, because the project reads its data from that file"
    return None


def unsafe_outputs(p: Pipeline, node_id: str, folder: Path) -> list[str]:
    """Why a step's files may not be written: each lands outside `folder` or over a file the project reads."""
    return [why for t in output_files(p, node_id) if (why := unsafe_write(p, t, folder))]
