"""Recovery copies of unsaved edits: where they live and which ones are dead."""
from __future__ import annotations

import os
import re
from pathlib import Path

from ..core import Pipeline, registry
from ..core.executor import _pid_alive


def _saved_paths(p: Pipeline, nid: str) -> list[str]:
    """The path settings of a step that saves files, as written."""
    nt = registry.get(p.nodes[nid].type)
    if nt.kind != "sink":
        return []
    return [str(v) for prm in nt.params if prm.kind == "path" and (v := p.nodes[nid].params.get(prm.name))]

def recovery_path(pid: int | None = None) -> Path:
    """Where this process keeps a copy of edits that are not in a project file yet (an unsaved project, or
    changes autosave has not written), offered back on the next start if the process dies."""
    from ..logsetup import log_path
    return log_path().parent / f"recovery-{os.getpid() if pid is None else pid}.json"


def dead_recovery_files() -> list[Path]:
    """Recovery copies left by DANCR processes that are no longer running."""
    out = []
    for p in sorted(recovery_path().parent.glob("recovery-*.json")):
        m = re.fullmatch(r"recovery-(\d+)\.json", p.name)
        if m and not _pid_alive(int(m.group(1))):
            out.append(p)
    return out
