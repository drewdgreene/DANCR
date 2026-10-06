"""One atomic text write, shared by every exporter and the project store."""
from __future__ import annotations

import os
import uuid
from pathlib import Path


def write_text_atomic(path: Path | str, text: str, encoding: str = "utf-8") -> Path:
    """Write text to ``path`` atomically: a unique temp file beside it, then one replace. A crash or a full disk
    mid-write leaves the existing file untouched, never a half-written one (the project file and every exported
    table already write this way)."""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        tmp.write_text(text, encoding=encoding)
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)
    return target
