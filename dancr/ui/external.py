"""Open a file with the desktop's default app without blocking the GUI thread."""
from __future__ import annotations


def open_external(path: str) -> None:
    import subprocess, sys, os
    try:
        if sys.platform == "win32":
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        pass
