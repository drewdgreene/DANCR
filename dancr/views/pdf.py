"""HTML -> PDF through Qt's text layout (headless). The only place outside dancr/ui that touches Qt."""
from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path


def html_to_pdf(doc: str, out: Path) -> Path:
    """Lay out ``doc`` on A4 pages and write them to ``out``. Qt's text/paint objects belong to the main thread,
    so anything off the main thread — the MCP server's worker threads, or a report node run by the window's run
    thread — writes the PDF in a short-lived helper process instead."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    if threading.current_thread() is not threading.main_thread():
        return _in_helper(doc, Path(out))
    QApplication.instance() or QApplication([])               # a QApplication must exist to lay out text
    return _write(doc, Path(out))


def _write(doc: str, out: Path) -> Path:
    from PySide6.QtGui import QTextDocument, QPdfWriter, QPageSize
    from PySide6.QtCore import QMarginsF
    td = QTextDocument()
    td.setHtml(doc)
    writer = QPdfWriter(str(out))
    writer.setPageSize(QPageSize(QPageSize.A4))
    writer.setPageMargins(QMarginsF(15, 15, 15, 15))
    td.print_(writer)
    return out


def _in_helper(doc: str, out: Path) -> Path:
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    cmd = [sys.executable, "-m", "dancr.views.pdf", str(out)]
    if getattr(sys, "frozen", False):                          # the packaged app: its own binary runs the helper
        cmd = [sys.executable, "pdf-helper", str(out)]
    r = subprocess.run(cmd, input=doc, text=True, encoding="utf-8", capture_output=True, env=env, timeout=300)
    if r.returncode != 0 or not out.exists():
        tail = (r.stderr or "").strip().splitlines()          # a helper killed with empty stderr must not IndexError
        raise RuntimeError(tail[-1] if tail else f"the PDF helper failed (exit {r.returncode})")
    return out


def _main(argv: list[str]) -> int:
    from PySide6.QtWidgets import QApplication
    app = QApplication([])                                      # kept alive (this process's main thread) while laying out
    _write(sys.stdin.read(), Path(argv[0]))
    app.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
