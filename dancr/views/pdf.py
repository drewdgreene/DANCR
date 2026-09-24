"""HTML -> PDF through Qt's text layout (headless). The only place outside dancr/ui that touches Qt."""
from __future__ import annotations

import os
from pathlib import Path


def html_to_pdf(doc: str, out: Path) -> Path:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QTextDocument, QPdfWriter, QPageSize
    from PySide6.QtCore import QMarginsF
    QApplication.instance() or QApplication([])               # a QApplication must exist to lay out text
    td = QTextDocument()
    td.setHtml(doc)
    writer = QPdfWriter(str(out))
    writer.setPageSize(QPageSize(QPageSize.A4))
    writer.setPageMargins(QMarginsF(15, 15, 15, 15))
    td.print_(writer)
    return out
