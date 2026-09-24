from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QVBoxLayout, QTextBrowser

DOCS = Path(__file__).resolve().parent.parent.parent / "docs"


class HelpDialog(QDialog):
    def __init__(self, parent=None, section: str = "") -> None:
        super().__init__(parent)
        self.setWindowTitle("DANCR help")
        self.resize(820, 700)
        self.setWindowFlag(Qt.Window)
        self.setAttribute(Qt.WA_DeleteOnClose)     # modeless: free it when closed, do not accumulate windows
        lay = QVBoxLayout(self)
        tb = QTextBrowser(); tb.setOpenExternalLinks(True)
        lay.addWidget(tb)
        name = {"formulas": "formulas.md", "agents": "AGENTS.md"}.get(section, "help.md")
        path = DOCS / name if name != "AGENTS.md" else DOCS.parent / "AGENTS.md"
        try:
            tb.setMarkdown(path.read_text(encoding="utf-8"))
        except OSError:
            tb.setPlainText(f"Help file not found: {path}")


class TourDialog(QDialog):
    """Four short pages shown on first start (Help → Show the tour again)."""

    PAGES = [
        ("Your data is a table", "Open a CSV, Excel or Parquet file and it appears as a table, however big it is. Scroll, find (Ctrl+F), jump to a time, copy cells to Excel."),
        ("Every change is a step", "Right-click a column header to filter, sort, chart, check a limit or make a new column. Each action becomes a step listed on the left. Nothing touches your original file."),
        ("Charts and reports", "Open a chart and use the chips along the top to change it. Press Add to report to collect charts and tables on one page you can send as a PDF."),
        ("Big data", "Small files recompute on every change. For big files DANCR shows a preview and waits until you press Run, then remembers the results so the next run is quick."),
    ]

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_DeleteOnClose)     # modeless: free it when closed
        self.setWindowTitle("Welcome to DANCR"); self.resize(520, 260)
        from PySide6.QtWidgets import QVBoxLayout, QLabel, QHBoxLayout, QPushButton
        self.i = 0
        lay = QVBoxLayout(self); lay.setContentsMargins(24, 20, 24, 16); lay.setSpacing(10)
        self.title = QLabel(""); self.title.setStyleSheet("font-size: 14pt; font-weight: 600;")
        self.body = QLabel(""); self.body.setWordWrap(True); self.body.setStyleSheet("font-size: 11pt;")
        self.dots = QLabel(""); self.dots.setObjectName("faint")
        lay.addWidget(self.title); lay.addWidget(self.body, 1); lay.addWidget(self.dots)
        row = QHBoxLayout()
        skip = QPushButton("Skip"); skip.clicked.connect(self.close)
        self.back = QPushButton("Back"); self.back.clicked.connect(lambda: self._go(-1))
        self.next = QPushButton("Next"); self.next.setObjectName("primary"); self.next.clicked.connect(lambda: self._go(1))
        row.addWidget(skip); row.addStretch(); row.addWidget(self.back); row.addWidget(self.next)
        lay.addLayout(row)
        self._go(0)

    def _go(self, d: int) -> None:
        self.i = max(0, self.i + d)
        if self.i >= len(self.PAGES):
            self.close(); return
        t, b = self.PAGES[self.i]
        self.title.setText(t); self.body.setText(b)
        self.dots.setText("  ".join("●" if k == self.i else "○" for k in range(len(self.PAGES))))
        self.back.setEnabled(self.i > 0)
        self.next.setText("Done" if self.i == len(self.PAGES) - 1 else "Next")
