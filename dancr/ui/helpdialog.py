from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QVBoxLayout, QTextBrowser

HELP = Path(__file__).resolve().parent.parent / "help"          # shipped inside the package (pip, Flatpak, PyInstaller)
AGENTS = Path(__file__).resolve().parent.parent.parent / "AGENTS.md"   # next to the package in a checkout or a frozen build


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
        path = AGENTS if section == "agents" else HELP / ("formulas.md" if section == "formulas" else "help.md")
        try:
            tb.setMarkdown(path.read_text(encoding="utf-8"))
        except OSError:
            if section == "agents":
                tb.setMarkdown("The guide for AI agents is AGENTS.md in the DANCR source. `dancr --help`, "
                               "`dancr nodes -v` and `dancr formulas` document every command, step and function.")
            else:
                tb.setPlainText(f"Help file not found: {path}")
