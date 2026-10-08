"""A one-page keyboard and mouse reference, opened with `?` or from Help."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QScrollArea, QVBoxLayout, QWidget

from .theme import T

# (group, key(s), what) — the same keys the window actually binds, so the sheet cannot drift from the app.
SHORTCUTS: list[tuple[str, str, str]] = [
    ("Files", "Ctrl+N", "New project"),
    ("Files", "Ctrl+O", "Open a project"),
    ("Files", "Ctrl+I", "Open a data file"),
    ("Files", "Ctrl+S  /  Ctrl+Shift+S", "Save / Save as"),
    ("Files", "Ctrl+Q", "Quit"),
    ("Editing", "Ctrl+Shift+P", "Command palette (anything, searchable)"),
    ("Editing", "Ctrl+Z  /  Ctrl+Y", "Undo / Redo"),
    ("Editing", "Ctrl+K  or  Insert", "Add a step"),
    ("Editing", "Delete  or  Backspace", "Delete the selected step"),
    ("Editing", "Ctrl+D", "Duplicate the selected step"),
    ("Editing", "Ctrl+Shift+N", "Add a note to the map"),
    ("Editing", "Ctrl+Shift+I", "Inputs (named values)"),
    ("Asking", "Ctrl+J", "Ask in plain words (rules, no model)"),
    ("Asking", "Ctrl+Shift+J", "Assistant (a model proposes steps)"),
    ("Running", "Ctrl+R  or  F5", "Run everything"),
    ("Running", "Ctrl+Shift+R", "Run up to the selected step"),
    ("Running", "Ctrl+.", "Stop after the current step"),
    ("View", "Ctrl+M", "Show or hide the result panel"),
    ("View", "Ctrl+,", "Show or hide the settings panel"),
    ("View", "Ctrl+0", "Fit the canvas in view"),
    ("View", "Ctrl+=  /  Ctrl+-", "Zoom in / out"),
    ("Table", "Ctrl+F", "Find, or jump to a date or time"),
    ("Table", "Ctrl+C", "Copy the selected cells"),
    ("Canvas", "Drag", "Pan the map"),
    ("Canvas", "Shift- or Ctrl-drag", "Select several steps"),
    ("Canvas", "Middle-drag", "Pan the map"),
    ("Canvas", "Scroll / pinch", "Zoom"),
    ("Help", "F1", "User guide"),
    ("Help", "Ctrl+/", "This sheet"),
    ("Help", "Esc", "Close a bar, or stop the Assistant"),
]


class ShortcutDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle("Keyboard and mouse")
        self.resize(560, 620)
        lay = QVBoxLayout(self)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QScrollArea.NoFrame)
        body = QWidget(); v = QVBoxLayout(body); v.setContentsMargins(4, 4, 4, 4); v.setSpacing(2)
        last = None
        for group, key, what in SHORTCUTS:
            if group != last:
                head = QLabel(group.upper()); head.setObjectName("section")
                head.setContentsMargins(0, 10 if last else 0, 0, 2)
                v.addWidget(head); last = group
            row = QLabel(f"<table width='100%' cellspacing='0'><tr>"
                         f"<td width='42%'><code>{key}</code></td>"
                         f"<td style='color:{T.muted}'>{what}</td></tr></table>")
            row.setTextFormat(Qt.RichText)
            v.addWidget(row)
        v.addStretch(1)
        scroll.setWidget(body)
        lay.addWidget(scroll, 1)
        bb = QDialogButtonBox(QDialogButtonBox.Close); bb.rejected.connect(self.accept); bb.accepted.connect(self.accept)
        lay.addWidget(bb)
