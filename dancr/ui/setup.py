"""The optional setup the person may do: connect the Assistant, wire DANCR into a coding agent, and check
what this build can do.

None of this is required to use DANCR — the window, the command line and the engine work with no key and no
agent. Each dialog writes only to this machine's QSettings, never into a project file.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout,
                               QLabel, QLineEdit, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget)

from ..core.assistant.client import DEFAULT_BASE_URL, DEFAULT_MODEL
from .assistant import SETTING_BASE, SETTING_CONSENT, SETTING_KEY, SETTING_MODEL, SETTING_SAMPLES
from .theme import T
from .icons import icon


# ---------------------------------------------------------------- coding-agent snippets
def mcp_command() -> list[str]:
    """The command that starts DANCR's MCP server in this install. A packaged app uses its `dancr-cli`
    binary; a source checkout runs the same module through the interpreter that is running it now."""
    if getattr(sys, "frozen", False):
        exe = Path(sys.executable)
        name = "dancr-cli.exe" if sys.platform == "win32" else "dancr-cli"
        cli = exe.with_name(name)
        return [str(cli if cli.exists() else exe)]
    return [sys.executable, "-m", "dancr"]


def mcp_snippets(root: str) -> dict[str, dict[str, str]]:
    """A ready-to-paste configuration for each supported MCP client, rooted at ``root``. The command is
    verbatim from the server's own docs, so a wrong entry is the client's, not a guess here."""
    cmd = mcp_command() + ["mcp", "--root", str(Path(root).expanduser().resolve())]
    joined = " ".join(f'"{c}"' if " " in c else c for c in cmd)
    opencode = json.dumps({"$schema": "https://opencode.ai/config.json",
                           "mcp": {"dancr": {"type": "local", "command": cmd, "enabled": True}}}, indent=2)
    desktop = json.dumps({"mcpServers": {"dancr": {"command": cmd[0], "args": cmd[1:]}}}, indent=2)
    return {
        "opencode": {"label": "opencode", "text": opencode,
                     "note": "Put this in opencode.json (a project file, or ~/.config/opencode/opencode.json)."},
        "claude-code": {"label": "Claude Code", "text": f"claude mcp add dancr -- {joined}",
                        "note": "Run this once in a terminal."},
        "claude-desktop": {"label": "Claude Desktop / Cursor", "text": desktop,
                           "note": "Add this to the client's MCP settings JSON."},
    }


class AssistantSetupDialog(QDialog):
    """Add or change the model key, endpoint and model. The key lives in this machine's settings, never in a
    project file. A reply may propose, but only the engine computes the numbers."""

    def __init__(self, parent: QWidget | None = None, *, on_saved: Any = None) -> None:
        super().__init__(parent)
        self._on_saved = on_saved
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle("Connect the Assistant")
        self.resize(560, 380)
        s = QSettings()
        lay = QVBoxLayout(self)
        intro = QLabel("The Assistant turns a plain question into real steps you approve. It needs an "
                       "OpenAI-style model endpoint — Fireworks by default. It can suggest, never compute: "
                       "a number in its reply that no calculation produced is flagged, not trusted.")
        intro.setWordWrap(True); intro.setObjectName("muted")
        lay.addWidget(intro)

        form = QFormLayout(); form.setLabelAlignment(Qt.AlignRight); form.setSpacing(8)
        self.key = QLineEdit(str(s.value(SETTING_KEY, "") or "")); self.key.setEchoMode(QLineEdit.Password)
        self.key.setPlaceholderText("sk-… (stored on this machine)")
        self.base = QLineEdit(str(s.value(SETTING_BASE, "") or "") or DEFAULT_BASE_URL)
        self.model = QLineEdit(str(s.value(SETTING_MODEL, "") or "") or DEFAULT_MODEL)
        form.addRow("API key", self.key)
        form.addRow("Endpoint", self.base)
        form.addRow("Model", self.model)
        lay.addLayout(form)

        self.samples = QCheckBox("Allow sending a few sample rows to the model (off by default)")
        self.samples.setChecked(bool(s.value(SETTING_SAMPLES, False, type=bool)))
        lay.addWidget(self.samples)

        self.consent = QCheckBox("I understand my project profile and questions are sent to this endpoint")
        self.consent.setChecked(bool(s.value(SETTING_CONSENT, False, type=bool)))
        lay.addWidget(self.consent)

        row = QHBoxLayout()
        self.test = QPushButton("Test connection"); self.test.clicked.connect(self._test)
        self.test_status = QLabel(""); self.test_status.setObjectName("muted")
        row.addWidget(self.test); row.addWidget(self.test_status, 1)
        lay.addLayout(row)

        bb = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        bb.accepted.connect(self._save); bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _settings(self):
        from ..core.assistant.client import ModelSettings
        return ModelSettings(base_url=self.base.text().strip() or DEFAULT_BASE_URL,
                             model=self.model.text().strip() or DEFAULT_MODEL,
                             api_key=self.key.text().strip())

    def _test(self) -> None:
        from ..core.assistant.client import Message, OpenAIProvider, ProviderError
        from .workers import Task, alive, view_pool
        settings = self._settings()
        if not settings.configured:
            self.test_status.setText("Add a key first."); return
        self.test.setEnabled(False); self.test_status.setText("Testing…")

        def ping() -> str:
            OpenAIProvider(settings).chat([Message("user", "Reply with the single word: ready")], [], settings)
            return "Connected."

        t = Task(ping); t.waits_for_run = False; t.quick = True
        t.signals.done.connect(lambda msg: (self.test.setEnabled(True), self.test_status.setText(msg)) if alive(self) else None)
        t.signals.failed.connect(lambda msg: (self.test.setEnabled(True),
                                              self.test_status.setText(f"Could not reach it: {msg}")) if alive(self) else None)
        view_pool().start(t)

    def _save(self) -> None:
        s = QSettings()
        s.setValue(SETTING_KEY, self.key.text().strip())
        s.setValue(SETTING_BASE, self.base.text().strip())
        s.setValue(SETTING_MODEL, self.model.text().strip())
        s.setValue(SETTING_SAMPLES, self.samples.isChecked())
        s.setValue(SETTING_CONSENT, self.consent.isChecked())
        if self._on_saved:
            self._on_saved()
        self.accept()


class AgentSetupDialog(QDialog):
    """Wire DANCR into a coding agent over MCP. The root folder is where agents may create projects; nothing
    outside it is reachable for writing."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle("Use DANCR from a coding agent")
        self.resize(680, 520)
        s = QSettings()
        lay = QVBoxLayout(self)
        intro = QLabel("DANCR is an MCP server. Point Claude, opencode or any MCP client at it and the agent can "
                       "build and run pipelines, read results and export — with every call logged, and a policy "
                       "you control. Agents may only create projects inside the root folder below.")
        intro.setWordWrap(True); intro.setObjectName("muted")
        lay.addWidget(intro)

        row = QHBoxLayout()
        self.root = QLineEdit(str(s.value("agents/root", "") or str(Path.home() / "DANCR")))
        browse = QPushButton("Choose…"); browse.clicked.connect(self._browse)
        row.addWidget(QLabel("Root folder")); row.addWidget(self.root, 1); row.addWidget(browse)
        lay.addLayout(row)

        top = QHBoxLayout()
        top.addWidget(QLabel("Configuration for"))
        self.client = QComboBox(); self.client.addItems(["opencode", "Claude Code", "Claude Desktop / Cursor"])
        self.client.currentTextChanged.connect(self._refresh)
        top.addWidget(self.client); top.addStretch()
        lay.addLayout(top)

        self.text = QPlainTextEdit(); self.text.setReadOnly(True); self.text.setObjectName("code")
        self.text.setStyleSheet(f"font-family: monospace; background: {T.bg};")
        lay.addWidget(self.text, 1)
        self.note = QLabel(""); self.note.setObjectName("faint"); self.note.setWordWrap(True)
        lay.addWidget(self.note)

        bottom = QHBoxLayout()
        copy = QPushButton("Copy"); copy.setIcon(icon("copy", T.text, 16)); copy.clicked.connect(self._copy)
        bottom.addWidget(copy); bottom.addStretch()
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(self._close); bb.accepted.connect(self._close)
        bottom.addWidget(bb)
        lay.addLayout(bottom)
        self._refresh()

    def _key(self) -> str:
        return {"opencode": "opencode", "Claude Code": "claude-code",
                "Claude Desktop / Cursor": "claude-desktop"}[self.client.currentText()]

    def _refresh(self) -> None:
        snippet = mcp_snippets(self.root.text().strip() or str(Path.home()))[self._key()]
        self.text.setPlainText(snippet["text"]); self.note.setText(snippet["note"])

    def _browse(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Root folder for agent-made projects", self.root.text() or str(Path.home()))
        if d:
            self.root.setText(d); self._refresh()

    def _copy(self) -> None:
        from PySide6.QtWidgets import QApplication
        QApplication.clipboard().setText(self.text.toPlainText())

    def _close(self) -> None:
        root = self.root.text().strip()
        if root:
            QSettings().setValue("agents/root", root)
        self.accept()


class CapabilitiesDialog(QDialog):
    """What this build can read and do, in plain words; the same report as `dancr doctor`."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle("Check this build")
        self.resize(520, 440)
        lay = QVBoxLayout(self)
        self.body = QLabel("Checking…"); self.body.setWordWrap(True); self.body.setTextFormat(Qt.RichText)
        lay.addWidget(self.body, 1)
        bb = QDialogButtonBox(QDialogButtonBox.Close); bb.rejected.connect(self.accept); bb.accepted.connect(self.accept)
        lay.addWidget(bb)

        from ..core.capabilities import UNLOCKS, capabilities
        from .workers import Task, alive, view_pool
        t = Task(capabilities); t.waits_for_run = False; t.quick = True
        t.signals.done.connect(lambda out: self._show(out) if alive(self) else None)
        t.signals.failed.connect(lambda msg: self.body.setText(f"Could not check the build: {msg}") if alive(self) else None)
        view_pool().start(t)
        self._unlocks = UNLOCKS

    def _show(self, out: dict[str, Any]) -> None:
        parts: list[str] = []
        if out.get("ok"):
            parts.append(f"<p style='color:{T.ok}'>Every reader is present.</p>")
        else:
            parts.append(f"<p style='color:{T.danger}'>Missing: {', '.join(out.get('missing', []))}. "
                         "Reinstall DANCR.</p>")
        parts.append("<ul style='margin-left:0'>")
        for name, ver in sorted(out.get("present", {}).items()):
            parts.append(f"<li>{self._unlocks.get(name, name)} <span style='color:{T.faint}'>({name} {ver})</span></li>")
        doc = out.get("documents") or {}
        if doc.get("present"):
            parts.append(f"<li>reading PDF and Office documents (MinerU {doc.get('version') or 'present'})</li>")
        else:
            parts.append(f"<li style='color:{T.muted}'>PDF and Office reading is off (MinerU not found)</li>")
        parts.append("</ul>")
        self.body.setText("".join(parts))
