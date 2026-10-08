"""The optional setup: the MCP snippets, the capability report, and the Assistant/agent dialogs."""
import json
from pathlib import Path

from PySide6.QtCore import QSettings

from dancr.core.capabilities import capabilities
from dancr.ui.setup import AgentSetupDialog, AssistantSetupDialog, mcp_snippets


def test_mcp_snippets_are_valid_json_and_carry_the_root(tmp_path):
    root = str(tmp_path / "work")
    resolved = str(Path(root).expanduser().resolve())
    snips = mcp_snippets(root)
    assert set(snips) == {"opencode", "claude-code", "claude-desktop"}
    for s in snips.values():
        assert "mcp" in s["text"] and resolved in s["text"]
    op = json.loads(snips["opencode"]["text"])
    assert op["mcp"]["dancr"]["type"] == "local" and op["mcp"]["dancr"]["enabled"] is True
    assert op["mcp"]["dancr"]["command"][-3:] == ["mcp", "--root", resolved]
    desk = json.loads(snips["claude-desktop"]["text"])
    assert desk["mcpServers"]["dancr"]["args"][-3:] == ["mcp", "--root", resolved]
    assert snips["claude-code"]["text"].startswith("claude mcp add dancr -- ")


def test_capabilities_reports_the_build():
    out = capabilities()
    assert out["ok"] is True and out["missing"] == []
    assert {"shapely", "pyproj", "sqlalchemy", "xarray", "h5py"} <= set(out["present"])
    assert "documents" in out


def test_assistant_setup_saves_settings(app):
    from dancr.ui.assistant import SETTING_BASE, SETTING_CONSENT, SETTING_KEY, SETTING_MODEL, SETTING_SAMPLES
    s = QSettings()
    keys = [SETTING_KEY, SETTING_BASE, SETTING_MODEL, SETTING_SAMPLES, SETTING_CONSENT]
    before = {k: s.value(k) for k in keys}
    try:
        dlg = AssistantSetupDialog()
        dlg.key.setText("sk-test"); dlg.base.setText("https://example.test/v1"); dlg.model.setText("m")
        dlg.samples.setChecked(True); dlg.consent.setChecked(True)
        dlg._save()
        assert s.value(SETTING_KEY) == "sk-test" and s.value(SETTING_MODEL) == "m"
        assert s.value(SETTING_SAMPLES, False, type=bool) is True
        assert s.value(SETTING_CONSENT, False, type=bool) is True
    finally:
        for k in keys:
            if before[k] is None:
                s.remove(k)
            else:
                s.setValue(k, before[k])


def test_agent_setup_persists_the_root(app, tmp_path):
    s = QSettings()
    before = s.value("agents/root")
    try:
        dlg = AgentSetupDialog()
        dlg.root.setText(str(tmp_path))
        dlg._close()
        assert s.value("agents/root") == str(tmp_path)
    finally:
        if before is None:
            s.remove("agents/root")
        else:
            s.setValue("agents/root", before)
