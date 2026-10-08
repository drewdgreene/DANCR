"""The command palette and the shortcut sheet."""
import polars as pl
from PySide6.QtCore import Qt

from helpers import wait_run


def test_command_palette_lists_actions_steps_and_filters(window, app, tmp_path):
    pl.DataFrame({"a": [1, 2, 3]}).write_csv(tmp_path / "d.csv")
    window._add_load_node(str(tmp_path / "d.csv"), None)
    wait_run(window, app)
    window.open_palette()
    pal = window._palette
    assert pal.isVisible()
    labels = [pal.list.item(i).text() for i in range(pal.list.count())]
    assert any("New project" in t for t in labels)
    assert "Go to step" in labels                       # the project's own steps are searchable too
    pal.search.setText("undo")
    labels = [pal.list.item(i).text() for i in range(pal.list.count())]
    assert any("Undo" in t for t in labels) and not any("New project" in t for t in labels)


def test_command_palette_runs_a_command(window, app, tmp_path):
    pl.DataFrame({"a": [1, 2, 3]}).write_csv(tmp_path / "d.csv")
    window._add_load_node(str(tmp_path / "d.csv"), None)
    wait_run(window, app)
    window.open_palette()
    pal = window._palette
    pal.search.setText("new project")
    item = next(pal.list.item(i) for i in range(pal.list.count()) if pal.list.item(i).data(Qt.UserRole))
    pal._run(item)
    assert not pal.isVisible() and window._entered          # the action really ran


def test_shortcut_sheet_lists_the_bindings(app):
    from dancr.ui.shortcuts import SHORTCUTS, ShortcutDialog
    keys = {k for _, k, _ in SHORTCUTS}
    assert any("Ctrl+Shift+P" in k for k in keys)
    assert any("Ctrl+Shift+J" in k for k in keys)
    dlg = ShortcutDialog()
    assert dlg.windowTitle() == "Keyboard and mouse"
