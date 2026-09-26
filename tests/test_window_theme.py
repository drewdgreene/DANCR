"""Switching between light and dark: the window is rebuilt and keeps its state and its controls."""
import os
import sys

import polars as pl
from PySide6.QtCore import QCoreApplication, QEvent, Qt

from helpers import pump, wait_run


def test_appearance_menu_switches_between_light_and_dark(window, app):
    from dancr.ui.theme import T, preference, set_preference, apply_app_style
    old = preference()
    try:
        window._set_theme("dark")
        assert T.dark and T.bg == "#1c1c1f" and window._theme_actions["dark"].isChecked()
        window._set_theme("light")
        assert not T.dark and T.bg == "#f2f2f4" and window._theme_actions["light"].isChecked()
    finally:
        set_preference(old)
        apply_app_style(app)


def test_ui_state_survives_a_theme_rebuild(window, app, sample):
    window._add_load_node(str(sample), None); wait_run(window, app)
    nid = window.current_table()
    state = window.capture_ui_state()
    assert state["current"] == nid

    from dancr.ui.mainwindow import MainWindow
    replacement = MainWindow(doc=window.doc)          # a rebuild carries the same Document over
    replacement.restore_ui_state(state)
    assert replacement._current == nid
    assert replacement.doc is window.doc
    replacement.dispose_for_theme()
    pump(app, 50)
    assert replacement._disposed


def test_a_theme_switch_during_a_run_keeps_stop_working(window, app, tmp_path, monkeypatch):
    from dancr.ui.mainwindow import MainWindow
    monkeypatch.setattr(type(window.doc), "running", property(lambda self: True))
    w2 = MainWindow(doc=window.doc)
    assert w2.a_stop.isEnabled() and not w2.a_run.isEnabled()
    monkeypatch.undo()
    w2.dispose_for_theme()


def test_a_theme_switch_leaves_no_listeners_behind(app, tmp_path, monkeypatch):
    from dancr.ui.app import _rebuild_for_theme
    from dancr.ui.mainwindow import MainWindow
    from dancr.ui.tableview import TableView
    import shiboken6
    errors = []
    monkeypatch.setattr(sys, "excepthook", lambda *e: errors.append(e))
    pl.DataFrame({"v": [1.0, 2.0]}).write_csv(tmp_path / "a.csv")
    w = MainWindow(); w.show()
    w._add_load_node(str(tmp_path / "a.csv"), None)
    ref = [w]
    for _ in range(3):
        _rebuild_for_theme(app, ref)
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        pump(app, 30)
    nid = next(iter(ref[0].doc.pipeline.nodes))
    ref[0].doc.rename(nid, "Renamed")
    ref[0].doc.undo.undo(); ref[0].doc.undo.redo()
    pump(app, 100)
    assert errors == []
    import gc
    gc.collect()
    live = [o for o in gc.get_objects() if isinstance(o, TableView) and shiboken6.isValid(o) and o.doc is ref[0].doc]
    assert len(live) == 1
    ref[0].doc.undo.setClean(); ref[0].close()


def test_gnome_uses_the_portal_theme_unless_one_is_chosen(monkeypatch):
    """Qt's GTK3 theme misses dark mode when the app starts in it; the portal theme does not."""
    from dancr.ui.app import use_desktop_portal
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "ubuntu:GNOME")
    monkeypatch.delenv("QT_QPA_PLATFORMTHEME", raising=False)
    use_desktop_portal()
    assert os.environ["QT_QPA_PLATFORMTHEME"] == "xdgdesktopportal"
    monkeypatch.setenv("QT_QPA_PLATFORMTHEME", "gtk3")                   # chosen by the person: kept
    use_desktop_portal()
    assert os.environ["QT_QPA_PLATFORMTHEME"] == "gtk3"
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "KDE")
    monkeypatch.delenv("QT_QPA_PLATFORMTHEME")
    use_desktop_portal()
    assert "QT_QPA_PLATFORMTHEME" not in os.environ


def test_a_desktop_switch_restyles_once_and_only_when_it_changes(app, monkeypatch):
    from dancr.ui import theme
    monkeypatch.setattr(theme, "preference", lambda: "system")
    tm = theme.theme_manager(app)
    applied = []
    def apply():
        applied.append(theme.is_dark()); theme.T.dark = theme.is_dark()
    monkeypatch.setattr(tm, "apply", apply)
    monkeypatch.setattr(theme.T, "dark", False)
    monkeypatch.setattr(theme, "is_dark", lambda: True)
    tm._on_system_changed(); tm._on_system_changed()     # one switch reported twice restyles once
    assert applied == [True]
