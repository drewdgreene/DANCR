"""The start page and the getting-started tour: the three ways in, the template and example cards, the setup
cards, and the first-run behaviour."""
from PySide6.QtCore import QSettings

from dancr.core.samples import EXAMPLES, TEMPLATES
from helpers import pump


def test_start_page_offers_every_way_in(window):
    for ident in ("new", "open", "import"):
        assert ident in window.start.cards
    for t in TEMPLATES:
        assert t["key"] in window.start.cards
    for e in EXAMPLES:
        assert e["key"] in window.start.cards
    for ident in ("assistant", "agents", "diagnostics", "guide"):
        assert ident in window.start.cards


def test_clicking_new_leaves_the_start_page(window, app):
    window.start.cards["new"].clicked.emit()
    pump(app, 50)
    assert window.doc.pipeline.nodes and not window._on_start_page()


def test_clicking_a_template_card_builds_and_runs(window, app, tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    window.start.cards["compare"].clicked.emit()
    pump(app, 200)
    types = {n.type for n in window.doc.pipeline.nodes.values()}
    assert "load_file" in types and len(types) > 1


def test_setup_cards_reach_the_window(window, app, monkeypatch):
    calls: list[str] = []
    import dancr.ui.setup as setup

    class _Stub:
        def __init__(self, *a, **k):
            pass

        def exec(self):
            calls.append("opened")
            return 0

    monkeypatch.setattr(setup, "AssistantSetupDialog", _Stub)
    monkeypatch.setattr(setup, "AgentSetupDialog", _Stub)
    monkeypatch.setattr(setup, "CapabilitiesDialog", _Stub)
    window.start.cards["assistant"].clicked.emit()
    window.start.cards["agents"].clicked.emit()
    window.start.cards["diagnostics"].clicked.emit()
    assert calls == ["opened", "opened", "opened"]


def test_welcome_banner_visibility(window):
    window.start.set_first_run(True)
    assert not window.start.welcome.isHidden()
    window.start.set_first_run(False)
    assert window.start.welcome.isHidden()


def test_first_run_opens_the_tour_once(app):
    QSettings().setValue("onboarding/seen", False)
    from dancr.ui.mainwindow import MainWindow
    w = MainWindow()
    w.show(); pump(app, 500)
    assert getattr(w, "_onboarding", None) is not None and w._onboarding.isVisible()
    w._onboarding._done()
    assert QSettings().value("onboarding/seen", False, type=bool) is True
    w.doc.stop(wait=True); w.doc.undo.setClean(); w.close(); w.deleteLater()
