"""The Assistant in the window: the dock tab, a build request, and the engine doing the work."""
import polars as pl
from PySide6.QtWidgets import QPushButton

from helpers import settle, wait_run
from dancr.core.assistant.client import ChatResult, FakeProvider, Message, ToolCall, Usage
from dancr.ui.assistant import AssistantCard, ChoiceCard, ChoiceRow, ProposalCard


def _call(name, **a):
    return ChatResult(Message("assistant", "", [ToolCall(f"c_{name}", name, a)]), Usage(3, 1, 4, 0), "tool_calls")


def _say(text):
    return ChatResult(Message("assistant", text), Usage(3, 2, 5, 0), "stop")


def _ready(window, app):
    settle(app, lambda: window.understanding.full and window.understanding.model is not None, 30)


def _load(window, app, tmp_path):
    pl.DataFrame({"region": ["North", "South", "North", "East"],
                  "amount": [10.0, 20.0, 30.0, 40.0]}).write_csv(tmp_path / "orders.csv")
    window._add_load_node(str(tmp_path / "orders.csv"), None)
    wait_run(window, app)
    _ready(window, app)
    return window.current_table()


def test_assistant_tab_shows_in_the_dock(window, app, tmp_path):
    _load(window, app, tmp_path)
    window.a_assistant.setChecked(True)
    window._apply_side_panels()
    assert window.side.isVisible() and window.side.stack.currentWidget() is window.assistant
    window.a_assistant.setChecked(False)
    window._apply_side_panels()
    assert window.side.stack.currentWidget() is window.inspector


def test_the_assistant_button_opens_and_closes_the_chat(window, app, tmp_path):
    # clicking the toolbar button triggers the action; it must reveal the Assistant, not cancel itself out
    _load(window, app, tmp_path)
    window.a_assistant.setChecked(False); window._apply_side_panels()
    window.a_assistant.trigger()
    assert window.a_assistant.isChecked()
    assert window.side.isVisible() and window.side.stack.currentWidget() is window.assistant
    window.a_assistant.trigger()
    assert not window.a_assistant.isChecked()
    assert window.side.stack.currentWidget() is window.inspector


def test_send_without_a_key_offers_setup(window, app, tmp_path, monkeypatch):
    _load(window, app, tmp_path)
    monkeypatch.delenv("DANCR_ASSISTANT_API_KEY", raising=False)
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    from dancr.ui.assistant import SETTING_KEY, SETTING_BASE, SETTING_MODEL
    s = window.settings
    for k in (SETTING_KEY, SETTING_BASE, SETTING_MODEL):
        s.remove(k)
    panel = window.assistant
    panel.edit.setPlainText("total amount by region")
    panel.send()
    buttons = [b.text() for b in panel.body.findChildren(QPushButton)]
    assert "Assistant settings…" in buttons                 # a clear way forward, not a doomed request
    assert not panel._busy and not panel._thread.turns


def test_command_hint_lists_matches(window, app, tmp_path):
    _load(window, app, tmp_path)
    panel = window.assistant
    panel.edit.setPlainText("/pro")
    assert panel.cmd_hint.isVisibleTo(panel) and "/profile" in panel.cmd_hint.text()
    panel.edit.setPlainText("hello")
    assert not panel.cmd_hint.isVisibleTo(panel)


def test_assistant_builds_and_runs_a_proposed_answer(window, app, tmp_path):
    src = _load(window, app, tmp_path)
    spec = {"recipe": "breakdown", "table": src, "measure": [src, "amount"], "stat": "sum", "by": [src, "region"]}
    window.assistant.set_provider(FakeProvider([
        _call("read_question", text="total amount by region"),
        _call("propose", reply="Total amount by region.", answer=spec),
    ]))
    window.a_assistant.setChecked(True); window._apply_side_panels()
    window.assistant.edit.setPlainText("total amount by region")
    window.assistant.send()
    settle(app, lambda: window.assistant._pending_card is not None, 20)
    card = window.assistant._pending_card
    assert card.proposal["kind"] == "answer" and "region" in str(card.proposal.get("title", ""))
    window.assistant._build(card)
    wait_run(window, app)
    assert len(window.doc.pipeline.answers) == 1
    a = window.doc.pipeline.answers[0]
    assert window.doc.state(a.terminal).status == "done"
    # the finding was handed back to the card, and the thread was recorded
    assert window.doc.pipeline.meta.get("assistant", {}).get("turns")
    # the steps are real, editable nodes
    assert a.terminal in window.doc.pipeline.nodes


def test_assistant_shows_choices_and_sends_the_pick(window, app, tmp_path):
    _load(window, app, tmp_path)
    window.assistant.set_provider(FakeProvider([
        _call("ask_choice", question="Which key links the tables?", options=["customer_id = id", "neither"]),
        _say("Understood."),
    ]))
    window.a_assistant.setChecked(True); window._apply_side_panels()
    panel = window.assistant
    panel.edit.setPlainText("how do they connect?"); panel.send()
    settle(app, lambda: panel.body.findChildren(ChoiceCard), 20)
    card = panel.body.findChildren(ChoiceCard)[0]
    options = card.findChildren(ChoiceRow)
    assert len(options) == 2
    before = len(panel.body.findChildren(AssistantCard))
    options[0].clicked.emit()
    settle(app, lambda: len(panel.body.findChildren(AssistantCard)) > before, 20)
    # the pick was sent as the next user turn
    assert any(t.role == "user" and "customer_id = id" in t.text for t in panel._thread.turns)


def test_slash_commands(window, app, tmp_path):
    _load(window, app, tmp_path)
    p = window.assistant
    text, done = p._command("/profile")
    assert done is None and "Profile every table" in text
    text, done = p._command("/connections")
    assert done is None and "how these tables relate" in text
    text, done = p._command("/build total amount by region")
    assert done is None and text == "total amount by region"
    assert p._command("/undo")[1] == "done"
    assert p._command("/nonsense")[1] == "done"


def test_dropping_a_table_offers_to_profile(window, app, tmp_path):
    _load(window, app, tmp_path)
    window.a_assistant.setChecked(True); window._apply_side_panels()
    panel = window.assistant
    before = len(panel.body.findChildren(AssistantCard))
    pl.DataFrame({"x": [1, 2, 3]}).write_csv(tmp_path / "extra.csv")
    window._add_load_node(str(tmp_path / "extra.csv"), None)
    settle(app, lambda: len(panel.body.findChildren(AssistantCard)) > before, 20)


def test_building_stays_in_the_assistant_tab(window, app, tmp_path):
    src = _load(window, app, tmp_path)
    # open the Assistant the way the person does: by clicking the dock tab, not the toolbar action
    window.side.set_assistant(True)
    assert window.a_assistant.isChecked() and window.side.stack.currentWidget() is window.assistant
    spec = {"recipe": "breakdown", "table": src, "measure": [src, "amount"], "stat": "sum", "by": [src, "region"]}
    window.assistant.set_provider(FakeProvider([_call("propose", reply="ok", answer=spec)]))
    window.assistant.edit.setPlainText("total by region"); window.assistant.send()
    settle(app, lambda: window.assistant._pending_card is not None, 20)
    window.assistant._build(window.assistant._pending_card)
    wait_run(window, app)
    # neither building nor the run may move the side column off the Assistant
    assert window.side.stack.currentWidget() is window.assistant and window.a_assistant.isChecked()


def test_assistant_applies_edits_undoably(window, app, tmp_path):
    src = _load(window, app, tmp_path)
    window.assistant.set_provider(FakeProvider([
        _call("propose_edits", reply="Friendlier names.", edits=[
            {"op": "rename", "node": src, "title": "Sales orders"},
        ]),
    ]))
    window.a_assistant.setChecked(True); window._apply_side_panels()
    panel = window.assistant
    panel.edit.setPlainText("name the steps plainly"); panel.send()
    settle(app, lambda: panel._pending_edits_card is not None, 20)
    assert window.doc.pipeline.nodes[src].title != "Sales orders"
    panel._apply_edits(panel._pending_edits_card)
    assert window.doc.pipeline.nodes[src].title == "Sales orders"
    window.doc.undo.undo()
    assert window.doc.pipeline.nodes[src].title != "Sales orders"


def test_thread_is_saved_into_the_project_and_undoes(window, app, tmp_path):
    _load(window, app, tmp_path)
    from dancr.core.assistant.store import Thread, Turn
    window.doc.set_thread(Thread(turns=[Turn("user", "hi")]).to_dict())
    assert window.doc.pipeline.meta.get("assistant") is not None
    window.doc.undo.undo()
    assert window.doc.pipeline.meta.get("assistant") is None


def test_assistant_replace_canvas_is_undoable(window, app, tmp_path):
    src = _load(window, app, tmp_path)
    pl.DataFrame({"foo": [1, 2], "bar": ["x", "y"]}).write_csv(tmp_path / "other.csv")
    window._add_load_node(str(tmp_path / "other.csv"), None)
    wait_run(window, app)
    other = window.current_table()
    wait_run(window, app)
    spec = {"recipe": "breakdown", "table": src, "measure": [src, "amount"], "stat": "sum", "by": [src, "region"]}
    window.assistant.set_provider(FakeProvider([_call("propose", reply="ok", answer=spec)]))
    window.a_assistant.setChecked(True); window._apply_side_panels()
    panel = window.assistant
    panel.edit.setPlainText("total by region"); panel.send()
    settle(app, lambda: panel._pending_card is not None, 20)
    panel._build(panel._pending_card)
    wait_run(window, app)
    a = window.doc.pipeline.answers[0]
    window.assistant_replace_canvas(a.terminal, confirm=False)
    wait_run(window, app)
    assert other not in window.doc.pipeline.nodes          # the unrelated table is gone
    assert a.terminal in window.doc.pipeline.nodes         # the answer's steps stay
    window.doc.undo.undo()
    assert other in window.doc.pipeline.nodes              # Ctrl+Z brings it all back


def test_a_built_plan_stays_built_after_a_reload(window, app, tmp_path):
    src = _load(window, app, tmp_path)
    spec = {"recipe": "breakdown", "table": src, "measure": [src, "amount"], "stat": "sum", "by": [src, "region"]}
    panel = window.assistant
    panel.set_provider(FakeProvider([_call("propose", reply="ok", answer=spec)]))
    window.a_assistant.setChecked(True); window._apply_side_panels()
    panel.edit.setPlainText("total by region"); panel.send()
    settle(app, lambda: panel._pending_card is not None, 20)
    panel._build(panel._pending_card)
    wait_run(window, app)
    from dancr.core.assistant.store import load_thread
    turn = load_thread(window.doc.pipeline).turns[-1]
    assert turn.node and turn.answer                       # the outcome was folded into the thread
    panel._reload_cards()                                  # what reopening a project does
    cards = panel.body.findChildren(ProposalCard)
    assert cards and cards[-1].build_btn.isHidden()        # shown as built, not offered to build again


def test_cost_meter_shows_the_session(window, app, tmp_path):
    _load(window, app, tmp_path)
    window.assistant.set_provider(FakeProvider([_say("hi")]))
    window.assistant.edit.setPlainText("hello"); window.assistant.send()
    settle(app, lambda: not window.assistant._busy, 20)
    assert "tokens" in window.assistant.cost.text()


def test_empty_state_gives_a_start(window, app, tmp_path):
    _load(window, app, tmp_path)
    panel = window.assistant
    assert panel.empty.isVisibleTo(panel) and not panel.scroll.isVisibleTo(panel)
    panel.set_provider(FakeProvider([_say("Hello.")]))
    panel.edit.setPlainText("hi"); panel.send()
    settle(app, lambda: not panel._busy, 20)
    assert not panel.empty.isVisibleTo(panel) and panel.scroll.isVisibleTo(panel)


def test_history_recall(window, app, tmp_path):
    _load(window, app, tmp_path)
    panel = window.assistant
    panel._history = ["first question", "second question"]; panel._hist = -1
    panel._history_step(-1); assert panel.edit.toPlainText() == "second question"
    panel._history_step(-1); assert panel.edit.toPlainText() == "first question"
    panel._history_step(1); assert panel.edit.toPlainText() == "second question"
    panel._history_step(1); assert panel.edit.toPlainText() == ""


def test_connection_map_card(window, app, tmp_path):
    _load(window, app, tmp_path)
    panel = window.assistant
    panel._thread.connections = [{"kind": "link", "tables": ["a", "b"], "left_on": "x", "right_on": "id",
                                  "match_pct": 94.0, "cardinality": "many-to-one"}]
    before = len(panel.body.findChildren(AssistantCard))
    panel.show_connections()
    assert len(panel.body.findChildren(AssistantCard)) == before + 1


def test_regenerate_replaces_the_reply_without_duplicating_the_question(window, app, tmp_path):
    _load(window, app, tmp_path)
    panel = window.assistant
    panel.set_provider(FakeProvider([_say("First answer."), _say("Second answer.")]))
    panel.edit.setPlainText("what is this?"); panel.send()
    settle(app, lambda: not panel._busy, 20)
    assert panel._thread.turns[-1].text == "First answer."
    assert len([t for t in panel._thread.turns if t.role == "user"]) == 1
    panel._regenerate()
    settle(app, lambda: not panel._busy and panel._thread.turns[-1].text == "Second answer.", 20)
    assert panel._thread.turns[-1].text == "Second answer."
    assert len([t for t in panel._thread.turns if t.role == "user"]) == 1      # the question is kept, once


def test_assistant_text_reply_changes_nothing(window, app, tmp_path):
    _load(window, app, tmp_path)
    before = set(window.doc.pipeline.nodes)
    window.assistant.set_provider(FakeProvider([_say("I can total a column by a group.")]))
    window.a_assistant.setChecked(True); window._apply_side_panels()
    window.assistant.edit.setPlainText("what can you do")
    window.assistant.send()
    settle(app, lambda: window.assistant._pending_card is None and not window.assistant._busy, 20)
    assert set(window.doc.pipeline.nodes) == before
    assert not window.doc.pipeline.answers
