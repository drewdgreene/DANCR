"""Answers in the window: files in, suggestions out, a click or a question builds one, chips change it."""
import time
from datetime import datetime, timedelta

import polars as pl
import pytest

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QMessageBox

from helpers import pump


@pytest.fixture
def window(app, tmp_path):
    from dancr.ui.mainwindow import MainWindow
    QSettings().setValue("tour_shown", True)
    w = MainWindow()
    w.show(); app.processEvents()
    yield w
    w.doc.stop(wait=True)                     # a run still going would ask "Stop the run and quit?"
    w.doc.undo.setClean()
    w.close()


def until(app, pred, secs=15.0):
    t = time.time()
    while time.time() - t < secs and not pred():
        app.processEvents(); time.sleep(0.01)
    return pred()


@pytest.fixture
def files(tmp_path):
    hours = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(60)]
    pl.DataFrame({"order_id": list(range(1, 61)), "customer_id": [i % 6 + 1 for i in range(60)],
                  "qty": [i % 4 + 1 for i in range(60)], "order_date": hours}).write_csv(tmp_path / "orders.csv")
    pl.DataFrame({"customer_id": list(range(1, 7)), "name": [f"C{i}" for i in range(1, 7)],
                  "region": ["n", "s", "e", "w", "n", "s"]}).write_csv(tmp_path / "customers.csv")
    return [str(tmp_path / "orders.csv"), str(tmp_path / "customers.csv")]


def ready(window, app):
    assert until(app, lambda: window.understanding.full and window.askbar.cards), "no suggestions appeared"


def test_opening_files_offers_answers(window, app, files):
    window._add_files(files)
    loads = [n for n in window.doc.pipeline.nodes.values() if n.type == "load_file"]
    assert len(loads) == 2 and window.doc.undo.count() == 1          # one undo step for the files
    assert loads[1].y > loads[0].y and loads[1].x == loads[0].x        # one under another in the first column
    ready(window, app)
    titles = [c.s.title for c in window.askbar.cards]
    assert "Total qty by region" in titles
    assert window.askbar.isVisible()


def test_a_card_builds_its_answer_and_shows_it(window, app, files):
    window._add_files(files)
    ready(window, app)
    card = next(c for c in window.askbar.cards if c.s.title == "Total qty by region")
    card.chosen.emit(card.s)
    assert until(app, lambda: bool(window.doc.pipeline.answers))
    a = window.doc.pipeline.answers[0]
    assert a.title == "Total qty by region" and a.terminal in window.doc.pipeline.nodes
    assert window._current_answer == a.id and a.id in window.scene.answers and a.id in window.rail._answer_items
    assert window.answer_bar.answer_id == a.id
    assert until(app, lambda: window.doc.state(a.terminal).status == "done")


def test_asking_in_words(window, app, files):
    window._add_files(files)
    ready(window, app)
    window.askbar.edit.setText("top 3 customers by qty"); window.askbar._ask()
    assert until(app, lambda: bool(window.doc.pipeline.answers))
    assert window.doc.pipeline.answers[0].title == "Top 3 customers by total qty"
    window.askbar.edit.setText("qty by colour"); window.askbar._ask()
    assert "colour" in window.askbar.message.text() and len(window.doc.pipeline.answers) == 1


def test_a_chip_changes_the_answer_in_place_and_undo_restores_it(window, app, files):
    window._add_files(files)
    ready(window, app)
    window.askbar.edit.setText("total qty by region"); window.askbar._ask()
    assert until(app, lambda: bool(window.doc.pipeline.answers))
    a = window.doc.pipeline.answers[0]
    nodes = set(window.doc.pipeline.nodes)
    chips = [window.answer_bar.chip_box.itemAt(i).widget().text() for i in range(window.answer_bar.chip_box.count())]
    assert chips[:3] == ["Total", "qty", "by region (customers)"]
    window.change_answer(a.id, "stat", "mean")
    assert until(app, lambda: window.doc.pipeline.answers[0].title == "Average qty by region")
    assert set(window.doc.pipeline.nodes) == nodes                        # the same steps, new settings
    window.doc.undo.undo()
    assert window.doc.pipeline.answers[0].title == "Total qty by region"
    assert window.doc.pipeline.nodes[a.steps["groups"]["node"]].params["default_stats"] == ["sum"]


def test_an_assumption_can_be_changed(window, app, tmp_path):
    for m in (1, 2):
        pl.DataFrame({"date": [datetime(2024, m, d) for d in range(1, 21)], "sales": [float(d) for d in range(1, 21)]}
                     ).write_csv(tmp_path / f"sales_2024-0{m}.csv")
    window._add_files([str(tmp_path / "sales_2024-01.csv"), str(tmp_path / "sales_2024-02.csv")])
    ready(window, app)
    window.askbar.edit.setText("sales per week"); window.askbar._ask()
    assert until(app, lambda: bool(window.doc.pipeline.answers))
    a = window.doc.pipeline.answers[0]
    together = next(x for x in a.assumptions if x["id"] == "together")
    window.change_answer(a.id, "set", together["choices"][0]["set"])
    assert until(app, lambda: window.doc.pipeline.answers[0].spec.get("together") is False)
    assert "stack" not in [window.doc.pipeline.nodes[n].type for n in window.doc.pipeline.upstream_closure(window.doc.pipeline.answers[0].terminal)]


def test_deleting_an_answer_can_keep_or_remove_its_steps(window, app, files):
    window._add_files(files)
    ready(window, app)
    window.askbar.edit.setText("total qty by region"); window.askbar._ask()
    assert until(app, lambda: bool(window.doc.pipeline.answers))
    a = window.doc.pipeline.answers[0]
    built = window.doc.answer_exclusive_nodes(a)
    assert built and not {"load_file_1", "load_file_2"} & set(built)
    before = set(window.doc.pipeline.nodes)
    window.doc.delete_answer(a.id, remove_steps=False)
    assert not window.doc.pipeline.answers and set(window.doc.pipeline.nodes) == before
    window.doc.undo.undo()
    window.doc.delete_answer(a.id, remove_steps=True)
    assert not set(built) & set(window.doc.pipeline.nodes)
    assert len([n for n in window.doc.pipeline.nodes.values() if n.type == "load_file"]) == 2


def test_hand_edits_are_kept_and_said_so(window, app, files):
    window._add_files(files)
    ready(window, app)
    window.askbar.edit.setText("total qty by region"); window.askbar._ask()
    assert until(app, lambda: bool(window.doc.pipeline.answers))
    a = window.doc.pipeline.answers[0]
    order, groups = a.steps["order"]["node"], a.steps["groups"]["node"]
    window.doc.set_params(order, {"descending": False})             # a step the change does not need to touch
    window.change_answer(a.id, "stat", "mean")
    assert until(app, lambda: window.doc.pipeline.answers[0].title == "Average qty by region")
    assert window.doc.pipeline.nodes[order].params["descending"] is False
    assert "Kept your changes" in window.answer_bar.kept.text()
    window.doc.set_params(groups, {"default_stats": ["max"]})       # a step the next change must change
    window.change_answer(a.id, "measure", None)
    assert until(app, lambda: window.doc.pipeline.answers[0].title == "Rows by region")
    assert window.doc.pipeline.nodes[groups].params["default_stats"] == ["max"]
    assert "left as you made it" in window.toast.label.text()


def test_the_same_question_twice_shows_the_answer_already_built(window, app, files):
    window._add_files(files)
    ready(window, app)
    card = next(c for c in window.askbar.cards if c.s.title == "Total qty by region")
    card.chosen.emit(card.s); card.chosen.emit(card.s)                # a double click
    assert until(app, lambda: bool(window.doc.pipeline.answers))
    pump(app, 300)
    window.askbar.edit.setText("total qty by region"); window.askbar._ask()
    pump(app, 300)
    assert len(window.doc.pipeline.answers) == 1


def test_a_build_waiting_for_the_data_is_dropped_when_another_project_opens(window, app, files, tmp_path):
    from dancr.core import Pipeline
    window._add_files(files)
    ready(window, app)
    window.understanding.full = False                               # as if every row were still being read
    window.build_answer({"recipe": "describe", "table": "load_file_1"})
    other = Pipeline("other"); other.save(tmp_path / "other.json")
    window.doc.undo.setClean()
    window.open_path(str(tmp_path / "other.json"))
    pump(app, 800)
    assert window.doc.pipeline.answers == [] and not window.understanding.waiting


def test_a_question_that_cannot_be_answered_says_why(window, app, files):
    window._add_files(files[:1])
    ready(window, app)
    window.build_answer({"recipe": "compare", "table": "load_file_1", "other": "nope", "measure": ["load_file_1", "qty"]})
    assert until(app, lambda: "do not record the same thing" in window.askbar.message.text())
    assert not window.doc.pipeline.answers


def test_selecting_a_step_focuses_the_suggestions(window, app, files):
    window._add_files(files)
    ready(window, app)
    window.understanding.set_focus("load_file_2")
    assert until(app, lambda: "customers" in window.askbar.heading.text())
