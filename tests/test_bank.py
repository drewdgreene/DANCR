"""The question bank: the answers a project can give, indexed by their words as a fallback behind the grammar.

The grammar stays authoritative, so these tests check two things at once: that the bank finds the right
question (by its canonical phrasing, and by a paraphrase of it), and that it never invents an answer for a
question the grammar could not read.
"""
import polars as pl
import pytest

from dancr.core import Pipeline
from dancr.core.ask import ask
from dancr.core.bank import Bank, MATCH_THRESHOLD
from dancr.core.executor import Executor
from dancr.core.recipes import plan
from dancr.core.understand import understand, deepen


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    """Orders that point at customers: two tables, one link, a measure and two groups."""
    d = tmp_path_factory.mktemp("bank")
    n = 60
    pl.DataFrame({"order_id": list(range(1, n + 1)),
                  "customer_id": [(i % 5) + 1 for i in range(n)],
                  "quantity": [(i % 4) + 1 for i in range(n)],
                  "price": [1.5 + (i % 7) for i in range(n)]}).write_csv(d / "orders.csv")
    pl.DataFrame({"customer_id": [1, 2, 3, 4, 5],
                  "region": ["North", "South", "East", "West", "North"],
                  "segment": ["Retail", "Trade", "Retail", "Trade", "Retail"]}).write_csv(d / "customers.csv")
    p = Pipeline("bank")
    p.path = d / "bank.json"
    p.add_node("load_file", id="orders", title="orders", params={"path": "orders.csv"})
    p.add_node("load_file", id="customers", title="customers", params={"path": "customers.csv"})
    ex = Executor(p)
    return deepen(p, ex, understand(p, ex))


def test_the_bank_lists_answers_for_every_table(model):
    bank = Bank.from_model(model)
    assert bank.entries()
    recipes = {q.recipe for q in bank.entries()}
    assert {"breakdown", "explain", "drivers", "quality", "describe"} <= recipes
    assert {q.table for q in bank.entries()} == {"orders", "customers"}
    for q in bank.entries():
        plan(model, q.spec)                               # every indexed question is buildable


def test_the_exact_canonical_phrasing_is_the_top_match(model):
    bank = Bank.from_model(model)
    top = bank.match("Total quantity by region")[0]
    assert top.question.canonical == "Total quantity by region"
    assert top.question.recipe == "breakdown" and top.question.table == "orders"
    assert top.score == pytest.approx(1.0)
    assert plan(model, top.question.spec)


def test_a_paraphrase_still_matches_the_right_question(model):
    bank = Bank.from_model(model)
    for text in ("quantity by region", "by region total quantity", "total quantity per region"):
        top = bank.match(text)[0]
        assert top.question.canonical == "Total quantity by region", text
        assert top.score >= MATCH_THRESHOLD, text


def test_matching_and_building_are_deterministic(model):
    first, second = Bank.from_model(model), Bank.from_model(model)
    assert [(q.key, q.canonical, q.spec) for q in first.entries()] == \
           [(q.key, q.canonical, q.spec) for q in second.entries()]
    a = [(m.question.key, m.score) for m in first.match("quantity by region")]
    b = [(m.question.key, m.score) for m in second.match("quantity by region")]
    assert a == b


def test_complete_lists_canonical_phrasings_starting_with_the_prefix(model):
    bank = Bank.from_model(model)
    got = bank.complete("total ")
    assert got and all(c.lower().startswith("total ") for c in got)
    assert got == bank.complete("total ")                     # stable, ranked
    assert bank.complete("zzz") == []


def test_ask_falls_back_to_the_bank_when_the_grammar_cannot_read(model):
    a = ask(model, "How the columns of orders move together")
    assert a.ok and a.source == "matched"
    assert a.matched == "How the columns of orders move together"
    assert a.spec == {"table": "orders", "recipe": "drivers"}
    plan(model, a.spec)


def test_ask_still_refuses_a_nonsense_question(model):
    a = ask(model, "flimble wobble")
    assert not a.ok and a.source == "grammar" and a.hints
    assert a.message and "I don't know" in a.message


def test_a_question_the_grammar_can_read_is_not_sent_to_the_bank(model):
    a = ask(model, "total quantity by region")
    assert a.ok and a.source == "grammar" and a.matched == ""
    assert a.title == "Total quantity by region"


def test_a_question_the_grammar_refuses_is_not_forced_onto_the_bank(model):
    a = ask(model, "total quantity by region and segment")
    assert not a.ok and a.source == "grammar"
    assert "one thing at a time" in a.message


def test_a_question_that_omits_the_number_is_not_matched(model):
    assert Bank.from_model(model).match("total by region") == []
    a = ask(model, "total by region")
    assert not a.ok and a.source == "grammar"
    assert "which number" in a.message


def test_scores_stay_between_0_and_1_and_the_question_itself_comes_first():
    from dancr.core.bank import Question
    exact = Question(spec={}, canonical="Quantity region", recipe="describe", table="t", key="exact")
    loud = Question(spec={}, canonical="Loud", recipe="trend", table="t", key="loud")      # ranks first on a tie
    short = Question(spec={}, canonical="Short", recipe="trend", table="t", key="short")
    bank = Bank([exact, loud, short], [(("quantity", "region"),),
                                       (("quantity", "quantity", "region", "region"),),    # says the words twice
                                       (("region",),)])                                   # shorter than the question
    got = bank.match("quantity region")
    assert got[0].question.key == "exact" and got[0].score == 1.0
    assert all(0.0 <= m.score <= 1.0 for m in got)
    assert bank.match("quantity region things stuff")[0].score < MATCH_THRESHOLD   # the question's words must be covered
    assert bank._score(("quantity", "region", "things"), ("region",)) <= 1.0


def test_scores_on_a_real_bank_never_exceed_1(model):
    bank = Bank.from_model(model)
    for text in ("quantity by region", "price per customer", "orders quantity price region segment", "total total quantity"):
        assert all(0.0 <= m.score <= 1.0 for m in bank.match(text, limit=100)), text
