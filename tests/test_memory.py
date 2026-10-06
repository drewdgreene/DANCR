"""Memory: the words a project learns to read, and the questions it remembers."""
from datetime import datetime

import polars as pl

from dancr.core import Pipeline
from dancr.core import memory
from dancr.core.ask import ask
from dancr.core.executor import Executor
from dancr.core.understand import deepen, understand
from dancr import headless as hl


def _pipe(tmp_path):
    rows = [{"time": datetime(2024, m, 5), "region": r, "sales": 100 * m}
            for m in range(1, 5) for r in ("North", "South")]
    df = pl.DataFrame(rows).with_columns(pl.col("time").cast(pl.Datetime("us")))
    df.write_parquet(tmp_path / "in.parquet")
    p = Pipeline("t")
    p.path = tmp_path / "p.json"
    p.add_node("load_file", params={"path": "in.parquet"}, id="src")
    return p


def _model(p):
    ex = Executor(p)
    return deepen(p, ex, understand(p, ex))


def test_alias_round_trip(tmp_path):
    p = _pipe(tmp_path)
    assert memory.aliases(p) == {}
    memory.remember_alias(p, "Zork", "sales")
    assert memory.aliases(p) == {"zork": "sales"}
    memory.remember_alias(p, "sales", "sales")          # identical: ignored
    memory.remember_alias(p, "", "x")                   # blank: ignored
    assert memory.aliases(p) == {"zork": "sales"}


def test_alias_survives_save_and_load(tmp_path):
    p = _pipe(tmp_path)
    memory.remember_alias(p, "zork", "sales")
    p.save()
    assert memory.aliases(Pipeline.load(tmp_path / "p.json")) == {"zork": "sales"}


def test_a_learned_alias_is_understood(tmp_path):
    p = _pipe(tmp_path)
    m = _model(p)
    assert not ask(m, "total zork by region").ok          # unknown before it is learned
    memory.remember_alias(p, "zork", "sales")
    a = ask(m, "total zork by region", aliases=memory.aliases(p))
    assert a.ok and a.spec["recipe"] == "breakdown"
    assert {"from": "zork", "to": "sales"} in a.corrected


def test_recent_questions_dedupe_and_cap(tmp_path):
    p = _pipe(tmp_path)
    for i in range(memory.MAX_RECENT + 4):
        memory.remember_question(p, f"question {i}")
    memory.remember_question(p, "question 3")                 # moves to the front, no duplicate
    r = memory.recent(p)
    assert len(r) == memory.MAX_RECENT and r[0] == "question 3"
    assert r.count("question 3") == 1


def test_aliases_ignore_a_malformed_store():
    """A hand-edited or damaged project must not raise when its learned words are read."""
    from types import SimpleNamespace
    assert memory.aliases(SimpleNamespace(meta={"ask": {"aliases": [1, 2]}})) == {}
    assert memory.aliases(SimpleNamespace(meta={"ask": {"aliases": "oops"}})) == {}


def test_ask_question_learns_a_repair(tmp_path):
    p = _pipe(tmp_path)
    out = hl.ask_question(p, "total saels by region")          # a transposition the lexicon repairs
    assert out["question"]["ok"], out["question"]["message"]
    assert memory.aliases(p).get("saels") == "sales"
    assert memory.recent(p)[0] == "total saels by region"
