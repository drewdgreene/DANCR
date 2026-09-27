"""Spelling repair: a misspelled word is matched to the project's own words before a question is refused."""
from datetime import datetime

import polars as pl
import pytest

from dancr.core import Pipeline
from dancr.core.ask import ask, vocabulary
from dancr.core.executor import Executor
from dancr.core.lexicon import correct, known_words
from dancr.core.understand import deepen, understand


def _model(tmp_path):
    rows = [{"time": datetime(2024, m, 5), "region": r, "sales": 100 * m}
            for m in range(1, 5) for r in ("North", "South")]
    df = pl.DataFrame(rows).with_columns(pl.col("time").cast(pl.Datetime("us")))
    df.write_parquet(tmp_path / "in.parquet")
    p = Pipeline("t")
    p.path = tmp_path / "p.json"
    p.add_node("load_file", params={"path": "in.parquet"}, id="src")
    ex = Executor(p)
    return deepen(p, ex, understand(p, ex))


def test_corrects_a_clear_typo(tmp_path):
    m = _model(tmp_path)
    vocab = vocabulary(m)
    fixed, fixes = correct(["total", "saels", "by", "regoin"], vocab)
    assert fixed == ["total", "sales", "by", "region"]
    assert dict(fixes) == {"saels": "sales", "regoin": "region"}


def test_leaves_gibberish_alone(tmp_path):
    vocab = vocabulary(_model(tmp_path))
    fixed, fixes = correct(["flibbertigibbet"], vocab)
    assert fixed == ["flibbertigibbet"] and fixes == []


def test_repair_is_deterministic(tmp_path):
    vocab = vocabulary(_model(tmp_path))
    assert correct(["saels", "presure"], vocab) == correct(["saels", "presure"], vocab)


def test_ask_reads_a_misspelled_question(tmp_path):
    m = _model(tmp_path)
    a = ask(m, "total saels by regoin")
    assert a.ok, (a.message, a.unknown)
    assert a.spec["recipe"] == "breakdown" and a.spec["table"] == "src"
    assert {c["from"] for c in a.corrected} == {"saels", "regoin"}


def test_ask_still_refuses_an_unknown_word(tmp_path):
    a = ask(_model(tmp_path), "a total of flibbertigibbet by region")
    assert not a.ok and "flibbertigibbet" in a.unknown


def test_known_words_come_from_the_project(tmp_path):
    words = known_words(vocabulary(_model(tmp_path)))
    assert "region" in words and "sales" in words and words == sorted(words)
