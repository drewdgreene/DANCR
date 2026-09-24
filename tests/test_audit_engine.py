"""Engine core, the wizard's link finder, versions, samples and undo groups (audit 2026-09-24, A15, A16, E1–E17, W8, W9)."""
import json
import os
import random

import polars as pl
import pytest

from dancr.core import Pipeline
from dancr.core.profile import profile_files, suggest_links, _looks_like_key, match_percent


# ------------------------------------------------------------------ link finder
def test_a_perfect_link_between_a_sorted_and_a_shuffled_table_is_found(tmp_path):
    ids = [f"C{i:05d}" for i in range(20_000)]
    shuffled = ids[:]
    random.Random(1).shuffle(shuffled)
    pl.DataFrame({"customer_id": ids, "amount": [1.0] * len(ids)}).write_csv(tmp_path / "orders.csv")
    pl.DataFrame({"customer_id": shuffled, "name": ["x"] * len(ids)}).write_csv(tmp_path / "customers.csv")
    profs, skipped = profile_files([tmp_path / "orders.csv", tmp_path / "customers.csv"])
    links = suggest_links(profs)
    assert links and links[0].left_col == links[0].right_col == "customer_id"
    assert links[0].match_pct == 100.0
    a, b = (p.column("customer_id") for p in profs)
    assert match_percent(a, b) == 100.0


@pytest.mark.parametrize("name,key", [("customer_id", True), ("CustomerID", True), ("order no", True), ("sku", True),
                                      ("Amount paid", False), ("valid", False), ("Humid", False), ("turkey", False),
                                      ("number", False)])
def test_key_names_are_whole_words(name, key):
    assert _looks_like_key(name) is key


# ------------------------------------------------------------------ versions
def test_saves_in_the_same_second_each_keep_a_version(tmp_path):
    p = Pipeline("v")
    path = tmp_path / "v.json"
    for i in range(4):
        p.meta["i"] = i
        p.save(path)
        os.utime(path, (1_700_000_000, 1_700_000_000))     # the same second every time
    kept = {json.loads(v.read_text())["meta"]["i"] for v in p.versions()}
    assert kept == {0, 1, 2}


# ------------------------------------------------------------------ samples
def test_a_file_named_like_the_sample_is_never_used_or_overwritten(tmp_path):
    from dancr.core.samples import write_sample, SAMPLE_NAME
    theirs = tmp_path / SAMPLE_NAME
    theirs.write_text("mine\n1\n")
    ours = write_sample(tmp_path, rows=3000)
    assert ours != theirs and theirs.read_text() == "mine\n1\n"
    assert write_sample(tmp_path, rows=3000) == ours           # our own sample of that size is reused
    with pytest.raises(ValueError):
        write_sample(tmp_path, rows=1)


def test_synthetic_truth_describes_probe_b(tmp_path):
    from dancr.synth import write_dataset
    t = write_dataset(tmp_path, hours=0.5, rate=20.0, seed=3)
    b = pl.read_csv(tmp_path / "probe_B.csv", try_parse_dates=True)
    assert len(t["gaps_b"]) == len(t["gaps"]) and b.height == t["rows_b"]
    g = t["gaps_b"][0]
    gap = b["time"].diff().dt.total_milliseconds().max() / 1000
    assert gap == pytest.approx(g["seconds"] + 1 / 20.0, abs=0.01)


# ------------------------------------------------------------------ model
def test_topological_order_is_stable_and_linear(tmp_path):
    p = Pipeline("t")
    p.add_node("enter_data", id="a")
    prev = "a"
    for i in range(300):
        p.add_node("sort", id=f"s{i}"); p.connect(prev, f"s{i}"); prev = f"s{i}"
    order = p.topological_order()
    assert order[0] == "a" and order[-1] == "s299" and len(order) == 301
    assert p.upstream_closure("s299") == set(order[:-1])


# ------------------------------------------------------------------ undo groups
@pytest.fixture
def doc(app):
    from dancr.ui.document import Document
    d = Document()
    yield d
    d.undo.setClean()
    d.shutdown()


def test_a_group_that_fails_half_way_leaves_nothing(doc):
    doc.add_node("enter_data", 0, 0)
    doc.undo.setClean()
    before = set(doc.pipeline.nodes)
    with pytest.raises(RuntimeError):
        with doc.macro("Two steps"):
            doc.add_node("sort", 0, 0)
            raise RuntimeError("the second part failed")
    assert set(doc.pipeline.nodes) == before and not doc.dirty


def test_a_group_that_changed_nothing_leaves_no_undo_step(doc):
    doc.undo.setClean()
    doc.duplicate_nodes([])
    with doc.macro("Nothing"):
        pass
    assert not doc.dirty and not doc.undo.canUndo()
