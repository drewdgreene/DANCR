"""Understanding tables: which numbers are codes, which columns really link two tables, and reading every row in
one pass."""
import random
from pathlib import Path

import polars as pl
import pytest

from dancr.core import Pipeline
from dancr.core.executor import Executor
from dancr.core.understand import Column, DataModel, ID, MEASURE, NUM, Relation, Table, _reorient_links, deepen, looks_like_key, understand


def model_of(tmp_path: Path, run: bool = False, **frames: pl.DataFrame):
    p = Pipeline("u"); p.path = tmp_path / "u.json"
    for name, df in frames.items():
        df.write_csv(tmp_path / f"{name}.csv")
        p.add_node("load_file", title=name, params={"path": f"{name}.csv"}, id=name)
    ex = Executor(p)
    if run:
        ex.run()
    return p, ex, deepen(p, ex, understand(p, ex))


def test_small_counting_ids_with_different_names_are_not_linked(tmp_path):
    _, _, m = model_of(tmp_path,
                       stores=pl.DataFrame({"store_id": list(range(1, 201)), "city": [f"C{i % 30}" for i in range(200)]}),
                       inventory=pl.DataFrame({"stock_id": [i % 150 + 1 for i in range(1000)], "units": [i % 9 for i in range(1000)]}))
    assert not [r for r in m.relations if r.kind == "link"]


def test_the_same_key_under_a_shortened_name_still_links(tmp_path):
    _, _, m = model_of(tmp_path,
                       orders=pl.DataFrame({"cust_id": [i % 20 + 1 for i in range(100)], "amount": [1.0] * 100}),
                       customers=pl.DataFrame({"customer_id": list(range(1, 21)), "name": [f"c{i}" for i in range(20)]}))
    links = [(r.tables, r.left_on, r.right_on, r.cardinality) for r in m.relations if r.kind == "link"]
    assert links == [(["orders", "customers"], "cust_id", "customer_id", "many-to-one")]


def test_codes_written_as_numbers_are_codes_not_amounts(tmp_path):
    n = 60
    _, _, m = model_of(tmp_path, people=pl.DataFrame({
        "zip": [10000 + 7 * i for i in range(n)], "phone": [5550000000 + 13 * i for i in range(n)],
        "account": [40000000 + i for i in range(n)], "ref_code": [3000000 + 17 * i for i in range(n)],
        "salary": [50000 + 311 * i for i in range(n)], "visits": [i % 7 for i in range(n)]}))
    roles = {c.name: c.role for c in m.tables["people"].columns}
    assert roles == {"zip": ID, "phone": ID, "account": ID, "ref_code": ID, "salary": MEASURE, "visits": MEASURE}


def test_amounts_named_after_a_code_word_or_as_wide_as_a_code_stay_amounts(tmp_path):
    n = 60
    _, _, m = model_of(tmp_path, branches=pl.DataFrame({
        "account_balance": [100 + 797 * i for i in range(n)], "phone_calls": [i % 20 for i in range(n)],
        "population": [1_000_000 + 149_993 * i for i in range(n)], "revenue": [2_000_000 + 131_071 * i for i in range(n)],
        "customer_phone": [5550000000 + 13 * i for i in range(n)], "zip_code": [10000 + 7 * i for i in range(n)],
        "account_no": [812 + 3 * i for i in range(n)], "member": [70001000 + 37 * i for i in range(n)]}))
    roles = {c.name: c.role for c in m.tables["branches"].columns}
    assert roles == {"account_balance": MEASURE, "phone_calls": MEASURE, "population": MEASURE, "revenue": MEASURE,
                     "customer_phone": ID, "zip_code": ID, "account_no": ID, "member": ID}


def test_a_link_turned_round_measures_its_match_from_the_new_side():
    def col(name, keys, unique):
        c = Column(name=name, dtype="Int64", kind=NUM, role=ID, distinct=len(keys), unique=unique)
        c._keys = set(keys)
        return c
    a = Table(node="a", title="a", columns=[col("k", [str(i) for i in range(10)], True)])        # the lookup
    b = Table(node="b", title="b", columns=[col("k", [str(i) for i in range(5, 25)], False)])     # half its keys found
    r = Relation(id="link:a.k>b.k", kind="link", tables=["a", "b"], left_on="k", right_on="k", match_pct=50.0)
    m = DataModel(tables={"a": a, "b": b}, relations=[r])
    _reorient_links(m)
    assert r.tables == ["b", "a"] and r.cardinality == "many-to-one" and r.match_pct == 25.0 and "25%" in r.why


def test_every_row_is_read_in_one_pass_per_table_and_one_per_link(tmp_path, monkeypatch):
    import dancr.core.understand as u
    monkeypatch.setattr(u, "SAMPLE_ROWS", 50)              # a sample that is not the whole table
    orders = pl.DataFrame({"order_id": list(range(1, 301)), "customer_id": [i % 30 + 1 for i in range(300)],
                           "amount": [float(i % 11) for i in range(300)]})
    customers = pl.DataFrame({"customer_id": list(range(1, 81)), "region": ["N", "S"] * 40})
    p, ex, _ = model_of(tmp_path, run=True, orders=orders, customers=customers)
    quick = understand(p, ex)
    calls = []
    real = pl.LazyFrame.collect
    monkeypatch.setattr(pl.LazyFrame, "collect", lambda self, *a, **k: calls.append(1) or real(self, *a, **k))
    m = deepen(p, ex, quick)
    links = [r for r in m.relations if r.kind == "link"]
    assert len(links) == 1 and links[0].match_pct == 100.0 and links[0].exact
    assert len(calls) == len(m.tables) + len(links)
    assert m.tables["orders"].rows == 300 and m.tables["orders"].column("order_id").unique_exact


def test_a_perfect_link_between_a_sorted_and_a_shuffled_table_is_found(tmp_path):
    ids = [f"C{i:05d}" for i in range(20_000)]
    shuffled = ids[:]
    random.Random(1).shuffle(shuffled)
    pl.DataFrame({"customer_id": ids, "amount": [1.0] * len(ids)}).write_csv(tmp_path / "orders.csv")
    pl.DataFrame({"customer_id": shuffled, "name": ["x"] * len(ids)}).write_csv(tmp_path / "customers.csv")
    p = Pipeline("t"); p.path = tmp_path / "t.json"
    for f in ("orders.csv", "customers.csv"):
        p.add_node("load_file", params={"path": f})
    links = [r for r in understand(p, Executor(p)).relations if r.kind == "link"]
    assert links and links[0].left_on == links[0].right_on == "customer_id"
    assert links[0].match_pct == 100.0


@pytest.mark.parametrize("name,key", [("customer_id", True), ("CustomerID", True), ("order no", True), ("sku", True),
                                      ("Amount paid", False), ("valid", False), ("Humid", False), ("turkey", False),
                                      ("number", False)])
def test_key_names_are_whole_words(name, key):
    assert looks_like_key(name) is key
