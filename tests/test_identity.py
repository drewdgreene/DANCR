"""Deterministic identity helpers: the same repository always yields the same ids."""
from pathlib import Path

from dancr.core.identity import (IDENTITY_VERSION, column_id, dataset_id, digest, edge_id, entity_id,
                                 file_digest, key_norm, project_id, source_id, split_dataset_id)


def test_identity_version_is_a_positive_int():
    assert isinstance(IDENTITY_VERSION, int) and IDENTITY_VERSION >= 1


def test_key_norm_collapses_spaces_and_underscores():
    assert key_norm("Customer ID") == "customer_id"
    assert key_norm("  order__total ") == "order_total"
    assert key_norm("") == ""
    assert key_norm(None) == ""


def test_digest_is_deterministic_and_key_order_independent():
    assert digest({"a": 1, "b": 2}) == digest({"b": 2, "a": 1})
    assert digest({"a": 1}) == digest({"a": 1})
    assert digest({"a": 1}) != digest({"a": 2})
    assert len(digest({"a": 1}, 8)) == 8


def test_project_id_is_relative_to_the_repo_root(tmp_path):
    root = tmp_path / "repo"
    (root / "data").mkdir(parents=True)
    project = root / "data" / "shop.json"
    project.write_text("{}")
    assert project_id(root, project) == "data/shop.json"


def test_project_id_outside_the_root_is_absolute(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    outside = tmp_path / "other.json"
    outside.write_text("{}")
    assert project_id(root, outside) == outside.resolve().as_posix()


def test_dataset_id_matches_the_catalog_key_convention():
    d = dataset_id("data/shop.json", "orders")
    assert d == "data/shop.json#orders"
    assert split_dataset_id(d) == ("data/shop.json", "orders")
    assert split_dataset_id("orders") == ("", "orders")


def test_column_and_entity_ids():
    assert column_id("Customer ID") == "column:customer_id"
    assert entity_id("Customer ID", 7) == "value:customer_id=7"


def test_source_id_and_edge_id():
    assert source_id("p.json", "src", "data/a.csv") == "source:p.json#src#a.csv"
    assert edge_id("link", "a", "b") == "link:a>b"
    assert edge_id("link", "a", "b", "id=id") != edge_id("link", "a", "b", "cust=id")


def test_file_digest(tmp_path):
    f = tmp_path / "x.json"
    f.write_text("hello")
    first = file_digest(f)
    assert first == file_digest(f)
    assert len(file_digest(f, 8)) == 8
    assert file_digest(tmp_path / "missing.json") == ""
    f.write_text("different")
    assert file_digest(f) != first
