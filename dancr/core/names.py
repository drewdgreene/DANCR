"""How column and table names are read: split into words, normalised, and judged to be a key.

These are pure functions over strings (no data, no registry), so a step that only needs a name
heuristic can import them without pulling in the answer engine (``understand``). ``understand``
re-exports them, so its callers are unchanged.
"""
from __future__ import annotations

import re

# The last word of a key-like name: customer_id, order_no, sku, registry_ref …
KEY_SUFFIXES = ("id", "key", "code", "no", "number", "ref", "sku", "uuid", "guid")


def name_words(name: str) -> list[str]:
    """'CustomerID' -> customer, id; 'order_no' -> order, no; 'Amount paid' -> amount, paid."""
    return [w.lower() for w in re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+", str(name))]


def norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def looks_like_key(name: str) -> bool:
    """A name whose last word is a key word (customer_id, OrderNo, sku), not one that merely ends in those
    letters (Amount paid, valid, Humid). A bare 'id' or 'sku' counts; a bare 'number' or 'no' does not."""
    w = name_words(name)
    if not w or w[-1] not in KEY_SUFFIXES:
        return False
    return len(w) > 1 or w[-1] in ("id", "key", "sku", "uuid", "guid", "code", "ref")
