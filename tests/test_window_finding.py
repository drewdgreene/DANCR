"""The result card: the finding it shows, and the quiet trust badges under it."""
from dancr.core.executor import NodeState
from dancr.ui.finding import FindingCard


def _labels(card) -> list[str]:
    out = []
    for i in range(card.lay.count()):
        w = card.lay.itemAt(i).widget()
        if w is not None and hasattr(w, "text"):
            out.append(w.text())
    return out


def test_card_shows_the_finding_and_trust_badges(app):
    st = NodeState(node_id="s", status="done", rows=1200,
                   report={"finding": {"kind": "change", "statement": "Sales rose 12% in April", "exact": False}})
    card = FindingCard()
    card.set_finding("compare_periods", st, lambda n: n)
    text = " | ".join(_labels(card))
    assert "Sales rose 12% in April" in text
    assert "from a sample" in text and "1,200 rows" in text


def test_card_shows_the_match_rate(app):
    st = NodeState(node_id="c", status="done", rows=400,
                   report={"match_percent": 97.5, "finding": {"kind": "summary", "statement": "97.5% matched", "exact": True}})
    card = FindingCard()
    card.set_finding("combine", st, lambda n: n)
    text = " | ".join(_labels(card))
    assert "98% of keys matched" in text and "exact" in text
