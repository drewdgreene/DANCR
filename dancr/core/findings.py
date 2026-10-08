"""Findings: the one thing a step found, said plainly.

Every step may attach a *finding* to its report — a short, honest sentence a person can read without
looking at the table. A finding is data, not prose generated at display time: the step that computed the
number writes the sentence, so it can never drift from what was computed.

    report["finding"] = finding("change", "Sales fell 12% this month (from 1,200 to 1,056), led by North",
                                magnitude=12.4, direction="down", exact=True)

The answer engine, the report and the result card all read findings the same way: they collect them,
rank them by how much they are worth telling, and show the best one as a headline. A step that has
nothing worth saying simply leaves the key out.

Pure core, no Qt.
"""
from __future__ import annotations

from typing import Any, Iterable


def finding(kind: str, statement: str, *, magnitude: float | None = None, direction: str | None = None,
            confidence: float = 1.0, exact: bool = True, **detail: Any) -> dict[str, Any]:
    """The normalized finding a step attaches to its report. ``kind`` groups findings (change, association,
    limit, gap, quality, forecast, fit, share…); ``statement`` is the sentence; ``magnitude`` is the size of
    the effect (for ranking); ``direction`` is up/down/flat; ``exact`` says whether it was computed over every
    row or from a sample. ``detail`` carries whatever the step wants to keep for the UI."""
    return {"kind": str(kind), "statement": " ".join(str(statement).split()), "magnitude": _finite(magnitude),
            "direction": direction, "confidence": float(confidence), "exact": bool(exact), "detail": detail}


def _finite(v: Any) -> float | None:
    try:
        f = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return f if f == f and f not in (float("inf"), float("-inf")) else None


def record(node_type: str, title: str, report: dict[str, Any] | None, status: str = "done") -> dict[str, Any] | None:
    """The finding of one step as a standalone record, or None when it has none (or did not run)."""
    if status != "done":
        return None
    f = (report or {}).get("finding")
    if not isinstance(f, dict) or not str(f.get("statement") or "").strip():
        return None
    return {"node_type": node_type, "title": title, **f}


def collect(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """The findings worth telling, most interesting first. ``items`` are the records ``record`` returns
    (None entries are ignored). Ties keep the order they were given in, so a run is always reported the
    same way."""
    found = [f for f in items if f]
    return sorted(found, key=lambda f: -interest(f))


def interest(f: dict[str, Any]) -> float:
    """How much a finding is worth telling: the size of the effect when there is one, else its confidence.
    A change of 40% outranks a change of 3%; a quality problem outranks a mild correlation."""
    m = f.get("magnitude")
    base = abs(float(m)) if isinstance(m, (int, float)) else 1.0
    bonus = _KIND_WEIGHT.get(str(f.get("kind")), 1.0)
    conf = float(f.get("confidence") or 0.0)
    return base * bonus * max(conf, 0.05)


_KIND_WEIGHT = {"limit": 3.0, "quality": 2.5, "gaps": 2.0, "outliers": 2.0, "groups": 2.0, "change": 1.5, "forecast": 1.3,
                "association": 1.0, "share": 1.0, "fit": 0.9, "summary": 0.5, "rhythm": 0.8}


def headline(findings: list[dict[str, Any]]) -> str:
    """The single best sentence from a run, or an empty string."""
    return str(findings[0]["statement"]) if findings else ""


def fmt_number(v: Any, digits: int = 3) -> str:
    """A number as a person reads it: thousands grouped, about ``digits`` significant figures, never 'nan' and
    never scientific notation for everyday magnitudes."""
    if v is None:
        return "–"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return f"{v:,}"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f != f or f in (float("inf"), float("-inf")):
        return "–"
    if f == 0:
        return "0"
    ax = abs(f)
    if ax >= 1e9 or ax < 1e-3:
        return f"{f:.{digits}g}"
    import math
    decimals = min(max(0, digits - 1 - math.floor(math.log10(ax))), 6)
    s = f"{f:,.{decimals}f}"
    return s.rstrip("0").rstrip(".") if "." in s else s


def fmt_pct(v: Any, digits: int = 1) -> str:
    """A percentage: 12.4% (no trailing .0, no '-0.0%')."""
    n = _finite(v)
    if n is None:
        return "–"
    s = f"{n:.{digits}f}".rstrip("0").rstrip(".")
    if s in ("-0", "-", ""):            # a small negative rounds to zero, which is not "-0%"
        s = "0"
    return f"{s}%"


def plural(n: Any, one: str, many: str | None = None) -> str:
    """'1 row', '2 rows'."""
    try:
        i = int(n)
    except (TypeError, ValueError):
        return many or one
    return one if i == 1 else (many or one + "s")
