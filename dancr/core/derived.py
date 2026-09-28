"""Columns worked out from other columns, found from the numbers alone — and the rows that break the rule.

Spreadsheets are full of columns someone calculated: ``D`` is polygon area minus leaf area, ``m/LA x 10000`` is
mass per leaf area in g/m², ``Total`` is price times quantity. Knowing that is worth two things:

- a row where the rule does not hold is almost always a typing slip (a D of 1.26 where polygon area minus leaf area
  is 54.1), and saying so catches what a summary never shows;
- a column and the columns it is made from are related by definition, so "D rises with polygon area" is not a
  finding.

A small, fixed set of rules is tried — ``a + b``, ``a − b``, ``k·a·b``, ``k·a/b`` and ``k·a`` (a unit change) — on
each number column against the number columns to its left (people calculate to the right of what they
calculate from). A rule counts when it holds, to the precision the column is written in, on nine rows in ten
and on at least five rows. Deterministic, NumPy only, no Qt.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

MAX_COLUMNS = 15            # number columns tried (the first ones)
MAX_ROWS = 5_000            # rows tried (the first ones)
MIN_ROWS = 5
HOLD_SHARE = 0.9
MAX_BREAKS = 5              # more rows than this breaking a rule, and it is not a rule


@dataclass
class Derivation:
    target: str
    op: str                         # sum | diff | product | ratio | scale
    operands: list[str]
    k: float = 1.0
    rows: int = 0                   # rows with every value filled
    holds: int = 0
    breaks: list[dict[str, Any]] = field(default_factory=list)   # {"row" (0-based), "value", "expected"}

    @property
    def formula(self) -> str:
        """The rule in DANCR's formula language: [polygon area] - [leaf area], [m] / [LA] * 10000."""
        a, b = (f"[{c}]" for c in (self.operands + [""])[:2])
        k = _nice(self.k)
        body = {"sum": f"{a} + {b}", "diff": f"{a} - {b}", "product": f"{a} * {b}", "ratio": f"{a} / {b}",
                "scale": a}[self.op]
        return body if k == "1" else f"{body} * {k}"

    @property
    def words(self) -> str:
        """The rule as people say it: polygon area minus leaf area."""
        from .units import header_parts
        a, b = (header_parts(c)[0] if c else "" for c in (self.operands + [""])[:2])
        body = {"sum": f"{a} plus {b}", "diff": f"{a} minus {b}", "product": f"{a} times {b}",
                "ratio": f"{a} divided by {b}", "scale": a}[self.op]
        k = _nice(self.k)
        return body if k == "1" else f"{body}, times {k}"

    def to_dict(self) -> dict[str, Any]:
        return {"target": self.target, "op": self.op, "operands": list(self.operands), "k": self.k,
                "formula": self.formula, "words": self.words, "rows": self.rows, "holds": self.holds,
                "breaks": list(self.breaks)}


def _nice(k: float) -> str:
    if abs(k - round(k)) < 1e-9 * max(1.0, abs(k)):
        return str(int(round(k)))
    return f"{k:.6g}"


def _decimals(v: float) -> int:
    """How many decimals a number is written with (19.1 -> 1, 98.27833572 -> 8), at most 9."""
    s = repr(float(v))
    if "e" in s or "E" in s:
        return 9
    return min(9, len(s.split(".")[1].rstrip("0"))) if "." in s else 0


def find(df: pl.DataFrame, columns: list[str] | None = None) -> list[Derivation]:
    """The columns of ``df`` that are worked out from others by one of the rules, with the rows that break it.
    ``columns`` limits the search (default: every number column, in order)."""
    schema = df.schema                                  # once: rebuilding it per column is slow on wide tables
    cols = [c for c in (columns or df.columns) if c in schema and schema[c].is_numeric()][:MAX_COLUMNS]
    if len(cols) < 2 or df.height < MIN_ROWS:
        return []
    d = df.head(MAX_ROWS).select([pl.col(c).cast(pl.Float64) for c in cols])
    arr = {c: d[c].to_numpy() for c in cols}
    tol = {}
    for c in cols:
        v = arr[c]
        ok = np.isfinite(v)
        dec = np.array([_decimals(x) if f else 0 for x, f in zip(v, ok)])
        tol[c] = 0.5 * 10.0 ** (-dec) + 1e-9
    out: list[Derivation] = []
    for ti, y in enumerate(cols):
        yv = arr[y]
        if _flat(yv):
            continue
        best: Derivation | None = None
        left = cols[:ti]
        for ai, a in enumerate(left):
            if _flat(arr[a]):
                continue
            cands = [("scale", [a])]
            for b in left[ai + 1:]:
                if _flat(arr[b]):
                    continue
                cands += [("sum", [a, b]), ("diff", [a, b]), ("diff", [b, a]), ("product", [a, b]), ("ratio", [a, b]),
                          ("ratio", [b, a])]
            for op, ops in cands:
                dv = _try(op, [arr[o] for o in ops], yv, tol[y], y, ops)
                if dv is not None and (best is None or (dv.holds, -_cost(dv.op)) > (best.holds, -_cost(best.op))):
                    best = dv
        if best is not None:
            out.append(best)
    return out


def _cost(op: str) -> int:
    return {"diff": 0, "sum": 0, "scale": 1, "ratio": 2, "product": 2}[op]


def _flat(v: np.ndarray) -> bool:
    f = v[np.isfinite(v)]
    return f.size < MIN_ROWS or float(np.nanmax(f) - np.nanmin(f)) == 0.0


def _try(op: str, xs: list[np.ndarray], y: np.ndarray, tol: np.ndarray, target: str, ops: list[str]) -> Derivation | None:
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if op == "sum":
            base = xs[0] + xs[1]
        elif op == "diff":
            base = xs[0] - xs[1]
        elif op == "product":
            base = xs[0] * xs[1]
        elif op == "ratio":
            base = xs[0] / xs[1]
        else:
            base = xs[0]
        ok = np.isfinite(base) & np.isfinite(y)
        n = int(ok.sum())
        if n < MIN_ROWS:
            return None
        k = 1.0
        if op in ("product", "ratio", "scale"):
            r = y[ok] / base[ok]
            r = r[np.isfinite(r) & (r != 0)]
            if r.size < MIN_ROWS:
                return None
            k = float(np.median(r))
            if not math.isfinite(k) or k == 0:
                return None
            k = _round_k(k)
            if op == "scale" and abs(k - 1.0) < 1e-12:
                return None                               # the same numbers twice is a copy, not a calculation
        expect = base * k
        close = ok & (np.abs(y - expect) <= tol + 1e-7 * np.abs(expect))
        telling = close & (np.abs(y) > tol)               # rows of zeros agree with any sum or difference
    if int(telling.sum()) < MIN_ROWS:
        return None
    holds = int(close.sum())
    breaks = n - holds
    if holds < MIN_ROWS or holds < HOLD_SHARE * n or breaks > MAX_BREAKS:
        return None
    bad = np.where(ok & ~close)[0]
    return Derivation(target, op, ops, k, n, holds,
                      [{"row": int(i), "value": float(y[i]), "expected": float(expect[i])} for i in bad])


def _round_k(k: float) -> float:
    """A constant as people write it: 10000 rather than 9999.99999997; else as found."""
    mag = 10 ** math.floor(math.log10(abs(k)))
    for step in (1, 0.5, 0.25, 0.1):
        r = round(k / (mag * step)) * mag * step
        if abs(r - k) <= 1e-6 * abs(k):
            return float(r)
    return k
