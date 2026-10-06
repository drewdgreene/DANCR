"""Deterministic scenario sets (roadmap F2).

A scenario is a named set of Input values. A scenario *set* is one of:

- **named** — an explicit list of ``{"id", "inputs"}``;
- **sweep** — the cartesian product of one or more input ranges (a grid);
- **monte_carlo** — a seeded random sample within input ranges;
- **sensitivity** — one-at-a-time: the base plus each single input varied.

Everything here is deterministic: a sweep is sorted, and Monte Carlo uses an
explicit seed, so the same spec always yields the same scenarios. Running them
lives in :mod:`dancr.headless._scenarios`; the per-scenario plan and output hashes
are folded into an attestation evidence block (``docs/adr/0007``).
"""
from __future__ import annotations

import random
from itertools import product
from typing import Any

SCENARIO_VERSION = 1
MAX_SWEEP = 100_000          # a sweep larger than this needs an explicit limit (it is generated lazily anyway)


def _scenario(sid: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {"id": str(sid), "inputs": {str(k): v for k, v in inputs.items()}}


def named(base: dict[str, Any], items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Explicit named scenarios. An item without ``inputs`` is a flat mapping of inputs."""
    out = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        inputs = dict(base)
        inputs.update(item.get("inputs") if isinstance(item.get("inputs"), dict) else
                      {k: v for k, v in item.items() if k not in ("id", "name")})
        out.append(_scenario(item.get("id") or item.get("name") or f"scenario_{i + 1}", inputs))
    return out


def _axis_values(values: Any) -> list[Any]:
    return list(values) if isinstance(values, (list, tuple)) else [values]


def sweep(base: dict[str, Any], axes: dict[str, Any], *, limit: int | None = None) -> list[dict[str, Any]]:
    """The cartesian product of the axes, deterministic (axes by name, values in order). Generated lazily, so a
    ``limit`` bounds the work; without one, a sweep past ``MAX_SWEEP`` scenarios is refused rather than built."""
    names = sorted(axes)
    value_lists = [_axis_values(axes[name]) for name in names]
    total = 1
    for values in value_lists:
        total *= len(values)
    if limit is None and total > MAX_SWEEP:
        raise ValueError(f"That sweep would make {total:,} scenarios. Pass a 'limit', or use fewer values")
    out: list[dict[str, Any]] = []
    for i, combo_values in enumerate(product(*value_lists)):
        if limit is not None and i >= limit:
            break
        combo = dict(zip(names, combo_values, strict=True))
        out.append(_scenario(name_value(combo), {**base, **combo}))
    return out


def name_value(combo: dict[str, Any]) -> str:
    return "_".join(f"{k}={combo[k]}" for k in sorted(combo)) or "base"


def monte_carlo(base: dict[str, Any], distributions: dict[str, Any], *, n: int = 100, seed: int = 0) -> list[dict[str, Any]]:
    """``n`` samples drawn within each input's range, from an explicit seed, so the set is reproducible.
    A distribution is ``{"min": .., "max": ..}`` (uniform, rounded to 6 dp) or ``{"values": [..]}`` (choice)."""
    rng = random.Random(int(seed))
    out = []
    for i in range(max(0, int(n))):
        inputs = dict(base)
        for name in sorted(distributions):
            spec = distributions[name] or {}
            if isinstance(spec, dict) and spec.get("values"):
                inputs[name] = rng.choice(list(spec["values"]))
            elif isinstance(spec, dict):
                lo, hi = spec.get("min", 0), spec.get("max", 1)
                inputs[name] = round(rng.uniform(float(lo), float(hi)), 6)
            else:
                inputs[name] = spec
        out.append(_scenario(f"mc_{i + 1}", inputs))
    return out


def sensitivity(base: dict[str, Any], ranges: dict[str, Any]) -> list[dict[str, Any]]:
    """One-at-a-time: the base, then each single input varied to each of its other values, the rest at base."""
    out = [_scenario("base", dict(base))]
    for name in sorted(ranges):
        for v in _axis_values(ranges[name]):
            if base.get(name) == v:
                continue
            out.append(_scenario(f"{name}={v}", {**base, name: v}))
    return out


def from_spec(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Build a scenario set from a spec: ``{"named": [...], "sweep": {...}, "monte_carlo": {...},
    "sensitivity": {...}, "base": {...}, "seed": N, "n": N}``. The named form wins when present."""
    if not isinstance(spec, dict):
        raise ValueError("A scenario spec is a JSON object")
    base = dict(spec.get("base") or {})
    if spec.get("scenarios"):
        return named(base, list(spec["scenarios"]))
    if spec.get("named"):
        return named(base, list(spec["named"]))
    if spec.get("sweep"):
        return sweep(base, dict(spec["sweep"]), limit=spec.get("limit"))
    if spec.get("monte_carlo"):
        return monte_carlo(base, dict(spec["monte_carlo"]), n=spec.get("n", 100), seed=spec.get("seed", 0))
    if spec.get("sensitivity"):
        return sensitivity(base, dict(spec["sensitivity"]))
    if base:
        return [_scenario("base", base)]
    raise ValueError("A scenario spec needs one of: scenarios, named, sweep, monte_carlo, sensitivity, base")
