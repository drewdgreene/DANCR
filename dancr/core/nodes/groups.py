"""Compare groups: are the numbers different from one group to another, by how much, and how sure is that?

The question behind most small studies — treated plots against control plots, before against after, this
machine against that one — and a question big tables ask too (do orders from the web differ from orders in
store?). For each number the step gives each group's count, average, standard deviation, standard error and
median, the difference, a test of whether it is more than chance, and how big it is, then says the clearest
result in a sentence.

It also notices what a person in a hurry would miss, and says it:

- which test fits and why (groups that vary differently: Welch's t-test, not Student's);
- values that are not bell-shaped in a small sample, and whether a rank test agrees;
- values unusually far from the rest of their group, named by the row they are in;
- a number that grows with the size of what is measured (a gap grows with the area it is measured on) in groups
  of different sizes: compared relative to size as well, and said when that turns the result round.

Counts, averages and spreads are computed over every row (streaming); the rank test, the checks of shape and the
size check use the first 100,000 rows. Deterministic, no Qt. The statistics live in ``groups_stats``.
"""
from __future__ import annotations

from typing import Any

import polars as pl

from ..params import Param
from ..registry import NodeType, Ctx, NodeResult, registry
from ..findings import finding, fmt_number
from ..expr import NUM, kind_of_dtype
from ._common import first_input, schema_of, require_column
from .groups_stats import ALPHA, _clear, _compare, _size_checks, _size_word, _test_one, _verdict, p_text, sentence

SAMPLE_ROWS = 100_000       # rows read for the rank test, the checks of shape and the size check
TESTS = [("auto", "choose for me"), ("welch", "Welch's t-test / ANOVA (groups may vary differently)"),
         ("student", "Student's t-test / ANOVA (groups vary alike)"), ("rank", "rank test (Mann–Whitney / Kruskal–Wallis)"),
         ("none", "no test")]


def _label(ctx: Ctx, name: str) -> str:
    """A column as a sentence names it: its label if it has one; else its header without the unit and short name,
    with what it measures in front of a cryptic one (a column named 'mass (m)' is 'mass', not 'm')."""
    from ..units import header_parts, quantity_of
    meta = (ctx.columns or {}).get(name) or {}
    if meta.get("label") and meta["label"] != name:
        return meta["label"]
    core, _abbrev, unit = header_parts(name)
    q = quantity_of(meta.get("unit") or unit)
    if q and (len(core) <= 3 or "/" in core) and q.lower() not in core.lower():
        return f"{q} ({core})"
    return core


def _unit(ctx: Ctx, name: str) -> str:
    from ..units import unit_from_name
    meta = (ctx.columns or {}).get(name) or {}
    return meta.get("unit") or unit_from_name(name)


def _apply(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any]) -> NodeResult:
    lf = first_input(inputs)
    schema = schema_of(lf)
    by = require_column(schema, params.get("by"), "column that names the groups")
    label = (params.get("label") or "").strip()
    label = require_column(schema, label, "column that names the rows") if label else ""
    pair_by = (params.get("pair_by") or "").strip()
    pair_by = require_column(schema, pair_by, "column that pairs the rows") if pair_by else ""
    given = [require_column(schema, c, "column to compare") for c in (params.get("columns") or [])]
    cols = given or [c for c, dt in schema.items() if kind_of_dtype(dt) == NUM and c not in (by, label, pair_by)]
    cols = [c for c in cols if c != by]
    if not cols:
        raise ValueError("Choose the numbers to compare between the groups")
    bad = [c for c in cols if kind_of_dtype(schema[c]) != NUM]
    if bad:
        raise ValueError(f"{', '.join(bad)} {'holds' if len(bad) == 1 else 'hold'} text, so {'it' if len(bad) == 1 else 'they'} "
                         f"can't be compared as numbers")
    test = params.get("test") or "auto"
    names = {c: _label(ctx, c) for c in cols}
    results, groups = _compare(lf, by, cols, names, test)
    if not results:
        raise ValueError("No group has any of these numbers filled in")
    keep = list(dict.fromkeys([by] + cols + [x for x in (label, pair_by) if x]))
    sample = lf.select(keep).head(SAMPLE_ROWS).collect(engine="streaming").with_columns(pl.col(by).cast(pl.Utf8))
    for r in results:
        _test_one(r, test, sample, by, pair_by)
    derivs = []
    if params.get("size_check", True) and len(cols) >= 2 and not pair_by:
        from ..derived import find
        try:
            derivs = find(sample, cols)
        except Exception:  # noqa: BLE001 - the size check is a bonus, never a failure
            derivs = []
        extra = _size_checks(sample, by, results, names, test, derivs)
    else:
        extra = []
    rel = (params.get("relative_to") or "").strip()
    if rel:
        rel = require_column(schema, rel, "column to compare relative to")
        for r in results:
            if r["column"] == rel:
                continue
            name = f"{names[r['column']]} per {_label(ctx, rel)}"                       # noqa: E501
            d = lf.select(pl.col(by).cast(pl.Utf8), (pl.col(r["column"]).cast(pl.Float64) / pl.col(rel).cast(pl.Float64)).alias(name))
            res, _ = _compare(d, by, [name], {name: name}, test)
            if res:
                _test_one(res[0], test, d.head(SAMPLE_ROWS).collect(), by)
                if not any(e["label"] == name for e in extra):
                    extra.append(res[0])
    rows = []
    for r in results + extra:
        unit = _unit(ctx, r["column"]) if r in results else ""
        row: dict[str, Any] = {"measure": r["label"], "unit": unit}
        for x in r["groups"]:
            row[f"{x['group']} n"] = x["n"]
            row[f"{x['group']} mean"] = x["mean"]
            row[f"{x['group']} SD"] = x["sd"]
            row[f"{x['group']} SE"] = x["se"]
            row[f"{x['group']} median"] = x["median"]
        hi = max(r["groups"], key=lambda x: x["mean"])
        lo = min(r["groups"], key=lambda x: x["mean"])
        first, second = r["groups"][0], r["groups"][1]
        diff = (first["mean"] - second["mean"]) if len(r["groups"]) == 2 else (hi["mean"] - lo["mean"])
        base = second["mean"] if len(r["groups"]) == 2 else lo["mean"]
        row["difference"] = diff
        row["difference (%)"] = (diff / base * 100) if base else None
        row["test"] = r.get("test", "")
        row["statistic"] = r.get("statistic")
        row["p value"] = r.get("p")
        row["effect size"] = r.get("effect")
        row["effect"] = f"{_size_word(r.get('effect'), r.get('effect_kind', 'g'))} ({r.get('effect_kind', 'g')} = {fmt_number(r.get('effect'))})" \
            if r.get("effect") is not None else ""
        row["result"] = _verdict(r, unit)
        r["sentence"] = sentence(r, unit)
        rows.append(row)
    out = pl.DataFrame(rows, infer_schema_length=None)
    # the clearest result first in the finding: a clear difference with the biggest effect
    clear = [r for r in results if r.get("p") is not None and r["p"] < ALPHA]
    best = max(clear or results, key=lambda r: (abs(r.get("effect") or 0.0), -results.index(r)))
    first_label = f" in {len(groups)} groups" if len(groups) > 2 else ""
    statement = best["sentence"]
    msgs = []
    tests = list(dict.fromkeys(r.get("test") for r in results if r.get("test")))
    why = ""
    if pair_by:
        why = f" ({tests[0] if tests else 'paired'}: each {pair_by} is compared with itself)"
    elif tests and any(r.get("uneven") for r in results):
        why = f" ({tests[0]}, which does not assume the groups vary alike: here they don't)"
    elif tests:
        why = f" ({', '.join(tests)})"
    among = ' and '.join(groups) if len(groups) <= 3 else f'the {len(groups)} groups'
    msgs.append(f"{len(clear)} of {len(results)} number{'s' if len(results) != 1 else ''} differ{'s' if len(results) == 1 else ''} "
                f"clearly between {among}{why}")
    for r in results:
        msgs.append(r["sentence"] + (f". Note: {'; '.join(r['notes'])}" if r.get("notes") else ""))
    for e in extra:
        if e.get("size"):
            of, size = names.get(e["of"], e["of"]), names.get(e["size"], e["size"])
            turned = "the other way round" if e["turned"] else ("clear" if _clear(e.get("p")) else "no longer clear")
            msgs.append(f"{of[:1].upper()}{of[1:]} grows with {size} (rank correlation {e['r']:.2f} within the groups), and the "
                        f"groups differ in {size}. Relative to {size} the result is {turned}: {e['sentence']}")
    named = {}
    if label:
        for r in results:
            for grp, v in r.get("unusual", []):
                hit = sample.filter((pl.col(by) == grp) & (pl.col(r["column"]).cast(pl.Float64) == v))
                if hit.height:
                    named.setdefault(r["label"], []).append(f"{fmt_number(v, 4)} ({label} {hit[label][0]}, {grp})")
    else:
        for r in results:
            for grp, v in r.get("unusual", []):
                named.setdefault(r["label"], []).append(f"{fmt_number(v, 4)} ({grp})")
    for what, items in list(named.items())[:4]:
        msgs.append(f"Unusual for its group — {what}: {', '.join(items[:4])}")
    turned = [e for e in extra if e.get("turned")]
    if turned:
        e = turned[0]
        top = max(e["groups"], key=lambda x: x["mean"])["group"]
        statement += f". Relative to {names.get(e['size'], e['size'])}, {names.get(e['of'], e['of'])} is the other way round: " \
                     f"{top} is higher ({p_text(e.get('p'))})"
    report = {"groups": groups, "compared": len(results), "clear": len(clear), "by": by,
              "results": [{"measure": r["label"], "test": r.get("test"), "p": r.get("p"), "effect": r.get("effect"),
                           "sentence": r["sentence"], "notes": r.get("notes", [])} for r in results],
              "relative": [{"measure": e["label"], "p": e.get("p"), "sentence": e["sentence"], "turned": e.get("turned", False)}
                           for e in extra],
              "finding": finding("groups", statement, magnitude=max(abs(best.get("effect") or 0.0), 1.0 if clear else 0.0),
                                 direction="flat", confidence=1.0 if best.get("p") is not None and best["p"] < ALPHA else 0.5,
                                 exact=True, groups=groups)}
    return NodeResult(out.lazy(), report=report, messages=msgs + ([f"Groups{first_label}: {', '.join(groups)}"] if len(groups) > 2 else []))


registry.register(NodeType(
    key="compare_groups", uses_labels=True, label="Compare groups", category="Analyse & model", icon="⫿",
    description="Are the numbers different from one group to another? Each group's count, average, spread and "
                "standard error, the difference, a test of whether it is more than chance, and how big it is.",
    apply=_apply,
    summary=lambda p: f"by {p.get('by')}" if p.get("by") else "choose the groups",
    params=[
        Param("by", "Groups are in", "column", required=True, help="The column that says which group each row is in"),
        Param("columns", "Numbers to compare", "columns", default=[], column_group="numeric",
              help="Blank = every number column"),
        Param("test", "Test", "choice", default="auto", choices=TESTS,
              help="'Chosen for me' uses Welch's test, which does not assume the groups vary alike, and checks it "
                   "with a rank test when a small sample is not bell-shaped"),
        Param("pair_by", "Pairs are matched by", "column", default="",
              help="When the same things are measured in both groups (each plant before and after), the column that "
                   "says which is which: the test is then on the difference within each pair"),
        Param("label", "Rows are named by", "column", default="", advanced=True,
              help="A column that names each row (sample number, id), used to point at unusual values"),
        Param("size_check", "Check for size effects", "bool", default=True, advanced=True,
              help="When a number grows with the size of what is measured and the groups differ in size, compare it "
                   "relative to size as well and say when that changes the result"),
        Param("relative_to", "Also compare relative to", "column", default="", column_group="numeric", advanced=True,
              help="Divide each number by this one (per area, per person) and compare that too"),
    ],
))
