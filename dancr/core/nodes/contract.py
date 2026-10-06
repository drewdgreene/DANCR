"""Stewardship steps: a data contract (checks that must hold) and a table diff (what changed).

A **data contract** is a small JSON document describing what a table must look like: each column's kind,
whether it may be blank, whether it is unique, a range, an allowed set, a pattern; cross-column rules written
in the formula language; and references to keys that must exist in another table. ``check_contract`` runs it
and reports every failure with the failing count and an example, or passes the rows straight through. This is
the steward's first job made into a step.

``diff_tables`` compares two versions of a table: columns added, removed or retyped, rows added or removed,
and cells whose value changed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from ..expr import BOOL, NUM, STR, compile_formula, kind_of_dtype
from ..findings import finding
from ..params import Param
from ..registry import NodeType, InputSpec, Ctx, NodeResult, registry
from ._common import first_input, schema_of, require_column

_KINDS = ("number", "text", "date/time", "true/false")


def _load_contract(ctx: Ctx, params: dict[str, Any]) -> dict[str, Any] | None:
    path = (params.get("contract_path") or "").strip()
    inline = (params.get("contract") or "").strip()
    if path:
        try:
            text = ctx.resolve(path).read_text(encoding="utf-8")
        except OSError as e:
            raise ValueError(f"Cannot read the contract file {path}: {e}") from e
    elif inline:
        text = inline
    else:
        return None
    try:
        doc = json.loads(text)
    except ValueError as e:
        raise ValueError(f"The contract is not valid JSON: {e}") from e
    if not isinstance(doc, dict):
        raise ValueError("A contract is a JSON object with 'columns', 'rules' and/or 'references'")
    return doc


def _infer(schema: dict[str, pl.DataType], lf: pl.LazyFrame) -> dict[str, Any]:
    """A draft contract from the data as it is now: each column's kind, and for a number its range; a text
    column with few distinct values lists them; a column with no blanks is marked required."""
    cols: dict[str, Any] = {}
    for c, dt in schema.items():
        kind = kind_of_dtype(dt)
        spec: dict[str, Any] = {"kind": kind}
        nulls = int(lf.select(pl.col(c).null_count()).collect(engine="streaming")[0, 0] or 0)
        if nulls == 0:
            spec["required"] = True
        if kind == NUM:
            lo, hi = lf.select([pl.col(c).min().alias("__lo"), pl.col(c).max().alias("__hi")]) \
                       .collect(engine="streaming").row(0)
            if lo is not None:
                spec["min"] = lo
            if hi is not None:
                spec["max"] = hi
        elif kind == STR:
            uniq = lf.select(pl.col(c).drop_nulls().unique().sort()).collect(engine="streaming").get_column(c).to_list()
            if len(uniq) <= 12:
                spec["allowed"] = uniq
        cols[c] = spec
    return {"columns": cols}


def _issue(check: str, column: str | None, failing: int, message: str, severity: str = "error",
           example: Any = None) -> dict[str, Any]:
    return {"check": check, "column": column, "severity": severity, "failing_rows": int(failing),
            "message": message, "example": None if example is None else str(example)[:120]}


def _column_checks(lf: pl.LazyFrame, schema: dict[str, pl.DataType], name: str, spec: dict[str, Any],
                   total: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    dtype = schema.get(name)
    if dtype is None:
        return [_issue("column present", name, total, f"There is no column called {name!r}")]
    kind = kind_of_dtype(dtype)
    want = spec.get("kind")
    if want and want != kind:
        out.append(_issue("kind", name, total, f"{name} is {kind}, but the contract says {want}"))
    if spec.get("required"):
        n = int(lf.select(pl.col(name).null_count()).collect(engine="streaming")[0, 0] or 0)
        if n:
            out.append(_issue("required", name, n, f"{n:,} blank value(s) where none are allowed"))
    if spec.get("unique"):
        row = lf.select([pl.col(name).count().alias("filled"), pl.col(name).n_unique().alias("uniq")]) \
                .collect(engine="streaming").row(0, named=True)
        dup = int(row["filled"] or 0) - int(row["uniq"] or 0)
        if dup > 0:
            out.append(_issue("unique", name, dup, f"{dup:,} repeated value(s) in a column that must be unique"))
    if kind == NUM:
        if spec.get("min") is not None:
            n = int(lf.select((pl.col(name) < spec["min"]).sum()).collect(engine="streaming")[0, 0] or 0)
            if n:
                out.append(_issue("minimum", name, n, f"{n:,} value(s) below {spec['min']}"))
        if spec.get("max") is not None:
            n = int(lf.select((pl.col(name) > spec["max"]).sum()).collect(engine="streaming")[0, 0] or 0)
            if n:
                out.append(_issue("maximum", name, n, f"{n:,} value(s) above {spec['max']}"))
    if spec.get("allowed"):
        allowed = list(spec["allowed"])
        bad = (pl.col(name).is_not_null()) & (~pl.col(name).cast(pl.Utf8).is_in([str(a) for a in allowed]))
        n = int(lf.select(bad.sum()).collect(engine="streaming")[0, 0] or 0)
        if n:
            example = lf.filter(bad).select(pl.col(name).cast(pl.Utf8)).head(1).collect(engine="streaming")
            ex = example[0, 0] if example.height else None
            out.append(_issue("allowed values", name, n, f"{n:,} value(s) outside the allowed set", example=ex,
                              severity="warning"))
    if spec.get("regex"):
        try:
            bad = (pl.col(name).is_not_null()) & (~pl.col(name).cast(pl.Utf8).str.contains(str(spec["regex"])))
            n = int(lf.select(bad.sum()).collect(engine="streaming")[0, 0] or 0)
        except Exception as e:  # noqa: BLE001 - a bad pattern is the contract's fault, reported plainly
            out.append(_issue("pattern", name, 0, f"Bad pattern {spec['regex']!r}: {e}"))
            n = 0
        if n:
            out.append(_issue("pattern", name, n, f"{n:,} value(s) do not match {spec['regex']!r}"))
    return out


def _check_contract(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any]) -> NodeResult:
    lf = first_input(inputs)
    schema = schema_of(lf)
    contract = _load_contract(ctx, params)
    inferred = None
    if contract is None:
        inferred = _infer(schema, lf)
        contract = inferred
    total = int(lf.select(pl.len()).collect(engine="streaming")[0, 0] or 0)
    issues: list[dict[str, Any]] = []

    for name, spec in (contract.get("columns") or {}).items():
        if not isinstance(spec, dict):
            continue
        issues += _column_checks(lf, schema, name, spec, total)

    for rule in contract.get("rules") or []:
        if not isinstance(rule, dict):
            continue
        expr_src = rule.get("expr") or ""
        try:
            expr, kind, _ = compile_formula(expr_src, schema, ctx.inputs)
        except Exception as e:  # noqa: BLE001
            issues.append(_issue("rule", rule.get("name"), 0, f"Rule {rule.get('name')!r} cannot be read: {e}"))
            continue
        if kind != BOOL:
            issues.append(_issue("rule", rule.get("name"), 0, f"Rule {rule.get('name')!r} must be true/false, not {kind}"))
            continue
        failing = (~expr.fill_null(True))
        n = int(lf.select(failing.sum()).collect(engine="streaming")[0, 0] or 0)
        if n:
            issues.append(_issue("rule", rule.get("name"), n, f"Rule failed on {n:,} row(s)",
                                 severity=rule.get("severity", "error")))

    ref_frames = inputs.get("references") or []
    ref_meta = (ctx.upstream_meta or {}).get("references") or []
    by_node = {m.get("node"): fr for m, fr in zip(ref_meta, ref_frames)}
    for ref in contract.get("references") or []:
        if not isinstance(ref, dict):
            continue
        col = ref.get("column")
        to_node = ref.get("to_node")
        to_col = ref.get("to_column") or col
        fr = by_node.get(to_node)
        if fr is None:
            issues.append(_issue("reference", col, 0,
                                 f"The referenced table {to_node!r} is not connected to this step"))
            continue
        if col not in schema:
            issues.append(_issue("reference", col, 0, f"There is no column called {col!r} to reference"))
            continue
        try:
            valid = fr.select(pl.col(to_col)).drop_nulls().unique()
            missing = (lf.select(pl.col(col)).filter(pl.col(col).is_not_null()).unique()
                         .join(valid, left_on=col, right_on=to_col, how="anti"))
            n = int(missing.select(pl.len()).collect(engine="streaming")[0, 0] or 0)
        except Exception as e:  # noqa: BLE001
            issues.append(_issue("reference", col, 0, f"Cannot check {col} against {to_node}.{to_col}: {e}"))
            continue
        if n:
            example = missing.head(1).collect(engine="streaming")
            issues.append(_issue("reference", col, n,
                                 f"{n:,} value(s) in {col} have no match in {to_node}.{to_col}",
                                 example=(example[0, 0] if example.height else None)))

    bad = sum(1 for i in issues if i["severity"] == "error")
    checks = len(contract.get("columns") or {}) + len(contract.get("rules") or []) + len(contract.get("references") or [])
    said = (f"{len(issues)} issue" + ("s" if len(issues) != 1 else "") + f" in {checks} check(s)"
            if issues else f"All {checks} check(s) passed")
    report: dict[str, Any] = {"checks": checks, "issues": len(issues), "errors": bad, "rows": total,
                              "finding": finding("quality", said, magnitude=float(bad or len(issues)), exact=True)}
    if inferred is not None:
        report["inferred_contract"] = inferred
        report["messages_note"] = "No contract given: a draft was inferred from the data"
    msgs = [said]
    if issues:
        msgs += [f"{i['severity']}: {i['message']}" for i in issues[:8]]

    if (params.get("output") or "rows") == "issues":
        out = pl.DataFrame({
            "check": pl.Series([i["check"] for i in issues], dtype=pl.Utf8),
            "column": pl.Series([i["column"] for i in issues], dtype=pl.Utf8),
            "severity": pl.Series([i["severity"] for i in issues], dtype=pl.Utf8),
            "failing_rows": pl.Series([i["failing_rows"] for i in issues], dtype=pl.Int64),
            "message": pl.Series([i["message"] for i in issues], dtype=pl.Utf8),
            "example": pl.Series([i["example"] for i in issues], dtype=pl.Utf8),
        }).lazy()
        return NodeResult(out, report=report, messages=msgs)
    return NodeResult(lf, report=report, messages=msgs)


registry.register(NodeType(
    key="check_contract", label="Check data contract", category="Check", icon="✓",
    description="Check a table against a data contract: each column's kind, blanks, uniqueness, range, allowed "
                "values and pattern; cross-column rules; and keys that must exist in another table. Reports every "
                "failure with a count and an example, or passes the rows through. Leave the contract empty to infer one.",
    apply=_check_contract,
    inputs=[InputSpec("in", "Table"), InputSpec("references", "Referenced tables", multiple=True, optional=True)],
    summary=lambda p: ("check the contract" if (p.get("contract") or p.get("contract_path")) else "infer a contract"),
    params=[
        Param("contract", "Contract (JSON)", "text", default="",
              help='e.g. {"columns": {"yield": {"kind": "number", "min": 0}}, "rules": [{"name": "n", "expr": "yield >= 0"}]}'),
        Param("contract_path", "…or a contract file", "path", default="", advanced=True,
              help="A JSON file (relative to the project folder); used when the box above is empty"),
        Param("output", "Result", "choice", default="rows", choices=[
            ("rows", "The table unchanged (with findings)"),
            ("issues", "One row per failed check")]),
    ],
))


# ----------------------------------------------------------------- diff two versions
def _diff_tables(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any]) -> NodeResult:
    before_frames = inputs.get("a") or []
    after_frames = inputs.get("b") or []
    if not before_frames or not after_frames:
        raise ValueError("Connect a 'before' table and an 'after' table")
    a, b = before_frames[0], after_frames[0]
    sa, sb = schema_of(a), schema_of(b)
    keys = [require_column(sa, c, "key column") for c in (params.get("key") or [])]
    if not keys:
        keys = [c for c in sa if c in sb]
        if not keys:
            raise ValueError("Choose the column(s) that identify a row (the two tables share no column names)")
    for c in keys:
        require_column(sb, c, "key column of the after table")
    only_a = [c for c in sa if c not in sb]
    only_b = [c for c in sb if c not in sa]
    retyped = [c for c in sa if c in sb and kind_of_dtype(sa[c]) != kind_of_dtype(sb[c])]
    compare = [c for c in (params.get("compare") or []) if c in sa and c in sb and c not in keys] \
        or [c for c in sa if c in sb and c not in keys]
    tol = params.get("tolerance")
    out_cols = [*keys, "change_type", "column", "before", "after", "delta"]
    empty = pl.DataFrame({c: pl.Series([], dtype=pl.Utf8) for c in out_cols}).lazy()

    if params.get("schema_only"):
        recs = [("column removed", c, str(sa[c]), None) for c in only_a]
        recs += [("column added", c, None, str(sb[c])) for c in only_b]
        recs += [("column retyped", c, str(sa[c]), str(sb[c])) for c in retyped]
        if not recs:
            return NodeResult(empty, messages=["Schemas match: no column was added, removed or retyped"])
        out = pl.DataFrame([{"column": c, "change_type": t, "before": bf, "after": af} for t, c, bf, af in recs]).lazy()
        report = {"columns_added": only_b, "columns_removed": only_a, "columns_retyped": retyped}
        return NodeResult(out, report=report, messages=[f"Schema: +{len(only_b)} -{len(only_a)} ~{len(retyped)} columns"])

    ja = a.with_columns(pl.lit(1).alias("__a"))
    jb = b.with_columns(pl.lit(1).alias("__b"))
    joined = ja.join(jb, on=keys, how="full", suffix="_after", coalesce=True)
    pieces: list[pl.LazyFrame] = []

    def key_cols(col: str | None = None) -> list[pl.Expr]:
        return [pl.col(k) for k in keys]

    for c in compare:
        ca, cb = c, f"{c}_after"
        if cb not in joined.collect_schema().names():
            continue
        both_null = pl.col(ca).is_null() & pl.col(cb).is_null()
        if tol not in (None, "") and kind_of_dtype(sa[c]) == NUM:
            cond = ((pl.col(ca).cast(pl.Float64) - pl.col(cb).cast(pl.Float64)).abs() > float(tol)) & ~both_null
        else:
            cond = (pl.col(ca) != pl.col(cb)) & ~both_null
        sel = [*key_cols(), pl.lit("changed").alias("change_type"), pl.lit(c).alias("column"),
               pl.col(ca).cast(pl.Utf8).alias("before"), pl.col(cb).cast(pl.Utf8).alias("after")]
        if kind_of_dtype(sa[c]) == NUM:
            sel.append((pl.col(cb).cast(pl.Float64) - pl.col(ca).cast(pl.Float64)).alias("delta"))
        pieces.append(joined.filter(cond).select(sel))

    removed = joined.filter(pl.col("__b").is_null()).select(
        [*key_cols(), pl.lit("removed").alias("change_type"), pl.lit(None).cast(pl.Utf8).alias("column"),
         pl.lit(None).cast(pl.Utf8).alias("before"), pl.lit(None).cast(pl.Utf8).alias("after"),
         pl.lit(None).cast(pl.Float64).alias("delta")])
    added = joined.filter(pl.col("__a").is_null()).select(
        [*key_cols(), pl.lit("added").alias("change_type"), pl.lit(None).cast(pl.Utf8).alias("column"),
         pl.lit(None).cast(pl.Utf8).alias("before"), pl.lit(None).cast(pl.Utf8).alias("after"),
         pl.lit(None).cast(pl.Float64).alias("delta")])
    pieces += [removed, added]
    out = pl.concat(pieces, how="diagonal_relaxed") if pieces else empty

    report: dict[str, Any] = {"columns_added": only_b, "columns_removed": only_a, "columns_retyped": retyped}
    msgs = []
    if not ctx.preview:
        counts = {
            "changed": sum(int(p.select(pl.len()).collect(engine="streaming")[0, 0] or 0)
                           for p in pieces if p is not removed and p is not added),
            "added": int(added.select(pl.len()).collect(engine="streaming")[0, 0] or 0),
            "removed": int(removed.select(pl.len()).collect(engine="streaming")[0, 0] or 0),
        }
        report.update(counts)
        said = (f"{counts['changed']:,} cell(s) changed, {counts['added']:,} row(s) added, "
                f"{counts['removed']:,} row(s) removed")
        if only_b or only_a or retyped:
            said += f"; columns +{len(only_b)} -{len(only_a)} ~{len(retyped)}"
        report["finding"] = finding("change", said, magnitude=float(counts["changed"] + counts["added"] + counts["removed"]),
                                    exact=True)
        msgs.append(said)
    return NodeResult(out, report=report, messages=msgs)


registry.register(NodeType(
    key="diff_tables", label="Compare two versions", category="Combine", icon="≠",
    description="Compare two versions of a table: columns added, removed or retyped, rows added or removed, and "
                "cells whose value changed. A before/after of this month's export, with the changes listed.",
    apply=_diff_tables,
    inputs=[InputSpec("a", "Before"), InputSpec("b", "After")],
    summary=lambda p: ("schema only" if p.get("schema_only") else f"diff by {', '.join(p.get('key') or ['shared columns'])}"),
    params=[
        Param("key", "Row identifier", "columns", default=[], port="a",
              help="The column(s) that identify a row. Leave empty to use every shared column"),
        Param("compare", "Columns to compare", "columns", default=[], port="b",
              help="Leave empty to compare every shared column except the key"),
        Param("tolerance", "Ignore number changes smaller than", "float", default=None, advanced=True,
              help="Blank means any difference counts"),
        Param("schema_only", "Only report added/removed/retyped columns", "bool", default=False, advanced=True),
    ],
))
