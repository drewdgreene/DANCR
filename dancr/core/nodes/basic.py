"""Everyday table operations."""
from __future__ import annotations

from typing import Any

import polars as pl

from ..conditions import build_mask, incomplete_rules, describe as describe_conditions
from ..expr import compile_formula, FormulaError, _kind_of_dtype
from ..params import Param
from ..registry import NodeType, InputSpec, Ctx, NodeResult, registry
from ._common import first_input, schema_of, require_column
from ..dtypes import datetime_literal, is_temporal, temp_name, text_to_bool, text_to_bool_expr, text_to_number_expr


# ------------------------------------------------------------- keep rows
def _keep_rows(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any]) -> NodeResult:
    lf = first_input(inputs)
    schema = schema_of(lf)
    mask = build_mask(schema, params.get("conditions") or {}, ctx.inputs)
    formula = (params.get("formula") or "").strip()
    if formula:
        expr, kind, _ = compile_formula(formula, schema, ctx.inputs)
        if kind not in ("true/false", "any"):
            raise FormulaError("The formula must give a true/false answer, e.g. Value > 100")
        mask = expr if mask is None else (mask & expr)
    n_skip = len(incomplete_rules(params.get("conditions") or {}))
    msgs = [f"{n_skip} condition{'s' if n_skip != 1 else ''} without a value yet — ignored until you fill it in"] if n_skip else []
    if mask is None:
        return NodeResult(lf, messages=msgs)
    if params.get("mode") == "remove":
        mask = ~mask.fill_null(False)
    return NodeResult(lf.filter(mask), messages=msgs)


registry.register(NodeType(
    key="keep_rows", label="Filter rows", category="Filter & sort", icon="▽",
    description="Keep only the rows that match your conditions (or remove them).",
    apply=_keep_rows,
    summary=lambda p: describe_conditions(p.get("conditions") or {}) if not (p.get("formula") or "").strip()
    else f"where {p['formula']}",
    params=[
        Param("mode", "Action", "choice", default="keep", choices=[("keep", "Keep matching rows"), ("remove", "Remove matching rows")]),
        Param("conditions", "Conditions", "conditions", default={"match": "all", "rules": []}),
        Param("formula", "Or a formula", "expr", default="", advanced=True,
              help="A true/false formula such as Value > 100 and Status = \"ok\""),
    ],
))


# --------------------------------------------------------- choose columns
def _choose_columns(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any]) -> NodeResult:
    lf = first_input(inputs)
    schema = schema_of(lf)
    cols = params.get("columns") or []
    mode = params.get("mode", "keep")
    if cols:
        missing = [c for c in cols if c not in schema]
        if missing:
            raise ValueError(f"These columns do not exist: {missing}")
        lf = lf.select(cols) if mode == "keep" else lf.drop(cols)
    renames = {k: v for k, v in (params.get("rename") or {}).items() if k and v and k != v}
    current = list(schema_of(lf))
    renames = {k: v.strip() for k, v in renames.items() if k in current}
    for old, new in renames.items():
        if new in current and new not in renames:
            raise ValueError(f"Cannot rename {old!r} to {new!r}: there is already a column called {new!r}")
    if len(set(renames.values())) != len(renames):
        raise ValueError("Two columns would get the same new name")
    if renames:
        lf = lf.rename(renames)
    return NodeResult(lf)


registry.register(NodeType(
    key="choose_columns", label="Pick columns", category="Filter & sort", icon="☰",
    description="Keep, remove, reorder or rename columns.",
    apply=_choose_columns,
    summary=lambda p: (f"{'keep' if p.get('mode', 'keep') == 'keep' else 'drop'} {len(p.get('columns') or [])} columns"
                       if p.get("columns") else "all columns") + (f", rename {len(p.get('rename') or {})}" if p.get("rename") else ""),
    params=[
        Param("mode", "Action", "choice", default="keep", choices=[("keep", "Keep only these"), ("drop", "Remove these")]),
        Param("columns", "Columns", "columns", default=[]),
        Param("rename", "Rename", "mapping", default={}, help="Give columns new names"),
    ],
))


# ------------------------------------------------------------------- sort
def _sort(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any]) -> NodeResult:
    lf = first_input(inputs)
    schema = schema_of(lf)
    cols = params.get("columns") or []
    if not cols:
        raise ValueError("Choose at least one column to sort by")
    cols = [require_column(schema, c, "sort column") for c in cols]
    desc = bool(params.get("descending") or False)
    return NodeResult(lf.sort(cols, descending=[desc] * len(cols), nulls_last=True))


registry.register(NodeType(
    key="sort", label="Sort", category="Filter & sort", icon="↕",
    description="Order rows by one or more columns.",
    apply=_sort,
    summary=lambda p: f"by {', '.join(p.get('columns') or [])} {'↓' if p.get('descending') else '↑'}",
    params=[
        Param("columns", "Sort by", "columns", default=[], required=True),
        Param("descending", "Largest first", "bool", default=False),
    ],
))


# -------------------------------------------------------------- calculate
def _calculate(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any]) -> NodeResult:
    lf = first_input(inputs)
    formulas = [f for f in (params.get("formulas") or []) if (f.get("name") or "").strip() and (f.get("expr") or "").strip()]
    if not formulas:
        raise ValueError("Add at least one formula (a column name and an expression)")
    messages = []
    for f in formulas:
        schema = schema_of(lf)      # each formula can use the previous ones
        name = f["name"].strip()
        try:
            expr, kind, used = compile_formula(f["expr"], schema, ctx.inputs)
        except FormulaError as e:
            raise ValueError(f"{name}: {e}") from e
        if name in schema:
            messages.append(f"{name} replaces the existing column of that name")
        lf = lf.with_columns(expr.alias(name))
        messages.append(f"{name} = {f['expr']}  ({kind})")
    if params.get("only_new"):
        lf = lf.select([f["name"].strip() for f in formulas])
    return NodeResult(lf, messages=messages)


registry.register(NodeType(
    key="calculate", label="New column (formula)", category="Calculate", icon="ƒ",
    description="Add columns computed with Excel-style formulas, e.g. [Price] * [Quantity].",
    apply=_calculate,
    summary=lambda p: ", ".join(f["name"] for f in (p.get("formulas") or []) if f.get("name")) or "no formulas yet",
    params=[
        Param("formulas", "Formulas", "formulas", default=[]),
        Param("only_new", "Keep only the new columns", "bool", default=False, advanced=True),
    ],
))


# ------------------------------------------------------------ fix missing
def _fix_missing(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any]) -> NodeResult:
    lf = first_input(inputs)
    schema = schema_of(lf)
    cols = [require_column(schema, c, "column") for c in (params.get("columns") or list(schema))]
    method = params.get("method") or "drop"
    if method == "drop":
        return NodeResult(lf.drop_nulls(subset=cols))
    if method == "drop_all":
        return NodeResult(lf.filter(~pl.all_horizontal([pl.col(c).is_null() for c in cols])))
    exprs = []
    skipped: list[str] = []
    for c in cols:
        e = pl.col(c)
        dt = schema[c]
        kind = _kind_of_dtype(dt)
        if method == "value":
            v = params.get("value")
            if v in (None, ""):
                raise ValueError("Enter the value to fill with")
            if dt.is_numeric():
                try:
                    lit = pl.lit(float(v))
                except ValueError:
                    raise ValueError(f"{v!r} is not a number, but {c} is a number column") from None
            elif is_temporal(dt):
                lit = datetime_literal(v, dt, c)
                if isinstance(dt, pl.Date):
                    lit = lit.cast(pl.Date)
            elif dt == pl.Boolean:
                lit = pl.lit(text_to_bool(v))
            else:
                lit = pl.lit(str(v)).cast(dt, strict=False)
            exprs.append(e.fill_null(lit).alias(c))
        elif method == "forward":
            exprs.append(e.fill_null(strategy="forward").alias(c))
        elif method == "backward":
            exprs.append(e.fill_null(strategy="backward").alias(c))
        elif method in ("interpolate", "mean"):
            if not dt.is_numeric():
                skipped.append(c)
                continue
            exprs.append((e.interpolate() if method == "interpolate" else e.fill_null(e.mean())).alias(c))
        elif method == "zero":
            if dt.is_numeric():
                exprs.append(e.fill_null(0).alias(c))
            elif kind == "text":
                exprs.append(e.fill_null("").alias(c))
            else:
                skipped.append(c)           # dates/booleans have no sensible "zero"
        else:
            raise ValueError(f"Unknown method {method!r}")
    msgs = [f"Left as they are (not number columns): {', '.join(skipped[:8])}" + (" …" if len(skipped) > 8 else "")] if skipped else []
    return NodeResult(lf.with_columns(exprs) if exprs else lf, messages=msgs)


registry.register(NodeType(
    key="fix_missing", label="Fill blanks", category="Clean up", icon="✚",
    description="Remove or fill in blank cells.",
    apply=_fix_missing,
    summary=lambda p: {"drop": "drop rows with blanks", "drop_all": "drop fully blank rows", "value": f"fill with {p.get('value')}",
                       "forward": "carry previous value forward", "backward": "use next value", "interpolate": "interpolate",
                       "mean": "fill with average", "zero": "fill with 0"}.get(p.get("method", "drop"), ""),
    params=[
        Param("method", "What to do", "choice", default="drop", choices=[
            ("drop", "Remove rows that have any blank"), ("drop_all", "Remove rows that are completely blank"),
            ("value", "Fill blanks with a value"), ("forward", "Carry the previous value forward"),
            ("backward", "Use the next value"), ("interpolate", "Interpolate between neighbours"),
            ("mean", "Fill with the column average"), ("zero", "Fill with 0")]),
        Param("value", "Value", "text", default="", visible_when={"method": "value"}),
        Param("columns", "Only these columns", "columns", default=[], help="Leave empty for all columns"),
    ],
))


# ------------------------------------------------------------ change type
def _change_type(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any]) -> NodeResult:
    lf = first_input(inputs)
    schema = schema_of(lf)
    cols = params.get("columns") or []
    if not cols:
        raise ValueError("Choose the column(s) to convert")
    cols = [require_column(schema, c, "column") for c in cols]
    to = params.get("to") or "number"
    exprs = []
    for c in cols:
        e = pl.col(c)
        dt = schema[c]
        as_number = text_to_number_expr(e) if dt in (pl.Utf8, pl.String) else e.cast(pl.Float64, strict=False)
        if to == "number":
            e = as_number
        elif to == "integer":
            e = as_number.round(0).cast(pl.Int64, strict=False)
        elif to == "text":
            e = e.cast(pl.Utf8)
        elif to == "datetime":
            fmt = (params.get("date_format") or "").strip() or None
            if dt in (pl.Utf8, pl.String):
                if fmt is None:
                    from ..timeutil import detect_datetime_format
                    fmt = detect_datetime_format(lf.select(c).head(2000).collect(engine="streaming")[c])
                    if fmt is None:
                        raise ValueError(f"Could not work out the date format of {c}. Set it under 'Date format' (e.g. %d/%m/%Y)")
                e = e.str.strip_chars().str.to_datetime(fmt, strict=False)
            elif dt.is_numeric():
                unit = params.get("epoch_unit") or "s"
                to_us = {"s": 1e6, "ms": 1e3, "us": 1.0}.get(unit)
                if to_us is None:
                    raise ValueError("'Number means' must be seconds, milliseconds or microseconds since 1970")
                e = pl.from_epoch((e.cast(pl.Float64) * to_us).round(0).cast(pl.Int64), time_unit="us")   # keeps fractional seconds
            else:
                e = e.cast(pl.Datetime("us"), strict=False)
        elif to == "bool":
            e = text_to_bool_expr(e) if dt in (pl.Utf8, pl.String) else e.cast(pl.Boolean, strict=False)
        else:
            raise ValueError(f"Unknown type {to!r}")
        exprs.append(e.alias(c))
    return NodeResult(lf.with_columns(exprs))


registry.register(NodeType(
    key="change_type", label="Fix numbers and dates", category="Clean up", icon="Aa",
    description="Turn text into numbers or dates, numbers into text, and so on.",
    apply=_change_type,
    summary=lambda p: f"{', '.join(p.get('columns') or [])} → {p.get('to', 'number')}",
    params=[
        Param("columns", "Columns", "columns", default=[], required=True),
        Param("to", "Convert to", "choice", default="number", choices=[
            ("number", "Number (decimal)"), ("integer", "Whole number"), ("text", "Text"),
            ("datetime", "Date / time"), ("bool", "True / false")]),
        Param("date_format", "Date format", "text", default="", visible_when={"to": "datetime"},
              help="Leave blank to detect. Examples: %Y-%m-%d %H:%M:%S, %d/%m/%Y"),
        Param("epoch_unit", "Number means", "choice", default="s", visible_when={"to": "datetime"},
              choices=[("s", "seconds since 1970"), ("ms", "milliseconds since 1970"), ("us", "microseconds since 1970")]),
    ],
))


# -------------------------------------------------------------- stack rows
def _stack(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any]) -> NodeResult:
    frames = inputs.get("tables") or []
    if len(frames) < 1:
        raise ValueError("Connect at least one table")
    label_col = (params.get("label_column") or "").strip()
    labels = params.get("labels") or []
    if label_col:
        for f in frames:
            if label_col in f.collect_schema():
                raise ValueError(f"There is already a column called {label_col!r}; choose another name for the label column")
        frames = [f.with_columns(pl.lit(str(labels[i]) if i < len(labels) and labels[i] is not None else f"table {i + 1}").alias(label_col))
                  for i, f in enumerate(frames)]
    return NodeResult(pl.concat(frames, how="diagonal_relaxed"))


registry.register(NodeType(
    key="stack", label="Stack tables", category="Combine", icon="⧉",
    description="Put several tables one after another (rows appended). Columns are matched by name.",
    apply=_stack,
    inputs=[InputSpec("tables", "Tables", multiple=True)],
    summary=lambda p: "append rows",
    params=[
        Param("label_column", "Add a column saying which table each row came from", "text", default="",
              placeholder="e.g. source"),
        Param("labels", "Labels (one per input, in order)", "text_list", default=[], advanced=True),
    ],
))


# --------------------------------------------------------- remove duplicates
def _dedupe(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any]) -> NodeResult:
    lf = first_input(inputs)
    schema = schema_of(lf)
    cols = [require_column(schema, c, "column") for c in (params.get("columns") or [])] or None
    keep = params.get("keep") or "first"
    return NodeResult(lf.unique(subset=cols, keep=keep, maintain_order=True))


registry.register(NodeType(
    key="remove_duplicates", label="Remove duplicates", category="Clean up", icon="⧈",
    description="Drop repeated rows.",
    apply=_dedupe,
    summary=lambda p: f"based on {', '.join(p['columns'])}" if p.get("columns") else "whole rows",
    params=[
        Param("columns", "Consider only these columns", "columns", default=[], help="Empty = the whole row must match"),
        Param("keep", "Keep", "choice", default="first", choices=[("first", "first occurrence"), ("last", "last occurrence"), ("none", "neither (drop all copies)")]),
    ],
))


# ----------------------------------------------------------------- sample
def _sample(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any]) -> NodeResult:
    lf = first_input(inputs)
    mode = params.get("mode") or "first"
    n = params.get("rows")
    n = 1000 if n is None else int(n)
    if n < 1:
        raise ValueError("N must be at least 1")
    if mode == "first":
        return NodeResult(lf.head(n))
    if mode == "last":
        return NodeResult(lf.tail(n))
    idx = temp_name("row", schema_of(lf))
    if mode == "every":
        k = max(1, int(params.get("every") or 10))
        return NodeResult(lf.with_row_index(idx).filter(pl.col(idx) % k == 0).drop(idx))
    if mode == "random":
        # streaming friendly-ish random sample: hash row index
        frac = float(params.get("fraction") or 0.01)
        if not 0 < frac <= 1:
            raise ValueError("Fraction must be between 0 and 1")
        return NodeResult(lf.with_row_index(idx).filter(pl.col(idx).hash(seed=int(params.get("seed") or 0)) % 1_000_000 < pl.lit(int(1_000_000 * frac))).drop(idx))
    raise ValueError(f"Unknown mode {mode!r}")


registry.register(NodeType(
    key="take_sample", label="Take a sample", category="Filter & sort", icon="✂",
    description="Keep just part of the data: the first rows, every Nth row, or a random fraction.",
    apply=_sample,
    summary=lambda p: {"first": f"first {p.get('rows')}", "last": f"last {p.get('rows')}", "every": f"every {p.get('every')}th row",
                       "random": f"random {float(p.get('fraction') or 0) * 100:g}%"}.get(p.get("mode", "first"), ""),
    params=[
        Param("mode", "Which rows", "choice", default="first", choices=[
            ("first", "First N rows"), ("last", "Last N rows"), ("every", "Every Nth row"), ("random", "Random fraction")]),
        Param("rows", "N", "int", default=1000, min=1, visible_when={"mode": ["first", "last"]}),
        Param("every", "N", "int", default=10, min=1, visible_when={"mode": "every"}),
        Param("fraction", "Fraction (0-1)", "float", default=0.01, min=0, max=1, visible_when={"mode": "random"}),
        Param("seed", "Random seed", "int", default=0, advanced=True, visible_when={"mode": "random"}),
    ],
))
