"""Everyday table operations."""
from __future__ import annotations

from typing import Any

import polars as pl

from ..conditions import build_mask, incomplete_rules, describe as describe_conditions
from ..expr import compile_formula, FormulaError, kind_of_dtype, excel_round
from ..params import Param
from ..registry import NodeType, InputSpec, Ctx, NodeResult, registry
from ._common import first_input, schema_of, require_column, number_param
from ..dtypes import datetime_literal, is_temporal, temp_name, text_to_bool, text_to_bool_expr, text_to_number_expr, typed_value


# ------------------------------------------------------------- keep rows
def _keep_rows(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any]) -> NodeResult:
    lf = first_input(inputs)
    schema = schema_of(lf)
    conditions = params.get("conditions") or {}
    # a rule made from a column's menu has no value until the person types one: the step passes rows
    # through meanwhile and says so, as Check limits does with no limit yet
    unfinished = incomplete_rules(conditions)
    n = len(unfinished)
    msgs = ([f"{n} conditions have no value yet, so they're skipped for now" if n > 1
             else "A condition has no value yet, so it's skipped for now"] if n else [])
    conditions = {**conditions, "rules": [r for r in conditions.get("rules", []) if r not in unfinished]}
    mask = build_mask(schema, conditions, ctx.inputs)
    formula = (params.get("formula") or "").strip()
    if formula:
        expr, kind, _ = compile_formula(formula, schema, ctx.inputs)
        if kind not in ("true/false", "any"):
            raise FormulaError("The formula must give a true/false answer, e.g. Value > 100")
        mask = expr if mask is None else (mask & expr)
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
            raise ValueError(f"Can't rename {old!r} to {new!r} because there is already a column called {new!r}")
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
    # NaN ("not a number") is a blank here too, as it is for "is empty" and ISBLANK
    floats = [c for c in cols if schema[c].is_float()]
    if floats:
        lf = lf.with_columns([pl.col(c).fill_nan(None) for c in floats])
    if method == "drop":
        return NodeResult(lf.drop_nulls(subset=cols))
    if method == "drop_all":
        return NodeResult(lf.filter(~pl.all_horizontal([pl.col(c).is_null() for c in cols])))
    exprs = []
    skipped: list[str] = []
    for c in cols:
        e = pl.col(c)
        dt = schema[c]
        kind = kind_of_dtype(dt)
        if method == "value":
            v = params.get("value")
            if v in (None, ""):
                raise ValueError("Enter the value to fill with")
            if dt.is_numeric():
                num = typed_value(str(v))           # '1,5' works; whole numbers stay exact
                if num is None:
                    raise ValueError(f"{v!r} is not a number, but {c} is a number column")
                if dt.is_integer() and not float(num).is_integer():
                    raise ValueError(f"{v!r} has decimals but {c} holds whole numbers. Fill it with a whole number or convert the column first")
                lit = pl.lit(int(num) if dt.is_integer() else num).cast(dt)   # keep the column's dtype
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
    msgs: list[str] = []
    for c in cols:
        e = pl.col(c)
        dt = schema[c]
        if isinstance(dt, (pl.Categorical, pl.Enum)):      # categories (common in Parquet) convert as their text
            e, dt = e.cast(pl.Utf8), pl.Utf8
        as_number = text_to_number_expr(e) if dt in (pl.Utf8, pl.String) else e.cast(pl.Float64, strict=False)
        if to == "number":
            e = as_number
        elif to == "integer":
            rounded = excel_round(as_number).cast(pl.Int64, strict=False)    # 2.5 -> 3, as in Excel
            if dt in (pl.Utf8, pl.String):
                # whole numbers written as text are read exactly (a 17-digit id would lose digits through a float)
                exact = e.str.strip_chars().str.replace_all(r"[\s,_']", "").cast(pl.Int64, strict=False)
                e = pl.coalesce([exact, rounded])
            elif dt.is_integer():
                e = e.cast(pl.Int64, strict=False)
            else:
                e = rounded
        elif to == "text":
            e = e.cast(pl.Utf8)
        elif to == "datetime":
            fmt = (params.get("date_format") or "").strip() or None
            if dt in (pl.Utf8, pl.String):
                from ..timeutil import detect_datetime_format, settle_day_month, offset_time_zone, has_offset
                sample = lf.select(e.alias(c)).head(2000).collect(engine="streaming")[c]
                if fmt is None:
                    fmt = detect_datetime_format(sample)
                    if fmt is None:
                        raise ValueError(f"Could not work out the date format of {c}. Set it under 'Date format' (e.g. %d/%m/%Y)")
                    fmt, settled = settle_day_month(lf, c, fmt, sample, whole=not ctx.preview)
                    if settled:
                        msgs.append(settled.replace("tick 'Day comes before month' (or set 'Date format')", "set 'Date format' (e.g. %d/%m/%Y)"))
                e = e.str.strip_chars().str.to_datetime(fmt, strict=False)
                if has_offset(fmt):
                    tz, note = offset_time_zone(c, sample, (params.get("time_zone") or "").strip() or None)
                    e = e.dt.convert_time_zone(tz)
                    if note:
                        msgs.append(note)
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
    return NodeResult(lf.with_columns(exprs), messages=msgs)


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
        Param("time_zone", "Time zone", "text", default="", visible_when={"to": "datetime"}, advanced=True,
              help="The zone to show times with a UTC offset (…+02:00) in, such as Europe/London. Leave empty to keep "
                   "the text's own offset if it's the same throughout, or UTC if not"),
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
                raise ValueError(f"There is already a column called {label_col!r}. Choose another name for the label column")
        frames = [f.with_columns(pl.lit(str(labels[i]) if i < len(labels) and labels[i] is not None else f"table {i + 1}").alias(label_col))
                  for i, f in enumerate(frames)]
    frames, msgs = _same_time_zones(frames)
    return NodeResult(pl.concat(frames, how="diagonal_relaxed"), messages=msgs)


def _same_time_zones(frames: list[pl.LazyFrame]) -> tuple[list[pl.LazyFrame], list[str]]:
    """Date/time columns that are in different time zones (or in one and in none) in different tables are put
    in UTC, so the stacked rows keep their true moments and the tables can be appended at all."""
    schemas = [f.collect_schema() for f in frames]
    zones: dict[str, set] = {}
    for sch in schemas:
        for c, dt in sch.items():
            if isinstance(dt, pl.Datetime):
                zones.setdefault(c, set()).add(dt.time_zone)
    mixed = [c for c, z in zones.items() if len(z) > 1]
    if not mixed:
        return frames, []
    out = []
    for f, sch in zip(frames, schemas):
        fixes = []
        for c in mixed:
            dt = sch.get(c)
            if isinstance(dt, pl.Datetime):
                e = pl.col(c).cast(pl.Datetime("us", dt.time_zone))
                e = e.dt.convert_time_zone("UTC") if dt.time_zone else e.dt.replace_time_zone("UTC")
                fixes.append(e.alias(c))
        out.append(f.with_columns(fixes) if fixes else f)
    return out, [f"{', '.join(mixed)}: the tables use different time zones, so all times are shown in UTC. "
                 "Times without a zone are taken as UTC"]


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
        k = int(number_param(params, "every", 10, "'Every Nth row'", whole=True, at_least=1))
        return NodeResult(lf.with_row_index(idx).filter(pl.col(idx) % k == 0).drop(idx))
    if mode == "random":
        # streaming friendly-ish random sample: hash row index
        frac = number_param(params, "fraction", 0.01, "The fraction", at_least=0, at_most=1)
        seed = int(number_param(params, "seed", 0, "The seed", whole=True))
        return NodeResult(lf.with_row_index(idx).filter(pl.col(idx).hash(seed=seed) % 1_000_000 < pl.lit(int(1_000_000 * frac))).drop(idx))
    raise ValueError(f"Unknown mode {mode!r}")


registry.register(NodeType(
    key="take_sample", label="Take a sample", category="Filter & sort", icon="✂",
    description="Keep part of the data: the first rows, every Nth row or a random fraction.",
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


# --------------------------------------------------------- columns into rows
MONTH_NAMES = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]


MONTH_FULL = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
              "november", "december"]


def month_of(name: str) -> int | None:
    """'Jan', 'January', 'Sept', 'Dec 24', 'jan-2024', '2024-01' -> the month's number; anything else None."""
    import re
    s = str(name).strip().lower()
    m = re.fullmatch(r"(\d{4})[-/ ](\d{1,2})", s)
    if m:
        return int(m.group(2)) if 1 <= int(m.group(2)) <= 12 else None
    m = re.fullmatch(r"([a-z]+)\.?(?:[\s\-/']?\d{2,4})?", s)
    if not m:
        return None
    w = m.group(1)
    for i, full in enumerate(MONTH_FULL, start=1):
        if w in (full, full[:3]) or (w == "sept" and i == 9):
            return i
    return None


def _unpivot(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any]) -> NodeResult:
    lf = first_input(inputs)
    schema = schema_of(lf)
    cols = [require_column(schema, c, "column") for c in (params.get("columns") or [])]
    if len(cols) < 2:
        raise ValueError("Choose at least two columns to turn into rows (for example Jan, Feb, Mar …)")
    name_col = (params.get("name_column") or "month").strip() or "month"
    value_col = (params.get("value_column") or "value").strip() or "value"
    keep = [c for c in schema if c not in cols]
    for n in (name_col, value_col):
        if n in keep:
            raise ValueError(f"There is already a column called {n!r}. Choose another name")
    kinds = {kind_of_dtype(schema[c]) for c in cols}
    if len(kinds) > 1:
        lf = lf.with_columns([pl.col(c).cast(pl.Utf8) for c in cols])
    out = lf.unpivot(on=cols, index=keep, variable_name=name_col, value_name=value_col)
    msgs = [f"Turned {len(cols)} columns into rows, one row per {', '.join(keep[:2]) or 'row'} and {name_col}"]
    year = params.get("year")
    months = [month_of(c) for c in cols]
    if year not in (None, "") and all(months):
        taken = set(keep) | {name_col, value_col}
        date_col = "date" if "date" not in taken else f"{name_col} date"
        order = pl.DataFrame({name_col: cols, "__m": months})
        out = (out.join(order.lazy(), on=name_col, how="left", maintain_order="left")
                  .with_columns(pl.date(int(year), pl.col("__m"), 1).cast(pl.Datetime("us")).alias(date_col)).drop("__m"))
        msgs.append(f"{name_col} as dates in {int(year)} in '{date_col}'")
    elif all(months):
        order = pl.DataFrame({name_col: cols, f"{name_col}_number": months})
        out = out.join(order.lazy(), on=name_col, how="left", maintain_order="left")
    return NodeResult(out, messages=msgs)


registry.register(NodeType(
    key="unpivot", label="Columns into rows", category="Combine", icon="⤓",
    description="Turn columns like Jan, Feb, Mar … into rows, giving one row per item and month with the numbers in one column. "
                "A wide spreadsheet with a column per month or year becomes a table you can total and chart over time.",
    apply=_unpivot,
    summary=lambda p: f"{len(p.get('columns') or [])} columns into {p.get('name_column') or 'month'}",
    params=[
        Param("columns", "Columns to turn into rows", "columns", default=[], required=True),
        Param("name_column", "Call their names", "text", default="month"),
        Param("value_column", "Call their values", "text", default="value"),
        Param("year", "Year of the months (makes real dates)", "text", default="", advanced=True,
              help="With month columns (Jan … Dec) and a year, each row gets the first day of its month as a date"),
    ],
))
