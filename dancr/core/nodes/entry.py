"""Type in a table, fix values, and Excel workbook output."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from ..params import Param
from ..registry import NodeType, InputSpec, Ctx, NodeResult, registry
from ._common import first_input, schema_of
from ..dtypes import datetime_literal, temp_name, text_to_bool, text_to_bool_expr, text_to_number_expr, number_from_text, is_date
from .outputs import excel_frame

TYPE_CHOICES = [("number", "number"), ("text", "text"), ("datetime", "date / time"), ("bool", "true / false")]


def _cast(series: pl.Series, kind: str) -> pl.Series:
    s = series.cast(pl.Utf8)
    if kind == "number":
        return pl.select(text_to_number_expr(s)).to_series()
    if kind == "datetime":
        from ..timeutil import detect_datetime_format
        fmt = detect_datetime_format(s) or "%Y-%m-%d %H:%M:%S"
        return s.str.strip_chars().str.to_datetime(fmt, strict=False)
    if kind == "bool":
        return pl.select(text_to_bool_expr(s)).to_series()
    return s


def _enter(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any]) -> NodeResult:
    cols = params.get("columns") or []
    if not cols:
        raise ValueError("Add at least one column")
    names = [str(c.get("name") or "").strip() or f"column_{i + 1}" for i, c in enumerate(cols)]
    if len(set(names)) != len(names):
        raise ValueError("Column names must be different from each other")
    rows = params.get("rows") or []
    width = len(names)
    data = {n: [] for n in names}
    for r in rows:
        r = list(r) + [None] * (width - len(r))
        for n, v in zip(names, r[:width]):
            data[n].append(None if v in ("", None) else str(v))
    df = pl.DataFrame({n: pl.Series(n, data[n], dtype=pl.Utf8) for n in names})
    df = df.with_columns([_cast(df[n], str(c.get("type") or "text")).alias(n) for n, c in zip(names, cols)])
    return NodeResult(df.lazy(), messages=[f"{len(rows)} rows typed in"])


registry.register(NodeType(
    key="enter_data", label="Type in a table", category="Get data", icon="⌨",
    description="A small table you fill in yourself: a short list of items, a lookup table, a few new values. Paste from Excel works.",
    apply=_enter, kind="source", inputs=[],
    summary=lambda p: f"{len(p.get('rows') or [])} rows × {len(p.get('columns') or [])} columns",
    params=[
        Param("columns", "Columns", "table_columns", default=[{"name": "column_1", "type": "text"}]),
        Param("rows", "Rows", "table_rows", default=[]),
    ],
))


# --------------------------------------------------------------- fix values
def _fix(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any]) -> NodeResult:
    lf = first_input(inputs)
    schema = schema_of(lf)
    fixes = [f for f in (params.get("fixes") or []) if f.get("column") in schema and f.get("row") is not None]
    if not fixes:
        return NodeResult(lf)
    rowc = temp_name("row", schema)
    lf = lf.with_row_index(rowc)
    exprs: dict[str, pl.Expr] = {}
    msgs = []
    for f in fixes:
        col = f["column"]; row = int(f["row"]) - 1
        if row < 0:
            raise ValueError(f"{col}: rows are numbered from 1, so row {row + 1} does not exist")
        dt = schema[col]
        val = f.get("value")
        lit: pl.Expr
        if val in (None, ""):
            lit = pl.lit(None).cast(dt)
        elif dt.is_numeric():
            num = number_from_text(val, f"Row {row + 1}, {col}")
            if dt.is_integer() and num != int(num):
                raise ValueError(f"Row {row + 1}, {col}: {val!r} has decimals but the column holds whole numbers. Convert the column with 'Fix numbers and dates' first.")
            lit = pl.lit(num).cast(dt)
        elif isinstance(dt, (pl.Datetime, pl.Date)):
            lit = datetime_literal(val, dt, f"row {row + 1}")
            if is_date(dt):
                lit = lit.cast(pl.Date)
        elif dt == pl.Boolean:
            lit = pl.lit(text_to_bool(val))
        else:
            lit = pl.lit(str(val))
        prev = exprs.get(col, pl.col(col))
        exprs[col] = pl.when(pl.col(rowc) == row).then(lit).otherwise(prev)
        msgs.append(f"Row {row + 1}: {col} {f.get('was')!r} → {val!r}" + (f" ({f['note']})" if f.get("note") else ""))
    out = lf.with_columns([e.alias(c) for c, e in exprs.items()]).drop(rowc)
    return NodeResult(out, messages=msgs)


registry.register(NodeType(
    key="fix_values", label="Fix values", category="Clean up", icon="✎",
    description="Correct individual cells. Every correction is recorded here with the old value and a note, so nothing is hidden.",
    apply=_fix,
    summary=lambda p: f"{len(p.get('fixes') or [])} corrections",
    params=[Param("fixes", "Corrections", "fixes", default=[])],
))


# --------------------------------------------------------------- workbook
def _workbook(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any]) -> NodeResult:
    frames = inputs.get("items") or []
    if not frames:
        raise ValueError("Connect the tables you want as sheets")
    path = (params.get("path") or "").strip()
    if not path:
        raise ValueError("Choose where to save the workbook (an .xlsx file)")
    out = ctx.resolve(path)
    if out.suffix.lower() != ".xlsx":
        raise ValueError("Save the workbook as an .xlsx file")
    if ctx.preview:
        return NodeResult(frames[0], messages=[f"Will write {out.name} when the pipeline runs"])
    import os
    from xlsxwriter import Workbook
    meta = (ctx.upstream_meta or {}).get("items") or []
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.{os.getpid()}.tmp.xlsx")
    used: set[str] = set()
    try:
        with Workbook(str(tmp)) as wb:
            for i, lf in enumerate(frames):
                title = (meta[i].get("title") if i < len(meta) else None) or f"Sheet {i + 1}"
                name = "".join(ch for ch in title if ch not in '[]:*?/\\')[:31] or f"Sheet {i + 1}"
                base, k = name, 2
                while name.lower() in used:
                    name = f"{base[:28]} {k}"; k += 1
                used.add(name.lower())
                excel_frame(lf, f"'{title}'").write_excel(wb, worksheet=name, autofit=True)
        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)
    return NodeResult(frames[0], messages=[f"Saved {len(frames)} sheets to {out}"], report={"path": str(out), "sheets": len(frames)})


registry.register(NodeType(
    key="workbook", label="Excel workbook", category="Share", icon="▦",
    description="Save several tables into one Excel file, one sheet each, named after the steps.",
    apply=_workbook, kind="sink", materialize=False,
    inputs=[InputSpec("items", "Tables", multiple=True)],
    summary=lambda p: Path(p.get("path") or "").name or "no file chosen",
    params=[Param("path", "Save as", "path", required=True, help="An .xlsx file")],
))
