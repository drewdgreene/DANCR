"""Type in a table, fix values, and Excel workbook output."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from ..params import Param
from ..registry import NodeType, InputSpec, Ctx, NodeResult, registry
from ._common import private_temp, first_input, schema_of
from ..dtypes import datetime_literal, temp_name, text_to_bool, text_to_bool_expr, text_to_number_expr, typed_value, is_date
from .outputs import excel_frame

TYPE_CHOICES = [("number", "number"), ("text", "text"), ("datetime", "date / time"), ("bool", "true / false")]


def _cast(series: pl.Series, kind: str) -> pl.Series:
    s = series.cast(pl.Utf8)
    if kind == "number":
        return pl.select(text_to_number_expr(s)).to_series()
    if kind == "datetime":
        # a typed-in table is small and people mix "2024-06-03 09:00" with "2024-06-10": read each cell
        # on its own rather than forcing one format on the whole column
        from ..timeutil import detect_datetime_format
        out = []
        for v in s.to_list():
            fmt = detect_datetime_format(pl.Series([v]), 1.0) if v is not None else None
            out.append(pl.Series([v]).str.strip_chars().str.to_datetime(fmt, strict=False).cast(pl.Datetime("us"))[0] if fmt else None)
        return pl.Series(series.name, out, dtype=pl.Datetime("us"))
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
    typed = df.with_columns([_cast(df[n], str(c.get("type") or "text")).alias(n) for n, c in zip(names, cols)])
    msgs = [f"{len(rows)} rows typed in"]
    for n, c in zip(names, cols):
        lost = df.filter(pl.col(n).is_not_null() & typed[n].is_null())[n].to_list()
        if lost:
            what = {"number": "numbers", "datetime": "dates"}.get(str(c.get("type")), "values")
            msgs.append(f"{n}: {len(lost)} cells could not be read as {what} and are blank: " + ", ".join(repr(v) for v in lost[:5])
                        + (" …" if len(lost) > 5 else ""))
    return NodeResult(typed.lazy(), messages=msgs)


registry.register(NodeType(
    key="enter_data", label="Type in a table", category="Get data", icon="⌨",
    description="A small table you type in yourself, such as a short list or a lookup table. You can paste from Excel.",
    apply=_enter, kind="source", inputs=[],
    summary=lambda p: f"{len(p.get('rows') or [])} rows × {len(p.get('columns') or [])} columns",
    params=[
        Param("columns", "Columns", "table_columns", default=[{"name": "column_1", "type": "text"}]),
        Param("rows", "Rows", "table_rows", default=[]),
    ],
))


# --------------------------------------------------------------- fix values
def _stale_fixes(lf: pl.LazyFrame, fixes: list[dict[str, Any]], rowc: str) -> list[str]:
    """Corrections that no longer fit the data: the row is past the end of the table, or the cell no longer
    holds the value it had when it was corrected (rows were added, removed or reordered upstream). A
    correction without a "was" is not checked; an earlier correction of the same cell counts as its value."""
    schema = lf.collect_schema()
    rows = sorted({int(f["row"]) - 1 for f in fixes if int(f["row"]) >= 1})
    cols = sorted({f["column"] for f in fixes})
    n = int(lf.select(pl.len()).collect(engine="streaming")[0, 0])
    cells = lf.with_row_index(rowc).filter(pl.col(rowc).is_in(rows)).select([rowc, *cols]).collect(engine="streaming")
    now: dict[tuple[int, str], Any] = {(r[rowc], c): r[c] for r in cells.iter_rows(named=True) for c in cols}
    out = []
    for f in fixes:
        row, col = int(f["row"]) - 1, f["column"]
        if row < 0:
            continue
        if row >= n:
            out.append(f"row {row + 1}, {col}: the table has only {n:,} rows")
            continue
        cur = now.get((row, col))
        if "was" in f and not _same(cur, f["was"], schema[col]):
            out.append(f"row {row + 1}, {col}: it holds {_shown(cur)} now but held {_shown(f['was'])} when it was corrected")
        now[(row, col)] = f.get("value")          # what a later correction of this cell sees
    return out


def _shown(v: Any) -> str:
    return "nothing" if v is None or (isinstance(v, str) and not v.strip()) else repr(str(v))


def _same(cur: Any, was: Any, dtype: pl.DataType) -> bool:
    """Whether a cell (a value, or the text of an earlier correction) still holds the recorded value."""
    def blank(v: Any) -> bool:
        return v is None or (isinstance(v, str) and not v.strip())
    if blank(was) or blank(cur):
        return blank(was) and blank(cur)
    if str(cur) == str(was):
        return True
    if isinstance(cur, bool):
        return str(was).strip().lower() in ("true", "false", "yes", "no", "1", "0", "y", "n", "t", "f", "on", "off") \
            and text_to_bool(was) == cur
    a, b = _as_number(str(cur)) if isinstance(cur, str) else (float(cur) if isinstance(cur, (int, float)) else None), _as_number(str(was))
    if a is not None or b is not None:
        return a is not None and a == b
    if isinstance(dtype, (pl.Datetime, pl.Date)):
        try:
            return pl.select(datetime_literal(was, dtype).alias("w"), datetime_literal(cur, dtype).alias("c")
                             if isinstance(cur, str) else pl.lit(cur).cast(pl.Datetime("us") if is_date(dtype) else dtype).alias("c")) \
                .select(pl.col("w") == pl.col("c")).item() is True
        except (ValueError, pl.exceptions.PolarsError):
            return False
    return False


def _as_number(text: str) -> float | None:
    from ..dtypes import typed_value
    v = typed_value(text)
    return None if v is None else float(v)


def _fix(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any]) -> NodeResult:
    lf = first_input(inputs)
    schema = schema_of(lf)
    all_fixes = [f for f in (params.get("fixes") or []) if f.get("row") is not None]
    fixes = [f for f in all_fixes if f.get("column") in schema]
    msgs = []
    gone = sorted({str(f.get("column")) for f in all_fixes if f.get("column") not in schema})
    if gone:
        msgs.append(f"{len(all_fixes) - len(fixes)} corrections are for columns that are no longer in the table "
                    f"({', '.join(gone)}) and changed nothing")
    if not fixes:
        return NodeResult(lf, messages=msgs)
    if ctx.preview and ctx.sample != "all":
        # a preview of a big table is a sample, whose row numbers are not the table's: a correction would
        # land on the wrong row
        return NodeResult(lf, messages=msgs + [f"The {len(fixes)} correction{'s are' if len(fixes) != 1 else ' is'} not shown in this "
                                               "preview, which is a sample of the rows. Run the step to see them."])
    rowc = temp_name("row", schema)
    stale = _stale_fixes(lf, fixes, rowc)
    if stale:
        # never write into a cell that is not the one that was corrected
        raise ValueError(f"{len(stale)} correction{'s no longer fit' if len(stale) != 1 else ' no longer fits'} the data because rows above this step were "
                         "added, removed or reordered. Nothing was changed. " + "; ".join(stale[:5])
                         + (" …" if len(stale) > 5 else "") + ". Remove or redo these corrections.")
    lf = lf.with_row_index(rowc)
    exprs: dict[str, pl.Expr] = {}
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
            num = typed_value(str(val))                 # whole numbers stay exact (no trip through float)
            if num is None:
                raise ValueError(f"Row {row + 1}, {col}: {val!r} is not a number")
            if dt.is_integer() and not float(num).is_integer():
                raise ValueError(f"Row {row + 1}, {col}: {val!r} has decimals but the column holds whole numbers. Convert the column with 'Fix numbers and dates' first.")
            lit = pl.lit(int(num) if dt.is_integer() else num).cast(dt)
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
    description="Correct individual cells. Each correction is kept here with the old value and a note.",
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
    out = ctx.resolve_output(path)
    if out.suffix.lower() != ".xlsx":
        raise ValueError("Save the workbook as an .xlsx file")
    if ctx.preview:
        return NodeResult(frames[0], messages=[f"Will write {out.name} when the project runs"])
    import os
    from xlsxwriter import Workbook
    meta = (ctx.upstream_meta or {}).get("items") or []
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = private_temp(out)
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
    return NodeResult(frames[0], messages=[f"Saved {len(frames)} sheets to {out}"], report={"path": str(out), "sheets": len(frames)},
                      files=[out])


registry.register(NodeType(
    key="workbook", label="Excel workbook", category="Share", icon="▦",
    description="Save several tables into one Excel file, one sheet each, named after the steps.",
    apply=_workbook, kind="sink", materialize=False,
    inputs=[InputSpec("items", "Tables", multiple=True)],
    summary=lambda p: Path(p.get("path") or "").name or "no file chosen",
    params=[Param("path", "Save as", "path", required=True, help="An .xlsx file")],
))
