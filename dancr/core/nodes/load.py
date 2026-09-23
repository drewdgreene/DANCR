"""Load File: CSV / TSV / text, Excel, Parquet, with sensible auto-detection."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from ..params import Param
from ..registry import NodeType, Ctx, NodeResult, registry
from ..timeutil import detect_datetime_format

CSV_EXT = {".csv", ".tsv", ".txt", ".dat", ".tab", ".log"}
EXCEL_EXT = {".xlsx", ".xlsm", ".xls", ".xlsb", ".ods"}
PARQUET_EXT = {".parquet", ".pq"}


def sniff_separator(path: Path, encoding: str = "utf8") -> str:
    with open(path, "r", encoding=encoding, errors="replace", newline="") as f:
        head = f.read(64 * 1024)
    lines = [ln for ln in head.splitlines() if ln.strip()][:50]
    if not lines:
        return ","
    best, best_score = ",", -1.0
    for sep in [",", "\t", ";", "|"]:
        counts = [ln.count(sep) for ln in lines]
        if not counts or max(counts) == 0:
            continue
        # consistent count across lines and > 0
        mode = max(set(counts), key=counts.count)
        consistency = counts.count(mode) / len(counts)
        score = consistency * 10 + min(mode, 20) / 20
        if mode > 0 and score > best_score:
            best, best_score = sep, score
    return best


NULLS = ["", "NA", "N/A", "NaN", "nan", "NULL", "null", "#N/A"]


def scan_file(ctx: Ctx, params: dict[str, Any]) -> tuple[pl.LazyFrame, list[str]]:
    messages: list[str] = []
    raw = params.get("path")
    if not raw or not str(raw).strip():
        raise ValueError("Choose a file to load")
    path = ctx.resolve(str(raw))
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.is_dir():
        raise ValueError(f"{path.name} is a folder. Choose a CSV, Excel or Parquet file inside it")
    if path.stat().st_size == 0:
        raise ValueError(f"{path.name} is empty")
    ext = path.suffix.lower()
    encoding = params.get("encoding") or "utf8"
    has_header = params.get("has_header")
    has_header = True if has_header is None else bool(has_header)
    skip_rows = int(params.get("skip_rows") or 0)
    if ext in PARQUET_EXT:
        lf = pl.scan_parquet(path)
    elif ext in EXCEL_EXT:
        sheet = params.get("sheet")
        kwargs: dict[str, Any] = {"has_header": has_header}
        if sheet not in (None, ""):
            if isinstance(sheet, int) or (isinstance(sheet, str) and sheet.strip().isdigit()):
                if int(sheet) < 1:
                    raise ValueError("Sheets are numbered from 1 (or give the sheet's name)")
                kwargs["sheet_id"] = int(sheet)
            else:
                kwargs["sheet_name"] = str(sheet)
        try:
            df = pl.read_excel(path, engine="calamine", read_options={"skip_rows": skip_rows} if skip_rows else None, **kwargs)
        except Exception as e:
            msg = str(e)
            if "sheet" in msg.lower():
                raise ValueError(f"Sheet {sheet!r} was not found in {path.name}. Sheets: {', '.join(list_sheets(path)) or '?'}") from e
            raise
        lf = df.lazy()
    else:
        sep = params.get("separator") or "auto"
        if sep == "auto":
            sep = sniff_separator(path, "utf8" if encoding == "utf8" else "latin-1")
        sep = {"tab": "\t", "\\t": "\t", "comma": ",", "semicolon": ";", "pipe": "|", "space": " "}.get(sep, sep)
        common = dict(separator=sep, has_header=has_header, skip_rows=skip_rows,
                      infer_schema_length=int(params.get("infer_rows") or 10000), try_parse_dates=False,
                      truncate_ragged_lines=True, ignore_errors=bool(params.get("ignore_errors", False)),
                      decimal_comma=bool(params.get("decimal_comma", False)), null_values=NULLS)
        try:
            if encoding == "utf8":
                lf = pl.scan_csv(path, encoding="utf8", **common)
                lf.head(5).collect(engine="streaming")          # surface bad-encoding errors early
            else:
                lf = pl.read_csv(path, encoding="latin-1", **common).lazy()
        except Exception as e:
            if "utf-8" in str(e).lower() or "utf8" in str(e).lower():
                raise ValueError(f"{path.name} is not UTF-8 text. Under More options set Text encoding to Latin-1 / Windows") from e
            if "NoDataError" in type(e).__name__ or "empty" in str(e).lower():
                raise ValueError(f"{path.name} has no data rows") from e
            raise
    # tidy column names, normalise datetime precision (keep time zones)
    schema = lf.collect_schema()
    dt_casts = [pl.col(c).cast(pl.Datetime("us", dt.time_zone)).alias(c) for c, dt in schema.items()
                if isinstance(dt, pl.Datetime) and dt.time_unit != "us"]
    if dt_casts:
        lf = lf.with_columns(dt_casts)
        schema = lf.collect_schema()
    renames = _tidy_names(list(schema))
    if renames:
        lf = lf.rename(renames)
        schema = lf.collect_schema()
        dup = [f"{a} → {b}" for a, b in renames.items() if a.strip() != b]
        if dup:
            messages.append("Renamed columns whose names only differed by spaces: " + ", ".join(dup))
    # optional subset
    keep = params.get("columns") or []
    if keep:
        missing = [c for c in keep if c not in schema]
        if missing:
            raise ValueError(f"These columns are not in the file: {missing}. File has: {list(schema)[:20]}")
        lf = lf.select(keep)
        schema = lf.collect_schema()
    # date parsing
    if params.get("parse_dates", True):
        forced = params.get("date_format") or None
        text_cols = [c for c, dt in schema.items() if dt in (pl.Utf8, pl.String)]
        if text_cols:
            sample = lf.select(text_cols).head(SAMPLE_ROWS).collect(engine="streaming")
            casts = []
            for c in text_cols:
                fmt = forced or detect_datetime_format(sample[c])
                if fmt:
                    e = pl.col(c).str.strip_chars().str.to_datetime(fmt, strict=False)
                    if "%z" in fmt:
                        e = e.dt.convert_time_zone("UTC")
                    casts.append(e.alias(c))
                    messages.append(f"Read '{c}' as date/time using {fmt}" + _unparsed_note(sample[c], fmt))
            if casts:
                lf = lf.with_columns(casts)
    return lf, messages


SAMPLE_ROWS = 2000


def _tidy_names(names: list[str]) -> dict[str, str]:
    """Strip surrounding spaces; names that then collide get _2, _3 … so nothing is lost."""
    out: dict[str, str] = {}
    taken: set[str] = set()
    for c in names:
        new = c.strip() or c
        base, k = new, 2
        while new in taken:
            new = f"{base}_{k}"
            k += 1
        taken.add(new)
        if new != c:
            out[c] = new
    return out


def _unparsed_note(sample: pl.Series, fmt: str) -> str:
    """How many filled cells of the sample do not match the chosen format (they become blank)."""
    filled = sample.drop_nulls().str.strip_chars()
    filled = filled.filter(filled != "")
    if len(filled) == 0:
        return ""
    bad = int(filled.str.to_datetime(fmt, strict=False).null_count())
    if not bad:
        return ""
    return f"; {bad:,} of the first {len(filled):,} values do not match and become blank (set 'Date format' under More options if that is wrong)"


def _apply(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any]) -> NodeResult:
    lf, messages = scan_file(ctx, params)
    return NodeResult(lf, messages=messages)


def _summary(p: dict[str, Any]) -> str:
    return Path(p.get("path") or "").name or "no file chosen"


registry.register(NodeType(
    key="load_file",
    label="Load file",
    category="Get data",
    icon="▤",
    kind="source",
    inputs=[],
    description="Open a CSV, text, Excel or Parquet file. Dates and numbers are detected automatically.",
    apply=_apply,
    summary=_summary,
    params=[
        Param("path", "File", "path", required=True, help="CSV, TSV, TXT, Excel (.xlsx) or Parquet"),
        Param("sheet", "Sheet", "text", default="", help="Excel only. Leave blank for the first sheet"),
        Param("has_header", "First row is column names", "bool", default=True),
        Param("skip_rows", "Skip rows at top", "int", default=0, min=0, help="Lines to ignore before the header"),
        Param("separator", "Column separator", "choice", default="auto", advanced=True,
              choices=[("auto", "detect automatically"), (",", "comma"), ("\t", "tab"), (";", "semicolon"), ("|", "pipe"), (" ", "space")]),
        Param("parse_dates", "Detect dates", "bool", default=True, advanced=True,
              help="Turn text columns that look like dates into real date/times"),
        Param("date_format", "Date format", "text", default="", advanced=True,
              help="Force a format like %d/%m/%Y %H:%M:%S when auto-detect gets it wrong"),
        Param("decimal_comma", "Numbers use decimal comma", "bool", default=False, advanced=True),
        Param("encoding", "Text encoding", "choice", default="utf8", advanced=True,
              choices=[("utf8", "UTF-8 (normal)"), ("latin1", "Latin-1 / Windows")]),
        Param("infer_rows", "Rows to inspect for types", "int", default=10000, min=100, advanced=True),
        Param("ignore_errors", "Skip values that will not parse", "bool", default=False, advanced=True),
        Param("columns", "Only load these columns", "columns", default=[], advanced=True),
    ],
))


def list_sheets(path: str | Path) -> list[str]:
    try:
        import fastexcel
        return list(fastexcel.read_excel(str(path)).sheet_names)
    except Exception:
        return []
