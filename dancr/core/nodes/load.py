"""Load File: CSV / TSV / text, Excel, Parquet, with sensible auto-detection."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from ..params import Param
from ..registry import NodeType, Ctx, NodeResult, registry
from ..timeutil import detect_datetime_format, settle_day_month

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
            if isinstance(sheet, str) and sheet.strip() in list_sheets(path):
                kwargs["sheet_name"] = sheet.strip()          # a sheet called "2024" is that sheet, not the 2024th
            elif isinstance(sheet, int) or (isinstance(sheet, str) and sheet.strip().isdigit()):
                if int(sheet) < 1:
                    raise ValueError("Sheets are numbered from 1 (or give the sheet's name)")
                kwargs["sheet_id"] = int(sheet)
            else:
                kwargs["sheet_name"] = str(sheet)
        try:
            import warnings
            with warnings.catch_warnings():      # a Polars notice about its own internals, not about the file
                warnings.simplefilter("ignore", FutureWarning)
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
        keep_text = _leading_zero_columns(path, sep, has_header, skip_rows, "utf8" if encoding == "utf8" else "latin-1")
        if keep_text:
            common["schema_overrides"] = {c: pl.Utf8 for c in keep_text}
            messages.append("Kept as text, so their leading zeros stay: " + ", ".join(keep_text))
        try:
            if encoding == "utf8":
                lf = pl.scan_csv(path, encoding="utf8", **common)
                try:
                    lf.head(5).collect(engine="streaming")      # surface bad-encoding errors early
                except Exception:
                    lf.head(5).collect()                        # some files trip only the streaming reader; try the other engine
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
            day_first = bool(params.get("day_first", False))
            forced_hits = 0
            for c in text_cols:
                if forced and not _mostly_reads(sample[c], forced):
                    continue                        # a forced format is for the date columns, not every text column
                fmt = forced or detect_datetime_format(sample[c], day_first=day_first)
                forced_hits += bool(forced)
                settled = None
                if fmt and not forced:
                    fmt, settled = settle_day_month(lf, c, fmt, sample[c], whole=not ctx.preview)
                if fmt:
                    e = pl.col(c).str.strip_chars().str.to_datetime(fmt, strict=False)
                    if "%z" in fmt:
                        e = e.dt.convert_time_zone("UTC")
                    casts.append(e.alias(c))
                    messages.append(f"Read '{c}' as date/time using {fmt}" + _unparsed_note(sample[c], fmt))
                    if settled:
                        messages.append(settled)
            if casts:
                lf = lf.with_columns(casts)
            if forced and not forced_hits:
                messages.append(f"No column reads as dates with the date format {forced!r}; every column was kept as it is")
        lf, schema = _date_and_time(lf, messages)
    if params.get("parse_numbers", True) and ext not in PARQUET_EXT:
        lf = _numbers_in_text(lf, lf.collect_schema(), messages, bool(params.get("decimal_comma", False)))
    if ext in EXCEL_EXT and params.get("sheet") in (None, ""):
        sheets = list_sheets(path)
        if len(sheets) > 1:
            messages.append(f"{path.name} has {len(sheets)} sheets ({', '.join(sheets[:6])}{'…' if len(sheets) > 6 else ''}); "
                            f"this reads the first, {sheets[0]!r}. Choose another under Sheet")
    return lf, messages


_LEADING_ZERO = r"^0\d+$"


def _leading_zero_columns(path: Path, sep: str, has_header: bool, skip_rows: int, encoding: str) -> list[str]:
    """Columns of whole numbers written with leading zeros (00123, 0042): codes and ids, which must stay text or
    the zeros are lost."""
    try:
        raw = pl.read_csv(path, separator=sep, has_header=has_header, skip_rows=skip_rows, n_rows=SAMPLE_ROWS,
                          infer_schema=False, truncate_ragged_lines=True, encoding=encoding, null_values=NULLS)
    except Exception:  # noqa: BLE001 - the real read reports any problem with the file
        return []
    out = []
    for c in raw.columns:
        v = raw[c].drop_nulls().str.strip_chars()
        v = v.filter(v != "")
        if len(v) and v.str.contains(r"^\d+$").all() and v.str.contains(_LEADING_ZERO).any():
            out.append(c.strip() or c)
    return out


_NUMBER_TEXT = r"^[(\-+]?\s*[$€£¥]?\s*[+-]?[\d.,' ]*\d[\d.,' ]*\s*%?\s*[)]?$"


CODE_WORDS = {"phone", "tel", "telephone", "mobile", "cell", "fax", "zip", "zipcode", "postcode", "postal", "code", "id",
              "sku", "ref", "reference", "account", "iban", "sort", "vat", "ean", "upc", "isbn", "barcode", "pin"}


def _looks_like_a_code(name: str, v: pl.Series) -> bool:
    """Phone numbers, postcodes, account and reference numbers are written with digits but are not quantities:
    their name says so, or they start with + or 0 once spaces are taken out."""
    import re
    words = {w.lower() for w in re.findall(r"[A-Za-z]+", name)}
    if words & CODE_WORDS:
        return True
    squeezed = v.str.replace_all(r"[\s\-]", "")
    return bool(squeezed.str.contains(r"^\+\d{6,}$").any() or squeezed.str.contains(r"^0\d{5,}$").any())


def _number_expr(col: str, decimal_comma: bool) -> pl.Expr:
    """Text like '£1,234.50', '(1,234)', '31.5%', '1 234,5' as the number a person means (a percent stays 31.5)."""
    from ..dtypes import text_to_number_expr
    s = pl.col(col).cast(pl.Utf8).str.strip_chars()
    negative = s.str.starts_with("(") & s.str.ends_with(")")
    body = s.str.replace_all(r"[()$€£¥%\s']", "")
    if decimal_comma:
        body = body.str.replace_all(".", "", literal=True).str.replace(",", ".", literal=True)
    n = text_to_number_expr(body)
    return pl.when(negative).then(-n).otherwise(n)


def _numbers_in_text(lf: pl.LazyFrame, schema, messages: list[str], decimal_comma: bool) -> pl.LazyFrame:
    """Text columns that are really numbers — with thousands separators, currency signs or percent signs, or a few
    notes like 'absent' among them — become numbers. At least nine in ten filled values must read as numbers;
    codes with leading zeros stay text. What could not be read is said."""
    text_cols = [c for c, dt in schema.items() if dt in (pl.Utf8, pl.String)]
    if not text_cols:
        return lf
    sample = lf.select(text_cols).head(SAMPLE_ROWS).collect(engine="streaming")
    casts = []
    for c in text_cols:
        v = sample[c].drop_nulls().str.strip_chars()
        v = v.filter(v != "")
        if len(v) < 3 or v.str.contains(_LEADING_ZERO).any() or _looks_like_a_code(c, v):
            continue
        looks = v.str.contains(_NUMBER_TEXT)
        parsed = pl.DataFrame({c: v}).select(_number_expr(c, decimal_comma))[c]
        ok = looks & parsed.is_not_null()
        share = float(ok.mean())
        if share < 0.9:
            continue
        casts.append(_number_expr(c, decimal_comma).alias(c))
        bad = v.filter(~ok)
        note = ""
        if len(bad):
            examples = ", ".join(repr(x) for x in bad.unique(maintain_order=True).head(3).to_list())
            note = f"; {len(bad):,} of the first {len(v):,} values are not numbers ({examples}) and are blank"
        signs = "".join(sorted({ch for ch in "".join(v.head(200).to_list()) if ch in "$€£¥%,()"}))
        messages.append(f"Read '{c}' as numbers" + (f" (ignoring {signs})" if signs else "") + note)
    return lf.with_columns(casts) if casts else lf


_TIME_OF_DAY = r"^\d{1,2}:\d{2}(:\d{2}(\.\d+)?)?(\s?[AaPp][Mm])?$"


def _date_and_time(lf: pl.LazyFrame, messages: list[str]):
    """A date column and a time-of-day column (Date 03/04/2024, Time 14:05:00) become one date/time column as
    well, named after both, so readings can be placed in time to the second."""
    schema = lf.collect_schema()
    dates = [c for c, dt in schema.items() if dt == pl.Date or (isinstance(dt, pl.Datetime) and _all_midnight(lf, c))]
    texts = [c for c, dt in schema.items() if dt in (pl.Utf8, pl.String)]
    if len(dates) != 1 or not texts:
        return lf, schema
    sample = lf.select(texts).head(SAMPLE_ROWS).collect(engine="streaming")
    times = []
    for c in texts:
        v = sample[c].drop_nulls().str.strip_chars()
        v = v.filter(v != "")
        if len(v) and float(v.str.contains(_TIME_OF_DAY).mean()) >= 0.9:
            times.append(c)
    if len(times) != 1:
        return lf, schema
    d, t = dates[0], times[0]
    name = f"{d} {t}"
    if name in schema:
        return lf, schema
    tod = pl.col(t).str.strip_chars().str.to_uppercase()
    parsed = pl.coalesce([tod.str.to_time(f, strict=False) for f in ("%H:%M:%S%.f", "%H:%M:%S", "%H:%M", "%I:%M:%S %p", "%I:%M %p", "%I:%M%p")])
    combined = pl.col(d).cast(pl.Date).cast(pl.Datetime("us")) + (parsed.cast(pl.Duration("ns")).cast(pl.Duration("us")))
    lf = lf.with_columns(combined.alias(name))
    messages.append(f"Combined '{d}' and '{t}' into '{name}' (a date and a time of day)")
    return lf, lf.collect_schema()


def _all_midnight(lf: pl.LazyFrame, c: str) -> bool:
    try:
        s = lf.select(pl.col(c)).head(SAMPLE_ROWS).collect(engine="streaming")[c].drop_nulls()
        return len(s) > 0 and bool((s.dt.hour() == 0).all() and (s.dt.minute() == 0).all() and (s.dt.second() == 0).all())
    except Exception:  # noqa: BLE001
        return False


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


def _mostly_reads(sample: pl.Series, fmt: str, share: float = 0.5) -> bool:
    """True when at least ``share`` of the filled cells in the sample read as dates with ``fmt``."""
    filled = sample.drop_nulls().str.strip_chars()
    filled = filled.filter(filled != "")
    if len(filled) == 0:
        return False
    try:
        ok = len(filled) - int(filled.str.to_datetime(fmt, strict=False).null_count())
    except Exception:  # noqa: BLE001 - a format Polars cannot use reads nothing
        return False
    return ok >= share * len(filled)


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
              help="Turn text columns that look like dates into real date/times (a date and a time-of-day column are also combined)"),
        Param("parse_numbers", "Detect numbers written as text", "bool", default=True, advanced=True,
              help="Read '1,234.50', '£99', '31.5%' and '(120)' as numbers"),
        Param("date_format", "Date format", "text", default="", advanced=True,
              help="Force a format like %d/%m/%Y %H:%M:%S when auto-detect gets it wrong"),
        Param("day_first", "Day comes before month (01/05 = 1 May)", "bool", default=False, advanced=True,
              help="Only matters when every date could be read either way"),
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


def tables_in(path: str | Path) -> list[tuple[str, dict[str, Any]]]:
    """The tables a file holds, as (title, load settings): one for a CSV or Parquet file, one per sheet of a
    workbook with several (Orders, Customers …), so dropping a workbook brings in every table in it."""
    p = Path(path)
    if p.suffix.lower() in EXCEL_EXT:
        sheets = list_sheets(p)
        if len(sheets) > 1:
            return [(sheet, {"sheet": sheet}) for sheet in sheets]
    return [(p.stem, {})]
