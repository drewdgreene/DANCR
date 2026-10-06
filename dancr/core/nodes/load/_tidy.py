"""Tidying a raw frame: names, numbers stored as text, leading-zero codes, dates and times."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from ...registry import Ctx
from ...timeutil import detect_datetime_format, settle_day_month, offset_time_zone, has_offset
from ...dtypes import LEADING_ZERO
from ._formats import EXCEL_EXT, NULLS, PARQUET_EXT, list_sheets  # noqa: E402


def _tidy_frame(ctx: Ctx, lf: pl.LazyFrame, params: dict[str, Any], messages: list[str], ext: str, path: Path) -> pl.LazyFrame:
    """Column names tidied, the chosen columns kept, dates and numbers written as text read as such."""
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
                    if has_offset(fmt):
                        tz, note = offset_time_zone(c, sample[c], (params.get("time_zone") or "").strip() or None)
                        e = e.dt.convert_time_zone(tz)
                        if note:
                            messages.append(note)
                    casts.append(e.alias(c))
                    messages.append(f"Read '{c}' as date/time using {fmt}" + _unparsed_note(sample[c], fmt))
                    if settled:
                        messages.append(settled)
            if casts:
                lf = lf.with_columns(casts)
            if forced and not forced_hits:
                messages.append(f"No column could be read as dates with the date format {forced!r}, so all columns were kept as they are")
        lf, schema = _date_and_time(lf, messages)
    if params.get("parse_numbers", True) and ext not in PARQUET_EXT:
        lf = _numbers_in_text(lf, lf.collect_schema(), messages, bool(params.get("decimal_comma", False)))
    if ext in EXCEL_EXT and params.get("sheet") in (None, ""):
        sheets = list_sheets(path)
        if len(sheets) > 1:
            messages.append(f"{path.name} has {len(sheets)} sheets ({', '.join(sheets[:6])}{'…' if len(sheets) > 6 else ''}). "
                            f"Reading the first one, {sheets[0]!r}. Choose another under Sheet")
    return lf




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
        if len(v) and v.str.contains(r"^\d+$").all() and v.str.contains(LEADING_ZERO).any():
            out.append(c)                                  # the raw header: a schema override must match it exactly
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
    from ...dtypes import text_to_number_expr
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
        if len(v) < 3 or v.str.contains(LEADING_ZERO).any() or _looks_like_a_code(c, v):
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
            note = f". {len(bad):,} of the first {len(v):,} values aren't numbers ({examples}) and are left blank"
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
    return f". {bad:,} of the first {len(filled):,} values don't match and are left blank. If that's wrong, set 'Date format' under More options"


