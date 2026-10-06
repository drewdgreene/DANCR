"""Load File: CSV / TSV / text, Excel, Parquet, with sensible auto-detection."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from ..params import Param
from ..registry import NodeType, Ctx, NodeResult, registry
from ..timeutil import detect_datetime_format, settle_day_month, offset_time_zone, has_offset
from ..dtypes import LEADING_ZERO

CSV_EXT = {".csv", ".tsv", ".txt", ".dat", ".tab", ".log"}
EXCEL_EXT = {".xlsx", ".xlsm", ".xls", ".xlsb", ".ods"}
PARQUET_EXT = {".parquet", ".pq"}
GEOJSON_EXT = {".geojson", ".json"}
VECTOR_EXT = {".gpkg", ".shp"}

_COMPRESS_SUFFIXES = (".gz", ".bgz")


def effective_ext(path: Path) -> str:
    """The format extension, seeing through a trailing .gz/.bgz (so x.vcf.gz is a .vcf)."""
    if path.suffix.lower() in _COMPRESS_SUFFIXES and len(path.suffixes) >= 2:
        return path.suffixes[-2].lower()
    return path.suffix.lower()

# Formats that are not tables. They are refused with a plain reason and a route forward, so a file is never
# silently misread as CSV (a FASTA became thousands of rows; an HDF5 file loaded as empty; a JSONL row as columns).
H5_EXT = {".h5", ".hdf5", ".hdf", ".he5"}
NC_EXT = {".nc", ".nc4", ".cdf", ".netcdf"}
NON_TABLE_EXT = {
    ".bcf": "a BCF variant file", ".embl": "an EMBL file",
    ".jsonl": "a line-delimited JSON file", ".ndjson": "a line-delimited JSON file",
    ".zip": "a zip archive", ".gz": "a compressed file", ".bz2": "a compressed file", ".xz": "a compressed file",
    ".tar": "a tar archive", ".xml": "an XML file", ".md": "a text document",
    ".yaml": "a YAML file", ".yml": "a YAML file",
    ".npy": "a NumPy array", ".npz": "a NumPy archive", ".mat": "a MATLAB file",
}

# Documents and images: not tables, read with the 'Load document' step (MinerU) — see dancr/core/nodes/document.py.
DOC_EXT = {
    ".pdf": "a PDF", ".doc": "a Word document", ".docx": "a Word document",
    ".ppt": "a PowerPoint file", ".pptx": "a PowerPoint file", ".rtf": "an RTF document",
    ".odt": "an OpenDocument text", ".ods": "an OpenDocument spreadsheet", ".odp": "an OpenDocument presentation",
    ".epub": "an EPUB", ".ofd": "an OFD document", ".html": "a web page", ".htm": "a web page",
    ".mhtml": "a web archive", ".mht": "a web archive",
}
IMAGE_EXT = {".png": "an image", ".jpg": "an image", ".jpeg": "an image", ".webp": "an image", ".gif": "an image",
             ".bmp": "an image", ".tif": "an image", ".tiff": "an image", ".jp2": "an image"}


# Scientific formats that DANCR reads with their own step (see dancr/core/nodes/bio.py): the guard points at it.
BIO_EXT = {
    ".fa": ("a FASTA sequence file", "Load sequences (FASTA/FASTQ)"),
    ".fasta": ("a FASTA sequence file", "Load sequences (FASTA/FASTQ)"),
    ".fna": ("a FASTA sequence file", "Load sequences (FASTA/FASTQ)"),
    ".faa": ("a FASTA protein file", "Load sequences (FASTA/FASTQ)"),
    ".ffn": ("a FASTA sequence file", "Load sequences (FASTA/FASTQ)"),
    ".fq": ("a FASTQ sequence file", "Load sequences (FASTA/FASTQ)"),
    ".fastq": ("a FASTQ sequence file", "Load sequences (FASTA/FASTQ)"),
    ".vcf": ("a VCF variant file", "Load variants (VCF)"),
    ".gff": ("a GFF/GTF feature file", "Load features (GFF/GTF/BED)"),
    ".gff3": ("a GFF3 feature file", "Load features (GFF/GTF/BED)"),
    ".gtf": ("a GTF feature file", "Load features (GFF/GTF/BED)"),
    ".bed": ("a BED interval file", "Load features (GFF/GTF/BED)"),
    ".gb": ("a GenBank file", "Load GenBank features"),
    ".gbk": ("a GenBank file", "Load GenBank features"),
    ".genbank": ("a GenBank file", "Load GenBank features"),
    ".ped": ("a PLINK genotype file", "Load markers (PLINK)"),
    ".bim": ("a PLINK .bim file", "Load markers (PLINK)"),
    ".fam": ("a PLINK .fam file", "Load markers (PLINK)"),
}


def _refuse_non_table(path: Path, ext: str) -> None:
    """Raise a plain reason for a file that is not a table, so it is never misread as CSV."""
    if ext in H5_EXT:
        raise ValueError(f"{path.name} is an HDF5 file, not a table. Add a 'Load HDF5' step and name the dataset inside it.")
    if ext in NC_EXT:
        raise ValueError(f"{path.name} is a NetCDF file, not a table. Add a 'Load NetCDF' step and name the variable.")
    if ext in BIO_EXT:
        what, step = BIO_EXT[ext]
        raise ValueError(f"{path.name} is {what}, not a plain table. Add a '{step}' step to read it.")
    if ext in DOC_EXT:
        raise ValueError(f"{path.name} is {DOC_EXT[ext]}, not a table. Add a 'Load document (PDF/Office)' step "
                         "to read it with MinerU.")
    if ext in IMAGE_EXT:
        raise ValueError(f"{path.name} is {IMAGE_EXT[ext]}. If it is a scanned page, screenshot or chart of data, "
                         "add a 'Load document (PDF/Office)' step to read it (OCR); otherwise it is not a table.")
    if ext in NON_TABLE_EXT:
        raise ValueError(f"{path.name} is {NON_TABLE_EXT[ext]}, not a table DANCR can read. Convert it to CSV, "
                         "Parquet or Excel, or read it with the tool that writes it.")


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


def scan_file(ctx: Ctx, params: dict[str, Any]) -> tuple[pl.LazyFrame, list[str], dict[str, Any]]:
    """A file as a table: (the rows, what was noticed on the way, a report of how the sheet was laid out)."""
    messages: list[str] = []
    report: dict[str, Any] = {}
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
    ext = effective_ext(path)
    _refuse_non_table(path, ext)
    if ext in GEOJSON_EXT:
        from ..geo import is_geojson, geojson_table
        if is_geojson(path):
            df = geojson_table(path)
            messages.append(f"Read a GeoJSON file: {len(df):,} feature{'s' if len(df) != 1 else ''}. "
                            "Each feature's properties are columns, with 'geometry' (WKT) and a centre point")
            return df.lazy(), messages, {"geojson": True}
        raise ValueError(f"{path.name} is JSON, not a table. 'Load from a URL' can read a JSON table over the "
                         "network; for a local file, convert it to CSV or Parquet.")
    if ext in VECTOR_EXT:
        from ..geo import vector_table
        layer = str(params.get("layer") or "").strip() or None
        df, notes = vector_table(path, layer)
        return df.lazy(), messages + notes, {"vector": True, "layer": layer or ""}
    encoding = params.get("encoding") or "utf8"
    has_header = params.get("has_header")
    has_header = True if has_header is None else bool(has_header)
    skip_rows = int(params.get("skip_rows") or 0)
    sep = None
    if ext not in PARQUET_EXT and ext not in EXCEL_EXT:
        sep = params.get("separator") or "auto"
        if sep == "auto":
            sep = sniff_separator(path, "utf8" if encoding == "utf8" else "latin-1")
        sep = {"tab": "\t", "\\t": "\t", "comma": ",", "semicolon": ";", "pipe": "|", "space": " "}.get(sep, sep)
    sheet = params.get("sheet")
    table = None
    retype: dict[str, pl.DataType] = {}
    keep_text: list[str] = []
    if has_header and ext not in PARQUET_EXT and (params.get("layout") or "auto") == "auto":
        table, k, n = _sheet_table(path, ext, sheet, sep, encoding, params, skip_rows or None)
    if table is not None:
        report["layout"] = table.describe()
        report["layout"]["tables_on_sheet"] = n
        messages += table.notes
        h = table.parts[0].header
        if n > 1:
            named = f" (“{table.title}”)" if table.title else ""
            messages.insert(0, f"This sheet holds {n} tables. Reading table {k}{named}, whose column names are in row {h + 1}. "
                               "Set 'Table on the sheet' to read another")
        elif table.plain and h and not skip_rows:
            messages.insert(0, f"Skipped {h} line{'s' if h > 1 else ''} above the column names because they look like a "
                               "title. Set 'Skip rows at top' to change that")
        if table.plain:
            skip_rows = h
        else:
            lf = _parts(path, ext, sheet, sep, encoding, table)
            return _tidy_frame(ctx, lf, params, messages, ext, path), messages, report
    if ext in PARQUET_EXT:
        lf = pl.scan_parquet(path)
    elif ext in EXCEL_EXT:
        kwargs: dict[str, Any] = {"has_header": has_header}
        if sheet not in (None, ""):
            ref = _sheet_ref(path, sheet)             # the same sheet the layout was read from, by one rule
            if isinstance(ref, str):
                kwargs["sheet_name"] = ref
            else:
                kwargs["sheet_id"] = ref + 1
        try:
            import warnings
            with warnings.catch_warnings():      # a Polars notice about its own internals, not about the file
                warnings.simplefilter("ignore", FutureWarning)
                # the header is the row after the skipped ones (calamine's skip_rows would skip data rows under it); a table
                # that ends above the bottom of the sheet is read to its end only, so rows below do not change its types
                opts: dict[str, Any] = {"header_row": skip_rows} if skip_rows else {}
                if table is not None and table.parts[0].stop is not None:
                    opts["n_rows"] = max(0, table.parts[0].stop - table.parts[0].first)
                df = pl.read_excel(path, engine="calamine", read_options=opts or None, **kwargs)
        except Exception as e:
            msg = str(e)
            if "sheet" in msg.lower():
                raise ValueError(f"Sheet {sheet!r} was not found in {path.name}. Sheets: {', '.join(list_sheets(path)) or '?'}") from e
            raise
        lf = df.lazy()
    else:
        common = dict(separator=sep, has_header=has_header, skip_rows=skip_rows,
                      infer_schema_length=int(params.get("infer_rows") or 10000), try_parse_dates=False,
                      truncate_ragged_lines=True, ignore_errors=bool(params.get("ignore_errors", False)),
                      decimal_comma=bool(params.get("decimal_comma", False)), null_values=NULLS)
        if table is not None and table.parts[0].stop is not None:
            common["n_rows"] = max(0, table.parts[0].stop - table.parts[0].first)
        if table is not None and table.parts[0].cut:
            # a big file with summary rows at the bottom: read up to them. The file is read in parallel pieces, so
            # the columns are read as text and given the types the first rows show once those rows are left off
            common["n_rows"] = max(0, _record_count(path, sep, encoding) - table.parts[0].first - table.parts[0].cut)
            try:
                retype = {c: dt for c, dt in pl.scan_csv(path, encoding="utf8-lossy", **{**common, "n_rows": 10_000})
                          .collect_schema().items() if dt != pl.Utf8}
            except Exception:  # noqa: BLE001 - then the file is read as usual and any problem reported
                retype = {}
            if retype:
                common["schema_overrides"] = {c: pl.Utf8 for c in retype}
        keep_text = _leading_zero_columns(path, sep, has_header, skip_rows, "utf8" if encoding == "utf8" else "latin-1")
        if keep_text:
            common["schema_overrides"] = {**(common.get("schema_overrides") or {}), **{c: pl.Utf8 for c in keep_text}}
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
    if retype:
        # strict: a value further down that is not of the type the first rows show fails the step with the value
        # named (as reading the file whole would), rather than quietly becoming a blank
        lf = lf.with_columns([pl.col(c).str.strip_chars().cast(dt, strict=True) for c, dt in retype.items() if c not in keep_text])
    if table is not None:
        lf = _without_bottom(lf, table.parts[0])
        names = lf.collect_schema().names()
        if len(names) > len(table.parts[0].cols):
            lf = lf.select(names[:len(table.parts[0].cols)])     # another table beside this one is not part of it
    return _tidy_frame(ctx, lf, params, messages, ext, path), messages, report


def _sheet_table(path: Path, ext: str, sheet: Any, sep: str | None, encoding: str, params: dict[str, Any],
                 header: int | None):
    """How the sheet is laid out (core.layout): the table to read from it, or None when its cells cannot be read
    (the usual read then reports why)."""
    from ..layout import excel_grid, csv_grid
    grid = excel_grid(path, _sheet_ref(path, sheet)) if ext in EXCEL_EXT else csv_grid(path, sep or ",", encoding)
    if grid is None:
        return None, 0, 0
    if grid.width == 0 or not any(any(c is not None for c in r) for r in grid.rows):
        where = f"The sheet {sheet!r} of {path.name}" if sheet not in (None, "") else (
            f"The first sheet of {path.name}" if ext in EXCEL_EXT else path.name)
        raise ValueError(f"{where} has nothing in it")
    tables = _tables_of(grid, header)
    if not tables:
        return None, 0, 0
    k = int(params.get("table") or 1)
    if not 1 <= k <= len(tables):
        raise ValueError(f"This sheet has {len(tables)} table{'s' if len(tables) != 1 else ''}. Set 'Table on the sheet' "
                         f"to a number from 1 to {len(tables)}")
    return tables[k - 1], k, len(tables)


_TABLES: dict[tuple[int, int | None], list] = {}


def _tables_of(grid, header: int | None) -> list:
    """The tables on a sheet, worked out once per grid read (the grids are kept per file, so a preview, a sample and
    a run of the same file all use one reading)."""
    from ..layout import find_tables
    key = (id(grid), header)
    if key not in _TABLES:
        if len(_TABLES) > 32:
            _TABLES.clear()
        _TABLES[key] = (grid, find_tables(grid, header))      # the grid is kept with it, so its id is not reused
    return _TABLES[key][1]


def _sheet_ref(path: Path, sheet: Any) -> int | str:
    """A Sheet setting as fastexcel names a sheet: its name (a sheet called "2024" is that sheet, whether the setting
    is the text or the number 2024), else its place counted from 0."""
    if sheet in (None, ""):
        return 0
    name = str(sheet).strip()
    if name in list_sheets(path):
        return name
    if name.isdigit():
        if int(name) < 1:
            raise ValueError("Sheets are numbered from 1 (or give the sheet's name)")
        return int(name) - 1
    return name


def _without_bottom(lf: pl.LazyFrame, p) -> pl.LazyFrame:
    """A plain table without the summary and empty rows under its data."""
    return lf.head(max(0, p.stop - p.first)) if p.stop is not None else lf


def _record_count(path: Path, sep: str, encoding: str) -> int:
    """The records of a CSV file as the reader counts them (a quoted value may hold line breaks), header included."""
    return int(pl.scan_csv(path, separator=sep, has_header=False, infer_schema=False, truncate_ragged_lines=True,
                           encoding="utf8-lossy").select(pl.len()).collect().item())


def _parts(path: Path, ext: str, sheet: Any, sep: str | None, encoding: str, table) -> pl.LazyFrame:
    """A table laid out in parts (side by side under banners, in blocks under titles, in sections) as one table,
    one part under another, with the group column saying which part each row came from."""
    frames = []
    for p in table.parts:
        df = _part_excel(path, sheet, p, table.col_offset) if ext in EXCEL_EXT else _part_csv(path, sep or ",", encoding, p)
        df = df.with_row_index("__row", offset=p.first)
        if p.fill_down:
            df = df.with_columns([pl.col(c).forward_fill() for c in p.fill_down if c in df.columns])
        if table.group:
            if p.sections:
                labels, cur = [], None
                for i in df["__row"].to_list():
                    cur = p.sections.get(i, cur)
                    labels.append(cur)
                df = df.with_columns(pl.Series(table.group, labels, dtype=pl.Utf8))
            else:
                df = df.with_columns(pl.lit(p.label, dtype=pl.Utf8).alias(table.group))
        if p.drop:
            df = df.filter(~pl.col("__row").is_in(p.drop))
        frames.append(df.drop("__row"))
    names = list(table.names)
    shared = len(set.intersection(*[set(q.cols) for q in table.parts])) if len(table.parts) > 1 else 0
    if shared >= len(names):
        shared = 0                                    # blocks one below another: the group comes first
    order = names[:shared] + ([table.group] if table.group else []) + names[shared:]
    kinds: dict[str, set] = {}
    for df in frames:
        for c, dt in df.schema.items():
            kinds.setdefault(c, set()).add(dt)
    mixed = {c for c, dts in kinds.items() if len({d for d in dts if d != pl.Null}) > 1}
    frames = [df.with_columns([pl.col(c).cast(pl.Utf8) for c in mixed if c in df.columns]) for df in frames]
    out = pl.concat(frames, how="diagonal_relaxed")
    return out.select([c for c in order if c in out.columns]).lazy()


def _part_excel(path: Path, sheet: Any, p, col_offset: int = 0) -> pl.DataFrame:
    import fastexcel
    import warnings
    n = None if p.stop is None else max(0, p.stop - p.first)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = fastexcel.read_excel(str(path)).load_sheet(_sheet_ref(path, sheet), header_row=p.header, n_rows=n,
                                                        use_columns=[j + col_offset for j in p.cols])
        df = s.to_polars()
    df = df.rename(dict(zip(df.columns, p.names)))
    # as Polars reads a sheet: dates to the microsecond, and a column of whole numbers as whole numbers
    whole = [c for c, dt in df.schema.items() if dt == pl.Float64 and df[c].drop_nulls().len()
             and bool((df[c].drop_nulls() == df[c].drop_nulls().round(0)).all()) and df[c].drop_nulls().abs().max() < 2 ** 53]
    return df.with_columns([pl.col(c).cast(pl.Datetime("us", dt.time_zone)) for c, dt in df.schema.items()
                            if isinstance(dt, pl.Datetime) and dt.time_unit != "us"]
                           + [pl.col(c).cast(pl.Int64) for c in whole])


def _part_csv(path: Path, sep: str, encoding: str, p) -> pl.DataFrame:
    width = max(p.cols) + 1
    schema = {f"c{j}": pl.Utf8 for j in range(width)}
    if encoding == "utf8":
        lf = pl.scan_csv(path, separator=sep, has_header=False, skip_rows=p.first, schema=schema, truncate_ragged_lines=True,
                         encoding="utf8-lossy", null_values=NULLS)
    else:
        lf = pl.read_csv(path, separator=sep, has_header=False, skip_rows=p.first, schema=schema, truncate_ragged_lines=True,
                         encoding="latin-1", null_values=NULLS).lazy()
    if p.stop is not None:
        lf = lf.head(max(0, p.stop - p.first))
    df = lf.select([pl.col(f"c{j}").str.strip_chars().alias(n) for j, n in zip(p.cols, p.names)]).collect()
    return df.with_columns([pl.when(pl.col(n) == "").then(None).otherwise(pl.col(n)).alias(n) for n in p.names])


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


def _apply(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any]) -> NodeResult:
    lf, messages, report = scan_file(ctx, params)
    return NodeResult(lf, report=report, messages=messages)


def _summary(p: dict[str, Any]) -> str:
    return Path(p.get("path") or "").name or "no file chosen"


registry.register(NodeType(
    key="load_file",
    label="Load file",
    category="Get data",
    icon="▤",
    kind="source",
    inputs=[],
    description="Open a CSV, text, Excel, Parquet, GeoJSON, GeoPackage or shapefile. Dates and numbers are "
                "detected automatically, and a map layer's geometry is reprojected to longitude/latitude.",
    apply=_apply,
    summary=_summary,
    params=[
        Param("path", "File", "path", required=True,
              help="CSV, TSV, TXT, Excel (.xlsx), Parquet, GeoJSON (.geojson), GeoPackage (.gpkg) or shapefile (.shp)"),
        Param("sheet", "Sheet", "text", default="", help="Excel only. Leave blank for the first sheet"),
        Param("layer", "Layer", "text", default="", help="GeoPackage only. Leave blank for the first layer"),
        Param("has_header", "First row is column names", "bool", default=True),
        Param("layout", "Layout", "choice", default="auto",
              choices=[("auto", "work out the layout"), ("as_is", "read the rows as they are")],
              help="Work out where the column names are, tables side by side or in blocks, summary rows (AVERAGE, "
                   "Total) and empty rows. 'As they are' reads every row under the column names"),
        Param("table", "Table on the sheet", "int", default=1, min=1, advanced=True,
              help="When a sheet holds several tables, which one to read (1 is the first)"),
        Param("skip_rows", "Skip rows at top", "int", default=0, min=0, help="Lines to ignore before the header"),
        Param("separator", "Column separator", "choice", default="auto", advanced=True,
              choices=[("auto", "detect automatically"), (",", "comma"), ("\t", "tab"), (";", "semicolon"), ("|", "pipe"), (" ", "space")]),
        Param("parse_dates", "Detect dates", "bool", default=True, advanced=True,
              help="Turn text columns that look like dates into date/times. A date column and a time-of-day column are combined into one"),
        Param("parse_numbers", "Detect numbers written as text", "bool", default=True, advanced=True,
              help="Read '1,234.50', '£99', '31.5%' and '(120)' as numbers"),
        Param("date_format", "Date format", "text", default="", advanced=True,
              help="Force a format like %d/%m/%Y %H:%M:%S when auto-detect gets it wrong"),
        Param("time_zone", "Time zone", "text", default="", advanced=True,
              help="The zone to show times with a UTC offset (…+02:00) in, such as Europe/London. Leave empty to keep "
                   "the file's own offset if it's the same throughout, or UTC if not"),
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
    import logging
    try:
        import fastexcel
        return list(fastexcel.read_excel(str(path)).sheet_names)
    except Exception as e:  # noqa: BLE001 - the caller reports "no such sheet"; log the underlying cause
        logging.getLogger("dancr").debug("could not list the sheets of %s: %s", path, e)
        return []


TABLES_IN_BYTES = 30 * 1024 * 1024     # a workbook up to this size is looked through for several tables on a sheet


def tables_in(path: str | Path) -> list[tuple[str, dict[str, Any]]]:
    """The tables a file holds, as (title, load settings): one for a CSV or Parquet file, one per sheet of a
    workbook with several (Orders, Customers …), and one per table on a sheet that holds several, so dropping a
    workbook brings in every table in it."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext in VECTOR_EXT:
        from ..geo import vector_layers
        layers = vector_layers(p)
        if not layers:
            return [(p.stem, {})]
        if len(layers) == 1:
            return [(layers[0], {"layer": layers[0]})]
        return [(name, {"layer": name}) for name in layers]
    if ext not in EXCEL_EXT:
        return [(p.stem, {})]
    sheets = list_sheets(p)
    out: list[tuple[str, dict[str, Any]]] = []
    for i, sheet in enumerate(sheets or [None]):
        found = _sheet_tables(p, i) if sheet is not None else []
        base = sheet if len(sheets) > 1 else p.stem
        extra = {"sheet": sheet} if len(sheets) > 1 else {}
        if len(found) <= 1:
            out.append((base, extra))
            continue
        for k, t in enumerate(found, start=1):
            out.append((t.title or f"{base} table {k}", {**extra, "table": k}))
    return out


def _sheet_tables(path: Path, sheet: int) -> list:
    try:
        if path.stat().st_size > TABLES_IN_BYTES:
            return []
        from ..layout import excel_grid, find_tables
        grid = excel_grid(path, sheet)
        return find_tables(grid) if grid is not None else []
    except Exception:  # noqa: BLE001 - the load step reports any problem with the file
        return []
