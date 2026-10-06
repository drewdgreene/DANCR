"""Load File: CSV / TSV / text, Excel, Parquet, with sensible auto-detection."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from ...params import Param
from ...registry import NodeType, Ctx, NodeResult, registry
from ._formats import (  # noqa: E402,F401
    BIO_EXT, CSV_EXT, DOC_EXT, EXCEL_EXT, GEOJSON_EXT, H5_EXT, IMAGE_EXT, NC_EXT, NON_TABLE_EXT,
    NULLS, PARQUET_EXT, VECTOR_EXT, _refuse_non_table, effective_ext, list_sheets, sniff_separator,
)
from ._tidy import _leading_zero_columns, _tidy_frame  # noqa: E402
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
        from ...geo import is_geojson, geojson_table
        if is_geojson(path):
            df = geojson_table(path)
            messages.append(f"Read a GeoJSON file: {len(df):,} feature{'s' if len(df) != 1 else ''}. "
                            "Each feature's properties are columns, with 'geometry' (WKT) and a centre point")
            return df.lazy(), messages, {"geojson": True}
        raise ValueError(f"{path.name} is JSON, not a table. 'Load from a URL' can read a JSON table over the "
                         "network; for a local file, convert it to CSV or Parquet.")
    if ext in VECTOR_EXT:
        from ...geo import vector_table
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
            messages.append("Kept as text, so their leading zeros stay: " + ", ".join(c.strip() for c in keep_text))
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
    from ...layout import excel_grid, csv_grid
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
    from ...layout import find_tables
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




TABLES_IN_BYTES = 30 * 1024 * 1024     # a workbook up to this size is looked through for several tables on a sheet


def tables_in(path: str | Path) -> list[tuple[str, dict[str, Any]]]:
    """The tables a file holds, as (title, load settings): one for a CSV or Parquet file, one per sheet of a
    workbook with several (Orders, Customers …), and one per table on a sheet that holds several, so dropping a
    workbook brings in every table in it."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext in VECTOR_EXT:
        from ...geo import vector_layers
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
        from ...layout import excel_grid, find_tables
        grid = excel_grid(path, sheet)
        return find_tables(grid) if grid is not None else []
    except Exception:  # noqa: BLE001 - the load step reports any problem with the file
        return []
