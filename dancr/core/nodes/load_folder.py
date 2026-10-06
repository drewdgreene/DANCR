"""Load Folder: every file matching a folder or a glob as one table.

The everyday shape of research and business data is *a folder of the same kind of file* — this month's
exports, one workbook per site, a directory of logs. This source reads them all into one table, with a
column naming the file each row came from, so a single pipeline reruns over a whole set. Each file is read
by the same reader as :mod:`load_file` (so layout, dates and numbers are handled identically).
"""
from __future__ import annotations

import glob as _glob
from pathlib import Path
from typing import Any

import polars as pl

from ..params import Param
from ..registry import NodeType, Ctx, NodeResult, registry
from .load import scan_file, tables_in, CSV_EXT, EXCEL_EXT, PARQUET_EXT, GEOJSON_EXT, VECTOR_EXT, BIO_EXT, effective_ext
from .bio import bio_node_for

DATA_EXT = CSV_EXT | EXCEL_EXT | PARQUET_EXT | GEOJSON_EXT | VECTOR_EXT | set(BIO_EXT)

# the read options shared by every file in the folder, passed straight to scan_file
SHARED = ("has_header", "layout", "skip_rows", "parse_dates", "parse_numbers", "date_format",
          "day_first", "decimal_comma", "encoding", "infer_rows", "ignore_errors")


def matched_files(directory: Path, params: dict[str, Any]) -> list[Path]:
    """The data files a folder/glob setting matches, sorted by path so a run is deterministic. Also used by the
    executor to fingerprint every member of the folder (see ``NodeType.source_files``)."""
    raw = str(params.get("path") or "").strip()
    if not raw:
        return []
    pattern = str(params.get("pattern") or "*").strip() or "*"
    recursive = bool(params.get("recursive", False))
    base = Path(raw).expanduser()
    if not base.is_absolute():
        base = directory / base
    try:
        if any(ch in raw for ch in "*?[") and not base.is_dir():
            found = [Path(p) for p in _glob.glob(str(base), recursive=recursive)]
        elif base.is_dir():
            found = base.rglob(pattern) if recursive else base.glob(pattern)
        elif base.is_file():
            found = [base]
        else:
            return []
    except OSError:
        return []
    out: list[Path] = []
    for f in found:
        try:
            if f.is_file() and effective_ext(f) in DATA_EXT and not f.name.startswith("."):
                out.append(f.resolve())
        except OSError:
            continue
    return sorted(set(out), key=lambda p: str(p))


def source_files(directory: Path, params: dict[str, Any]) -> list[Path]:
    """The files this source reads, for the executor's cache fingerprint."""
    return matched_files(directory, params)


def _apply(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any]) -> NodeResult:
    files = matched_files(ctx.pipeline_dir, params)
    raw = str(params.get("path") or "").strip()
    pattern = str(params.get("pattern") or "*").strip() or "*"
    if not files:
        raise ValueError(f"No data files matched {raw!r} (pattern {pattern!r}). Give a folder such as 'exports' "
                         "or a glob such as 'exports/*.csv'")
    source_column = str(params.get("source_column") or "").strip()
    table_column = str(params.get("table_name_column") or "").strip()
    on_error = str(params.get("on_error") or "fail")
    unify = str(params.get("unify") or "diagonal")
    mode = str(params.get("tables") or "first")
    wanted = str(params.get("table_match") or "").strip().lower()
    shared = {k: params.get(k) for k in SHARED if params.get(k) is not None}

    frames: list[pl.LazyFrame] = []
    names: list[str] = []
    labels: list[str] = []
    report_files: list[dict[str, Any]] = []
    skipped: list[str] = []
    bio_files: set[str] = set()
    bio_params = params.get("bio_params") or {}
    for f in files:
        # A scientific file (FASTA/FASTQ, VCF, GFF/GTF/BED, GenBank, PLINK) is one table read by its own step; a
        # table file can hold several (a sheet, a table on a sheet, a layer), chosen by `tables`.
        bio_type = bio_node_for(f)
        parts: list[tuple[str, dict[str, Any]]] = []
        if bio_type or mode == "first":
            parts = [(f.stem, {})]
        else:
            try:
                found = tables_in(f)
            except Exception as e:  # noqa: BLE001 - reported below with the file named
                found = []
                report_files.append({"file": f.name, "tables_error": str(e)})
            if mode == "match":
                found = [t for t in found if wanted and (wanted in t[0].lower()
                                                         or any(wanted == str(v).lower() for v in t[1].values()))]
                if not found:
                    why = f"no table or layer named {params.get('table_match')!r}"
                    if on_error == "skip":
                        skipped.append(f"{f.name}: {why}")
                        report_files.append({"file": f.name, "skipped": why})
                        continue
                    raise ValueError(f"{f.name}: {why}. It holds: {', '.join(t[0] for t in tables_in(f)) or 'none'}")
            parts = found or [(f.stem, {})]
        for title, extra in parts:
            try:
                if bio_type:
                    res = registry.get(bio_type).apply(ctx, {}, {"path": str(f), **bio_params})
                    lf = getattr(res, "frame", res)
                    msgs = list(getattr(res, "messages", []) or [])
                    rep = dict(getattr(res, "report", {}) or {})
                    bio_files.add(f.name)
                else:
                    lf, msgs, rep = scan_file(ctx, {**shared, **extra, "path": str(f)})
            except Exception as e:  # noqa: BLE001 - reported with the file named, or skipped when asked
                from ..executor import friendly_error
                why = friendly_error(e)
                if on_error == "skip":
                    skipped.append(f"{f.name}: {why}")
                    report_files.append({"file": f.name, "skipped": why})
                    continue
                raise ValueError(f"{f.name}: {why}") from e
            if source_column:
                lf = lf.with_columns(pl.lit(f.name, dtype=pl.Utf8).alias(source_column))
            if table_column and mode != "first":
                lf = lf.with_columns(pl.lit(title, dtype=pl.Utf8).alias(table_column))
            frames.append(lf)
            names.append(f.name)
            labels.append(title)
            report_files.append({"file": f.name, "table": title, "messages": msgs, "layout": rep.get("layout")})
    if not frames:
        raise ValueError("Every file matched was skipped: " + "; ".join(skipped))

    where = [f"{n} › {lbl}" if mode != "first" else n for n, lbl in zip(names, labels)]
    if unify == "strict":
        base_schema = dict(frames[0].collect_schema())
        for name, fr in zip(where[1:], frames[1:]):
            sch = dict(fr.collect_schema())
            if sch != base_schema:
                raise ValueError(f"{name} does not have the same columns as {where[0]} "
                                 f"({_schema_diff(base_schema, sch)}). Set 'Combine files' to 'union' or 'text' "
                                 "to read files or tables that differ")
        # same set of columns, but they may be in a different order per file: vertical concat needs one order,
        # so line every table up with the first (data is keyed by column name, so this cannot change values)
        order = list(base_schema)
        out = pl.concat([fr.select(order) for fr in frames], how="vertical")
    elif unify == "text":
        as_text = [fr.with_columns([pl.col(c).cast(pl.Utf8) for c in fr.collect_schema().names()]) for fr in frames]
        out = pl.concat(as_text, how="diagonal_relaxed")
    else:                                              # "diagonal": union of the columns, missing filled with blank
        out = pl.concat(frames, how="diagonal_relaxed")

    files_n = len(set(names))
    messages = [f"Read {len(frames)} table{'s' if len(frames) != 1 else ''} from {files_n} "
                f"file{'s' if files_n != 1 else ''} in {raw!r}"
                + (f" matching {pattern!r}" if pattern != "*" else "")]
    if source_column:
        messages.append(f"Each row's file is in '{source_column}'")
    if table_column and mode != "first":
        messages.append(f"Which table each row came from is in '{table_column}'")
    if bio_files:
        messages.append(f"{len(bio_files)} scientific file{'s' if len(bio_files) != 1 else ''} read with "
                        f"{'their' if len(bio_files) != 1 else 'its'} own reader")
    if skipped:
        messages.append(f"Skipped {len(skipped)} file(s): " + "; ".join(skipped))
    return NodeResult(out, report={"kind": "folder", "path": raw, "pattern": pattern, "count": len(frames),
                                   "files": report_files, "skipped": skipped, "tables": mode},
                      messages=messages)


def _schema_diff(a: dict, b: dict) -> str:
    only_a = [c for c in a if c not in b]
    only_b = [c for c in b if c not in a]
    changed = [c for c in a if c in b and a[c] != b[c]]
    bits = []
    if only_a:
        bits.append(f"missing {', '.join(only_a[:5])}")
    if only_b:
        bits.append(f"extra {', '.join(only_b[:5])}")
    if changed:
        bits.append(f"different type for {', '.join(changed[:5])}")
    return "; ".join(bits) or "different columns"


def _summary(p: dict[str, Any]) -> str:
    raw = str(p.get("path") or "").strip()
    if not raw:
        return "no folder chosen"
    name = Path(raw).name or raw
    pat = str(p.get("pattern") or "*").strip()
    return f"{name}/{pat}" if pat and pat != "*" else name


registry.register(NodeType(
    key="load_folder",
    label="Load folder",
    category="Get data",
    icon="▤",
    kind="source",
    inputs=[],
    description="Read every CSV, text, Excel, Parquet, geo or scientific file matching a folder (and a pattern) as "
                "one table, with a column naming the file each row came from. The same reader as Load file, and "
                "FASTA/FASTQ, VCF, GFF/GTF/BED, GenBank and PLINK files are read by their own step — so a folder of "
                "VCFs or FASTAs is one table.",
    apply=_apply,
    summary=_summary,
    source_files=source_files,
    params=[
        Param("path", "Folder or glob", "dir", required=True,
              help="A folder ('exports') or a glob ('exports/*.csv'). Relative paths are taken from the project's folder"),
        Param("pattern", "File name pattern", "text", default="*",
              help="Which files inside the folder to read, e.g. *.csv or *.xlsx (ignored when the folder path is itself a glob)"),
        Param("recursive", "Look inside sub-folders", "bool", default=False),
        Param("bio_params", "Settings for scientific files", "mapping", default={}, advanced=True,
              help="Passed to the reader for FASTA/FASTQ, VCF, GFF/GTF/BED, GenBank and PLINK files, e.g. "
                   "{\"samples\": \"genotype\"} for a folder of VCFs"),
        Param("source_column", "Put the file name in", "text", default="source_file",
              help="A column naming the file each row came from. Leave blank for none"),
        Param("tables", "Which tables to read", "choice", default="first",
              choices=[("first", "the first table of each file"),
                       ("all", "every sheet, table and layer each file holds"),
                       ("match", "only tables whose name matches")],
              help="A workbook or GeoPackage can hold several tables; read the first, all of them, or the ones you name"),
        Param("table_match", "Table or layer name", "text", default="",
              visible_when={"tables": "match"}, help="Part of a sheet, table or layer name to keep"),
        Param("table_name_column", "Put the table name in", "text", default="source_table",
              visible_when={"tables": ["all", "match"]}, help="A column naming the table each row came from. Leave blank for none"),
        Param("unify", "Combine files", "choice", default="diagonal",
              choices=[("diagonal", "union of the columns (missing filled with blank)"),
                       ("strict", "only files with exactly the same columns"),
                       ("text", "read every column as text")]),
        Param("on_error", "When a file cannot be read", "choice", default="fail",
              choices=[("fail", "stop and name the file"), ("skip", "skip it and carry on")], advanced=True),
        Param("has_header", "First row is column names", "bool", default=True, advanced=True),
        Param("layout", "Layout", "choice", default="auto", advanced=True,
              choices=[("auto", "work out the layout"), ("as_is", "read the rows as they are")]),
        Param("parse_dates", "Detect dates", "bool", default=True, advanced=True),
        Param("parse_numbers", "Detect numbers written as text", "bool", default=True, advanced=True),
        Param("date_format", "Date format", "text", default="", advanced=True),
        Param("day_first", "Day comes before month (01/05 = 1 May)", "bool", default=False, advanced=True),
        Param("decimal_comma", "Numbers use decimal comma", "bool", default=False, advanced=True),
        Param("encoding", "Text encoding", "choice", default="utf8", advanced=True,
              choices=[("utf8", "UTF-8 (normal)"), ("latin1", "Latin-1 / Windows")]),
        Param("infer_rows", "Rows to inspect for types", "int", default=10000, min=100, advanced=True),
        Param("ignore_errors", "Skip values that will not parse", "bool", default=False, advanced=True),
    ],
))
