"""Chart and Export nodes. Both pass their input through unchanged."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from ..params import Param
from ..registry import NodeType, Ctx, NodeResult, registry
from ._common import first_input, schema_of, require_column

CHART_KINDS = [("line", "Line over time / x"), ("scatter", "Scatter (x vs y)"), ("histogram", "Histogram"), ("bar", "Bar (category totals)")]


def _chart(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any]) -> NodeResult:
    """Validation only: every column the chart names must exist (the drawing happens in the views)."""
    lf = first_input(inputs)
    schema = schema_of(lf)
    kind = params.get("kind", "line")
    if kind in ("line", "scatter"):
        if params.get("x"):
            require_column(schema, params["x"], "x column")
        for s in params.get("series") or []:
            if s.get("column"):
                require_column(schema, s["column"], "series column")
        if params.get("color_by"):
            require_column(schema, params["color_by"], "colour-by column")
    if kind == "histogram" and params.get("column"):
        require_column(schema, params["column"], "histogram column")
    if kind == "bar":
        if params.get("category"):
            require_column(schema, params["category"], "category column")
        if params.get("value"):
            require_column(schema, params["value"], "value column")
    if kind in ("line", "scatter", "histogram") and params.get("split_by"):
        require_column(schema, params["split_by"], "split-by column")
    return NodeResult(lf)


registry.register(NodeType(
    key="chart", label="Chart", category="Share", icon="◢",
    description="Draw the data. Big data is summarised per pixel so even 100 million points draw instantly.",
    apply=_chart,
    materialize=False,
    summary=lambda p: f"{p.get('kind', 'line')}: {', '.join(s.get('column', '') for s in (p.get('series') or []))}",
    params=[
        Param("kind", "Chart type", "choice", default="line", choices=CHART_KINDS),
        Param("x", "X axis", "column", help="Blank = first date/time column, else row number", visible_when={"kind": ["line", "scatter"]}),
        Param("series", "Series (Y)", "series", default=[], column_group="numeric", visible_when={"kind": ["line", "scatter"]}),
        Param("column", "Column", "column", column_group="numeric", visible_when={"kind": "histogram"}),
        Param("bins", "Bins", "int", default=50, min=2, max=2000, visible_when={"kind": "histogram"}),
        Param("category", "Category", "column", visible_when={"kind": "bar"}),
        Param("value", "Value", "column", column_group="numeric", visible_when={"kind": "bar"}),
        Param("stat", "Statistic", "choice", default="mean", visible_when={"kind": "bar"},
              choices=[("mean", "average"), ("sum", "total"), ("count", "count"), ("min", "minimum"), ("max", "maximum"), ("median", "median")]),
        Param("color_by", "Colour by", "column", visible_when={"kind": ["line", "scatter"]}, help="A category column; one colour per value"),
        Param("split_by", "Split into panels by", "column", visible_when={"kind": ["line", "scatter", "histogram"]},
              help="A category column; one panel per value, stacked with a shared X axis"),
        Param("limits", "Limit lines", "limits", default=[], visible_when={"kind": ["line", "scatter", "histogram"]}),
        Param("fit", "Fitted curve", "choice", default="", visible_when={"kind": "scatter"},
              choices=[("", "none"), ("linear", "straight line"), ("saturating", "levels off"), ("exponential", "exponential"), ("power", "power law"), ("logarithmic", "logarithmic"), ("polynomial", "curve (polynomial)")]),
        Param("mean_line", "Show the average as a line", "bool", default=False, visible_when={"kind": ["line", "scatter"]}),
        Param("title", "Title", "text", default=""),
        Param("y_label", "Y axis label (units)", "text", default="", placeholder="e.g. Value (units)"),
        Param("break_gaps", "Show gaps in the data as breaks", "bool", default=True, advanced=True),
        Param("log_y", "Log scale Y", "bool", default=False, advanced=True),
    ],
))


def _export(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any]) -> NodeResult:
    lf = first_input(inputs)
    path = (params.get("path") or "").strip()
    if not path:
        raise ValueError("Choose where to save the file")
    out = ctx.resolve(path)
    if ctx.preview:
        return NodeResult(lf, messages=[f"Will write {out.name} when the pipeline runs"])
    write_table(lf, out)
    return NodeResult(lf, messages=[f"Saved {out}"], report={"path": str(out)})


EXCEL_MAX_ROWS = 1_048_576


def excel_frame(lf: pl.LazyFrame, what: str = "This table") -> pl.DataFrame:
    """The whole table in memory, ready for Excel (which holds at most 1,048,576 rows and no time zones)."""
    from ..dtypes import strip_time_zones
    n = int(lf.select(pl.len()).collect(engine="streaming")[0, 0])
    if n > EXCEL_MAX_ROWS:
        raise ValueError(f"{what} has {n:,} rows; Excel sheets hold at most {EXCEL_MAX_ROWS:,}. Save as CSV or Parquet, or use 'Average over time' first.")
    return strip_time_zones(lf.collect(engine="streaming"))


def write_table(lf: pl.LazyFrame, out: Path) -> None:
    """Write a LazyFrame to csv/tsv/txt/parquet/xlsx safely: temp file, then atomic replace."""
    import os
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ext = out.suffix.lower()
    if ext not in (".parquet", ".pq", ".xlsx", ".csv", ".txt", ".tsv"):
        raise ValueError("Use a .csv, .tsv, .txt, .parquet or .xlsx file name")
    tmp = out.with_name(f".{out.name}.{os.getpid()}.tmp{ext}")
    try:
        if ext in (".parquet", ".pq"):
            lf.sink_parquet(tmp)
        elif ext == ".xlsx":
            excel_frame(lf).write_excel(tmp)
        else:
            lf.sink_csv(tmp, separator="\t" if ext == ".tsv" else ",")
        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)


registry.register(NodeType(
    key="export", label="Save to file", category="Share", icon="⇩",
    description="Write the table to CSV, Excel or Parquet.",
    apply=_export,
    kind="sink",
    summary=lambda p: Path(p.get("path") or "").name or "no file chosen",
    params=[Param("path", "Save as", "path", required=True, help=".csv, .tsv, .xlsx or .parquet")],
))
