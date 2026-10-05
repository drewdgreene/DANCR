"""Chart and Export nodes. Both pass their input through unchanged."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from ..params import Param
from ..registry import NodeType, Ctx, NodeResult, registry
from ._common import private_temp, first_input, schema_of, require_column
from ..findings import finding

CHART_KINDS = [("line", "Line over time / x"), ("scatter", "Scatter (x vs y)"), ("histogram", "Histogram"), ("bar", "Bar (category totals)")]


def _chart(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any]) -> NodeResult:
    """Validation only: every column the chart names must exist (the drawing happens in the views)."""
    lf = first_input(inputs)
    validate_chart(schema_of(lf), params)
    kind = params.get("kind", "line")
    bits: list[str] = []
    if kind in ("line", "scatter"):
        if params.get("x"):
            bits.append(f"over {params['x']}")
        names = [s.get("column") for s in (params.get("series") or []) if s.get("column")]
        if names:
            bits.append(", ".join(names))
    elif kind == "histogram" and params.get("column"):
        bits.append(str(params["column"]))
    elif kind == "bar" and params.get("category"):
        bits.append(f"{params.get('stat') or 'mean'} of {params.get('value') or 'rows'} by {params['category']}")
    what = " ".join(b for b in bits if b)
    said = f"{kind.capitalize()} chart" + (f": {what}" if what else "")
    return NodeResult(lf, report={"kind": kind, "finding": finding("summary", said, exact=True)})


def validate_chart(schema: dict[str, pl.DataType], params: dict[str, Any]) -> None:
    """Every column a chart names must exist; raises a plain-English ValueError otherwise."""
    kind = params.get("kind", "line")
    if kind not in {k for k, _ in CHART_KINDS}:
        raise ValueError(f"Unknown chart kind {kind!r}. Use one of: {', '.join(k for k, _ in CHART_KINDS)}")
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


registry.register(NodeType(
    key="chart", label="Chart", category="Share", icon="◢",
    description="Draw the data as a chart. Large tables are summarised per pixel, so even 100 million points draw quickly.",
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
        Param("error", "Error bars", "choice", default="", visible_when={"kind": "bar"},
              choices=[("", "none"), ("se", "± standard error"), ("sd", "± standard deviation"), ("ci95", "95% confidence interval")],
              help="With averages, a bar for each group's uncertainty; the bars keep the order the groups appear in"),
        Param("color_by", "Colour by", "column", visible_when={"kind": ["line", "scatter"]}, help="A category column. Each value gets its own colour"),
        Param("split_by", "Split into panels by", "column", visible_when={"kind": ["line", "scatter", "histogram"]},
              help="A category column. Each value gets its own panel, stacked on a shared X axis"),
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
    out = ctx.resolve_output(path)
    if ctx.preview:
        return NodeResult(lf, messages=[f"Will write {out.name} when the project runs"])
    write_table(lf, out)
    return NodeResult(lf, messages=[f"Saved {out}"], report={"path": str(out)}, files=[out])


EXCEL_MAX_ROWS = 1_048_576


def excel_frame(lf: pl.LazyFrame, what: str = "This table") -> pl.DataFrame:
    """The whole table in memory, ready for Excel (which holds at most 1,048,576 rows and no time zones)."""
    from ..dtypes import strip_time_zones
    n = int(lf.select(pl.len()).collect(engine="streaming")[0, 0])
    if n > EXCEL_MAX_ROWS:
        raise ValueError(f"{what} has {n:,} rows, but an Excel sheet holds at most {EXCEL_MAX_ROWS:,}. Save as CSV or Parquet, or use 'Average over time' first.")
    return strip_time_zones(lf.collect(engine="streaming"))


def write_table(lf: pl.LazyFrame, out: Path) -> None:
    """Write a LazyFrame to csv/tsv/txt/parquet/xlsx/geojson safely: temp file, then atomic replace."""
    import os
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ext = out.suffix.lower()
    if ext not in (".parquet", ".pq", ".xlsx", ".csv", ".txt", ".tsv", ".geojson"):
        raise ValueError("Use a .csv, .tsv, .txt, .parquet, .xlsx or .geojson file name")
    tmp = private_temp(out)
    try:
        if ext in (".parquet", ".pq"):
            lf.sink_parquet(tmp)
        elif ext == ".xlsx":
            excel_frame(lf).write_excel(tmp)
        elif ext == ".geojson":
            write_geojson(lf, tmp)
        else:
            lf.sink_csv(tmp, separator="\t" if ext == ".tsv" else ",")
        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)


def write_geojson(lf: pl.LazyFrame, out: Path) -> None:
    """Write a table of points as GeoJSON. Needs a latitude and a longitude column (a WKT `geometry`
    column is used as-is when present); every other column becomes a property."""
    import json
    from ..dtypes import json_safe
    from ..geo import lat_lon_pair
    schema = dict(lf.collect_schema())
    df = lf.collect(engine="streaming")
    pair = None
    if "geometry" in schema:
        geom_col = "geometry"
    else:
        numeric = [(c, df[c].min(), df[c].max()) for c, dt in schema.items() if dt.is_numeric()]
        pair = lat_lon_pair(numeric)
        if pair is None:
            raise ValueError("To save GeoJSON the table needs latitude and longitude columns (or a 'geometry' column). "
                             "Use 'Make a point' first.")
        geom_col = None
    lat, lon = pair if pair else (None, None)
    prop_cols = [c for c in df.columns if c not in (lat, lon, "geometry")]
    features = []
    for row in df.iter_rows(named=True):
        if geom_col:
            geom = _wkt_geometry(row.get(geom_col))
        else:
            la, lo = row.get(lat), row.get(lon)
            if la is None or lo is None:
                continue
            geom = {"type": "Point", "coordinates": [float(lo), float(la)]}
        if geom is None:
            continue
        features.append({"type": "Feature", "geometry": geom,
                         "properties": json_safe({c: _json_value(row.get(c)) for c in prop_cols})})
    out.write_text(json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False, default=str),
                   encoding="utf-8")


def _json_value(v: Any) -> Any:
    import datetime
    if isinstance(v, (datetime.date, datetime.datetime, datetime.time)):
        return v.isoformat()
    return v


def _wkt_geometry(text: Any) -> dict[str, Any] | None:
    """A WKT Point/LineString/Polygon (and Multi*) as GeoJSON geometry, or None if unreadable."""
    s = str(text or "").strip()
    if not s or "(" not in s:
        return None
    kind = s.split("(", 1)[0].strip().upper()
    body = s[s.index("("):]
    try:
        if kind == "POINT":
            x, y = [float(t) for t in body[1:-1].split()[:2]]
            return {"type": "Point", "coordinates": [x, y]}
        if kind == "LINESTRING":
            coords = [[float(t) for t in pt.split()[:2]] for pt in body[1:-1].split(",")]
            return {"type": "LineString", "coordinates": coords}
        if kind in ("POLYGON", "MULTIPOLYGON"):
            return _wkt_polygon(kind, body)
    except (ValueError, IndexError):
        return None
    return None


def _wkt_polygon(kind: str, body: str) -> dict[str, Any] | None:
    import re
    rings = re.findall(r"\(([^()]*)\)", body)
    parsed = [[[float(t) for t in pt.split()[:2]] for pt in ring.split(",")] for ring in rings]
    if kind == "POLYGON":
        return {"type": "Polygon", "coordinates": parsed}
    return {"type": "MultiPolygon", "coordinates": [[r] for r in parsed]}


registry.register(NodeType(
    key="export", label="Save to file", category="Share", icon="⇩",
    description="Write the table to CSV, Excel, Parquet or GeoJSON (a table with latitude/longitude columns).",
    apply=_export,
    kind="sink",
    materialize=False,      # the table is the upstream result; keeping a second copy in the cache wastes the disk
    summary=lambda p: Path(p.get("path") or "").name or "no file chosen",
    params=[Param("path", "Save as", "path", required=True, help=".csv, .tsv, .xlsx, .parquet or .geojson")],
))
