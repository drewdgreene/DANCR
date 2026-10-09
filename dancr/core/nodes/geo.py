"""Where things are: turn a latitude/longitude pair into a clean point, measure distances, and
summarise points into grid cells.

Coordinates are ordinary columns, so these steps are ordinary transforms: they stay lazy, run over
millions of rows and cache like every other step. No geometry type, no projection library, no network.
"""
from __future__ import annotations

from typing import Any

import polars as pl

from ..expr import NUM
from ..findings import finding
from ..geo import (
    UNIT_CHOICES,
    UNITS_M,
    distance_m_expr,
    format_distance,
    grid_degrees,
    valid_lat_expr,
    valid_lon_expr,
)
from ..params import Param
from ..registry import Ctx, NodeResult, NodeType, registry
from ._common import (
    STAT_HELP,
    build_aggregations,
    first_input,
    number_param,
    require_column,
    schema_of,
)


def _free_name(base: str, taken: set[str]) -> str:
    """A column name not already used by the table ('latitude', 'latitude_2', …)."""
    if base not in taken:
        return base
    i = 2
    while f"{base}_{i}" in taken:
        i += 1
    return f"{base}_{i}"


# =================================================================== make a point
def _make_point(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any]) -> NodeResult:
    lf = first_input(inputs)
    schema = schema_of(lf)
    latc = require_column(schema, params.get("lat"), "latitude column", NUM)
    lonc = require_column(schema, params.get("lon"), "longitude column", NUM)
    if latc == lonc:
        raise ValueError("The latitude and longitude columns must be different")
    taken = set(schema)
    lat_out = (params.get("lat_out") or "").strip() or _free_name("latitude", taken)
    lon_out = (params.get("lon_out") or "").strip() or _free_name("longitude", taken | {lat_out})
    for out, src, what in ((lat_out, latc, "latitude"), (lon_out, lonc, "longitude")):
        if out in schema and out != src:
            raise ValueError(f"There is already a column called {out!r}. Choose another name for the clean {what}")
    validate = bool(params.get("validate", True))
    drop_invalid = bool(params.get("drop_invalid", False))

    la = pl.col(latc).cast(pl.Float64)
    lo = pl.col(lonc).cast(pl.Float64)
    ok = valid_lat_expr(latc) & valid_lon_expr(lonc)
    if validate:
        la = pl.when(ok).then(la).otherwise(None)
        lo = pl.when(ok).then(lo).otherwise(None)
    out = lf.with_columns(la.alias(lat_out), lo.alias(lon_out))
    invalid = 0
    dropped = 0
    if not ctx.preview:
        if validate:
            invalid = int(lf.select((~ok).sum()).collect(engine="streaming")[0, 0] or 0)
        if drop_invalid:
            dropped = int(out.filter(pl.col(lat_out).is_null() | pl.col(lon_out).is_null()).select(pl.len())
                          .collect(engine="streaming")[0, 0] or 0)
    if drop_invalid:
        out = out.filter(pl.col(lat_out).is_not_null() & pl.col(lon_out).is_not_null())

    msgs: list[str] = []
    if invalid:
        verb = "left blank" if not drop_invalid else "turned into a blank coordinate"
        msgs.append(f"{invalid:,} row{'s' if invalid != 1 else ''} had a blank or impossible coordinate; "
                    f"they were {verb}")
    if dropped:
        msgs.append(f"{dropped:,} row{'s' if dropped != 1 else ''} with no usable location were left out")
    report: dict[str, Any] = {}
    if not ctx.preview:
        points = int(out.select((pl.col(lat_out).is_not_null() & pl.col(lon_out).is_not_null()).sum())
                     .collect(engine="streaming")[0, 0] or 0)
        statement = f"{points:,} point{'s' if points != 1 else ''} with a location"
        if invalid:
            statement += f"; {invalid:,} left blank"
        report = {"points": points, "invalid": invalid, "dropped": dropped,
                  "finding": finding("summary", statement, magnitude=points, exact=True)}
        msgs.append(statement)
    return NodeResult(out, report=report, messages=msgs)


registry.register(NodeType(
    key="make_point", label="Make a point", category="Location", icon="◍",
    description="Turn a latitude and a longitude column (or easting/northing, or a named pair) into a clean "
                "point: impossible coordinates are left blank, and you can drop the rows that have none. "
                "The map, distance and grid steps can then use the pair.",
    apply=_make_point,
    summary=lambda p: f"{p.get('lat') or '?'}, {p.get('lon') or '?'}",
    params=[
        Param("lat", "Latitude column", "column", column_group="numeric", required=True),
        Param("lon", "Longitude column", "column", column_group="numeric", required=True),
        Param("validate", "Leave impossible coordinates blank", "bool", default=True,
              help="A latitude outside −90…90 or a longitude outside −180…180 is not a place"),
        Param("drop_invalid", "Leave out rows with no location", "bool", default=False),
        Param("lat_out", "Name for the clean latitude", "text", default="", advanced=True, placeholder="latitude"),
        Param("lon_out", "Name for the clean longitude", "text", default="", advanced=True, placeholder="longitude"),
    ],
))


# =================================================================== distance
def _fixed_point(params: dict[str, Any]) -> tuple[float, float]:
    lat = number_param(params, "to_lat", 0.0, "The latitude of the fixed point")
    lon = number_param(params, "to_lon", 0.0, "The longitude of the fixed point")
    if not -90.0 <= lat <= 90.0:
        raise ValueError("The fixed point's latitude must be within −90…90")
    if not -180.0 <= lon <= 180.0:
        raise ValueError("The fixed point's longitude must be within −180…180")
    return lat, lon


def _distance(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any]) -> NodeResult:
    lf = first_input(inputs)
    schema = schema_of(lf)
    method = params.get("method") or "between"
    unit = params.get("units") or "km"
    if unit not in UNITS_M:
        raise ValueError(f"Unknown distance unit {unit!r}")
    factor = UNITS_M[unit]
    out_name = (params.get("output") or "").strip() or f"distance_{unit}"
    if out_name in schema:
        raise ValueError(f"There is already a column called {out_name!r}. Choose another name for the distance")

    if method == "from":
        latc = require_column(schema, params.get("lat"), "latitude column", NUM)
        lonc = require_column(schema, params.get("lon"), "longitude column", NUM)
        to_lat, to_lon = _fixed_point(params)
        lf = lf.with_columns(pl.lit(float(to_lat)).alias("__to_lat"), pl.lit(float(to_lon)).alias("__to_lon"))
        metres = distance_m_expr(latc, lonc, "__to_lat", "__to_lon")
        out = lf.with_columns((metres / factor).alias(out_name)).drop("__to_lat", "__to_lon")
        what = f"the fixed point {to_lat:g}, {to_lon:g}"
    else:
        lat1 = require_column(schema, params.get("lat1"), "first latitude column", NUM)
        lon1 = require_column(schema, params.get("lon1"), "first longitude column", NUM)
        lat2 = require_column(schema, params.get("lat2"), "second latitude column", NUM)
        lon2 = require_column(schema, params.get("lon2"), "second longitude column", NUM)
        out = lf.with_columns((distance_m_expr(lat1, lon1, lat2, lon2) / factor).alias(out_name))
        what = "the two points in each row"

    report: dict[str, Any] = {}
    msgs = [f"Added {out_name}: great-circle distance to {what}"]
    if not ctx.preview:
        stats = out.select([pl.col(out_name).is_not_null().sum().alias("n"),
                            pl.col(out_name).median().alias("med"),
                            pl.col(out_name).max().alias("mx")]).collect(engine="streaming").row(0, named=True)
        n = int(stats["n"] or 0)
        if n:
            report = {"distances": n, "median": stats["med"], "longest": stats["mx"],
                      "finding": finding("summary",
                                         f"{n:,} distance{'s' if n != 1 else ''} to {what}; median "
                                         f"{format_distance(stats['med'], unit)}, longest {format_distance(stats['mx'], unit)}",
                                         magnitude=stats["mx"], exact=True)}
    return NodeResult(out, report=report, messages=msgs)


registry.register(NodeType(
    key="distance", label="Distance", category="Location", icon="↔",
    description="Measure the great-circle distance between two points: two coordinate pairs in the same row, "
                "or one coordinate pair and a fixed point you type in. Choose the unit. The sphere is a fixed, "
                "deterministic model, so the same points always give the same answer.",
    apply=_distance,
    summary=lambda p: ("from a fixed point" if p.get("method") == "from" else "between rows") + f" in {p.get('units') or 'km'}",
    params=[
        Param("method", "What to measure", "choice", default="between", choices=[
            ("between", "Between two points in the same row"),
            ("from", "From one fixed point")]),
        Param("lat1", "First latitude", "column", column_group="numeric", visible_when={"method": "between"}),
        Param("lon1", "First longitude", "column", column_group="numeric", visible_when={"method": "between"}),
        Param("lat2", "Second latitude", "column", column_group="numeric", visible_when={"method": "between"}),
        Param("lon2", "Second longitude", "column", column_group="numeric", visible_when={"method": "between"}),
        Param("lat", "Latitude", "column", column_group="numeric", visible_when={"method": "from"}),
        Param("lon", "Longitude", "column", column_group="numeric", visible_when={"method": "from"}),
        Param("to_lat", "Latitude of the fixed point", "float", default=0.0, min=-90.0, max=90.0, visible_when={"method": "from"}),
        Param("to_lon", "Longitude of the fixed point", "float", default=0.0, min=-180.0, max=180.0, visible_when={"method": "from"}),
        Param("units", "Unit", "choice", default="km", choices=UNIT_CHOICES),
        Param("output", "Name for the distance column", "text", default="", advanced=True, placeholder="distance_km"),
    ],
))


# =================================================================== grid
def _points_grid(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any]) -> NodeResult:
    lf = first_input(inputs)
    schema = schema_of(lf)
    latc = require_column(schema, params.get("lat"), "latitude column", NUM)
    lonc = require_column(schema, params.get("lon"), "longitude column", NUM)
    size_deg = grid_degrees(params.get("size") or "0.1")
    only = [require_column(schema, c, "column", NUM) for c in (params.get("columns") or [])]
    if latc in only or lonc in only:
        raise ValueError("Choose value columns to summarise, not the coordinate columns themselves")
    count_col = (params.get("count_column") or "").strip()
    aggs = build_aggregations(schema, params.get("aggregations"), exclude=[latc, lonc],
                              default_stats=tuple(params.get("default_stats") or ["mean"]), only=only or None)
    if not aggs and not count_col:
        raise ValueError("There are no number columns to summarise. Choose at least one, or add a count column")

    cell_lat = (pl.col(latc) / size_deg).floor() * size_deg
    cell_lon = (pl.col(lonc) / size_deg).floor() * size_deg
    msgs: list[str] = []
    # The count column must not clash with a summarised column's output or the cell-centre columns. A single
    # default statistic keeps the summarised column's own name, so a 'points' value column would collide with
    # the default count of 'points'; give the count a free name and say so rather than raise a duplicate-column error.
    reserved = {a.meta.output_name() for a in aggs} | {"cell_lat", "cell_lon"}
    if count_col and count_col in reserved:
        renamed = _free_name(count_col, reserved)
        msgs.append(f"The point count is called {renamed!r} because {count_col!r} is already a column in the output")
        count_col = renamed
    dropped = 0
    if not ctx.preview:
        dropped = int(lf.select((pl.col(latc).is_null() | pl.col(lonc).is_null()).sum())
                      .collect(engine="streaming")[0, 0] or 0)
    work = lf.filter(pl.col(latc).is_not_null() & pl.col(lonc).is_not_null())
    aggs_out = list(aggs)
    if count_col:
        aggs_out.append(pl.len().alias(count_col))
    grouped = work.group_by([cell_lat.alias("__clat"), cell_lon.alias("__clon")]).agg(aggs_out)
    out = grouped.with_columns(
        (pl.col("__clat") + size_deg / 2).alias("cell_lat"),
        (pl.col("__clon") + size_deg / 2).alias("cell_lon"),
    )
    names = out.collect_schema().names()
    lead = ["cell_lat", "cell_lon"] + [c for c in names if c not in ("cell_lat", "cell_lon", "__clat", "__clon")]
    out = out.select(lead).sort(["cell_lat", "cell_lon"])

    report: dict[str, Any] = {}
    if not ctx.preview:
        cells = int(out.select(pl.len()).collect(engine="streaming")[0, 0] or 0)
        pts = int(work.select(pl.len()).collect(engine="streaming")[0, 0] or 0)
        med = pts / cells if cells else 0
        report = {"cells": cells, "points": pts, "size_degrees": size_deg,
                  "finding": finding("summary", f"{pts:,} point{'s' if pts != 1 else ''} in {cells:,} "
                                     f"cell{'s' if cells != 1 else ''} (about {med:.0f} per cell)", magnitude=cells, exact=True)}
    if dropped:
        msgs.append(f"{dropped:,} row{'s' if dropped != 1 else ''} without coordinates were left out")
    msgs.append(f"Grouped points into cells of {size_deg:g}° by {size_deg:g}°")
    return NodeResult(out, report=report, messages=msgs)


registry.register(NodeType(
    key="points_grid", label="Count points per cell", category="Location", icon="▦",
    description="Group points into square grid cells and summarise each cell. Use it to turn a cloud of points "
                "into a density map or a table of hotspots; the output carries each cell's centre, ready to map.",
    apply=_points_grid,
    summary=lambda p: f"cells of {p.get('size') or '0.1'}",
    params=[
        Param("lat", "Latitude column", "column", column_group="numeric", required=True),
        Param("lon", "Longitude column", "column", column_group="numeric", required=True),
        Param("size", "Cell size", "text", default="0.1", required=True,
              placeholder="0.1 (degrees) or 5km", help="A plain number is degrees; 5km is converted to degrees of latitude"),
        Param("columns", "Columns to summarise", "columns", column_group="numeric", default=[],
              help="Empty = every number column except the coordinates"),
        Param("default_stats", "Statistics", "text_list", default=["mean"], help=STAT_HELP),
        Param("count_column", "Add a column counting the points in each cell", "text", default="points", advanced=True),
        Param("aggregations", "Choose statistics column by column", "aggregations", default=[], column_group="numeric", advanced=True),
    ],
))


# =================================================================== map
MAP_EXTENTS = [("auto", "Fit to the data"), ("world", "The whole world")]
MAP_PROJECTIONS = [("equirectangular", "Equirectangular (equal degrees)"), ("mercator", "Mercator (web-map look)")]


def validate_map(schema: dict[str, pl.DataType], params: dict[str, Any]) -> None:
    """Every column a map names must exist and suit its role; raises a plain-English ValueError otherwise."""
    require_column(schema, params.get("lat"), "latitude column", NUM)
    require_column(schema, params.get("lon"), "longitude column", NUM)
    color = params.get("color_by")
    if color:
        require_column(schema, color, "colour column")
    size = params.get("size_by")
    if size:
        require_column(schema, size, "size column", NUM)
    label = params.get("label")
    if label:
        require_column(schema, label, "label column")
    raw = str(params.get("cell_size") or "").strip()
    if raw:
        grid_degrees(raw)
    if params.get("extent") not in (None, "auto", "world"):
        raise ValueError(f"Unknown extent {params.get('extent')!r}")
    if params.get("projection") not in (None, "equirectangular", "mercator"):
        raise ValueError(f"Unknown projection {params.get('projection')!r}")


def _map(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any]) -> NodeResult:
    """Validation only: every column the map names must exist (the drawing happens in the views)."""
    lf = first_input(inputs)
    schema = schema_of(lf)
    validate_map(schema, params)
    cells = str(params.get("cell_size") or "").strip()
    if cells:
        said = f"Map of grid cells of {cells}"
    else:
        said = f"Map of points at {params.get('lon')}, {params.get('lat')}"
    if params.get("color_by"):
        said += f", coloured by {params['color_by']}"
    if params.get("extent") == "world":
        said += ", over the whole world"
    return NodeResult(lf, report={"kind": "cells" if cells else "points",
                                  "finding": finding("summary", said, exact=True)})


registry.register(NodeType(
    key="map", label="Map", category="Location", icon="◍",
    description="Draw the data on a map: one point per row, or a square per grid cell when the cell size is set. "
                "Country outlines come from a basemap bundled with the app, so a map needs no internet and no "
                "tile server — the locations never leave the machine.",
    apply=_map,
    kind="sink",
    materialize=False,
    summary=lambda p: f"{p.get('lon') or '?'}, {p.get('lat') or '?'}" + (f" by {p['color_by']}" if p.get("color_by") else ""),
    params=[
        Param("lat", "Latitude column", "column", column_group="numeric", required=True),
        Param("lon", "Longitude column", "column", column_group="numeric", required=True),
        Param("color_by", "Colour by", "column", help="A number (a colour scale) or a category (a colour each)"),
        Param("size_by", "Size by", "column", column_group="numeric", help="A number: bigger points for bigger values"),
        Param("label", "Label points by", "column", help="A column to print beside each point (only when there are few)"),
        Param("cell_size", "Draw squares of this size", "text", default="",
              placeholder="blank = points, or e.g. 0.1 or 5km",
              help="Set this to draw grid cells (the output of 'Count points per cell') instead of points"),
        Param("basemap", "Draw country outlines", "bool", default=True),
        Param("extent", "Show", "choice", default="auto", choices=MAP_EXTENTS),
        Param("projection", "Projection", "choice", default="equirectangular", choices=MAP_PROJECTIONS,
              help="Equirectangular keeps counts undistorted; Mercator looks like familiar web maps"),
        Param("title", "Title", "text", default=""),
    ],
))


# =================================================================== projected coordinates
def _project(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any]) -> NodeResult:
    """Turn projected coordinates (UTM easting/northing, or an EPSG code via pyproj) into longitude/latitude.

    UTM is built in (a metre-accurate series, no dependency); any other EPSG code uses pyproj when it is
    installed, with a plain message when it is not."""
    lf = first_input(inputs)
    schema = schema_of(lf)
    ec = require_column(schema, params.get("easting"), "easting column", NUM)
    nc = require_column(schema, params.get("northing"), "northing column", NUM)
    zone = params.get("utm_zone")
    south = bool(params.get("south"))
    crs = str(params.get("crs") or "").strip()
    taken = set(schema)
    lon_out = (params.get("lon_out") or "").strip() or _free_name("longitude", taken)
    lat_out = (params.get("lat_out") or "").strip() or _free_name("latitude", taken | {lon_out})
    for out, src in ((lon_out, ec), (lat_out, nc)):
        if out in schema and out != src:
            raise ValueError(f"There is already a column called {out!r}. Choose another name for the coordinate")

    if crs and not crs.lower().startswith("utm"):
        # any EPSG code: pyproj if it is there
        try:
            from pyproj import Transformer
        except ImportError:
            raise ValueError("Reading a coordinate reference system other than UTM needs the 'pyproj' package, "
                             "which is missing from this build. Reinstall DANCR, or set the CRS to a UTM zone "
                             "(e.g. EPSG:32737)") from None
        import numpy as np
        tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        e = pl.col(ec).cast(pl.Float64)
        n = pl.col(nc).cast(pl.Float64)
        out = lf.with_columns(e.alias("__e"), n.alias("__n"))
        df = out.select(["__e", "__n"]).collect(engine="streaming")
        # transform the valid coordinates as arrays in one call (pyproj is vectorized); a blank stays blank
        ev, nv = df["__e"].to_numpy(), df["__n"].to_numpy()
        valid = np.isfinite(ev) & np.isfinite(nv)
        lo = np.full(ev.shape, np.nan)
        la = np.full(nv.shape, np.nan)
        if bool(valid.any()):
            xs, ys = tr.transform(ev[valid], nv[valid])
            lo[valid], la[valid] = xs, ys
        out = out.with_columns(pl.Series(lon_out, lo).fill_nan(None), pl.Series(lat_out, la).fill_nan(None)).drop("__e", "__n")
        msgs = [f"Reprojected {ec}, {nc} from {crs} to longitude/latitude (pyproj)"]
        return NodeResult(out, messages=msgs)

    # UTM: a zone if given, else the EPSG code (32600 + zone is north, 32700 + zone is south)
    if not zone and crs.lower().startswith("utm"):
        zone = crs[3:].strip()
    if not zone and crs.startswith("EPSG:"):
        code = int(crs.split(":")[1])
        if 32601 <= code <= 32660:
            zone, south = code - 32600, False
        elif 32701 <= code <= 32760:
            zone, south = code - 32700, True
    if not zone:
        raise ValueError("Give the UTM zone (or an EPSG code such as EPSG:32737), or the CRS to reproject from")
    z = int(str(zone).strip())
    if not 1 <= z <= 60:
        raise ValueError("The UTM zone must be between 1 and 60")
    import numpy as np
    from ..geo import utm_to_latlon_arrays
    e = pl.col(ec).cast(pl.Float64)
    n = pl.col(nc).cast(pl.Float64)
    # vectorized, so a projected table of millions of rows is reprojected without a Python loop
    pts = lf.select([e.alias("__e"), n.alias("__n")]).collect(engine="streaming")
    ev, nv = pts["__e"].to_numpy(), pts["__n"].to_numpy()
    valid = np.isfinite(ev) & np.isfinite(nv)         # a blank (None/NaN/inf) coordinate stays blank
    lo = np.full(ev.shape, np.nan)
    la = np.full(ev.shape, np.nan)
    if bool(valid.any()):
        la_v, lo_v = utm_to_latlon_arrays(ev[valid], nv[valid], z, south)
        la[valid], lo[valid] = la_v, lo_v
    out = lf.with_columns(pl.Series(lon_out, lo).fill_nan(None), pl.Series(lat_out, la).fill_nan(None))
    msgs = [f"Reprojected {ec}, {nc} from UTM zone {z}{'S' if south else 'N'} to longitude/latitude"]
    report: dict[str, Any] = {}
    if not ctx.preview:
        n_pts = int(valid.sum())
        report = {"points": n_pts, "zone": z, "south": south,
                  "finding": finding("summary", f"{n_pts:,} projected points read as longitude/latitude", magnitude=n_pts, exact=True)}
    return NodeResult(out, report=report, messages=msgs)


registry.register(NodeType(
    key="project", label="Projected to lat/long", category="Location", icon="↔",
    description="Turn projected coordinates (UTM easting/northing, or another EPSG code) into ordinary "
                "longitude/latitude that the map and distance steps use. UTM is built in; other systems need "
                "the optional pyproj package.",
    apply=_project,
    summary=lambda p: f"{p.get('easting') or '?'}, {p.get('northing') or '?'}",
    params=[
        Param("easting", "Easting / X", "column", column_group="numeric", required=True),
        Param("northing", "Northing / Y", "column", column_group="numeric", required=True),
        Param("utm_zone", "UTM zone", "text", default="", placeholder="e.g. 37 (or leave blank and give a CRS)"),
        Param("south", "Southern hemisphere", "bool", default=False),
        Param("crs", "Or a CRS / EPSG code", "text", default="", advanced=True, placeholder="e.g. EPSG:32737"),
        Param("lon_out", "Name for the longitude column", "text", default="", advanced=True, placeholder="longitude"),
        Param("lat_out", "Name for the latitude column", "text", default="", advanced=True, placeholder="latitude"),
    ],
))
