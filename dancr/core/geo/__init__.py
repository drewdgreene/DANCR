"""Geospatial helpers: coordinates, distances, grids and geometry.

Coordinates in DANCR are plain columns (``lat`` and ``lon``, or easting/northing): there is no geometry
type, so every step stays a Polars transform, keeps working on millions of rows and stores in the cache
like any other. This module holds the maths — great-circle distance, distance units, grid cells and the
recognition of coordinate columns — and nothing else. It is pure core: no Qt, no network.

Everything here is deterministic. The Earth is treated as a sphere of one fixed mean radius
(:data:`EARTH_RADIUS_M`), so the same two points always give the same distance, to the metre. For the
data this tool is built for — water points, clinics, villages, events — the sphere is accurate to well
under the width of the points being measured, and it removes the dependency, and the surprise, of a
projection library.
"""
from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import polars as pl

# Mean Earth radius (IUGG mean radius). One constant everywhere, so every distance in DANCR is comparable.
EARTH_RADIUS_M = 6_371_008.8
EARTH_RADIUS_KM = EARTH_RADIUS_M / 1000.0

# Metres per degree of latitude: a constant on a sphere (one degree of latitude is always the same).
METRES_PER_DEGREE_LAT = math.pi * EARTH_RADIUS_M / 180.0


# =================================================================== distance units
# The canonical short name of each unit, and how many metres it is.
UNITS_M = {"m": 1.0, "km": 1000.0, "mi": 1609.344, "nmi": 1852.0, "ft": 0.3048}
UNIT_CHOICES = [("km", "kilometres (km)"), ("m", "metres (m)"), ("mi", "miles (mi)"),
                ("nmi", "nautical miles"), ("ft", "feet (ft)")]
UNIT_WORD = {"m": "m", "km": "km", "mi": "mi", "nmi": "nmi", "ft": "ft"}

_DIST_RE = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?|\.\d+)\s*([a-zA-Z]*)\s*$")
_DIST_ALIASES = {
    "m": "m", "meter": "m", "meters": "m", "metre": "m", "metres": "m",
    "km": "km", "k": "km", "kilometer": "km", "kilometers": "km", "kilometre": "km", "kilometres": "km", "kms": "km",
    "mi": "mi", "mile": "mi", "miles": "mi",
    "nm": "nmi", "nmi": "nmi", "nautical": "nmi", "nauticalmile": "nmi", "nauticalmiles": "nmi",
    "ft": "ft", "foot": "ft", "feet": "ft",
}


def parse_distance(text: Any, default_unit: str = "km") -> tuple[str, float]:
    """'5km', '3 miles', '500 m' -> (canonical text like '5km', metres). A bare number takes
    ``default_unit``. Raises a plain-English error otherwise."""
    s = str(text or "").strip()
    if not s:
        raise ValueError("Enter a distance, for example 5km, 3 miles or 500 m")
    m = _DIST_RE.match(s)
    if not m:
        raise ValueError(f"Cannot read {text!r} as a distance. Try 5km, 3 miles or 500 m")
    n = float(m.group(1).replace(",", ""))
    unit = (m.group(2) or default_unit).strip().lower()
    if unit not in _DIST_ALIASES:
        raise ValueError(f"Unknown distance unit {unit!r}. Use m, km, miles, nautical miles or feet")
    if n <= 0:
        raise ValueError(f"{text!r} must be greater than zero")
    cu = _DIST_ALIASES[unit]
    return f"{n:g}{cu}", n * UNITS_M[cu]


def convert_distance(metres: float, unit: str) -> float:
    """Metres as the chosen unit."""
    return float(metres) / UNITS_M[unit]


def format_distance(metres: Any, unit: str | None = None) -> str:
    """A distance as a person reads it: 850 m, 1.24 km, 12 km. ``unit`` fixes the unit; otherwise metres
    below a kilometre and kilometres above."""
    if metres is None:
        return "–"
    try:
        m = float(metres)
    except (TypeError, ValueError):
        return str(metres)
    if m != m or m in (float("inf"), float("-inf")):
        return "–"
    if unit:
        return f"{convert_distance(m, unit):g} {UNIT_WORD.get(unit, unit)}"
    ax = abs(m)
    if ax < 1:
        return f"{m * 100:.0f} cm"
    if ax < 1000:
        return f"{m:.0f} m"
    if ax < 10_000:
        return f"{m / 1000:.2f} km"
    return f"{m / 1000:,.0f} km"


def length_to_metres(text: Any, default_unit: str = "km") -> float:
    """A length as metres ('5km' -> 5000)."""
    return parse_distance(text, default_unit)[1]


# =================================================================== great-circle distance
def distance_m_expr(lat1: str, lon1: str, lat2: str, lon2: str, *, radius_m: float = EARTH_RADIUS_M) -> pl.Expr:
    """Great-circle (haversine) distance in metres between two points, as a Polars expression.

    Null in gives null out, and the result is symmetric: the distance from A to B equals B to A. Longitude
    is taken modulo its direction (the difference is fed through ``sin``), so points either side of the
    antimeridian are measured the short way, not the long way round the globe.
    """
    p1 = pl.col(lat1).radians()
    p2 = pl.col(lat2).radians()
    dphi = p2 - p1
    dlam = (pl.col(lon2) - pl.col(lon1)).radians()
    h = (dphi / 2).sin() ** 2 + p1.cos() * p2.cos() * (dlam / 2).sin() ** 2
    h = h.clip(0.0, 1.0)                     # guard against a hair over 1 from rounding at the antipodes
    return 2.0 * radius_m * h.sqrt().arcsin()


# =================================================================== coordinate validity
def valid_lat_expr(name: str) -> pl.Expr:
    """Latitude within −90…90 (blank counts as invalid)."""
    return pl.col(name).is_between(-90.0, 90.0).fill_null(False)


def valid_lon_expr(name: str) -> pl.Expr:
    """Longitude within −180…180 (blank counts as invalid)."""
    return pl.col(name).is_between(-180.0, 180.0).fill_null(False)


def normalize_lon_expr(name: str) -> pl.Expr:
    """A longitude folded into −180…180, so 190° becomes −170°. Leaves blanks blank."""
    x = pl.col(name)
    return pl.when(x > 180.0).then(x - 360.0).when(x < -180.0).then(x + 360.0).otherwise(x)


def swap_looks_like(lo: float, hi: float) -> bool:
    """True when a value range fits longitude but not latitude: the two coordinates may be swapped."""
    return lo < -90.0 and hi > 90.0


# =================================================================== grids
_DEG_RE = re.compile(r"^\s*(\d+(?:\.\d+)?|\.\d+)\s*°?\s*$")


def grid_degrees(size: Any, default_unit: str = "km") -> float:
    """A grid cell size as degrees per cell.

    A bare number is degrees ('0.1' is a tenth of a degree). A length ('5km') is turned into degrees of
    latitude at the mean radius; applied to both axes it makes *rectangular* degree cells, so away from the
    equator a '5km' cell is narrower east–west than north–south. That is the honest, projection-free choice:
    it never distorts a count, and the map draws the cells where they really are.
    """
    s = str(size or "").strip()
    if not s:
        raise ValueError("Enter a cell size, for example 0.1 (degrees) or 5km")
    if _DEG_RE.match(s):
        deg = float(_DEG_RE.match(s).group(1))            # type: ignore[union-attr]
    else:
        deg = length_to_metres(s, default_unit) / METRES_PER_DEGREE_LAT
    if deg <= 0:
        raise ValueError("The cell size must be greater than zero")
    if deg > 90:
        raise ValueError("The cell size is larger than the whole grid (90°). Use a smaller size")
    return deg


def grid_cell_exprs(lat: str, lon: str, size_deg: float, *, prefix: str = "cell") -> tuple[pl.Expr, pl.Expr]:
    """(cell centre latitude, cell centre longitude) for each point: the centre of the degree cell it falls in.

    Cells are half-open ``[k·size, (k+1)·size)``, so every point lands in exactly one; the centre is offset
    by half a cell and is what a map plots and a table sorts by.
    """
    lat0 = (pl.col(lat) / size_deg).floor() * size_deg
    lon0 = (pl.col(lon) / size_deg).floor() * size_deg
    return (lat0 + size_deg / 2).alias(f"{prefix}_lat"), (lon0 + size_deg / 2).alias(f"{prefix}_lon")


def cell_key_exprs(lat: str, lon: str, size_deg: float) -> tuple[pl.Expr, pl.Expr]:
    """The integer cell indices a point falls in, for grouping."""
    return (pl.col(lat) / size_deg).floor().cast(pl.Int64), (pl.col(lon) / size_deg).floor().cast(pl.Int64)


# =================================================================== recognising coordinates
# Names that name a coordinate. Kept deliberately tight: a range or a pair still has to agree before a
# column is treated as a location, so an ordinary x/y scatter is never mistaken for a map.
LAT_NAMES = {"lat", "lats", "latitude", "latitudes", "lattitude", "ycoord", "y_coord", "ycoordinate",
             "northing", "northings", "gpslat", "gps_lat", "lat_deg", "latitude_deg", "y"}
LON_NAMES = {"lon", "lons", "lng", "long", "longitude", "longitudes", "xcoord", "x_coord", "xcoordinate",
             "easting", "eastings", "gpslon", "gps_lon", "gpslng", "lon_deg", "longitude_deg", "x"}
X_NAMES = {"x", "easting", "eastings", "xcoord", "x_coord", "xcoordinate"}
Y_NAMES = {"y", "northing", "northings", "ycoord", "y_coord", "ycoordinate"}


def _norm(name: str) -> str:
    return re.sub(r"[\s\-]+", "_", str(name or "").strip().lower())


def _strong_lat(name: str) -> bool:
    """A name that says latitude on its own: lat, latitude, y, northing, or anything with a separating word
    ('clinic_lat', 'gps_latitude')."""
    n = _norm(name)
    return n in LAT_NAMES or n.endswith("_lat") or n.endswith("_latitude") or n.startswith("lat_")


def _strong_lon(name: str) -> bool:
    n = _norm(name)
    return n in LON_NAMES or n.endswith("_lon") or n.endswith("_lng") or n.endswith("_long") \
        or n.endswith("_longitude") or n.startswith("lon_") or n.startswith("lng_")


def name_suggests_lat(name: str) -> bool:
    n = _norm(name)
    return _strong_lat(name) or (n.endswith("lat") and len(n) <= 6)     # clat, vlat, ylat


def name_suggests_lon(name: str) -> bool:
    n = _norm(name)
    return _strong_lon(name) or (n.endswith(("lon", "lng")) and len(n) <= 6)   # clon, vlon, xlon


def plausible_lat(vmin: float, vmax: float) -> bool:
    """A latitude range: inside −90…90, with some spread. A single value is not a location."""
    return vmin is not None and vmax is not None and -90.0 <= vmin <= vmax <= 90.0 and vmax > vmin


def plausible_lon(vmin: float, vmax: float) -> bool:
    return vmin is not None and vmax is not None and -180.0 <= vmin <= vmax <= 180.0 and vmax > vmin


def lat_lon_pair(columns: list[tuple[str, Any, Any]]) -> tuple[str, str] | None:
    """The best (lat, lon) pair among ``columns`` = [(name, minimum, maximum), …], or None.

    A pair needs one name that says latitude and one that says longitude, and both ranges must fit their
    coordinate. Plain ``x``/``y`` are only accepted when the ranges leave no doubt — x within ±180 and y
    within ±90 with a real spread — because x/y is a scatter as often as it is a location.
    """
    def rng(c: tuple[str, Any, Any]) -> tuple[float, float] | None:
        try:
            return (float(c[1]), float(c[2]))
        except (TypeError, ValueError):
            return None

    lat_c = next((c for c in columns if name_suggests_lat(c[0]) and rng(c) and plausible_lat(*rng(c))), None)
    lon_c = next((c for c in columns if name_suggests_lon(c[0]) and rng(c) and plausible_lon(*rng(c))), None)
    if lat_c and lon_c:
        # a weak name (clat, clon) only counts when its partner names its coordinate strongly (clon + clinic_lat
        # is a pair), or when the two names share a prefix and the ranges are a small window: an ordinary column
        # ending in 'lon' beside an unrelated 'lat'-ish column is not read as a coordinate
        strong = _strong_lat(lat_c[0]) or _strong_lon(lon_c[0])
        shared = _same_stem(lat_c[0], lon_c[0]) and _narrow_pair(rng(lat_c), rng(lon_c))
        if strong or shared:
            return lat_c[0], lon_c[0]
    # x / y, but only when neither name is already spoken for and the ranges fit tightly
    xs = [c for c in columns if _norm(c[0]) in X_NAMES and rng(c)]
    ys = [c for c in columns if _norm(c[0]) in Y_NAMES and rng(c)]
    for y in ys:
        ry = rng(y)
        for x in xs:
            rx = rng(x)
            if rx and ry and plausible_lon(*rx) and plausible_lat(*ry):
                return y[0], x[0]
    return None




def _same_stem(a: str, b: str) -> bool:
    """Two coordinate names that share a prefix once 'lat'/'lon' (or 'y'/'x') is stripped: clat/clon,
    vlat/vlon, gpslat/gpslon."""
    import re as _re
    strip = lambda s: _re.sub(r"(lat|lon|lng|latitude|longitude)$", "", _norm(s))
    sa, sb = strip(a), strip(b)
    return sa == sb and len(sa) <= 3


def _narrow_pair(lat_rng: tuple[float, float] | None, lon_rng: tuple[float, float] | None) -> bool:
    """A lat/lon pair whose ranges are a real window: lat inside ±90 and lon a span under 180°, so two ordinary
    wide-ranging columns are not mistaken for coordinates."""
    if not lat_rng or not lon_rng:
        return False
    la0, la1 = lat_rng
    lo0, lo1 = lon_rng
    return -90.0 <= la0 <= la1 <= 90.0 and (lo1 - lo0) <= 180.0


# =================================================================== reading a GeoJSON file
def is_geojson(path: Path) -> bool:
    """True when a .json/.geojson file holds GeoJSON (a FeatureCollection, Feature or bare geometry)."""
    if path.suffix.lower() not in (".json", ".geojson"):
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            head = f.read(4096)
    except OSError:
        return False
    return '"FeatureCollection"' in head or '"Feature"' in head or '"geometry"' in head or '"coordinates"' in head


def geojson_features(path: Path) -> list[dict[str, Any]]:
    """The features of a GeoJSON file, as plain dicts. Accepts a FeatureCollection, a single Feature, a bare
    geometry (turned into one feature) and newline-delimited GeoJSON. Raises a plain-English error otherwise."""
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError as e:
        raise ValueError(f"Cannot read {path.name}: {e}") from None
    if not text:
        raise ValueError(f"{path.name} is empty")
    docs: list[Any] = []
    try:
        docs.append(json.loads(text))
    except json.JSONDecodeError:
        # newline-delimited GeoJSON: one object per line
        for line in text.splitlines():
            line = line.strip().rstrip(",")
            if line and line not in ("[", "]"):
                try:
                    docs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        if not docs:
            raise ValueError(f"{path.name} is not valid GeoJSON") from None
    out: list[dict[str, Any]] = []
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        t = doc.get("type")
        if t == "FeatureCollection":
            out += [f for f in (doc.get("features") or []) if isinstance(f, dict)]
        elif t == "Feature":
            out.append(doc)
        elif t in ("Point", "LineString", "Polygon", "MultiPoint", "MultiLineString", "MultiPolygon", "GeometryCollection"):
            out.append({"type": "Feature", "properties": {}, "geometry": doc})
    if not out:
        raise ValueError(f"{path.name} has no GeoJSON features in it")
    return out


def geometry_wkt(geom: dict[str, Any] | None) -> str | None:
    """A GeoJSON geometry as WKT text (what a `geometry` column holds), or None for a null geometry."""
    if not geom:
        return None
    gtype = geom.get("type")
    coords = geom.get("coordinates")
    if gtype == "Point" and coords:
        return f"POINT ({_num(coords[0])} {_num(coords[1])})"
    if gtype == "LineString" and coords:
        return "LINESTRING (" + ", ".join(f"{_num(x)} {_num(y)}" for x, y in coords) + ")"
    if gtype == "Polygon" and coords:
        rings = ["(" + ", ".join(f"{_num(x)} {_num(y)}" for x, y in ring) + ")" for ring in coords]
        return "POLYGON (" + ", ".join(rings) + ")"
    if gtype in ("MultiPoint", "MultiLineString", "MultiPolygon") and coords:
        parts = []
        for part in coords:
            if gtype == "MultiPoint":
                parts.append(f"({_num(part[0])} {_num(part[1])})")
            elif gtype == "MultiLineString":
                parts.append("(" + ", ".join(f"{_num(x)} {_num(y)}" for x, y in part) + ")")
            else:
                parts.append("(" + ", ".join("(" + ", ".join(f"{_num(x)} {_num(y)}" for x, y in ring) + ")" for ring in part) + ")")
        return f"{gtype.upper()} (" + ", ".join(parts) + ")"
    return None


def geometry_point(geom: dict[str, Any] | None) -> tuple[float | None, float | None]:
    """A representative (lon, lat) for a geometry: its coordinates for a point, else the centre of its bounds."""
    if not geom:
        return None, None
    coords = geom.get("coordinates")
    if geom.get("type") == "Point" and coords:
        return float(coords[0]), float(coords[1])
    flat = list(_flatten_coords(coords))
    if not flat:
        return None, None
    xs = [c[0] for c in flat]
    ys = [c[1] for c in flat]
    return (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0


def _flatten_coords(coords: Any):
    if not coords:
        return
    if isinstance(coords[0], (int, float)):
        yield coords
        return
    for c in coords:
        yield from _flatten_coords(c)


def _num(v: Any) -> str:
    f = float(v)
    return f"{f:g}"


def geojson_table(path: Path) -> "pl.DataFrame":
    """A GeoJSON file as a table: one row per feature, its properties as columns, plus a `geometry` column
    (WKT text) and a `longitude`/`latitude` centre. Imported here so ``core`` stays free of a hard Polars
    import at module load."""
    feats = geojson_features(path)
    props: list[str] = []
    for f in feats:
        for k in (f.get("properties") or {}):
            if k not in props:
                props.append(k)
    rows: list[dict[str, Any]] = []
    for f in feats:
        p = f.get("properties") or {}
        row = {k: p.get(k) for k in props}
        geom = f.get("geometry")
        row["geometry"] = geometry_wkt(geom)
        lon, lat = geometry_point(geom)
        row["longitude"] = lon
        row["latitude"] = lat
        rows.append(row)
    if not rows:
        raise ValueError(f"{path.name} has no GeoJSON features in it")
    return pl.DataFrame(rows, infer_schema_length=None)


# =================================================================== reading a GeoPackage or shapefile
VECTOR_EXT = {".gpkg", ".shp"}


def vector_layers(path: Path) -> list[str]:
    """The layer names a GeoPackage or shapefile holds, in file order. Empty when it cannot be read (the load
    step reports why) or the optional 'pyogrio' package is not installed."""
    try:
        import pyogrio
    except ImportError:
        return []
    try:
        return [str(name) for name, _geom in pyogrio.list_layers(str(path))]
    except Exception:  # noqa: BLE001 - the load step reports the real problem, in plain words
        return []


def _is_wgs84(crs: str) -> bool:
    """Whether a coordinate reference system is WGS84 lon/lat (the spelling pyogrio usually returns, or a full
    definition understood by pyproj when it is installed)."""
    if crs.strip().lower() in ("epsg:4326", "4326", "wgs84", "wgs 84", "crs84", "ogc:crs84", "urn:ogc:def:crs:ogc:1.3:crs84"):
        return True
    try:
        from pyproj import CRS
        return CRS.from_user_input(crs).to_epsg() == 4326
    except Exception:  # noqa: BLE001 - pyproj absent, or a CRS it does not know
        return False


def vector_table(path: Path, layer: str | None = None) -> tuple["pl.DataFrame", list[str]]:
    """A GeoPackage or shapefile as a table: one row per feature, its attributes as columns, plus a `geometry`
    column (WKT text) and a `longitude`/`latitude` centre. A layer in another coordinate system is reprojected
    to WGS84 lon/lat (pyproj), so the coordinates are degrees wherever the file came from.

    Returns (table, notes). The optional 'pyogrio' and 'shapely' packages do the reading; a plain message says
    so when they are missing."""
    try:
        from pyogrio import raw as ogr
    except ImportError:
        raise ValueError("Reading a GeoPackage or shapefile needs the 'pyogrio' package, which is missing from "
                         "this build. Reinstall DANCR (or, in a source checkout, run 'uv sync')") from None
    try:
        import shapely
        from shapely.ops import transform as _shp_transform
    except ImportError:
        raise ValueError("Reading the geometry of a GeoPackage or shapefile needs the 'shapely' package, which "
                         "is missing from this build. Reinstall DANCR (or, in a source checkout, run 'uv sync')") from None

    layers = vector_layers(path)
    ref = str(layer).strip() if layer else None
    if ref and layers and ref not in layers:
        raise ValueError(f"{path.name} has no layer called {ref!r}. Layers: {', '.join(layers)}")
    try:
        meta, _fids, geometry, field_data = ogr.read(str(path), layer=ref, force_2d=True)
    except Exception as e:  # noqa: BLE001 - report the file, not a driver stack trace
        where = f" (layer {ref})" if ref else ""
        raise ValueError(f"Cannot read {path.name}{where}: {e}") from e

    notes: list[str] = []
    raw_fields = meta.get("fields") if meta else None
    fields = [str(f) for f in (raw_fields if raw_fields is not None else [])]
    crs = str(meta.get("crs")) if meta and meta.get("crs") else ""
    tr = None
    if crs and not _is_wgs84(crs):
        try:
            from pyproj import Transformer
        except ImportError:
            raise ValueError(f"{path.name} is in {crs}, which needs the 'pyproj' package to read as "
                             "longitude/latitude. It is missing from this build — reinstall DANCR, or reproject "
                             "the file before loading it") from None
        tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        notes.append(f"Reprojected from {crs} to longitude/latitude (WGS84)")
    elif not crs:
        notes.append("This layer names no coordinate reference system, so its coordinates are taken as they are")

    geoms = list(geometry) if geometry is not None else []
    count = len(field_data[0]) if field_data is not None and len(field_data) else len(geoms)
    rows: list[dict[str, Any]] = []
    for i in range(count):
        row = {name: field_data[j][i] for j, name in enumerate(fields)}
        g = geoms[i] if i < len(geoms) else None
        if g is not None:
            geom = shapely.from_wkb(bytes(g))
            if tr is not None:
                geom = _shp_transform(tr.transform, geom)
            row["geometry"] = geom.wkt
            minx, miny, maxx, maxy = geom.bounds
            row["longitude"] = (minx + maxx) / 2.0
            row["latitude"] = (miny + maxy) / 2.0
        else:
            row["geometry"] = None
            row["longitude"] = None
            row["latitude"] = None
        rows.append(row)
    if not rows:
        raise ValueError(f"{path.name} has no features in it")
    kind = "GeoPackage" if path.suffix.lower() == ".gpkg" else "shapefile"
    notes.insert(0, f"Read a {kind}: {len(rows):,} feature{'s' if len(rows) != 1 else ''}. Attributes are columns, "
                    "with 'geometry' (WKT) and a centre point")
    return pl.DataFrame(rows, infer_schema_length=None), notes


# extracted leaf modules, re-exported so `geo.world_outlines` etc. stay valid
from ._geometry import parse_wkt_rings, point_in_polygon, ring_bounds  # noqa: E402
from ._projection import (latlon_to_utm, looks_projected, utm_to_latlon, utm_to_latlon_arrays,  # noqa: E402
                         utm_zone)
from ._world import world_bounds, world_outlines  # noqa: E402
