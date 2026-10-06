"""The bundled Natural Earth basemap outlines."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

_WORLD_ASSET = Path(__file__).resolve().parents[2] / "assets" / "geo" / "world.geojson"


@lru_cache(maxsize=1)
def world_outlines() -> tuple[tuple[tuple[float, float], ...], ...]:
    """The bundled world country outlines as (lon, lat) rings, read once and kept.

    Public-domain Natural Earth 1:110m countries, simplified and rounded to two decimals. The app ships it,
    so a map draws with no network and no tile server ever learns where the data is. Returns an empty tuple
    if the asset is missing (a map then draws only the data).
    """
    try:
        data = json.loads(_WORLD_ASSET.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ()
    rings: list[tuple[tuple[float, float], ...]] = []
    for feat in data.get("features", []):
        geom = feat.get("geometry") or {}
        coords = geom.get("coordinates")
        if not coords:
            continue
        polys = [coords] if geom.get("type") == "Polygon" else coords
        for poly in polys:
            for ring in poly:
                rings.append(tuple((float(x), float(y)) for x, y in ring))
    return tuple(rings)


def world_bounds() -> tuple[float, float, float, float]:
    """(lon_min, lon_max, lat_min, lat_max) of the whole world."""
    return (-180.0, 180.0, -90.0, 90.0)
