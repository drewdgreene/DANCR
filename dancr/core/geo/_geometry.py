"""WKT polygon parsing and point-in-polygon tests."""
from __future__ import annotations

from typing import Any

# =================================================================== polygons
def parse_wkt_rings(wkt: Any) -> list[list[tuple[float, float]]]:
    """A WKT Polygon/MultiPolygon as a list of rings, each a list of (lon, lat). Outer and inner rings are all
    returned; ``point_in_polygon`` treats a point inside an even number of rings as outside. Empty for other
    geometry (points, lines) or unreadable text."""
    s = str(wkt or "").strip()
    if "(" not in s:
        return []
    kind = s.split("(", 1)[0].strip().upper()
    if kind not in ("POLYGON", "MULTIPOLYGON"):
        return []
    import re as _re
    rings: list[list[tuple[float, float]]] = []
    for body in _re.findall(r"\(([^()]*)\)", s):
        pts = []
        for pair in body.split(","):
            parts = pair.split()
            if len(parts) >= 2:
                try:
                    pts.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    pass
        if len(pts) >= 3:
            rings.append(pts)
    return rings


def point_in_polygon(lon: float, lat: float, rings: list[list[tuple[float, float]]]) -> bool:
    """Whether (lon, lat) is inside the polygon whose WKT parsed to ``rings`` (ray casting, holes handled by
    the even-odd rule). Deterministic and dependency-free; good for admin boundaries and catchments."""
    inside = False
    for ring in rings:
        n = len(ring)
        j = n - 1
        for i in range(n):
            xi, yi = ring[i]
            xj, yj = ring[j]
            if (yi > lat) != (yj > lat) and lon < (xj - xi) * (lat - yi) / (yj - yi) + xi:
                inside = not inside
            j = i
    return inside


def ring_bounds(rings: list[list[tuple[float, float]]]) -> tuple[float, float, float, float] | None:
    """(lon_min, lon_max, lat_min, lat_max) of a ring set, or None."""
    pts = [p for ring in rings for p in ring]
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), max(xs), min(ys), max(ys)
