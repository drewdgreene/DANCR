"""Dependency-free UTM inverse and forward (WGS84)."""
from __future__ import annotations

import math
from typing import Any

# =================================================================== projected coordinates (UTM)
# A dependency-free UTM inverse and forward (WGS84), good to about a metre — enough to bring projected
# eastings/northings onto the map. For other projections, pyproj is used when it is installed.
_A, _F = 6378137.0, 1 / 298.257223563
_E2 = _F * (2 - _F)
_K0 = 0.9996
_EP2 = _E2 / (1 - _E2)
_E1 = (1 - math.sqrt(1 - _E2)) / (1 + math.sqrt(1 - _E2))


def utm_to_latlon(easting: float, northing: float, zone: int, south: bool) -> tuple[float, float]:
    """(lat, lon) of a UTM easting/northing in a zone (WGS84)."""
    x = float(easting) - 500000.0
    y = float(northing) - (10_000_000.0 if south else 0.0)
    m = y / _K0
    mu = m / (_A * (1 - _E2 / 4 - 3 * _E2 ** 2 / 64 - 5 * _E2 ** 3 / 256))
    phi1 = (mu + (3 * _E1 / 2 - 27 * _E1 ** 3 / 32) * math.sin(2 * mu)
            + (21 * _E1 ** 2 / 16) * math.sin(4 * mu) + (151 * _E1 ** 3 / 96) * math.sin(6 * mu))
    c1 = _EP2 * math.cos(phi1) ** 2
    t1 = math.tan(phi1) ** 2
    n1 = _A / math.sqrt(1 - _E2 * math.sin(phi1) ** 2)
    r1 = n1 * (1 - _E2) / (1 - _E2 * math.sin(phi1) ** 2)
    d = x / (n1 * _K0)
    lat = phi1 - (n1 * math.tan(phi1) / r1) * (
        d ** 2 / 2 - (5 + 3 * t1 + 10 * c1 - 4 * c1 ** 2 - 9 * _EP2) * d ** 4 / 24
        + (61 + 90 * t1 + 298 * c1 + 45 * t1 ** 2 - 252 * _EP2 - 3 * c1 ** 2) * d ** 6 / 720)
    lon = (d - (1 + 2 * t1 + c1) * d ** 3 / 6
           + (5 - 2 * c1 + 28 * t1 - 3 * c1 ** 2 + 8 * _EP2 + 24 * t1 ** 2) * d ** 5 / 120) / math.cos(phi1)
    lon0 = math.radians((zone - 1) * 6 - 180 + 3)
    return math.degrees(lat), math.degrees(lon0 + lon)


def utm_to_latlon_arrays(easting: Any, northing: Any, zone: int, south: bool) -> tuple[Any, Any]:
    """The inverse of the UTM series for arrays of eastings/northings, in one vectorized pass (NumPy).

    The same series as :func:`utm_to_latlon`, applied elementwise, so a projected table of millions of rows is
    reprojected without a Python loop. Non-finite inputs give non-finite outputs (the caller blanks them)."""
    import numpy as np
    x = np.asarray(easting, dtype=float) - 500000.0
    y = np.asarray(northing, dtype=float) - (10_000_000.0 if south else 0.0)
    m = y / _K0
    mu = m / (_A * (1 - _E2 / 4 - 3 * _E2 ** 2 / 64 - 5 * _E2 ** 3 / 256))
    phi1 = (mu + (3 * _E1 / 2 - 27 * _E1 ** 3 / 32) * np.sin(2 * mu)
            + (21 * _E1 ** 2 / 16) * np.sin(4 * mu) + (151 * _E1 ** 3 / 96) * np.sin(6 * mu))
    c1 = _EP2 * np.cos(phi1) ** 2
    t1 = np.tan(phi1) ** 2
    n1 = _A / np.sqrt(1 - _E2 * np.sin(phi1) ** 2)
    r1 = n1 * (1 - _E2) / (1 - _E2 * np.sin(phi1) ** 2)
    d = x / (n1 * _K0)
    lat = phi1 - (n1 * np.tan(phi1) / r1) * (
        d ** 2 / 2 - (5 + 3 * t1 + 10 * c1 - 4 * c1 ** 2 - 9 * _EP2) * d ** 4 / 24
        + (61 + 90 * t1 + 298 * c1 + 45 * t1 ** 2 - 252 * _EP2 - 3 * c1 ** 2) * d ** 6 / 720)
    lon = (d - (1 + 2 * t1 + c1) * d ** 3 / 6
           + (5 - 2 * c1 + 28 * t1 - 3 * c1 ** 2 + 8 * _EP2 + 24 * t1 ** 2) * d ** 5 / 120) / np.cos(phi1)
    lon0 = math.radians((zone - 1) * 6 - 180 + 3)
    return np.degrees(lat), np.degrees(lon0 + lon)


def latlon_to_utm(lat: float, lon: float, zone: int, south: bool) -> tuple[float, float]:
    """(easting, northing) of a lat/lon in a UTM zone (WGS84)."""
    lat_r = math.radians(lat)
    lon0 = math.radians((zone - 1) * 6 - 180 + 3)
    n = _A / math.sqrt(1 - _E2 * math.sin(lat_r) ** 2)
    t = math.tan(lat_r) ** 2
    c = _EP2 * math.cos(lat_r) ** 2
    a = math.cos(lat_r) * (math.radians(lon) - lon0)
    m = _A * ((1 - _E2 / 4 - 3 * _E2 ** 2 / 64 - 5 * _E2 ** 3 / 256) * lat_r
              - (3 * _E2 / 8 + 3 * _E2 ** 2 / 32 + 45 * _E2 ** 3 / 1024) * math.sin(2 * lat_r)
              + (15 * _E2 ** 2 / 256 + 45 * _E2 ** 3 / 1024) * math.sin(4 * lat_r)
              - (35 * _E2 ** 3 / 3072) * math.sin(6 * lat_r))
    easting = _K0 * n * (a + (1 - t + c) * a ** 3 / 6 + (5 - 18 * t + t ** 2 + 72 * c - 58 * _EP2) * a ** 5 / 120) + 500000.0
    northing = _K0 * (m + n * math.tan(lat_r) * (a ** 2 / 2 + (5 - t + 9 * c + 4 * c ** 2) * a ** 4 / 24
                       + (61 - 58 * t + t ** 2 + 600 * c - 330 * _EP2) * a ** 6 / 720))
    if south:
        northing += 10_000_000.0
    return easting, northing


def utm_zone(lon: float, lat: float) -> tuple[int, bool]:
    """The UTM zone number (1–60) and whether it is the southern hemisphere for a lon/lat."""
    zone = int((lon + 180) / 6) + 1
    zone = min(60, max(1, zone))
    return zone, lat < 0


def looks_projected(emin: float, emax: float, nmin: float, nmax: float) -> bool:
    """Easting/northing ranges too large to be degrees: a projected coordinate system (UTM values are in the
    hundreds of thousands)."""
    return abs(emax) > 180 or abs(nmin) > 90 or (abs(emax) > 1e5 and abs(nmax) > 1e5)
