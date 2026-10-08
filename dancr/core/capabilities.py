"""What this build of DANCR can do: the optional readers and the document engine.

`dancr doctor` and the window's "Check this build" panel both report this, so the two never disagree.
Importing the module is cheap; importing the capabilities is not, which is why the window runs the check on a
worker. Every capability ships in the one install, so a "missing" entry means a broken build, not an extra to
download."""
from __future__ import annotations

from typing import Any

PACKAGES = ("shapely", "pyproj", "pyogrio", "sqlalchemy", "psycopg", "xarray", "netCDF4", "h5py", "httpx")

# what each package unlocks, in the person's words
UNLOCKS = {
    "shapely": "polygons, spatial joins and point-in-polygon",
    "pyproj": "reprojecting any EPSG coordinate system",
    "pyogrio": "GeoPackage and shapefile files",
    "sqlalchemy": "databases other than SQLite",
    "psycopg": "PostgreSQL",
    "xarray": "NetCDF files",
    "netCDF4": "NetCDF files",
    "h5py": "HDF5 files",
    "httpx": "the Assistant's model client",
}


def capabilities() -> dict[str, Any]:
    """The installed optional packages, with versions, and whether a document reader (MinerU) is present."""
    import importlib
    import importlib.metadata as md
    present: dict[str, str] = {}
    missing: list[str] = []
    for name in PACKAGES:
        try:
            importlib.import_module(name)
            try:
                present[name] = md.version(name)
            except Exception:  # noqa: BLE001 - a version we cannot read is still present
                present[name] = "?"
        except Exception:  # noqa: BLE001 - any failure to import counts as missing
            missing.append(name)
    out: dict[str, Any] = {"ok": not missing, "present": present, "missing": missing}
    try:
        from .nodes.document import mineru_home, mineru_tool, mineru_version
        tool = mineru_tool({})
        out["documents"] = {"present": bool(tool), "command": tool or "",
                            "version": mineru_version(tool) or "", "models": mineru_home() or ""}
    except Exception:  # noqa: BLE001 - a diagnostic must never fail
        out["documents"] = {"present": False, "command": "", "version": "", "models": ""}
    return out
