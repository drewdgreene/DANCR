"""Shared constants and small helpers for the FAIR descriptors."""
from __future__ import annotations

from typing import Any

SCHEMA_ORG_CONTEXT = "https://schema.org/"
FRICTIONLESS_VERSION = "https://specs.frictionlessdata.io/data-package/"
RO_CRATE_CONTEXT = "https://w3id.org/ro/crate/1.1/context"
RO_CRATE_CONFORMS = "https://w3id.org/ro/crate/1.1"
MANIFEST_KIND = "dancr.manifest"
MANIFEST_VERSION = 1


def version() -> str:
    from dancr import __version__
    return __version__


def slug(text: Any) -> str:
    """A short identifier for a resource name, safe in a Data Package."""
    import re
    s = re.sub(r"[^0-9a-zA-Z]+", "-", str(text or "")).strip("-").lower()
    return s or "table"


def as_list(v: Any) -> list[Any]:
    if v is None or v == "":
        return []
    return list(v) if isinstance(v, (list, tuple)) else [v]


def person(v: Any) -> dict[str, Any]:
    """A creator/contact as a schema.org Person, from a name, a ``{name,email,orcid}`` object or a string."""
    if isinstance(v, dict):
        p: dict[str, Any] = {"@type": "Person"}
        if v.get("name"):
            p["name"] = v["name"]
        if v.get("email"):
            p["email"] = v["email"]
        orcid = v.get("orcid") or v.get("identifier")
        if orcid:
            p["identifier"] = orcid
            if str(orcid).startswith(("http://", "https://")):
                p["@id"] = orcid
        return p if len(p) > 1 else {"@type": "Person", "name": str(v)}
    return {"@type": "Person", "name": str(v)}


def people(value: Any) -> list[dict[str, Any]]:
    return [person(v) for v in as_list(value)]


def license_url(meta: dict[str, Any]) -> str | None:
    lic = meta.get("license")
    if not lic:
        return None
    s = str(lic).strip()
    if s.lower().startswith(("http://", "https://")):
        return s
    # an SPDX id (MIT, CC-BY-4.0): point at its canonical URL, as schema.org expects a URL for a license
    return f"https://spdx.org/licenses/{s}.html"


def context_tables(ctx: dict[str, Any]) -> list[dict[str, Any]]:
    return [t for t in (ctx.get("tables") or []) if isinstance(t, dict)]


def columns(table: dict[str, Any]) -> list[dict[str, Any]]:
    return [c for c in (table.get("columns") or []) if isinstance(c, dict)]


def dataset_name(ctx: dict[str, Any], meta: dict[str, Any]) -> str:
    return str(meta.get("title") or (ctx.get("project") or {}).get("name") or "Untitled")


def encoding_for(path: str) -> str:
    ext = str(path).lower().rsplit(".", 1)[-1] if "." in str(path) else ""
    return {"csv": "text/csv", "tsv": "text/tab-separated-values", "txt": "text/plain",
            "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "xls": "application/vnd.ms-excel", "parquet": "application/vnd.apache.parquet",
            "geojson": "application/geo+json", "json": "application/json"}.get(ext, "application/octet-stream")


def now() -> str:
    from datetime import datetime
    return datetime.now().isoformat(timespec="seconds")
