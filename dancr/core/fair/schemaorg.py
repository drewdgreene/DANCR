"""schema.org/Dataset JSON-LD, for Google Dataset Search and any schema.org consumer."""
from __future__ import annotations

from typing import Any

from ..units import to_ucum
from ._common import SCHEMA_ORG_CONTEXT, as_list, columns as _columns, context_tables, dataset_name, encoding_for, \
    license_url, people, slug


def dataset_jsonld(ctx: dict[str, Any], meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """The project's datasets as a schema.org ``Dataset`` (JSON-LD), for Google Dataset Search and any
    schema.org consumer. ``variableMeasured`` is one ``PropertyValue`` per column (name, unit, meaning),
    ``distribution`` the file each table comes from, plus temporal and spatial coverage where known."""
    meta = dict(meta or ctx.get("dataset") or {})
    tables = context_tables(ctx)
    name = dataset_name(ctx, meta)
    doc = {
        "@context": SCHEMA_ORG_CONTEXT,
        "@type": "Dataset",
        "name": name,
        "identifier": meta.get("identifier") or (ctx.get("project") or {}).get("file") or slug(name),
        "dateModified": ctx.get("generated_at"),
        "isAccessibleForFree": True,
    }
    if meta.get("description"):
        doc["description"] = meta["description"]
    if meta.get("version"):
        doc["version"] = meta["version"]
    if meta.get("keywords"):
        doc["keywords"] = as_list(meta["keywords"])
    if meta.get("citation"):
        doc["citation"] = meta["citation"]
    if meta.get("language"):
        doc["inLanguage"] = meta["language"]
    creators = people(meta.get("creator"))
    if creators:
        doc["creator"] = creators
    contacts = people(meta.get("contact"))
    if contacts:
        doc["contactPoint"] = contacts
    if meta.get("publisher"):
        doc["publisher"] = {"@type": "Organization", "name": str(meta["publisher"])}
    licence = license_url(meta)
    if licence:
        doc["license"] = licence
    if ctx.get("engine_version"):
        doc["creator"] = doc.get("creator", []) + [{"@type": "SoftwareApplication", "name": "DANCR",
                                                    "softwareVersion": ctx["engine_version"]}]

    measured: list[dict[str, Any]] = []
    distributions: list[dict[str, Any]] = []
    temporal: list[str] = []
    for t in tables:
        for c in _columns(t):
            pv: dict[str, Any] = {"@type": "PropertyValue", "name": c.get("name")}
            if c.get("unit"):
                # unitText is the UCUM code when the unit is known, so a machine can interpret it; the raw unit
                # (what the header said) is kept beside it
                pv["unitText"] = to_ucum(c["unit"]) or c["unit"]
                if to_ucum(c["unit"]):
                    pv["unitCode"] = to_ucum(c["unit"])
            meaning = c.get("label") or c.get("role")
            if meaning and meaning != c.get("name"):
                pv["description"] = str(meaning)
            measured.append(pv)
        if t.get("from") is not None and t.get("to") is not None:
            temporal.append(f"{t['from']}/{t['to']}")
        dist: dict[str, Any] = {"@type": "DataDownload", "name": t.get("title") or t.get("node")}
        if t.get("file"):
            dist["contentUrl"] = t["file"]
            dist["encodingFormat"] = encoding_for(t["file"])
        dist["description"] = t.get("shape") or "table"
        distributions.append(dist)
    if measured:
        doc["variableMeasured"] = measured
    if distributions:
        doc["distribution"] = distributions
    if temporal:
        doc["temporalCoverage"] = temporal if len(temporal) > 1 else temporal[0]
    spatial = _spatial(tables)
    if spatial:
        doc["spatialCoverage"] = spatial
    if ctx.get("graph"):
        # DANCR extension: the cross-project relations incident to this project's datasets (a schema.org
        # consumer ignores an unknown property; the graph block is plain JSON)
        doc["dancrGraph"] = ctx["graph"]
    return doc


def _spatial(tables: list[dict[str, Any]]) -> Any:
    """A bounding box from a table's latitude/longitude columns, when their range is known."""
    for t in tables:
        place = t.get("place")
        if not isinstance(place, dict) or not place.get("lat") or not place.get("lon"):
            continue
        by_name = {c.get("name"): c for c in _columns(t)}
        lat = by_name.get(place["lat"])
        lon = by_name.get(place["lon"])
        if not lat or not lon or lat.get("min") is None or lat.get("max") is None or lon.get("min") is None or lon.get("max") is None:
            continue
        return {"@type": "Place",
                "geo": {"@type": "GeoShape",
                        "box": f"{lat['min']} {lon['min']} {lat['max']} {lon['max']}"}}
    return None
