"""A Frictionless Data Package (``datapackage.json``)."""
from __future__ import annotations

from typing import Any

from ..units import to_ucum
from ._common import as_list, columns as _columns, context_tables, dataset_name, license_url, people, slug


def _frictionless_type(c: dict[str, Any]) -> tuple[str, str | None]:
    """(type, format) for a Frictionless field, from a profile column."""
    dtype = str(c.get("dtype") or "").lower()
    kind = str(c.get("kind") or "").lower()
    if "int" in dtype or dtype.startswith(("u", "i")) and "int" in dtype:
        return "integer", None
    if "bool" in dtype or kind == "true/false":
        return "boolean", None
    if "datetime" in dtype or "timestamp" in dtype:
        tz = "default" if "time_zone" in dtype and "none" not in dtype.replace(" ", "") else "naive"
        return "datetime", tz
    if dtype.startswith("date") or dtype == "date":
        return "date", None
    if dtype == "time" or dtype.startswith("time"):
        return "time", None
    if "float" in dtype or "decimal" in dtype or kind == "number":
        return "number", None
    return "string", None


def datapackage(ctx: dict[str, Any], meta: dict[str, Any] | None = None, *, path_root: str = "") -> dict[str, Any]:
    """The project's datasets as a Frictionless Data Package (``datapackage.json``): one resource per table,
    each with its fields' names, types, units and meanings. ``path_root`` prefixes resource paths (blank for the
    project folder itself)."""
    meta = dict(meta or ctx.get("dataset") or {})
    tables = context_tables(ctx)
    name = dataset_name(ctx, meta)

    def rel(p: str) -> str:
        p = str(p)
        return f"{path_root.rstrip('/')}/{p}" if path_root else p

    package: dict[str, Any] = {
        "profile": "tabular-data-package",
        "name": slug(name),
        "title": name,
        "created": meta.get("created") or ctx.get("generated_at"),
    }
    if meta.get("description"):
        package["description"] = meta["description"]
    if meta.get("version"):
        package["version"] = meta["version"]
    if meta.get("keywords"):
        package["keywords"] = as_list(meta["keywords"])
    if meta.get("citation"):
        package["citation"] = meta["citation"]
    licence = license_url(meta)
    if licence:
        package["licenses"] = [{"name": str(meta.get("license")), "path": licence}]
    contributors = []
    for i, p in enumerate(people(meta.get("creator"))):
        entry = {"title": p.get("name", "")}
        if p.get("email"):
            entry["email"] = p["email"]
        if p.get("identifier"):
            entry["path"] = p["identifier"]
        entry["role"] = "author"
        contributors.append(entry)
    if contributors:
        package["contributors"] = contributors

    resources = []
    for t in tables:
        fields = []
        primary = None
        for c in _columns(t):
            ftype, fmt = _frictionless_type(c)
            field: dict[str, Any] = {"name": c.get("name"), "type": ftype}
            if fmt:
                field["format"] = fmt
            if c.get("label") and c["label"] != c.get("name"):
                field["title"] = c["label"]
            if c.get("role") and c["role"] not in ("constant", "blank"):
                field["description"] = str(c["role"])
            if c.get("unit"):
                field["unit"] = to_ucum(c["unit"]) or c["unit"]
            fields.append(field)
            if primary is None and c.get("unique") and c.get("role") == "id":
                primary = c.get("name")
        schema: dict[str, Any] = {"fields": fields}
        if primary:
            schema["primaryKey"] = primary
        resource: dict[str, Any] = {
            "name": slug(t.get("node") or t.get("title")),
            "title": t.get("title") or t.get("node"),
            "profile": "tabular-data-resource",
            "schema": schema,
        }
        if t.get("file"):
            resource["path"] = rel(t["file"])
            resource["format"] = str(t["file"]).rsplit(".", 1)[-1].lower()
        if t.get("summary"):
            resource["description"] = t["summary"]
        resources.append(resource)
    package["resources"] = resources
    if ctx.get("graph"):
        package["dancrGraph"] = ctx["graph"]      # DANCR extension: cross-project relations (ignored by consumers)
    return package
