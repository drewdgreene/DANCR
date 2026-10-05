"""FAIR descriptors for a project's datasets: schema.org/Dataset, a Frictionless Data Package, a run manifest
(provenance), and a minimal RO-Crate metadata graph.

These are *descriptors*, deterministic and standard-shaped, built from what DANCR already knows — the
knowledge-base context document (:func:`dancr.headless.build_context`) and the executor's plan hashes. They
make a project's tables findable (Google Dataset Search, any schema.org consumer), interoperable (Frictionless
consumers, generic metadata), and reusable (who made it, under what license, how to cite it, and exactly what
was run to produce it). Nothing is inferred by a model and nothing here computes data: it is a mapping.

Pure core: no Qt, no execution. The dataset-level fields come from ``Pipeline.dataset_meta()`` (kept in
``meta["dataset"]``); the schema, ranges and relations come from the context document.
"""
from __future__ import annotations

import json
from typing import Any

from .units import to_ucum

SCHEMA_ORG_CONTEXT = "https://schema.org/"
FRICTIONLESS_VERSION = "https://specs.frictionlessdata.io/data-package/"
RO_CRATE_CONTEXT = "https://w3id.org/ro/crate/1.1/context"
RO_CRATE_CONFORMS = "https://w3id.org/ro/crate/1.1"
MANIFEST_KIND = "dancr.manifest"
MANIFEST_VERSION = 1


def _version() -> str:
    from dancr import __version__
    return __version__


# ----------------------------------------------------------------- shared helpers
def _slug(text: Any) -> str:
    """A short identifier for a resource name, safe in a Data Package."""
    import re
    s = re.sub(r"[^0-9a-zA-Z]+", "-", str(text or "")).strip("-").lower()
    return s or "table"


def _as_list(v: Any) -> list[Any]:
    if v is None or v == "":
        return []
    return list(v) if isinstance(v, (list, tuple)) else [v]


def _person(v: Any) -> dict[str, Any]:
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


def _people(value: Any) -> list[dict[str, Any]]:
    return [_person(v) for v in _as_list(value)]


def _license(meta: dict[str, Any]) -> str | None:
    lic = meta.get("license")
    if not lic:
        return None
    s = str(lic).strip()
    if s.lower().startswith(("http://", "https://")):
        return s
    # an SPDX id (MIT, CC-BY-4.0): point at its canonical URL, as schema.org expects a URL for a license
    return f"https://spdx.org/licenses/{s}.html"


def _context_tables(ctx: dict[str, Any]) -> list[dict[str, Any]]:
    return [t for t in (ctx.get("tables") or []) if isinstance(t, dict)]


def _columns(table: dict[str, Any]) -> list[dict[str, Any]]:
    return [c for c in (table.get("columns") or []) if isinstance(c, dict)]


def _dataset_name(ctx: dict[str, Any], meta: dict[str, Any]) -> str:
    return str(meta.get("title") or (ctx.get("project") or {}).get("name") or "Untitled")


# ----------------------------------------------------------------- schema.org/Dataset
def dataset_jsonld(ctx: dict[str, Any], meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """The project's datasets as a schema.org ``Dataset`` (JSON-LD), for Google Dataset Search and any
    schema.org consumer. ``variableMeasured`` is one ``PropertyValue`` per column (name, unit, meaning),
    ``distribution`` the file each table comes from, plus temporal and spatial coverage where known."""
    meta = dict(meta or ctx.get("dataset") or {})
    tables = _context_tables(ctx)
    doc = {
        "@context": SCHEMA_ORG_CONTEXT,
        "@type": "Dataset",
        "name": _dataset_name(ctx, meta),
        "identifier": meta.get("identifier") or (ctx.get("project") or {}).get("file") or _slug(_dataset_name(ctx, meta)),
        "dateModified": ctx.get("generated_at"),
        "isAccessibleForFree": True,
    }
    if meta.get("description"):
        doc["description"] = meta["description"]
    if meta.get("version"):
        doc["version"] = meta["version"]
    if meta.get("keywords"):
        doc["keywords"] = _as_list(meta["keywords"])
    if meta.get("citation"):
        doc["citation"] = meta["citation"]
    if meta.get("language"):
        doc["inLanguage"] = meta["language"]
    people = _people(meta.get("creator"))
    if people:
        doc["creator"] = people
    contacts = _people(meta.get("contact"))
    if contacts:
        doc["contactPoint"] = contacts
    if meta.get("publisher"):
        doc["publisher"] = {"@type": "Organization", "name": str(meta["publisher"])}
    licence = _license(meta)
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
            dist["encodingFormat"] = _encoding_for(t["file"])
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
    return doc


def _encoding_for(path: str) -> str:
    ext = str(path).lower().rsplit(".", 1)[-1] if "." in str(path) else ""
    return {"csv": "text/csv", "tsv": "text/tab-separated-values", "txt": "text/plain",
            "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "xls": "application/vnd.ms-excel", "parquet": "application/vnd.apache.parquet",
            "geojson": "application/geo+json", "json": "application/json"}.get(ext, "application/octet-stream")


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


# ----------------------------------------------------------------- Frictionless Data Package
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
    tables = _context_tables(ctx)

    def rel(p: str) -> str:
        p = str(p)
        return f"{path_root.rstrip('/')}/{p}" if path_root else p

    package: dict[str, Any] = {
        "profile": "tabular-data-package",
        "name": _slug(_dataset_name(ctx, meta)),
        "title": _dataset_name(ctx, meta),
        "created": meta.get("created") or ctx.get("generated_at"),
    }
    if meta.get("description"):
        package["description"] = meta["description"]
    if meta.get("version"):
        package["version"] = meta["version"]
    if meta.get("keywords"):
        package["keywords"] = _as_list(meta["keywords"])
    if meta.get("citation"):
        package["citation"] = meta["citation"]
    licence = _license(meta)
    if licence:
        package["licenses"] = [{"name": str(meta.get("license")), "path": licence}]
    contributors = []
    for i, person in enumerate(_people(meta.get("creator"))):
        entry = {"title": person.get("name", "")}
        if person.get("email"):
            entry["email"] = person["email"]
        if person.get("identifier"):
            entry["path"] = person["identifier"]
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
            "name": _slug(t.get("node") or t.get("title")),
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
    return package


# ----------------------------------------------------------------- run manifest (provenance)
def _source_fingerprint(pipe, node_id: str) -> list[dict[str, Any]]:
    """What a source step reads, with the size/time/content sample the executor fingerprints it by."""
    from .registry import registry, resolve_path
    from .executor import _content_sample
    node = pipe.nodes[node_id]
    nt = registry.get(node.type)
    out: list[dict[str, Any]] = []
    for prm in nt.params:
        if prm.kind != "path":
            continue
        v = node.params.get(prm.name)
        if not (isinstance(v, str) and v.strip()):
            continue
        try:
            target = resolve_path(pipe.directory, v)
            st = target.stat()
            out.append({"path": str(target.resolve()), "size": st.st_size, "mtime_ns": st.st_mtime_ns,
                        "sample": _content_sample(target, st.st_size)})
        except (OSError, ValueError, TypeError):
            out.append({"path": str(v), "missing": True})
    return out


def run_manifest(pipe, executor=None, states: dict[str, Any] | None = None,
                 meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """Exactly what produced the project's results: the engine and library versions, the code fingerprint, the
    source files (with the stamp they are fingerprinted by), and every step's plan hash, row count and elapsed
    time. This is the provenance half of a FAIR record, and it is already what the cache keys on."""
    from .executor import engine_versions, CODE_FINGERPRINT, IMPL_VERSION
    if executor is None:
        from .executor import Executor
        executor = Executor(pipe)
    if states is None:
        states = executor.states()
    dataset_meta = dict(meta if meta is not None else pipe.dataset_meta())

    nodes = []
    for nid in pipe.topological_order():
        node = pipe.nodes[nid]
        st = states.get(nid)
        h = None
        try:
            h = executor.plan_hash(nid)
        except Exception:  # noqa: BLE001 - a step whose hash cannot be worked out simply reports none
            h = None
        entry: dict[str, Any] = {"id": nid, "type": node.type, "title": node.title, "hash": h}
        if st is not None:
            entry["status"] = st.status
            entry["rows"] = st.rows
            entry["columns"] = list(st.columns or [])
            entry["elapsed"] = st.elapsed
            entry["finished_at"] = st.finished_at
            if st.files:
                entry["files"] = sorted(st.files)
        nodes.append(entry)

    sources = []
    from .registry import registry
    for nid, node in pipe.nodes.items():
        if registry.get(node.type).kind == "source":
            sources.append({"node": nid, "title": node.title, "files": _source_fingerprint(pipe, nid)})

    return {
        "kind": MANIFEST_KIND,
        "version": MANIFEST_VERSION,
        "generated_at": _now(),
        "engine": {"name": "DANCR", "version": _version(), "impl_version": IMPL_VERSION,
                   "fingerprint": CODE_FINGERPRINT},
        "libraries": engine_versions(),
        "project": {"name": pipe.name, "file": pipe.path.name if pipe.path else None},
        "dataset": dataset_meta,
        "sources": sources,
        "nodes": nodes,
    }


def _now() -> str:
    from datetime import datetime
    return datetime.now().isoformat(timespec="seconds")


# ----------------------------------------------------------------- RO-Crate (metadata graph)
def ro_crate_graph(ctx: dict[str, Any], meta: dict[str, Any], files: list[dict[str, Any]]) -> dict[str, Any]:
    """The RO-Crate metadata graph for the given files. Each ``files`` entry is ``{path, name?, encodingFormat?,
    description?}``; duplicates by path are merged. The root ``Dataset`` ``hasPart`` exactly these files, so a
    crate's manifest describes precisely what was packaged."""
    meta = dict(meta or {})
    root_id = "./"
    graph: list[dict[str, Any]] = [
        {"@id": "ro-crate-metadata.json", "@type": "CreativeWork", "about": {"@id": root_id},
         "conformsTo": {"@id": RO_CRATE_CONFORMS}},
    ]
    root: dict[str, Any] = {"@id": root_id, "@type": "Dataset", "name": _dataset_name(ctx, meta)}
    if meta.get("description"):
        root["description"] = meta["description"]
    if meta.get("keywords"):
        root["keywords"] = _as_list(meta["keywords"])
    licence = _license(meta)
    if licence:
        root["license"] = {"@id": licence}
    people = _people(meta.get("creator"))
    if people:
        root["author"] = [{"@id": p.get("@id") or f"#person-{i}"} for i, p in enumerate(people)]
        for i, p in enumerate(people):
            node = {"@type": "Person", **{k: v for k, v in p.items() if k != "@type"}}
            node["@id"] = p.get("@id") or f"#person-{i}"
            graph.append(node)
    if ctx.get("generated_at"):
        root["datePublished"] = ctx["generated_at"]
    parts: list[dict[str, Any]] = []
    seen: set[str] = set()
    for f in files:
        path = str(f.get("path") or "").strip()
        if not path or path in seen:
            continue
        seen.add(path)
        parts.append({"@id": path})
        entity: dict[str, Any] = {"@id": path, "@type": "File"}
        if f.get("name"):
            entity["name"] = f["name"]
        entity["encodingFormat"] = f.get("encodingFormat") or _encoding_for(path)
        if f.get("description"):
            entity["description"] = f["description"]
        if f.get("source"):
            entity["isBasedOn"] = {"@id": str(f["source"])}
        graph.append(entity)
    root["hasPart"] = parts
    graph.insert(1, root)
    return {"@context": RO_CRATE_CONTEXT, "@graph": graph}


def ro_crate(ctx: dict[str, Any], manifest: dict[str, Any] | None = None, meta: dict[str, Any] | None = None,
             *, pipeline_file: str | None = None, extra_files: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """A minimal RO-Crate metadata graph (``ro-crate-metadata.json``): the root Dataset with each table as a
    file, the pipeline and the run manifest as parts. Metadata-only — it describes the files by relative path;
    pair it with the project's own files to form a crate, or use ``package_rocrate`` to write a full crate."""
    meta = dict(meta or ctx.get("dataset") or {})
    files: list[dict[str, Any]] = []
    for t in _context_tables(ctx):
        if t.get("file"):
            files.append({"path": str(t["file"]), "name": t.get("title") or t.get("node"),
                          "encodingFormat": _encoding_for(t["file"])})
    if pipeline_file:
        files.append({"path": str(pipeline_file), "name": "DANCR pipeline", "encodingFormat": "application/json"})
    if manifest is not None:
        files.append({"path": "dancr-manifest.json", "name": "DANCR run manifest",
                      "encodingFormat": "application/json"})
    files.append({"path": "dancr-datapackage.json", "name": "Frictionless data package",
                  "encodingFormat": "application/json"})
    files += list(extra_files or [])
    return ro_crate_graph(ctx, meta, files)


# ----------------------------------------------------------------- top-level convenience
FORMATS = ("schema.org", "frictionless", "manifest", "rocrate")


def fair_document(ctx: dict[str, Any], fmt: str, *, pipe=None, executor=None, manifest: dict[str, Any] | None = None,
                  meta: dict[str, Any] | None = None, pipeline_file: str | None = None) -> Any:
    """One descriptor by name: ``schema.org`` | ``frictionless`` | ``manifest`` | ``rocrate``."""
    key = (fmt or "").strip().lower().replace("_", ".")
    if key in ("schema.org", "schemaorg", "schema", "jsonld", "dataset"):
        return dataset_jsonld(ctx, meta)
    if key in ("frictionless", "datapackage", "data-package"):
        return datapackage(ctx, meta)
    if key in ("manifest", "provenance", "run"):
        if manifest is not None:
            return manifest
        if pipe is None:
            raise ValueError("A run manifest needs the project")
        return run_manifest(pipe, executor, meta=meta)
    if key in ("rocrate", "ro-crate"):
        return ro_crate(ctx, manifest, meta, pipeline_file=pipeline_file)
    raise ValueError(f"Unknown format {fmt!r}. Choose one of: {', '.join(FORMATS)}")


def dump(document: Any) -> str:
    """A descriptor as indented JSON text."""
    from .dtypes import json_safe
    return json.dumps(json_safe(document), indent=2, ensure_ascii=False, default=str) + "\n"
