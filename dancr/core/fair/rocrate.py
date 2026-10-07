"""A minimal RO-Crate metadata graph."""
from __future__ import annotations

from typing import Any

from ._common import RO_CRATE_CONFORMS, RO_CRATE_CONTEXT, as_list, context_tables, dataset_name, encoding_for, \
    license_url, people


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
    root: dict[str, Any] = {"@id": root_id, "@type": "Dataset", "name": dataset_name(ctx, meta)}
    if meta.get("description"):
        root["description"] = meta["description"]
    if meta.get("keywords"):
        root["keywords"] = as_list(meta["keywords"])
    licence = license_url(meta)
    if licence:
        root["license"] = {"@id": licence}
    creators = people(meta.get("creator"))
    if creators:
        root["author"] = [{"@id": p.get("@id") or f"#person-{i}"} for i, p in enumerate(creators)]
        for i, p in enumerate(creators):
            node = {"@type": "Person", **{k: v for k, v in p.items() if k != "@type"}}
            node["@id"] = p.get("@id") or f"#person-{i}"
            graph.append(node)
    if ctx.get("generated_at"):
        root["datePublished"] = ctx["generated_at"]
    if ctx.get("graph"):
        root["dancrGraph"] = ctx["graph"]         # DANCR extension: cross-project relations (plain JSON)
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
        entity["encodingFormat"] = f.get("encodingFormat") or encoding_for(path)
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
    for t in context_tables(ctx):
        if t.get("file"):
            files.append({"path": str(t["file"]), "name": t.get("title") or t.get("node"),
                          "encodingFormat": encoding_for(t["file"])})
    if pipeline_file:
        files.append({"path": str(pipeline_file), "name": "DANCR pipeline", "encodingFormat": "application/json"})
    if manifest is not None:
        files.append({"path": "dancr-manifest.json", "name": "DANCR run manifest",
                      "encodingFormat": "application/json"})
    files.append({"path": "dancr-datapackage.json", "name": "Frictionless data package",
                  "encodingFormat": "application/json"})
    files += list(extra_files or [])
    return ro_crate_graph(ctx, meta, files)
