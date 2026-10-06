"""Retrieval steps: build a search index over a project's text, then search it.

The index is a table with one row per passage and a ``vector`` column (see :mod:`dancr.core.rag`). It is
offline and deterministic: no model, no network. A table is indexed by a text column, or, lacking one, by a
line built from all of its columns; a loaded document is indexed by its ``text`` blocks. Search returns the
best passages by cosine similarity, each carrying the source row it came from.
"""
from __future__ import annotations

from typing import Any

import polars as pl

from ..params import Param
from ..registry import NodeType, InputSpec, Ctx, NodeResult, registry
from ..findings import finding, plural
from ..rag import (DEFAULT_DIM, doc_content_hash, embed, load_index, matrix, merge_index, read_context_feed,
                   save_index, top_k)
from ._common import first_input, schema_of

_RESERVED = ("chunk_id", "chunk_index", "text", "vector")


def _build_index(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any]) -> NodeResult:
    frames = inputs.get("items") or []
    if not frames:
        raise ValueError("Connect at least one table or document to index")
    text_col = (params.get("text_column") or "").strip()
    size = int(params.get("chunk_chars") or 800)
    overlap = int(params.get("chunk_overlap") or 100)
    dim = int(params.get("dim") or DEFAULT_DIM)
    limit = int(params.get("max_chunks") or 200000)
    id_col = (params.get("id_column") or "").strip()
    index_path = (params.get("index_path") or "").strip()
    changed_path = (params.get("changed_path") or "").strip()
    target = ctx.resolve_output(index_path) if (index_path and not ctx.preview) else None
    if target is not None and target.suffix.lower() not in (".parquet", ".pq"):
        raise ValueError("Keep the persistent index in a .parquet file (for example search_index.parquet)")
    upstream = (ctx.upstream_meta or {}).get("items") or []
    existing = load_index(target) if target is not None else None
    feed_note = ""

    # Which datasets can be skipped entirely? One whose plan hash is unchanged since the feed, and whose
    # passages are already in the persistent index (so they can be carried over without reading it again).
    skip: set[str] = set()
    if changed_path and not ctx.preview:
        try:
            is_changed_feed, listed = read_context_feed(ctx.resolve(changed_path))
        except OSError as e:
            raise ValueError(f"Cannot read the changed feed {changed_path}: {e}") from e
        if existing is None or "dataset" not in existing.columns:
            feed_note = "Skipping unchanged datasets needs a persistent index; indexing every dataset this run"
        else:
            have = set(existing["dataset"].drop_nulls().unique().to_list())
            for m in upstream:
                nid, cur = m.get("node"), m.get("hash")
                if not nid or nid not in have:
                    continue
                if is_changed_feed:
                    if nid not in listed:                 # a changed feed lists only what moved
                        skip.add(nid)
                elif nid in listed and listed[nid] == cur:  # a full context: compare hashes
                    skip.add(nid)

    documents: list[dict[str, Any]] = []
    rows_seen = 0
    for fi, lf in enumerate(frames):
        src_node = (upstream[fi].get("node") if fi < len(upstream) else None) or f"in{fi}"
        if src_node in skip:
            continue
        df = lf.collect(engine="streaming")
        schema = df.schema
        tcol = text_col if text_col in schema else ("text" if "text" in schema else None)
        meta_cols = [c for c in df.columns if c != tcol]
        rename = {c: (f"{c}_src" if c in _RESERVED else c) for c in meta_cols}
        for ridx, row in enumerate(df.iter_rows(named=True)):
            if tcol is not None:
                text = row.get(tcol)
            else:
                text = "  ".join(f"{c}: {row[c]}" for c in df.columns if row.get(c) is not None)
            meta = {rename[c]: row.get(c) for c in meta_cols}
            key = f"{src_node}:{row.get(id_col)}" if (id_col and id_col in schema) else f"{src_node}:{ridx}"
            documents.append({"dataset": src_node, "doc_key": key, "content_hash": doc_content_hash(text, meta),
                              "text": text, "meta": meta})
        rows_seen += df.height

    # merge only the included datasets, so a skipped one is not seen as removed; then carry its passages over
    included = {m.get("node") for m in upstream if m.get("node")} - skip
    if existing is not None and "dataset" in existing.columns and any(m.get("node") for m in upstream):
        existing_included = existing.filter(pl.col("dataset").is_in(sorted(included)))
    else:
        existing_included = existing
    out, changes = merge_index(existing_included, documents, dim=dim, chunk_chars=size, overlap=overlap)
    carried = None
    if skip and existing is not None and "dataset" in existing.columns:
        carried = existing.filter(pl.col("dataset").is_in(sorted(skip)))
        if carried.height:
            out = pl.concat([out, carried], how="diagonal_relaxed").sort("chunk_id")
            changes["carried_chunks"] = int(carried.height)
    if out.height > limit:
        raise ValueError(f"The index would pass {limit:,} passages. Raise 'Most passages', index fewer files, "
                         "or use larger chunks")
    if target is not None:
        save_index(out, target)
        changes["index_path"] = str(target)

    said = (f"Indexed {changes['documents']:,} document(s) into {out.height:,} passage(s): "
            f"{changes['added']} new, {changes['changed']} changed, {changes['unchanged']} unchanged"
            + (f", {changes['removed']} removed" if changes["removed"] else "")
            + (f" ({changes['reused_chunks']:,} passages reused, {changes['embedded_chunks']:,} embedded)"
               if existing is not None else "")
            + (f"; skipped {len(skip)} unchanged dataset(s)" if skip else ""))
    report = {"documents": changes["documents"], "chunks": out.height, "dim": dim, "index": changes,
              "skipped_datasets": sorted(skip),
              "finding": finding("summary", said, magnitude=float(changes["embedded_chunks"]), exact=True)}
    return NodeResult(out.lazy(), report=report, messages=([said] + ([feed_note] if feed_note else [])))


registry.register(NodeType(
    key="build_index", label="Build search index", category="AI", icon="⌕",
    description="Turn tables and documents into a searchable index: one row per passage, each with an offline "
                "vector. Feed it to 'Search index' to find passages by meaning, with citations back to the source row.",
    apply=_build_index,
    inputs=[InputSpec("items", "Tables and documents", multiple=True)],
    summary=lambda p: (f"index {p.get('text_column') or 'all columns'} · chunks of {p.get('chunk_chars') or 800}"),
    params=[
        Param("text_column", "Text column", "column", column_group="string", default="",
              help="The column that holds the text. Leave empty to use a column named 'text', or every column joined"),
        Param("chunk_chars", "Passage length (characters)", "int", default=800, min=50, max=20000, advanced=True),
        Param("chunk_overlap", "Overlap between passages", "int", default=100, min=0, max=5000, advanced=True),
        Param("dim", "Vector size", "int", default=DEFAULT_DIM, min=32, max=4096, advanced=True),
        Param("id_column", "Column that identifies a document", "column", default="", advanced=True,
              help="Used to match documents between runs for incremental re-indexing; blank uses each source row's position"),
        Param("index_path", "Keep a persistent index file", "path", default="", advanced=True,
              help="e.g. search_index.parquet. With it, only documents whose content changed are embedded again "
                   "(added/changed/unchanged/removed are reported). Blank keeps the index in the cache only"),
        Param("changed_path", "…or a changed feed (skip unchanged datasets)", "path", default="", advanced=True,
              help="A knowledge-base export (dancr context --jsonl, or its --changed form). Datasets whose content "
                   "has not moved are skipped and carried over from the persistent index without being read again"),
        Param("max_chunks", "Most passages to index", "int", default=200000, min=1, advanced=True),
    ],
))


def _retrieve(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any]) -> NodeResult:
    lf = first_input(inputs)
    schema = schema_of(lf)
    query = (params.get("query") or "").strip()
    if not query:
        raise ValueError("Type what to search for")
    k = int(params.get("k") or 5)
    min_score = float(params.get("min_score") if params.get("min_score") is not None else 0.0)
    df = lf.collect(engine="streaming")
    if "vector" not in df.columns:
        raise ValueError("Connect a 'Build search index' step to this one (it provides the 'vector' column)")
    withheld = 0
    if "sensitivity" in df.columns and not params.get("allow_restricted"):
        restricted = df["sensitivity"].cast(pl.Utf8).str.to_lowercase().is_in(["confidential", "restricted"])
        withheld = int(restricted.sum())
        df = df.filter(~restricted)
    if df.height == 0:
        empty = df.drop("vector").with_columns(pl.Series("rank", [], dtype=pl.Int64),
                                               pl.Series("score", [], dtype=pl.Float64)).lazy()
        return NodeResult(empty, report={"hits": 0, "withheld": withheld}, messages=["The index has no searchable passages"])
    mat = matrix(df["vector"].to_list())
    scores = mat @ embed(query, mat.shape[1])
    order = top_k(scores, k, min_score)
    picked = df.drop("vector")[order]
    picked = picked.with_columns(pl.Series("rank", list(range(1, len(order) + 1)), dtype=pl.Int64),
                                 pl.Series("score", [round(float(scores[i]), 4) for i in order], dtype=pl.Float64))
    cols = ["rank", "score"] + [c for c in picked.columns if c not in ("rank", "score")]
    out = picked.select(cols)
    top = float(scores[order[0]]) if len(order) else None
    said = (f"Found {plural(len(order), 'passage')}" + (f" (best match {top:.2f})" if top is not None else "")
            + (f" for {query!r}; {withheld:,} restricted withheld" if withheld else f" for {query!r}"))
    report: dict[str, Any] = {"hits": len(order), "withheld": withheld,
                              "top_score": (round(top, 4) if top is not None else None)}
    return NodeResult(out.lazy(), report=report, messages=[said])


registry.register(NodeType(
    key="retrieve", label="Search index", category="AI", icon="🔎",
    description="Search an index built by 'Build search index': give a query, get the closest passages ranked, "
                "each with its source columns. Use it to ground a summary or feed a report.",
    apply=_retrieve,
    summary=lambda p: (f"search {p.get('query') or '?'}" + (f" · top {p.get('k')}" if p.get("k") else "")),
    params=[
        Param("query", "Search for", "text", default="", required=True, placeholder="e.g. drought tolerance in 2025 trials"),
        Param("k", "How many passages", "int", default=5, min=1, max=200),
        Param("min_score", "Minimum similarity", "float", default=0.0, min=0.0, max=1.0, advanced=True,
              help="0 keeps the top k whatever their score; raise it to drop weak matches"),
        Param("allow_restricted", "Include confidential/restricted passages", "bool", default=False, advanced=True,
              help="When the index has a 'sensitivity' column, restricted passages are withheld unless this is on"),
    ],
))
