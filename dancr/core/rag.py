"""In-product retrieval: a small, deterministic, offline embedding and a search over a project's own text.

No model, no network and no new dependency: a text becomes a bag of words, word bigrams and character
trigrams hashed into a fixed-width vector and L2-normalised, so two passages that share vocabulary land
close together. It is a lexical index, not a neural one, but it makes a project's documents and tables
searchable today, on the machine, and the retrieval contract (a ``vector`` column, cosine scores, cited
source rows) is exactly what a model embedding would fill later without changing anything else.

Pure standard library + numpy (already a core dependency). No Qt.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import unicodedata
import uuid
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_DIM = 256
_WORD = re.compile(r"[0-9a-zA-Z]+")


def _features(text: Any) -> list[str]:
    t = unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode("ascii").lower()
    words = _WORD.findall(t)
    toks = list(words)
    toks += [f"{a}_{b}" for a, b in zip(words, words[1:])]          # word bigrams
    for w in words:
        if len(w) > 3:
            toks += [w[i:i + 3] for i in range(len(w) - 2)]         # character trigrams
    return toks


def embed(text: Any, dim: int = DEFAULT_DIM) -> np.ndarray:
    """A unit-norm float32 vector for a passage. Deterministic: the same text always gives the same vector."""
    v: np.ndarray = np.zeros(int(dim), dtype=np.float32)
    for tok in _features(text):
        h = int.from_bytes(hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest(), "big")
        v[h % dim] += 1.0 if (h >> 63) & 1 else -1.0
    n = float(np.linalg.norm(v))
    return v / n if n else v


def chunks(text: Any, size: int = 800, overlap: int = 100) -> list[str]:
    """Split a long text into overlapping pieces, cutting only on a character count."""
    s = str(text or "")
    if not s.strip():
        return []
    size = max(1, int(size))
    overlap = max(0, min(int(overlap), size - 1))
    step = size - overlap
    out: list[str] = []
    for start in range(0, len(s), step):
        piece = s[start:start + size]
        if piece.strip():
            out.append(piece)
        if start + size >= len(s):
            break
    return out


def matrix(vectors: Any) -> np.ndarray:
    """A (n, dim) float32 matrix from a column of vectors (a numpy array passes through; a list is converted)."""
    if vectors is None or len(vectors) == 0:
        return np.zeros((0, DEFAULT_DIM), dtype=np.float32)
    return np.asarray(vectors, dtype=np.float32)


def top_k(scores: np.ndarray, k: int, min_score: float = 0.0) -> np.ndarray:
    """The indices of the best ``k`` scores at or above ``min_score``, best first."""
    if scores.size == 0:
        return np.empty(0, dtype=np.int64)
    order = np.argsort(-scores)
    if min_score > 0.0:
        order = order[scores[order] >= min_score]
    return order[:max(0, int(k))].astype(np.int64)


# ---------------------------------------------------------------- hybrid retrieval (C1)
# Deterministic and offline: a lexical vector retriever (the default), a standard-library BM25 keyword
# retriever, and a deterministic fusion of the two. No randomness, no model, no vector database. A neural
# embedder, if one is ever added, must be pinned by id and version and recorded in the index sidecar so a run
# stays auditable (docs/adr/0004). The provenance block on every result states which retrievers were used.
EMBEDDER_ID = "dancr-lexical-v1"
RETRIEVERS = ("lexical", "bm25", "hybrid")
INDEX_META_SUFFIX = ".meta.json"


def _tokens(text: Any) -> list[str]:
    return [w.lower() for w in _WORD.findall(str(text or ""))]


def bm25_scores(texts: list[Any], query: str, *, k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    """Okapi BM25 of every passage against the query. Pure standard library and deterministic; computed at
    query time from the stored passage text, so adding it needs no re-embedding."""
    docs = [_tokens(t) for t in texts]
    n = len(docs)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    q = _tokens(query)
    avgdl = (sum(len(d) for d in docs) / n) or 1.0
    df_count: dict[str, int] = {}
    for d in docs:
        for w in set(d):
            df_count[w] = df_count.get(w, 0) + 1
    scores: np.ndarray = np.zeros(n, dtype=np.float32)
    for term in q:
        df = df_count.get(term, 0)
        if df == 0:
            continue
        idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
        for i, d in enumerate(docs):
            tf = d.count(term)
            if tf == 0:
                continue
            denom = tf + k1 * (1.0 - b + b * (len(d) / avgdl))
            scores[i] += idf * (tf * (k1 + 1.0)) / denom
    return scores


def _minmax(scores: np.ndarray) -> np.ndarray:
    """Scale scores to [0, 1] for fusion; all-equal (or empty) becomes zeros. Deterministic."""
    if scores.size == 0:
        return scores
    lo, hi = float(scores.min()), float(scores.max())
    if hi <= lo:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


def lexical_scores(query: str, vectors: list[Any], dim: int) -> np.ndarray:
    """Cosine similarity of the query's offline embedding against the stored passage vectors (the default)."""
    mat = matrix(vectors)
    if mat.size == 0:
        return np.zeros(0, dtype=np.float32)
    return mat @ embed(query, int(dim or mat.shape[1]))


def rank(query: str, *, texts: list[Any], vectors: list[Any] | None, dim: int | None = None,
         retriever: str = "lexical", k: int = 5, min_score: float = 0.0,
         weight: float = 0.5) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Rank passages by the chosen retriever. Returns ``(order, scores, provenance)``.

    ``lexical`` uses the offline embedding (the default, byte-identical to before); ``bm25`` uses keyword
    scoring with no vectors at all; ``hybrid`` fuses both (weighted, each min-max scaled). ``min_score`` is a
    floor on the reported score and is ignored for BM25, whose scores are unbounded."""
    if retriever not in RETRIEVERS:
        raise ValueError(f"Unknown retriever {retriever!r}. Choose one of: {', '.join(RETRIEVERS)}")
    vecs: list[Any] = vectors if vectors is not None else []
    has_vectors = len(vecs) > 0
    provenance: dict[str, Any] = {"retriever": retriever, "embedder": None, "retrievers": []}
    if retriever == "bm25" or not has_vectors:
        scores = bm25_scores(texts, query)
        provenance["retrievers"] = ["bm25"]
        floor = 0.0
    else:
        size = int(dim or (len(vecs[0]) if has_vectors else DEFAULT_DIM))
        lex = lexical_scores(query, vecs, size)
        provenance["embedder"] = EMBEDDER_ID
        if retriever == "hybrid":
            bm = bm25_scores(texts, query)
            w = min(1.0, max(0.0, float(weight)))
            scores = w * _minmax(lex) + (1.0 - w) * _minmax(bm)
            provenance["retrievers"] = ["lexical", "bm25"]
            provenance["fusion"] = {"method": "weighted_minmax", "lexical_weight": w}
        else:
            scores = lex
            provenance["retrievers"] = ["lexical"]
        floor = min_score
    order = top_k(scores, k, floor)
    return order, scores, provenance


# ---------------------------------------------------------------- persistent index metadata
def save_index_meta(index_path: Path | str, *, embedder: str = EMBEDDER_ID, dim: int = DEFAULT_DIM,
                    retriever: str = "lexical") -> Path | None:
    """Write the small sidecar next to a persistent index that records how its vectors were made. A neural
    embedder (future) would be pinned here by id and version; today it is the offline lexical embedder."""
    from .repo import write_json_atomic
    p = Path(index_path).expanduser()
    return write_json_atomic(str(p) + INDEX_META_SUFFIX,
                             {"kind": "dancr.index", "version": 1, "embedder": embedder, "dim": int(dim),
                              "retriever": retriever})


def load_index_meta(index_path: Path | str) -> dict[str, Any]:
    """The sidecar for a persistent index, or an empty mapping for an index built before it existed (read as
    the lexical embedder — the only one there was)."""
    from .repo import read_json
    data = read_json(str(Path(index_path).expanduser()) + INDEX_META_SUFFIX)
    return data if isinstance(data, dict) else {}


# ----------------------------------------------------------------- incremental, persistent index
def doc_content_hash(text: Any, meta: dict[str, Any]) -> str:
    """A stable hash of a whole source document (its text plus its metadata), the unit an incremental index
    reuses or re-embeds. If any part changes, the hash changes and that document is embedded again."""
    payload = {"text": str(text or ""),
               "meta": {k: (None if v is None else str(v)) for k, v in sorted(meta.items())}}
    return hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def load_index(path: Path | str) -> Any:
    """An index saved by :func:`save_index`, or None when the file is absent or unreadable."""
    import polars as pl
    p = Path(path).expanduser()
    try:
        return pl.read_parquet(p) if p.is_file() else None
    except Exception:  # noqa: BLE001 - a damaged index is rebuilt from scratch, never fatal
        return None


def save_index(df: Any, path: Path | str) -> None:
    """Write the index atomically: a temp file beside it, then one replace, so a crash mid-write leaves the
    previous index intact."""
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    # the pid alone collides when two threads of one process (a batch/scenario/catalog run) write the same index
    tmp = p.with_name(f".{p.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp.parquet")
    try:
        df.write_parquet(tmp)
        os.replace(tmp, p)
    finally:
        tmp.unlink(missing_ok=True)


def merge_index(existing: Any, documents: list[dict[str, Any]], *, dim: int, chunk_chars: int,
                overlap: int) -> tuple[Any, dict[str, Any]]:
    """Build the index from ``documents`` (each ``{doc_key, content_hash, text, meta}``), reusing the stored
    passage rows (text and vector) of every document whose ``content_hash`` is unchanged and embedding only the
    added and changed ones. Returns the new index and a change summary (``added``/``changed``/``unchanged``/
    ``removed`` documents, and ``reused_chunks``/``embedded_chunks``)."""
    import polars as pl
    want = pl.Array(pl.Float32, dim)
    stored: dict[str, Any] = {}
    if existing is not None and existing.height and "doc_key" in existing.columns \
            and "content_hash" in existing.columns:
        for grp in existing.partition_by("doc_key", maintain_order=True):
            stored[str(grp["doc_key"][0])] = grp

    reused: list[Any] = []
    fresh: list[dict[str, Any]] = []
    added = changed = unchanged = reused_chunks = 0
    current: set[str] = set()
    for d in documents:
        key = str(d["doc_key"])
        current.add(key)
        grp = stored.get(key)
        if grp is not None and grp["content_hash"][0] == d["content_hash"] and grp.schema.get("vector") == want:
            reused.append(grp)                     # unchanged: keep its passages and vectors as they are
            unchanged += 1
            reused_chunks += len(grp)
            continue
        if grp is not None:
            changed += 1
        else:
            added += 1
        for ci, piece in enumerate(chunks(d["text"], chunk_chars, overlap)):
            fresh.append({**d["meta"], "dataset": d.get("dataset"), "doc_key": key, "content_hash": d["content_hash"],
                          "chunk_id": f"{key}::{ci}", "chunk_index": ci, "text": piece,
                          "vector": embed(piece, dim).tolist()})
    removed = sum(1 for k in stored if k not in current)

    if fresh:
        reused.append(pl.DataFrame(fresh, infer_schema_length=None).with_columns(pl.col("vector").cast(want)))
    if reused:
        out = pl.concat(reused, how="diagonal_relaxed")
    else:
        out = pl.DataFrame(schema={"dataset": pl.Utf8, "doc_key": pl.Utf8, "content_hash": pl.Utf8, "chunk_id": pl.Utf8,
                                   "chunk_index": pl.Int64, "text": pl.Utf8, "vector": want})
    out = out.sort("chunk_id")                     # deterministic order, so the result (and its hash) is stable
    changes = {"documents": len(documents), "added": added, "changed": changed, "unchanged": unchanged,
               "removed": removed, "reused_chunks": reused_chunks, "embedded_chunks": len(fresh)}
    return out, changes


def read_context_feed(source: Any) -> tuple[bool, dict[str, str | None]]:
    """A knowledge-base context export (a ``dancr context`` document, or its JSON Lines) read as
    ``(is_changed_feed, {node: content_hash})``. A ``--changed`` feed carries a ``changes`` block; a full context
    does not. An incremental index uses it to skip the datasets that did not move."""
    text = ""
    if isinstance(source, (str, Path)):
        p = Path(str(source)).expanduser()
        text = p.read_text(encoding="utf-8") if p.is_file() else str(source)
    else:
        text = str(source)
    changed = False
    docs: list[dict[str, Any]] = []
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            changed = "changes" in data
            if isinstance(data.get("documents"), list):
                docs = [d for d in data["documents"] if isinstance(d, dict)]
            elif data.get("node") or data.get("id"):
                docs = [data]                      # a one-line JSON Lines export parses as a single document
        elif isinstance(data, list):
            docs = [d for d in data if isinstance(d, dict)]
    except ValueError:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if isinstance(d, dict):
                docs.append(d)
    out: dict[str, str | None] = {}
    for d in docs:
        nid = d.get("node") or d.get("id")
        if nid is not None:
            out[str(nid)] = d.get("content_hash")
    if not changed:
        changed = any("changes" in d for d in docs)     # a --changed export keeps the marker on every JSONL line
    return changed, out


def search_knowledge(pipe, query: str, *, node: str | None = None, k: int = 5, min_score: float = 0.0,
                     allow_restricted: bool = False, restricted_levels: tuple[str, ...] = ("confidential", "restricted"),
                     retriever: str = "lexical", executor: Any = None) -> dict[str, Any]:
    """Search a project's own text index: rank the indexed chunks by the chosen retriever (the offline lexical
    embedding by default; ``bm25`` and ``hybrid`` are offline alternatives), and return the best ones with their
    source columns and a provenance block. ``node`` names the 'Build search index' step; when there is exactly
    one it is used automatically. The index is computed first if it has not been.

    When the index carries a ``sensitivity`` column, passages marked confidential or restricted are **withheld**
    by default (their count is reported); pass ``allow_restricted=True`` to include them."""
    import polars as pl

    from .executor import Executor
    from .registry import registry

    indexes = [nid for nid, n in pipe.nodes.items() if n.type == "build_index"]
    if not indexes:
        raise ValueError("This project has no search index yet. Add a 'Build search index' step and run it.")
    if node is not None:
        if node not in pipe.nodes:
            raise ValueError(f"No step called {node!r}. Steps: {list(pipe.nodes)}")
        if pipe.nodes[node].type != "build_index":
            raise ValueError(f"{pipe.nodes[node].title} is not a 'Build search index' step")
    elif len(indexes) == 1:
        node = indexes[0]
    else:
        raise ValueError(f"Which index? Name one of: {', '.join(indexes)}")
    assert node is not None

    ex = executor if executor is not None else Executor(pipe)
    st = ex.state(node)
    if st.status != "done":
        st = ex.run(targets=[node])[node]
    if st.status != "done" or not st.output:
        raise ValueError(f"{pipe.nodes[node].title} could not be built: {st.error or st.status}")
    df = pl.read_parquet(st.output)
    withheld = 0
    if "sensitivity" in df.columns and not allow_restricted:
        # a null label means "not marked", i.e. visible: only rows explicitly labelled at a restricted level are
        # withheld. (A null compared with is_in gives null, and ~null is null, which filter would drop.)
        level = df["sensitivity"].cast(pl.Utf8).str.strip_chars().str.to_lowercase()
        restricted = level.is_not_null() & level.is_in([s.lower() for s in restricted_levels])
        withheld = int(restricted.sum())
        df = df.filter(~restricted)
    if df.height == 0:
        return {"kind": "dancr.search", "query": query, "node": node, "count": 0, "withheld": withheld,
                "provenance": {"retriever": retriever, "retrievers": [], "embedder": None}, "hits": []}
    texts = df["text"].to_list() if "text" in df.columns else [""] * df.height
    # a numpy matrix straight from the Arrow buffer, not a Python list of lists: a 200k-chunk index is held once,
    # not copied into interpreter objects
    vectors = df["vector"].to_numpy() if "vector" in df.columns else []
    use = retriever if (len(vectors) or retriever == "bm25") else "bm25"   # an index with no vectors still searches by keywords
    order, scores, provenance = rank(query, texts=texts, vectors=vectors, dim=None, retriever=use,
                                     k=k, min_score=min_score)
    provenance["node"] = node
    idx_path = str(pipe.nodes[node].params.get("index_path") or "")
    if idx_path and "lexical" in provenance["retrievers"]:
        from .registry import resolve_path
        meta = load_index_meta(resolve_path(pipe.directory, idx_path))
        if meta.get("embedder"):
            provenance["embedder"] = meta["embedder"]
        if meta.get("dim"):
            provenance["dim"] = meta["dim"]
    view = df.drop("vector") if "vector" in df.columns else df
    picked = view[order]
    hits = []
    for pos, i in enumerate(order.tolist(), start=1):
        rec = picked.row(pos - 1, named=True)
        rec["rank"] = pos
        rec["score"] = round(float(scores[i]), 4)
        hits.append(rec)
    return {"kind": "dancr.search", "query": query, "node": node, "count": len(hits), "withheld": withheld,
            "top_score": (round(float(scores[order[0]]), 4) if len(order) else None),
            "provenance": provenance, "hits": hits}
