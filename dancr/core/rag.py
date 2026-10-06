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
import os
import re
import unicodedata
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
    v = np.zeros(int(dim), dtype=np.float32)
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


def matrix(vectors: list[Any]) -> np.ndarray:
    """A (n, dim) float32 matrix from a column of vectors (lists or arrays)."""
    if not vectors:
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
    tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp.parquet")
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
    if existing is not None and getattr(existing, "height", 0) and "doc_key" in existing.columns \
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
                     executor: Any = None) -> dict[str, Any]:
    """Search a project's own text index: embed the query, rank the indexed chunks by cosine similarity, and
    return the best ones with their source columns. ``node`` names the 'Build search index' step; when there is
    exactly one it is used automatically. The index is computed first if it has not been.

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
        restricted = df["sensitivity"].cast(pl.Utf8).str.to_lowercase().is_in([s.lower() for s in restricted_levels])
        withheld = int(restricted.sum())
        df = df.filter(~restricted)
    if "vector" not in df.columns or df.height == 0:
        return {"kind": "dancr.search", "query": query, "node": node, "count": 0, "withheld": withheld, "hits": []}
    mat = matrix(df["vector"].to_list())
    dim = mat.shape[1]
    scores = mat @ embed(query, dim)
    order = top_k(scores, k, min_score)
    view = df.drop("vector")[order]
    hits = []
    for rank, i in enumerate(order.tolist(), start=1):
        rec = view.row(rank - 1, named=True)
        rec["rank"] = rank
        rec["score"] = round(float(scores[i]), 4)
        hits.append(rec)
    return {"kind": "dancr.search", "query": query, "node": node, "count": len(hits), "withheld": withheld,
            "top_score": (round(float(scores[order[0]]), 4) if len(order) else None), "hits": hits}
