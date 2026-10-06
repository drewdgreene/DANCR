# Retrieval (C1): the offline index, three retrievers, and provenance

`dancr/core/rag.py` is the whole retrieval engine: an offline, deterministic
lexical embedder, a standard-library BM25 retriever, a deterministic fusion of the
two, and an incremental persistent index. Nothing here reaches the network and
nothing needs a new dependency. See `docs/adr/0004-retrieval-embedder-policy.md`.

## The retrievers

| retriever | what it does |
|---|---|
| `lexical` (default) | a bag of words, word bigrams and character trigrams hashed into a fixed-width, unit-norm vector; cosine similarity. Byte-identical to the behaviour before hybrid retrieval existed. |
| `bm25` | Okapi BM25 over the stored passage text, computed at query time; no vectors needed. |
| `hybrid` | both, each min-max scaled, fused with a fixed weight (0.5 lexical / 0.5 keyword). |

All three are deterministic: the same index and query always give the same
ranking. `rank(query, texts=…, vectors=…, retriever=…)` returns
`(order, scores, provenance)`.

## The contract

A persistent index is a Parquet table with one row per passage and a `vector`
column (the lexical embedding). Search returns the best passages ranked, each with
its source columns, and every result carries a **provenance** block:

```json
{"retriever": "hybrid", "retrievers": ["lexical", "bm25"], "embedder": "dancr-lexical-v1",
 "fusion": {"method": "weighted_minmax", "lexical_weight": 0.5}, "node": "idx", "dim": 256}
```

The embedder's identity and dimension are recorded in a small sidecar next to a
persistent index (`<index>.meta.json`), written when the index is built. An index
built before the sidecar existed reads as the lexical embedder — the only one
there was. That is the extension point for a future neural embedder: it would be
pinned by id and version there, so a result stays auditable even though it is not
bit-reproducible.

## Using it

```bash
dancr search shop.json "drought tolerance" --retriever hybrid
dancr --json search shop.json "drought" --retriever bm25
```

- A `Build search index` step builds the index (`build_index`), optionally to a
  persistent `index_path`; it writes the sidecar and reports the embedder.
- A `Search index` step (`retrieve`) has a `retriever` setting
  (`lexical` | `bm25` | `hybrid`); its report carries the retrieval provenance.
- The MCP `search_knowledge` tool and `dancr.search_knowledge` take the same
  `retriever` argument.
- When the index has a `sensitivity` column, confidential/restricted passages are
  withheld unless `allow_restricted` is set — on every retriever.

## Honest limits

BM25 and the fusion are computed at query time from the stored text, so a very
large index is slower to search than the vector-only path. The fusion weight is
fixed. There is no neural embedder yet — only the recorded interface for one.
