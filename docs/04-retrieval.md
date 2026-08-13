# 04 — Retrieval: vector DB, embeddings, hybrid search, reranking

Everything here is governed by one rule from [01-architecture.md](01-architecture.md): **no
network hops inside the retrieval loop.** A 200 ms budget cannot absorb a hosted service call
between stages.

---

## Vector database

### Options evaluated

| Option | Deploy shape | Query p50 @2 M | Hybrid | Filters | Quantisation | Multi-vector | Ops cost | Verdict |
|---|---|---|---|---|---|---|---|---|
| **Qdrant** (local/embedded or sidecar) | In-process or same-container | ~3–8 ms | Native sparse+dense w/ built-in fusion | Rich, indexed payload filters | Scalar int8, binary, product | **Named vectors per point** | Low | **Chosen** |
| LanceDB | Embedded, file-based | ~5–12 ms | FTS + vector | Good, SQL-ish | IVF-PQ | Awkward | Very low | Strong runner-up |
| FAISS (raw) | In-process library | ~2–6 ms | None — build it yourself | None — build it yourself | Excellent | Manual | Medium (you are the DB) | Fastest, least featured |
| hnswlib | In-process library | ~2–5 ms | No | No | No | No | Medium | Too bare |
| USearch | In-process library | ~2–5 ms | No | Basic | Yes | No | Medium | Fast, thin |
| Milvus | Server + etcd + object store | ~5–15 ms | Yes | Yes | Yes | Yes | **High** | Over-engineered for this |
| Weaviate | Server / cloud | ~8–20 ms local | Yes | Yes | Yes | Yes | Medium-high | Heavier than needed |
| pgvector | Postgres extension | ~10–30 ms | With tsvector | Excellent (SQL) | Halfvec, binary | No | Low if PG exists | Slower; no PG here anyway |
| Vespa | JVM cluster | ~5–15 ms | Best-in-class | Excellent | Yes | Yes | **Very high** | Wrong scale of commitment |
| Chroma | Embedded / server | ~15–40 ms | Basic | Basic | Limited | No | Very low | Not fast enough |
| Pinecone | **Hosted only** | 20–80 ms **network** | Yes | Yes | Yes | Yes | None | **Disqualified — network hop** |
| Turbopuffer | **Hosted only** | 30–100 ms **network** | Yes | Yes | Yes | No | None | **Disqualified — network hop** |

Latency figures are indicative for ~2 M × 384-d on 4 vCPU and will be replaced by our own measured
numbers in `bench/results/`.

### Decision: Qdrant, colocated

Reasoning, in order of weight:

1. **Named vectors per point.** A single point can hold multiple vectors under different names.
   This is exactly the primitive [03-chunking.md](03-chunking.md) needs: all chunking strategies
   coexist in one collection, and the ablation becomes a query-parameter change rather than
   twelve separate indexes to build, host and keep in sync. No other option on the list makes the
   ablation this cheap.
2. **Native hybrid with server-side fusion.** Sparse and dense vectors in one query with RRF
   applied inside the engine — one call instead of two calls plus client-side merging.
3. **Payload filters with proper indexing.** Filtering by `language`, `query_type` or
   `has_numbers` is a first-class indexed operation, which makes metadata-aware retrieval a real
   mechanism rather than post-hoc filtering that breaks top-k.
4. **Scalar int8 quantisation with `always_ram`.** ~4× memory reduction with small recall loss,
   and an optional rescore pass against full-precision vectors when accuracy matters. This is
   what lets a multi-million-vector index sit in RAM on an 8 GB machine.
5. **Runs colocated.** Local mode or a same-container process. Loopback, no network.

Configuration intent:

```
HNSW:         m = 32, ef_construct = 256
Search:       ef = 64 (tuned against recall/latency curve, not guessed)
Quantisation: scalar int8, quantile 0.99, always_ram = true
Rescore:      on, oversampling ×2 — cheap accuracy recovery
Storage:      mmap'd on a Fly volume; payload on disk, vectors in RAM
Payload idx:  language, query_type, has_numbers, strategy
```

`ef` is the primary latency/recall dial. We publish the sweep (`ef` ∈ {32, 64, 128, 256} × recall
× p99 latency) rather than picking a value and hoping.

---

## Embedding model

### Options evaluated

| Model | Params | Dim | Indic quality | Query encode (CPU, int8) | Notes |
|---|---|---|---|---|---|
| **`intfloat/multilingual-e5-small`** | 118 M | 384 | Good | **~4–8 ms** | Best speed/quality/size balance. 100+ languages. Needs `query:` / `passage:` prefixes. **Chosen.** |
| `intfloat/multilingual-e5-base` | 278 M | 768 | Better | ~12–20 ms | Fallback if `small` recall is inadequate; 2× index size |
| `BAAI/bge-m3` | 568 M | 1024 | **Best** | ~35–60 ms | Excellent multilingual + native sparse/dense/ColBERT in one model. Too slow on CPU for the budget; a GPU deploy would make it the top pick |
| `jinaai/jina-embeddings-v3` | 572 M | 1024 (Matryoshka) | Very good | ~35–60 ms | Task LoRAs, truncatable dims, late-chunking friendly. Same speed problem |
| `paraphrase-multilingual-MiniLM-L12-v2` | 118 M | 384 | Fair | ~4–8 ms | Older, weaker on retrieval than e5-small |
| `google/muril-*` | 236 M | 768 | Indic-specialised | ~15–25 ms | Pretrained for Indian languages but not a trained *retriever* — needs fine-tuning to be useful |
| Cohere `embed-multilingual-v3` | hosted | 1024 | Very good | **40–120 ms network** | Disqualified — network hop |
| OpenAI `text-embedding-3-*` | hosted | 1536 | Good | **40–120 ms network** | Disqualified — network hop |

**Note:** Sarvam provides STT, TTS, translation and chat, but we are not assuming a production
embedding endpoint. Even if one exists it would be a network hop and therefore disqualified from
the retrieval loop.

### Decision: `multilingual-e5-small`, ONNX, int8 dynamic quantisation, 384-d

- Covers all five target languages plus the English backbone
- 384 dims keeps the index small enough to stay in RAM at T2/T3 scale
- ONNX Runtime with int8 quantisation puts query encoding at single-digit milliseconds on CPU
- Documented upgrade path: if `small` costs too much recall, move to `base` and pay ~12 ms and 2×
  index size; if we ever add a GPU, `bge-m3` becomes the obvious choice

**Correctness detail that is easy to get wrong:** e5 models require asymmetric prefixes —
`"query: …"` at search time and `"passage: …"` at index time. Getting this wrong degrades recall
badly while looking like it works. It goes in a unit test, not just a comment.

**Ingest-time acceleration:** embedding millions of chunks on CPU is the ingest bottleneck. Ingest
runs on a rented GPU or uses `sentence-transformers` batched inference on a larger machine; only
*query-time* encoding must be CPU-fast. These are separate concerns and we do not conflate them.

---

## Hybrid search

Dense-only retrieval fails on exact strings — names, numbers, rare entities, acronyms. On a
QA corpus full of `PERSON`, `LOCATION` and `NUMERIC` questions, that failure mode is common.

### Design

```
        query
          │
    ┌─────┴─────┐          both arms run concurrently on separate threads,
    ▼           ▼          so the cost is max(dense, sparse), not the sum
 dense ANN    BM25
 (Qdrant)    (bm25s)
 top-50      top-50
    └─────┬─────┘
          ▼
   Reciprocal Rank Fusion
   score = Σ 1/(k + rank_i),  k = 60
          ▼
     top-50 fused
```

- **Sparse index:** `bm25s` (Scipy-sparse, memory-mapped, sub-10 ms at this scale) or Qdrant's
  native sparse vectors. Preference is Qdrant-native so fusion happens server-side in one call and
  the two arms cannot drift out of sync; `bm25s` is the fallback if we need tokenisation control.
- **Indic tokenisation for BM25 is not free.** Whitespace tokenisation is wrong for
  morphologically rich languages like Tamil and Malayalam, where inflection buries the stem. We
  use `indic-nlp-library` normalisation plus light stemming per language. English gets standard
  analysis. Reporting BM25 performance per language will show whether this actually worked.
- **RRF over score normalisation:** RRF uses ranks, so it is immune to the incomparable score
  scales of cosine similarity and BM25. No tuning, no calibration drift.
- **Weighted RRF by `query_type`:** `NUMERIC` and `ENTITY` questions get the sparse arm up-weighted
  (exact tokens matter); `DESCRIPTION` questions get the dense arm up-weighted (semantics matter).
  The weights are fitted on a dev split, not hand-waved.

---

## Reranking

The biggest single quality lever and the biggest single latency cost — 45 ms of a 200 ms budget.
So it is **gated**, not unconditional.

### Options evaluated

| Reranker | Params | Multilingual | 50 docs, CPU int8 | Notes |
|---|---|---|---|---|
| **`jina-reranker-v2-base-multilingual`** | 278 M | Yes | ~40–60 ms | Best multilingual quality/speed trade-off. **Chosen.** |
| `BAAI/bge-reranker-v2-m3` | 568 M | Yes, strong | ~90–140 ms | Better quality, roughly 2× the cost. Too slow unless we get a GPU |
| `mixedbread-ai/mxbai-rerank-xsmall` | 70 M | Weak on Indic | ~15–25 ms | Fast but English-centric — wrong for this corpus |
| ColBERT late interaction | — | Via bge-m3 | ~10–20 ms | Attractive: precompute token vectors at ingest, MaxSim at query time. Big index inflation. Documented stretch option |
| Cohere Rerank 3 | hosted | Yes | **100–300 ms network** | Disqualified |
| LLM-as-reranker | hosted | Yes | 200 ms+ | Disqualified |

### Adaptive rerank gating

Reranking a candidate list that is already correctly ordered is 45 ms spent for nothing. So we skip
it when it cannot help, or when we cannot afford it:

```python
def should_rerank(candidates, remaining_budget_ms) -> bool:
    if remaining_budget_ms < RERANK_P95_MS + GUARD_OUT_MS + ANSWER_MS:
        return False                    # cannot afford it — budget guard
    margin = candidates[0].score - candidates[1].score
    if margin > MARGIN_HIGH:
        return False                    # top-1 already dominant — no gain expected
    if candidates[0].score < SCORE_FLOOR:
        return False                    # nothing is relevant; we are about to
                                        # refuse anyway (see coverage gate)
    return True
```

Three distinct reasons to skip, all measurable. We report the **skip rate** and the **quality delta
on skipped queries** to prove the gate is not silently degrading answers — a gate that fires on 60 %
of queries and costs 4 points of nDCG is a bad gate, and we would rather know.

This is also where the harness's deadline propagation becomes concrete: `remaining_budget_ms` is
computed from the actual elapsed time of prior stages, not assumed. See [06-harness.md](06-harness.md).

---

## Context assembly

Between reranking and answering, and it costs real milliseconds, so it is counted.

1. **Parent resolution** — small-to-big strategies (S5) return `parent_id`; fetch parent text.
2. **Deduplication** — multiple children of one parent collapse to one entry. Without this, top-5
   can be five children of the same passage and the context is one passage repeated.
3. **Span merging** — adjacent or overlapping `char_span`s on the same parent merge into one range.
4. **Twin resolution** — for S10 hits, choose the language version matching the query.
5. **Token budgeting** — fill to a fixed token budget in fused-rank order; truncate at sentence
   boundaries, never mid-sentence.
6. **Citation mapping** — each assembled block keeps its `chunk_id` so the answer can cite exactly
   and the groundedness gate has something to check against.

---

## Caching

An exact-match and near-match query cache is legitimate — production RAG systems all have one —
but it can also be used to fake a benchmark, so we are explicit:

- **Exact cache:** normalised query hash → full response. TTL-bounded.
- **Semantic cache:** query embedding, cosine > 0.97 against recent queries → reuse.
- **Benchmark policy:** all headline latency numbers are reported **cache-cold**. Cached numbers
  are reported separately and clearly labelled. The benchmark harness has a `--no-cache` flag that
  is **on by default**, and the committed CSVs record the cache state per row.

Reporting a cache-warm P50 as the headline figure would be the easiest way to "hit" 200 ms and the
fastest way to lose credibility with anyone who reads the code.
