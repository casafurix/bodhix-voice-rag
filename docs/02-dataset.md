# 02 — Dataset: MSMARCO-XI

> **MVP scope note:** shipped language set is **English + Hindi + Bengali** (3, not the 5 —
> en/hi/ta/bn/mr — discussed below), and ingest currently pulls `validation/hinval.parquet` +
> `validation/benval.parquet` directly rather than the row-group HTTP-streaming approach
> described in "Step 0" (two ~460MB files don't need that machinery — see
> `ingest/stream_corpus.py`). Live status, real dedup/chunk counts:
> [docs/13-build-status.md](13-build-status.md).

## What we are actually given

[`ai4bharat/MSMARCO-XI`](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI), from the
[IndicRAGSuite](https://arxiv.org/abs/2506.01615) paper — MS MARCO machine-translated into Indic
languages, with the source English preserved on every row.

| Property | Value |
|---|---|
| Rows | 11,451,314 |
| Size | 55.6 GB Parquet |
| Splits | `train` 10.1 M · `validation` 1.37 M |
| Languages | 14 Indic + English source |
| Passages per row | ~10, with binary relevance labels |
| HF dataset viewer | **Broken** (`JobManagerCrashedError`) — no browsing, must download |

### Row schema

```jsonc
{
  "source_lang": "eng_Latn",
  "target_lang": "asm_Beng",
  "meta": { "model_name": "ckpt-3epochs-sft-then-400k-kd", "temperature": 0.0,
            "max_tokens": 4096, "top_p": 1.0,
            "frequency_penalty": 0.0, "presence_penalty": 0.0 },

  "query":       "…translated query…",
  "Answer":      "…translated answer…",
  "query_id":    1185869,
  "query_type":  "DESCRIPTION",

  "passages": {
    "is_selected":        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "English_passages":   ["…", "…", …],
    "Translated_passages":["…", "…", …]
  },

  "Eng_Query":  "…original English query…",
  "Eng_Answer": "…original English answer…"
}
```

### Files on the hub

`train/` — `asm ben guj hin kan mal mar nep ori pan san tam urd` (13 files, ~3.3–4.0 GB each)
`validation/` — the same 13 **plus `telval.parquet`** (~0.42–0.49 GB each)

**Note a real gap:** there is a `validation/telval.parquet` but **no `train/teltrain.parquet`**.
The README's language table lists Telugu with a `teltrain.jsonl`, but the file is absent from the
repository tree. Telugu is therefore validation-only. We will not silently pretend otherwise; if
Telugu appears in our language set it is sourced from validation and labelled as such.

---

## Three properties that change the design

### 1. `is_selected` is a free, labelled retrieval benchmark

Each row ships ~10 candidate passages with a binary relevance label. That is a ready-made
qrels file. We can compute **Recall@k, nDCG@10, MRR and MAP** for any chunking or retrieval
configuration without writing a single annotation.

This is the most valuable thing in the dataset and most submissions will ignore it. It converts
the brief's "chunking should be vast" from a subjective claim into a measured leaderboard — we do
not argue that our chunking is thoughtful, we publish the table.
See [03-chunking.md](03-chunking.md) and [09-evaluation.md](09-evaluation.md).

Caveat, stated up front: MS MARCO relevance labels are famously **sparse and incomplete** — a
passage labelled `0` is often actually relevant, just unjudged. So we treat these metrics as a
*relative* ranking signal between our own configurations, which is exactly what we need, and not
as an absolute quality claim comparable to published leaderboards.

### 2. Every row is bilingual and parallel

`Translated_passages[i]` and `English_passages[i]` are the same passage in two languages, aligned
by index. Likewise `query` / `Eng_Query` and `Answer` / `Eng_Answer`.

This enables things a monolingual corpus cannot:

- **Cross-lingual twin indexing.** Index both language versions of a passage as linked twins. A
  Hindi query that fails to match the Hindi translation may still match the English original,
  and the answer is returned in the query's language from the twin. Free recall.
- **Translation-quality filtering.** We can score the fidelity of a translation (embedding
  similarity between twins, length ratio, script purity) and *down-weight or drop* bad
  translations at ingest. The data is machine-translated by a distilled checkpoint, so some of it
  is bad. Most submissions will index the noise as-is.
- **Language-consistency guardrail.** We have a reference for what language the answer should be
  in, and a parallel text to check against. See [07-guardrails.md](07-guardrails.md).
- **Zero-cost eval pairs.** The same question in 14 languages against the same passage set is a
  ready-made cross-lingual consistency test: does the pipeline give the same answer to the same
  question asked in Tamil and in Bengali? That is a genuinely interesting number to publish.

### 3. Passages are already chunk-sized, and heavily duplicated

MS MARCO passages are web snippets of roughly 50–120 words — already close to an ideal retrieval
unit. That means:

- Naive fixed-size splitting is not just lazy here, it is **actively harmful**: it cuts coherent
  passages in half. This is worth saying in the demo, because it inverts the usual assumption.
- The interesting chunking work is therefore *not* "how do I split a long document" but
  "what is the right retrieval unit, and what context do I return around it" — sub-passage
  propositions, super-passage windows, and derived units like generated questions.

And because MS MARCO reuses the same passages across many queries, the ~10 passages × 11.45 M rows
≈ **114 M passage instances collapse to far fewer unique passages**. Dedup is mandatory, not an
optimisation: it is the difference between an index we cannot host and one we can.

---

## Corpus construction plan

### Step 0 — Do not download 55 GB

Stream with `datasets` in streaming mode or read Parquet row-groups directly from the hub with
`pyarrow`/`polars` over HTTP range requests. We only ever materialise the columns we need
(`query`, `Answer`, `query_id`, `query_type`, `passages`, `Eng_*`), which drops the read volume
substantially. Target: never hold more than a few GB on disk.

### Step 1 — Language and split selection

We do not need all 14 languages to satisfy the brief, and indexing all of them would trade
timeline for no marks. Selection:

| Language | Why included |
|---|---|
| **English** | The source text. Every judge can verify the answers themselves — essential for a demo video. |
| **Hindi** | Largest speaker base; strongest Sarvam support; a judge in Goa can test it live. |
| **Tamil** | Different script family and morphologically rich — stresses tokenisation and script-aware chunk sizing. |
| **Bengali** | Second-largest speaker base, third script family. |
| **Marathi** | Devanagari like Hindi but a distinct language — tests whether we are actually doing language ID or just script ID. |

Five languages, three script families, one shared English backbone. Stretch: add Kannada,
Malayalam and Gujarati if ingest time allows — the pipeline is language-agnostic, so this is a
config change and a compute bill, not code.

Splits: build the index from **`validation`** primarily (1.37 M rows, ~0.47 GB/language — an order
of magnitude cheaper to process than `train`) and pull additional passages from `train` only if we
need corpus volume. Held-out queries for benchmarking are drawn from rows whose passages are in
the index, since the point of the eval is retrieval quality, not generalisation.

### Step 2 — Explode and deduplicate

```
row  ──explode──▶  10 passage instances
                    │
                    ├─ text (translated)     ─┐
                    ├─ text (english)         ├─ twin pair
                    ├─ is_selected            │
                    ├─ query_id, query_type   │
                    └─ language, script       ┘
```

Dedup key: `blake3(normalise(text))` where `normalise` does NFC Unicode normalisation, whitespace
collapse, and lowercasing for Latin script only (case is meaningless in Indic scripts). Keep the
first occurrence; accumulate the *set* of `query_id`s that reference it, because a passage
referenced by many queries is a popularity signal we can use as a retrieval prior.

Expected reduction: we will measure and publish the exact figure, but MS MARCO's passage reuse
means a 3–10× collapse is the realistic range.

### Step 3 — Quality filtering

Drop or flag, with counts published:

| Filter | Reason |
|---|---|
| Length < 20 chars or > 4000 chars | Fragments and scrapes, not answerable content |
| Script purity below threshold | A "Hindi" passage that is 90 % Latin is a failed translation |
| Twin length ratio outside [0.4, 2.5] | Translation truncated or hallucinated extra content |
| Twin embedding cosine < 0.6 | Translation is semantically wrong — the distilled checkpoint drifted |
| Near-duplicate (MinHash / SimHash, Jaccard > 0.9) | Boilerplate web repetition inflating the index |
| Boilerplate patterns (nav bars, cookie notices) | Noise that outranks real content on short queries |

The twin-similarity filter is only possible because the dataset is parallel, and it is a genuine
quality edge. We publish before/after retrieval metrics to show the filtering earned its place —
if it does not improve nDCG, it comes out.

### Step 4 — Metadata attachment

Every chunk carries a payload, because metadata-aware chunking is explicitly named in the brief
and because filtered search is a latency win (a filtered HNSW search over a partition is faster
than an unfiltered search over everything).

```jsonc
{
  "chunk_id":     "hi/1185869/p0/c2",
  "doc_id":       "hi/1185869/p0",
  "twin_id":      "en/1185869/p0",
  "language":     "hi",
  "script":       "Deva",
  "query_type":   "DESCRIPTION",       // DESCRIPTION | NUMERIC | ENTITY | PERSON | LOCATION
  "is_selected":  true,
  "ref_count":    7,                   // how many queries cite this passage
  "strategy":     "parent_child",      // which chunker produced this unit
  "parent_id":    "hi/1185869/p0",
  "char_span":    [104, 312],
  "n_tokens":     87,
  "twin_cosine":  0.91,                // translation fidelity
  "has_numbers":  true                 // enables the numeric guardrail cheaply
}
```

`query_type` deserves a note: MS MARCO's type taxonomy lets us **route by question type**. A
`NUMERIC` question should prefer chunks with `has_numbers: true` and should trigger a stricter
numeric groundedness check. A `PERSON` or `LOCATION` question benefits from entity-boosted
lexical matching. This is metadata-aware retrieval with an actual mechanism behind it rather than
a payload field nobody reads.

### Step 5 — Tiered index scale

Build in tiers so there is always a working demo, and so the scaling curve is publishable.

| Tier | Unique chunks | Purpose | Index size (384-d, int8) |
|---|---|---|---|
| **T0 dev** | ~50 K | Fast iteration; full test suite runs in seconds | ~20 MB |
| **T1 demo** | ~500 K | The live link and the demo video | ~190 MB |
| **T2 target** | ~2 M | The submitted benchmark numbers | ~750 MB |
| **T3 stretch** | ~8 M | Only if ingest and RAM allow; proves the latency holds at scale | ~3 GB |

Reporting `t_core` at every tier turns "we hit 200 ms" into "here is how latency scales with
corpus size, and here is where it breaks" — a much stronger claim, and it costs one extra
benchmark run per tier.

---

## Ingest pipeline

```
hub Parquet (streamed)
   ↓  polars lazy scan, column projection
explode passages → instance rows
   ↓  blake3 dedup, MinHash near-dedup
quality filters (+ counts logged)
   ↓
metadata enrichment (lang id, script, query_type, twin cosine)
   ↓
chunkers × N strategies  ──────────────┐   (see 03-chunking.md)
   ↓                                   │
embed in batches (ONNX, int8, CPU/GPU) │
   ↓                                   │
Qdrant upsert (named vectors per       │
strategy so all strategies coexist     │
in one collection)                     │
   ↓                                   │
bm25s sparse index build ◀─────────────┘
   ↓
snapshot → single artefact for deploy
```

Idempotent, resumable and checkpointed per shard — a 2 M-chunk ingest must survive a laptop lid
closing. Ingest is `ingest/` in the repo and is versioned: every index artefact records the
commit, chunker config and model revision that produced it, so a benchmark number can always be
traced back to the exact index that produced it.

## Open questions to resolve during build

- [ ] Actual dedup ratio — determines whether T2 needs `train` data or `validation` suffices
- [ ] Measured translation-quality distribution — how much of the corpus fails the twin filters
- [ ] `query_type` distribution across the selected languages
- [ ] Whether Telugu is worth including as validation-only, or cleaner to exclude and say why
- [ ] Whether `ref_count` as a retrieval prior helps or just biases toward popular passages
