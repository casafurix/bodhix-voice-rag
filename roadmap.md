# Voice-RAG over MSMARCO-XI — Build Plan

Voice-enabled Retrieval-Augmented Generation. A user speaks a question, we transcribe it,
retrieve grounded context from the MSMARCO-XI passage corpus, and generate an answer from
that context only.

```
Voice → Sarvam STT → embed query → FAISS top-k → guardrail check → grounded answer
```

The dataset is **not** for training. Its passages are the knowledge base we chunk, embed and
index. Its `query` / `is_selected` labels are our **answer key** for grading retrieval — they
never touch the live pipeline.

---

## 1. Tech stack

| Layer               | Choice                                                    | Why                                                                                                         |
| ------------------- | --------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| Speech-to-text      | **Sarvam**                                                | Built for Indic languages; matches the dataset. Pick one — we pick Sarvam.                                  |
| Embeddings          | **`intfloat/multilingual-e5-small`** (384-dim)            | CPU-fast (~10–30ms/query), multilingual, handles Hindi + English. Needs `query:` / `passage:` prefixes.     |
| Vector store        | **FAISS**, in-process                                     | Local, no network hop. `IndexFlatIP` (exact cosine after L2-normalize) for a small index; HNSW if it grows. |
| Metadata store      | **Parquet / SQLite sidecar**                              | Holds `source_passage_id`, `query_type`, `lang`, raw text per chunk. Required for provenance + scoring.     |
| Generation LLM      | **Groq (Llama-3.x)** or **Gemini 2.0 Flash**              | Fast, free-tier. NVIDIA NIM as a free alternative. Generation is **outside** the latency budget.            |
| Harness             | **Python + Pydantic** (typed I/O), **tenacity** (retries) | Structured request/response objects, retry + error recovery around every external call.                     |
| Serving / live link | **Gradio on Hugging Face Spaces**                         | Built-in mic capture, deployable in minutes — right call with a 1-day deadline.                             |

> **Language decision.** Pick **one** language for the demo (Hindi is the safe choice).
> Embed the **Translated** (Hindi) passages and embed the Hindi query with the same
> multilingual model — same-language matching in a shared space gives the best recall.
> Keep the English passages indexed too if you want to demo cross-lingual retrieval.

---

## 2. Repo structure

```
app/
  config.py       # LATENCY_BUDGET_MS and other constants  (their benchmark imports this)
  ingest.py       # OFFLINE: load dataset → dedup → chunk → embed → build FAISS index + metadata
  retriever.py    # warmup() + search(query, top_k=5) -> SearchResponse  (their benchmark imports this)
  stt.py          # Sarvam speech-to-text wrapper (retry + error recovery)
  generator.py    # grounded LLM call (Groq/Gemini), refuses when context is weak
  guardrails.py   # input safety + off-topic + grounding threshold
  harness.py      # orchestration: stt → retrieve → guardrail → generate, structured I/O
  benchmark.py    # THEIR script (drop in as-is, then tweak percentiles to P50/P70/P100)
  eval.py         # is_selected → recall@k + MRR, and the chunking comparison table
  server.py       # FastAPI or Gradio app (mic in, answer out)
data/
  index.faiss     # built by ingest.py
  chunks.parquet  # chunk_id, text, source_passage_id, query_type, lang
```

The two files their reference benchmark imports are `app/config.py` (`LATENCY_BUDGET_MS`) and
`app/retriever.py` (`warmup()`, `search()`). Build those to spec and their harness runs
against us unmodified.

---

## 3. Data prep & chunking

### 3a. Prep (offline, once)

1. Load one language subset (`load_dataset("ai4bharat/MSMARCO-XI", "hi", split="train")`), and
   **scope down** — a few thousand unique passages is plenty for a demo and keeps search fast.
2. **Flatten** all `passages` across all queries into one pile.
3. **Dedup** — the same passage recurs across many queries. Hash text, keep one copy, but keep
   a `passage_id → [query_ids]` map so we don't lose the `is_selected` links.
4. **Tag** every chunk with metadata: `source_passage_id`, `query_type`, `lang`. This provenance
   is what makes scoring possible — without `source_passage_id` you cannot check hits.

### 3b. Chunking — a menu, not one strategy (requirement 2)

> MSMARCO passages are already short (~50–100 words), so aggressive splitting can _hurt_.
> That's a legitimate, report-worthy finding — we measure it rather than assume.

1. **Passage-as-chunk (baseline / control)** — each passage is one chunk.
2. **Semantic chunking** — embed sentences, cut where cosine similarity between consecutive
   sentences drops (topic-shift "valleys"). The natural-language analog of AST chunking: both
   cut on _meaning boundaries_, not fixed sizes. (AST itself is for source code — wrong tool for prose.)
3. **Small-to-big / parent-document** — index _sentences_ for precise matching, but return the
   whole parent _passage_ to the LLM for context. Usually the quiet winner.
4. **Metadata-aware** — the tagging above, so retrieval can filter/boost by `query_type` or `lang`.

Fixed-size + overlap is included only as a second baseline to show we considered it — not our
submission strategy.

**We benchmark all of these against each other** (§7) and ship the winner, with the comparison
table in the repo.

---

## 4. Storage & retrieval

- **Normalize** all embeddings to unit length, then inner-product (`IndexFlatIP`) == cosine.
- Prefix correctly for e5: passages embedded as `"passage: <text>"`, queries as `"query: <text>"`.
- Persist `index.faiss` + `chunks.parquet` from `ingest.py`. The server loads them once at boot.

Runtime search is pure math and **blind to `is_selected`**: embed the query, inner-product
against every chunk, return the top-k highest. It has no idea which chunk is "supposed" to be
right — it just returns what's closest. That blindness is the point; grading happens separately.

### The `search()` contract (matches their benchmark)

```python
# app/retriever.py
class SearchResponse(BaseModel):
    chunks: list[dict]     # [{text, source_passage_id, score, ...}]
    total_ms: float
    embed_ms: float
    search_ms: float

def warmup() -> None:
    """Load model + FAISS, run one throwaway inference so the first real query isn't cold."""

def search(query: str, top_k: int = 5) -> SearchResponse:
    """Embed → FAISS top-k. Times embed and search separately."""
```

---

## 5. Harness (requirement 5)

Not a raw prompt-in/text-out call. `harness.py` orchestrates the stages with structured I/O and
recovery:

- **Typed I/O** — Pydantic models for the request (audio/text, lang, top_k) and the response
  (answer, retrieved chunks, scores, timings, `grounded: bool`).
- **Retries** — `tenacity` around Sarvam STT and the LLM call (transient network / rate-limit).
- **Error recovery** — STT fails → ask user to retry; LLM fails → fall back to returning the top
  chunk verbatim rather than crashing.
- **Stage timing** — every stage timed and returned, so the demo can show where time goes.

---

## 6. Guardrails (requirement 6)

Show the system knows **when not to answer**, not just how.

1. **Input safety** — reject unsafe/inappropriate transcribed queries before retrieval.
2. **Off-topic / not-in-corpus** — if the top-k **max cosine score < τ**, don't answer; return
   _"I don't have information on that in my knowledge base."_
3. **Grounded generation** — system prompt: _answer ONLY from the provided context; if it's not
   there, say you don't know._ No outside knowledge.
4. **Hallucination check (optional, cheap)** — verify the answer's claims are supported by the
   retrieved chunks (lexical overlap or a light LLM-as-judge pass; generation is off the clock).

**Calibrate τ with `is_selected`** — plot the score distribution of known-relevant vs
known-irrelevant passages, pick the threshold that separates them. This ties the guardrail
directly to the eval data — a strong point for the writeup.

---

## 7. Evaluation — the differentiator

`is_selected` is the answer key. For each test query, the flagged passage (`is_selected == 1`) is
the **gold** passage. Run the query through the blind retriever and check whether any retrieved
chunk traces back (via `source_passage_id`) to that gold passage.

- **Recall@k** — fraction of queries where gold is anywhere in the top-k. Headline number.
- **MRR** — rewards ranking gold _high_ (rank 1 → 1.0, rank 3 → 0.33, absent → 0).

Run every chunking strategy from §3b through the same eval and ship a table:

| Strategy             | Recall@5 | MRR | Notes           |
| -------------------- | -------- | --- | --------------- |
| Passage-as-chunk     | …        | …   | control         |
| Semantic             | …        | …   |                 |
| Small-to-big         | …        | …   | expected winner |
| Fixed-size + overlap | …        | …   | naive baseline  |

"We measured; strategy X won at recall@5" beats asserting it. Measure over 200+ queries, not one run.

---

## 8. Latency plan

- The reference benchmark measures **`embed_ms + search_ms` only** (`resp.total_ms`) — STT and
  LLM generation are **not** in the budget. State this interpretation explicitly in the submission.
- The docstring budget is **50ms** (`LATENCY_BUDGET_MS` in `config.py` is authoritative — the
  task text's 200ms is the looser number; hit 50ms).
- To hit it: `warmup()` before timing, small CPU embedder, normalized vectors, `IndexFlatIP`
  on a small index, in-process FAISS (no network).
- **Report P50 / P70 / P100** (P100 = max/worst case) over 200+ queries. Their script emits
  p50/p95/p99 — override the percentile calls to the required P50/P70/P100 for the submission.

---

## 9. Build order (≈1 day)

1. `ingest.py` — load Hindi subset, dedup, passage-as-chunk baseline, embed, build FAISS + parquet.
2. `retriever.py` — `warmup()` + `search()` to spec; run **their** `benchmark.py` → confirm < 50ms.
3. `eval.py` — recall@k + MRR on the baseline. Get a real number on the board.
4. Add semantic + small-to-big chunking → rerun eval → fill the comparison table → pick winner.
5. `stt.py` (Sarvam) + `generator.py` (grounded Groq/Gemini) + `guardrails.py` (calibrate τ).
6. `harness.py` wiring + `server.py` (Gradio mic UI) → deploy to HF Spaces for the live link.
7. Record demo video + team video. Post both to Instagram + X, every member, `#RAGInGoa`.

---

## 10. Submission checklist

- [ ] GitHub repo (this structure)
- [ ] Live link (HF Spaces)
- [ ] Latency: P50 / P70 / P100 over 200+ queries, embed+search under budget
- [ ] Chunking comparison table (recall@k + MRR) in repo
- [ ] Harness: typed I/O, retries, error recovery
- [ ] Guardrails: off-topic + unsafe + grounding threshold, demoed refusing a bad query
- [ ] Video 1 — team/process, 90s
- [ ] Video 2 — end-to-end demo
- [ ] Both videos on Instagram **and** X, **by every member**, ≥1 public IG, tag **#RAGInGoa**
- [ ] Submit the form — **no resubmissions**, submit only when final
