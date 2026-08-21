# 13 — Build status (planned vs. actual)

> **This is the living source of truth.** Docs 00-12 describe the original
> plan, designed for a 3-person team over 10 days. That staffing
> assumption stopped holding early — one person (`casafurix`) is doing
> the bulk of the work, including initiation, on a compressed timeline.
> Rather than silently under-deliver against the original plan or
> rewrite 13 documents' worth of design rationale every time scope
> changes, this doc tracks **what is actually true right now** and is
> updated every time real status changes. Docs 00-12 keep their original
> design thinking (still valid context for *why* a choice was made) with
> a short scope-note pointing here wherever the shipped scope diverges.

Last updated: see the most recent commit touching this file.

---

## Scope cuts vs. the original plan, and why

None of these change what the brief itself requires (re-verified against
the brief's literal text, not just our own `docs/00-task-brief.md`
reading, before cutting anything) — every cut is *ambition the team
added on top of the brief*, not a graded requirement.

| Area | Planned (docs/) | Actual (shipped) | Why |
|---|---|---|---|
| Team | 3 people, parallel tracks (`docs/11-roadmap.md`) | 1 person doing all tracks | Staffing reality changed after the plan was written |
| Languages | 5 (en, hi, ta, bn, mr) | 3 (**en, hi, bn**) | Solo build: fewer languages = less QA/verification surface area for one person. Tamil→Bengali swap: the builder reads Bengali, not Tamil, and can verify correctness personally. Not a brief requirement either way. |
| Chunking strategies | 12 (S1-S12) | 6 (**S1, S2, S3, S5, S9, S10**) | The plan's own pre-declared "minimum viable scope" (`docs/03-chunking.md`) — still satisfies "vast, not a single naive split" with a real ablation once `bench/` lands |
| Embedding model | `intfloat/multilingual-e5-small` (384-dim, ONNX int8) | `paraphrase-multilingual-MiniLM-L12-v2` (384-dim) | The installed `fastembed` (0.8.0) doesn't bundle e5-small, only the 1024-dim `-large` variant. MiniLM is the doc's own documented fallback (`docs/04-retrieval.md`). Revisit via custom ONNX export if the ablation shows a real quality gap. |
| Reranker | `jina-reranker-v2-base-multilingual`, gated | **Not built** | Real extra latency + a third ONNX model to tune; extractive answer uses lexical-overlap span scoring instead. The harness's "optional stage, skippable" architecture already accounts for this — rerank is stage 6, cut cleanly. |
| Groundedness check | NLI entailment cross-encoder | **Lexical-overlap** | The doc's own documented fallback (`docs/07-guardrails.md`) when no NLI model is fast/available enough |
| Answer path | Dual: extractive (fast) + abstractive (rich, LLM) | **Both wired** — extractive default/fallback, abstractive via `options.answer_mode` (NVIDIA NIM) with automatic fallback to extractive on LLM failure or guard_out veto | The original "extractive only" cut held until the backend had an LLM client; the dual-path design from docs/08-latency.md is now real |
| STT | Sarvam primary + ElevenLabs failover | **Sarvam only**, ElevenLabs interface stubbed, not wired | Brief says "pick one"; failover was demo-resilience insurance, not a requirement |
| Speculative retrieval, `/diagnose`, semantic cache | Planned stretch features | **Not built** | Per the plan's own descoping order (`docs/11-roadmap.md`) |

---

## What's actually verified working right now

Not "written" — **run, with real data, and the output inspected by hand**:

- `uv sync` installs cleanly; every module in `api/`, `ingest/`, `bench/` imports with no errors
- `ingest/load_cached_embeddings.py` populates Qdrant (embedded mode, both `dense` MiniLM and `dense_nvidia` vector fields) + BM25 from the committed parquet cache in ~60s, zero API calls
- `POST /ask`: text query in → grounded extractive answer out, end-to-end in **~60ms warm** (inside the 200ms `t_core` budget); Hindi questions retrieve cross-lingually and return grounded answers
- `POST /listen`: audio upload → Sarvam STT → same DAG, with transcript preserved on refusals so the UI shows what was actually said; language codes like `en-IN` normalise to `en`
- Retrieval latency (the old #1 gap): FIXED. Six per-strategy filtered searches (~100-150ms each in qdrant local mode, ~600ms stage total) replaced by ONE unfiltered grouped search bucketed client-side. Bench n=200: **p50 34ms / p95 45ms**, PASS vs the 50ms sub-budget (`bench/run_retrieval_latency.py`)
- Coverage gate: CALIBRATED with real data (`bench/run_guardrails_calibration.py`, 30 in-domain + 10 out-of-domain queries). Gates on raw cosine, not RRF ranks — RRF is rank-based and statistically identical for covered/uncovered queries, which is why off-topic questions used to get gibberish answers. Out-of-corpus questions now refuse cleanly with `OUT_OF_SCOPE`; thresholds are provider-aware (voice path always re-scores with the local model — the NVIDIA embedding space shows poor separation)
- Guardrails verified live: `OUT_OF_SCOPE`, `UNSUPPORTED_LANGUAGE`, `INJECTION_DETECTED`, `NO_SPEECH`, plus numeric/citation/groundedness checks on the output side
- Frontend: React NOVA app (`web-react/`) — hold-to-talk mic recording 16kHz WAV in-browser, conversation history, browser TTS, citations panel, human-readable refusal messages. Voice E2E through the UI verified by hand

## Known gaps — found by testing, not yet fixed

Listed here rather than silently patched, per the project's own stated
principle of measuring and publishing honestly rather than picking the
flattering interpretation.

| Gap | Evidence | Status |
|---|---|---|
| Corpus is tiny | 899 docs / 8,197 chunks, from 30 rows/language | Common-sense questions ("who is Lionel Messi") correctly REFUSE — the guardrail works — but topic coverage limits the demo. Scaling `ROWS_PER_LANGUAGE` up is queued; build is ~16 min at current size |
| No full-pipeline benchmark | Only `bench/run_retrieval_latency.py` + `bench/run_guardrails_calibration.py` exist | `queries.jsonl`, `run_latency.py` (full `/ask` P50/P70/P100 over ≥300 queries), `run_retrieval.py` (Recall@k/nDCG/MRR chunking ablation), `report.py` — none built yet. These produce the graded tables/CSVs |
| Not deployed | Local dev only | Dockerfile is Render-ready (verified locally with docker build/run); actual deploy pending |
| Videos + promotion | — | Both videos and the 9 mandatory posts outstanding |

---

## Bugs found and fixed while wiring real data

Worth recording because they're the kind of thing that only shows up once
real data flows through the system — every one of the earlier "verified
by hand" claims in commit messages meant "imports and returns a plausible
shape," not "tested against real data," until the ingest pipeline landed.

1. `Chunk` was missing a `query_type` field needed for payload filtering
2. `bm25s` round-trips a plain-string corpus through disk as
   `{"id","text"}` records on reload, not the original strings — broke
   sparse search on the first real query
3. `coverage_gate.py`'s thresholds were scaled for raw cosine similarity
   (0-1) but the gate is fed RRF fusion scores (~0.01-0.12) — refused a
   genuinely on-topic query before the fix
4. `harness/pipeline.py` and `bench/run_retrieval_latency.py` were using
   Qdrant's internal point UUID (not shared with BM25's identifier space)
   as the fusion key instead of the payload's `chunk_id` — would have
   silently prevented dense/sparse hits for the same chunk from ever
   fusing correctly
5. The coverage gate could never discriminate on RRF scores even at the
   right scale: fused scores are rank-based, so an uncovered query's
   distribution looks identical to a covered one. Re-derived from data:
   the gate now runs on raw cosine similarity with calibrated thresholds,
   and the voice path always gates in the local embedding space (the
   NVIDIA space's in/out-of-domain distributions overlap)
6. Per-strategy filtered dense searches cost 100-150ms each in qdrant
   embedded mode (payload filter forces a brute-force scan) — 6 of them
   put the retrieve stage at ~600ms. One unfiltered grouped search +
   client-side bucketing gives identical ranked lists at p50 34ms

---

## Next up, in priority order

1. Scale ingest past 30 rows/language (corpus expansion for demo coverage)
2. Build the remaining `bench/` scripts (`queries.jsonl`, `run_latency.py`,
   `run_retrieval.py`, `report.py`) — the graded P50/P70/P100 numbers and
   the chunking ablation table come from here
3. Deploy backend to Render (live link requirement)
4. Record both videos; publish the 9 promotion posts
