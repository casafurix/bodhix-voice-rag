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
| Answer path | Dual: extractive (fast) + abstractive (rich, LLM) | **Extractive only** | Satisfies "returns an answer"; grounded by construction; no LLM cost or latency in the loop |
| STT | Sarvam primary + ElevenLabs failover | **Sarvam only**, ElevenLabs interface stubbed, not wired | Brief says "pick one"; failover was demo-resilience insurance, not a requirement |
| Speculative retrieval, `/diagnose`, semantic cache | Planned stretch features | **Not built** | Per the plan's own descoping order (`docs/11-roadmap.md`) |

---

## What's actually verified working right now

Not "written" — **run, with real data, and the output inspected by hand**:

- `uv sync` installs cleanly; every module in `api/`, `ingest/`, `bench/` imports with no errors
- `ingest/build_index.py` successfully streams real MSMARCO-XI data (`validation/hinval.parquet` + `validation/benval.parquet`), explodes + dedupes it, runs all 6 chunkers, embeds, and indexes into Qdrant (embedded mode) + BM25 — confirmed with a 30-rows-per-language run: **899 deduped passage docs → 8,197 chunks in ~88s**
- `POST /ask` returns a real `ANSWERED` response with a correct, grounded citation for an English query ("what is a corporation") against the real ingested index
- The same question **in Hindi** correctly retrieves the same underlying content cross-lingually (via the `s9_doc2query` strategy matching the dataset's own free question field) and returns a grounded **Hindi** answer — language consistency holds
- `guard_in` correctly refuses on `UNSUPPORTED_LANGUAGE`, `INJECTION_DETECTED` (tested with a Bengali injection phrase)
- `guard_out` correctly passes numeric grounding, citation integrity, extractive span verification, and lexical-overlap groundedness on a real answer
- The harness correctly propagates a full per-stage timing trace (`timings_ms`) on every response, answered or refused

## Known gaps — found by testing, not yet fixed

Listed here rather than silently patched, per the project's own stated
principle of measuring and publishing honestly rather than picking the
flattering interpretation.

| Gap | Evidence | Status |
|---|---|---|
| Retrieval is far over its sub-budget | `bench/run_retrieval_latency.py`: embed ~4.7ms avg (on target), but embed+search **~285ms avg / ~480ms P95** vs. a 50ms sub-budget | The 6 strategy searches run **sequentially** in `harness/pipeline.py`, not concurrently as `docs/04-retrieval.md` specifies ("dense and sparse arms run concurrently... cost is max(), not sum()"). Fix identified, not yet applied. |
| Coverage gate isn't reliably discriminative yet | A genuinely off-topic query ("what's the weather in Panaji") was incorrectly `ANSWERED` instead of refused | `TAU_*` thresholds in `coverage_gate.py` are a rescaled *placeholder* (fixed once already — see the bug-fix commit — for being on the wrong numeric scale entirely), not a real calibration. Needs `bench/run_guardrails.py` with real in-domain/out-of-domain query sets, which doesn't exist yet. At T0's tiny scale (899 docs) the RRF score distribution isn't very discriminative regardless. |
| Corpus is tiny | 899 docs / 8,197 chunks, from 30 rows/language | Nowhere near T0 (50K chunks) target yet. `ROWS_PER_LANGUAGE` is a one-line change in `ingest/build_index.py`; not scaled up yet pending the concurrency fix (no point indexing more data into a retrieval path that's already 5-10x over budget). |
| No `bench/` scripts beyond retrieval-latency | Only `run_retrieval_latency.py` exists | `queries.jsonl`, `run_latency.py` (full `/ask` P50/P70/P100), `run_retrieval.py` (Recall@k/nDCG/MRR ablation), `report.py` — none built yet |
| No frontend | — | `web/` doesn't exist yet; backend-first per the agreed priority |

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

Full detail in the `fix(retrieval): three real bugs found by running the
pipeline on real data` commit.

---

## Next up, in priority order

1. Fix `harness/pipeline.py`'s retrieve stage to search all 6 strategies
   concurrently (asyncio, not a sequential `for` loop) — the single
   biggest lever toward the 200ms `t_core` target
2. Scale ingest past 30 rows/language once retrieval is fast enough that
   more data doesn't just mean a slower-and-still-failing benchmark
3. Build `bench/run_guardrails.py` with real in-domain/out-of-domain
   query sets and actually calibrate the coverage-gate thresholds
4. Build the remaining `bench/` scripts (`run_latency.py`,
   `run_retrieval.py`, `report.py`) — this is where the graded
   P50/P70/P100 numbers and the chunking ablation table come from
5. `web/` frontend, once the backend numbers are real
