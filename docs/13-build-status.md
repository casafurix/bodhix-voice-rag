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

## External eval: rag-local-eval-loop (organizer-provided)

Wired per `wiring-in-the-eval-loop.pdf` — `app/embedder.py` and
`app/generator.py` at the repo root are thin adapters onto the REAL
production code (`api/retrieval/embed.py`'s MiniLM model, the calibrated
`api/guardrails/coverage_gate.py`, `api/answer/extractive.py`'s span
picker), so this suite grades exactly what a plain-text `POST /ask`
request would do — not a reimplementation. Extractive only (not
abstractive): the suite calls `generate_answer()` once per sampled example
across `--workers` threads, and the NVIDIA abstractive path takes 8-25s
per call by its own docstring's measurement — impractical to run at any
real sample size. Full report: `bench/results/eval_loop_report.json`
(`--num-answerable 25 --num-unanswerable 25`, seed 42, Anthropic judge).

| Check | Result | Read |
|---|---|---|
| Retrieval | Recall@1 0.48, Recall@5 0.88, MRR 0.61 | Real MiniLM quality against the suite's own throwaway FAISS index — informative on the embedding model alone, not our full hybrid pipeline (see the suite's own scope note) |
| Faithfulness | 98% faithful, 2% hallucination, 96.9% self-report precision | When we answer, it's honest — extractive-by-construction pays off here |
| Correctness | 24% match MSMARCO's exact reference answer | Expected gap, not a bug: extractive picks a real sentence from a real candidate, not a reworded match to MSMARCO's curated phrasing. Some misses are retrieval misses (`top_k=5` didn't surface the right candidate), not extraction failures |
| Reliability | False refusal 16%, **false confidence 44%** | Real finding: the coverage gate's TAU thresholds (calibrated on our own OOD-vs-in-domain query set, `bench/run_guardrails_calibration.py`) don't cleanly separate MSMARCO's specific "near-miss, not-selected" candidates — a harder negative than the "fully off-topic question" case `/ask` mostly faces in practice |
| Latency | embed p95 40.9ms (PASS vs 50ms), generation p95 475ms (PASS vs 1500ms) | Consistent with `bench/run_retrieval_latency.py`'s numbers |

**Tried and reverted:** raised `TAU_ABSOLUTE`/`TAU_MEAN` (0.70/0.62 →
0.78/0.70) to test the false-confidence gap. Result: false confidence
0.44→0.16, but false refusal 0.16→0.48 — a straight tradeoff, not an
improvement (combined error rate got slightly worse, 0.60→0.64). Kept the
original values, which are calibrated against our own corpus's actual
traffic shape; recalibrating against this specific harness's harder
negatives risks overfitting to one eval tool at the expense of the
real corpus. Flagged here as a known, measured gap rather than
silently tuned away.

---

## Graded benchmark numbers (bench/results/)

Real numbers, reproducible from committed scripts — see
`bench/results/report.md`, `latency_full.csv`, `retrieval_ablation.csv`.

**Latency** (`bench/run_latency.py`, 320 real `POST /ask` requests, 55
distinct real MSMARCO-XI queries cycled to reach the sample size + 20
out-of-domain):

| metric | p50 | p70 | p95 | p99 | p100 |
|---|---|---|---|---|---|
| t_core | 42.1 | 44.6 | 63.5 | 152.3 | 5975.8ms |
| t_e2e | 44.4 | 46.9 | 66.3 | 155.3 | 5987.9ms |

Degradation rate: 1/320 (0.3%) over the 200ms budget — that one request is
the run's first (cold model load + Qdrant mmap warmup right after a
restart, the same cold-start cost documented for every fresh process in
this repo), not a steady-state failure. Every other request in the run
landed under budget.

Over-refusal rate: 3.3% (10/305 in-domain, non-cold-start requests) —
found and fixed one real cause along the way: `langdetect` misclassified
short, unambiguously-English MSMARCO queries ("defination arbitrary",
"does delta fly to bangalore") as unsupported languages, over-refusing 19
of them with `UNSUPPORTED_LANGUAGE`. Fixed in `api/normalise.py`: text
that's pure ASCII (none of hi/bn/ta/mr use Latin script, so this can't
misfire on them) biases toward `en` over a shaky non-English guess. This
dropped in-domain over-refusals from 28→10 in the same run.

**Chunking ablation** (`bench/run_retrieval.py`, 55 relevance-labelled
queries against real `is_selected` ground truth):

| arm | recall@10 | nDCG@10 | MRR |
|---|---|---|---|
| **s5_parent_child (champion)** | 0.600 | **0.231** | 0.272 |
| s3_sentence_window | 0.509 | 0.216 | 0.259 |
| ENSEMBLE_rrf (production) | 0.909 | 0.205 | 0.321 |
| sparse_bm25 | 0.909 | 0.189 | 0.339 |
| s9_doc2query | 0.945 | 0.186 | 0.321 |
| s1_fixed | 0.491 | 0.164 | 0.260 |
| s2_passage_native | 0.491 | 0.164 | 0.260 |
| s10_crosslingual_twin | 0.382 | 0.158 | 0.177 |

Real, non-obvious finding: s5_parent_child wins on nDCG@10 despite far
lower recall@10 than the ensemble/sparse/doc2query arms (0.60 vs 0.91+) —
it finds fewer relevant docs overall, but ranks the ones it does find much
higher. Recall and ranking quality aren't the same question, and this is
exactly the "champion by measured data, not the strategy you'd have
guessed" result docs/12-submission.md's video script calls for.

---

## Deployment: real memory constraint + a real Marathi bug, found deploying to Render

**Render free tier OOM (exit 137 / SIGKILL).** First deploy attempt crashed
before ever opening a port. Root cause: `dense_nvidia` (2048-dim, one per
chunk) was the only unquantized vector field — ~90MB raw float32 across the
full corpus, inside a hard 512MB container memory cap. This was never
actually caught locally because `docker run` without an explicit
`--memory 512m` flag doesn't reproduce the constraint — Docker Desktop's
default VM has multiple GB available. Fixed: `dense_nvidia` now uses the
same int8 scalar quantization `dense` already had
(`api/retrieval/qdrant_store.py`), applied automatically on the next
`load_cached_embeddings` run against the same committed cache — no
re-embedding needed. Also trimmed the cache ~45% (10,994 → 6,045 chunks,
sampled proportionally per language) as a safety margin on top of the
quantization fix, since the exact memory ceiling couldn't be measured
without a real capped container to test against.

**Marathi bug, found while trimming the cache.** Checking the language
distribution before trimming turned up zero `mr` chunks in the entire
corpus — despite Marathi being accepted everywhere (settings, guard_in,
Sarvam). Root cause: `ingest/filters.py`'s script-purity range for Marathi
was `0x0A80-0x0AFF`, which is the **Gujarati** Unicode block, not
Devanagari — Marathi actually uses the same Devanagari block as Hindi
(`0x0900-0x097F`). Every Marathi passage failed script-purity at 0% and
was silently dropped during ingest; nothing errored because a fully-empty
language just looks like "no positive labels in this sample" rather than
a crash. Fixed in code; **not yet reflected in the built index** — that
needs a real re-embed (~15 min, NVIDIA API calls), deferred past the
Render OOM fix given the deadline. Until that rebuild runs, a genuine
Marathi query will correctly `OUT_OF_SCOPE`-refuse (no indexed content to
find) rather than silently misbehave — the guardrail masks the gap
without hiding it from an honest read of this doc.

---

## Deployment, round 2: the OOM was actually the embedding model itself

The quantization + trim fix above solved the *startup* OOM — the deploy
went fully live (`Uvicorn running`, health checks passing). But the
**first real `/ask` request** then crashed the whole container (502, then
`/healthz` itself started failing too — the process died, not just that
route). Root cause, confirmed by inspecting the actual cached model file:
`paraphrase-multilingual-MiniLM-L12-v2`'s ONNX file is **224MB on disk**
(already int8-quantized) and is lazy-loaded on first use — untouched
during `load_cached_embeddings`, which is exactly why that phase looked
fine. onnxruntime's default session options (a memory arena that
pre-allocates larger reusable blocks, per-thread scratch buffers) pushed
loading that model over the 512MB cap mid-request.

Fixed in `api/retrieval/embed.py`: `TextEmbedding(threads=1,
enable_cpu_mem_arena=False)` — both onnxruntime's own documented levers
for memory-constrained deployments. Trimmed the cache further as
additional safety margin (6,045 → 3,927 chunks, still proportional across
all 4 real languages) since there's no way to verify the exact ceiling
without a real memory-capped container to test against locally.

Honest note on Render's pricing: **Starter is also 512MB RAM** — it only
removes the spin-down/SSH restrictions, not the memory cap. Only
**Standard ($25/month) jumps to 2GB.** If this round of fixes still isn't
enough, the free-tier path runs out of levers beyond trimming the corpus
further or swapping to a smaller embedding model (the latter needs a full
re-embed + gate recalibration — expensive this close to the deadline).

---

## Next up, in priority order

1. Re-run `ingest/build_index.py` now that the Marathi script-purity bug is
   fixed, to actually get real `mr` content into the corpus (and
   re-export + re-trim the cache)
2. Verify the Render redeploy succeeds under the real 512MB cap; if it
   still OOMs, the next lever is trimming further (the quantization + 45%
   trim is a best-effort given no way to test against a real memory-capped
   container locally)
3. Record both videos; publish the 9 promotion posts
