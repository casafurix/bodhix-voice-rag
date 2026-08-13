# 08 — Latency: budget, measurement, reporting

> The brief: *"The full process — chunking + vector DB retrieval + everything through output —
> should complete in under 200 ms."* and *"Submit P50 / P70 / P100 latency numbers for your
> pipeline, measured across a reasonable number of test queries — not a single best-case run."*

## The three numbers, defined precisely

Ambiguity here is the whole game. See [00-task-brief.md](00-task-brief.md) §A for why the literal
reading is unachievable. Definitions we commit to, and which appear on every chart:

| Metric | From | To | Target |
|---|---|---|---|
| **`t_core`** | Normalised transcript available | Guarded answer ready to emit | **P100 < 200 ms** |
| **`t_stt`** | End of speech (VAD-detected) | Final transcript received | Measured, per provider |
| **`t_e2e`** | End of speech | Answer rendered in browser | Measured, reported honestly |

`t_core` is the pipeline the brief describes: chunk-time retrieval work, vector DB search, fusion,
reranking, context assembly, answer construction, output guardrails. It is the part we build and
control, and it is the number we are held to.

`t_stt` and `t_e2e` are published alongside it at equal prominence. We are not going to print one
flattering number and hope nobody asks about the microphone.

---

## Budget allocation for `t_core`

| Stage | Nominal | P95 est. | Optional | Notes |
|---|---|---|---|---|
| `normalise` | 1 ms | 2 ms | | NFC, script/lang detect |
| `guard_in` | 2 ms | 4 ms | | 8 ordered checks |
| `embed` | 5 ms | 9 ms | | e5-small ONNX int8, 384-d |
| `retrieve_dense` ∥ `retrieve_sparse` | 12 ms | 22 ms | | Concurrent → `max`, not sum |
| `fuse` | 1 ms | 2 ms | | RRF over 100 candidates |
| `coverage_gate` | 1 ms | 1 ms | | Score-distribution stats |
| `rerank` | 45 ms | 70 ms | **yes** | 50 → 5, cross-encoder int8 |
| `assemble` | 4 ms | 8 ms | | Parent expand, dedup, token budget |
| `answer_fast` | 20 ms | 32 ms | | Extractive span selection |
| `guard_out` | 25 ms | 38 ms | | NLI + numeric + citations + language |
| **Total** | **116 ms** | **188 ms** | | |

Sum-of-P95 is a deliberately pessimistic model — stage tails do not all coincide — so the real P95
should land comfortably below 188 ms. But **P100 is what the brief asks for**, and a P100 is a tail
event by definition. Sum-of-P95 at 188 ms leaves only 12 ms of margin, which is not enough to
promise a P100.

That is precisely why the degradation ladder exists. `rerank` is 45–70 ms, is the only stage large
enough to matter, and is marked optional. When the budget check before `rerank` shows the remaining
stages will not fit, it is skipped and the request lands around 120 ms instead of 190 ms. **P100 is
enforced structurally by shedding work, not hoped for.** See [06-harness.md](06-harness.md).

The honest framing of our claim: *`t_core` P100 < 200 ms, with a measured degradation rate of X %,*
where X is published. That is a real engineering guarantee. "P100 < 200 ms" with no degradation
figure would be an unfalsifiable one.

---

## Where the milliseconds were won

Worth stating because these are the decisions, not micro-optimisations:

| Decision | Saved | Doc |
|---|---|---|
| Colocated vector DB instead of hosted | 20–80 ms | [04](04-retrieval.md) |
| Local ONNX embeddings instead of a hosted API | 40–120 ms | [04](04-retrieval.md) |
| int8 quantisation on all three local models | ~2–3× per model | [04](04-retrieval.md) |
| Dense and sparse arms concurrent, not sequential | ~8 ms | [04](04-retrieval.md) |
| Extractive fast path instead of waiting on an LLM | 300–700 ms | [01](01-architecture.md) |
| Mumbai region for API, browser, and STT | 50–200 ms of RTT | [10](10-deployment.md) |
| Adaptive rerank gating | 45 ms on gated queries | [04](04-retrieval.md) |
| Streaming STT + client VAD | 200–800 ms of `t_stt` | [05](05-speech-to-text.md) |
| Speculative retrieval on stable partials | up to all of `t_core` | [05](05-speech-to-text.md) |
| Warm process, mmap'd index, no cold start | seconds | [10](10-deployment.md) |

The two biggest wins — region choice and eliminating network hops from the retrieval loop — are
architectural and were decided before any code. No amount of profiling recovers a bad choice on
either.

---

## Measurement methodology

The methodology is the deliverable. Anyone can print a fast number; a reviewer's real question is
"what exactly did you measure, and can I re-run it?"

### Query set

- **N = 300 minimum**, target 500
- **Stratified by language:** 5 languages, ~60–100 each
- **Stratified by `query_type`:** DESCRIPTION / NUMERIC / ENTITY / PERSON / LOCATION, proportional
  to the corpus distribution
- **Stratified by length:** short (< 6 words), medium, long (> 15 words)
- **Sampled from held-out `query` fields** whose passages are in the index — so these are genuinely
  answerable and we are timing the real path, not the refusal path
- **Plus a refusal cohort** of 50 off-topic queries, timed separately. Refusals short-circuit and are
  therefore *faster*; mixing them into the main set would deflate the numbers. This is a specific way
  benchmarks get accidentally gamed and we exclude it explicitly.
- Query set committed as `bench/queries.jsonl` — fixed, versioned, re-runnable

### Conditions

Every condition reported separately. No averaging across conditions.

| Dimension | Values |
|---|---|
| Cache | **cold (default, headline)** · warm (labelled separately) |
| Concurrency | 1 · 4 · 16 |
| Corpus tier | T0 50 K · T1 500 K · T2 2 M · T3 8 M |
| Rerank | forced on · adaptive · forced off |
| Process state | warm (after 50 warmup queries) · cold start (reported once) |

### Protocol

1. Start container, load index, warm all three ONNX models
2. Run 50 warmup queries, discard — measuring JIT and page-cache effects is not the point
3. Run the full query set, record **every stage timing for every query**
4. Repeat 3× on different runs; report the median run and the spread across runs
5. Emit per-query CSV → `bench/results/`
6. Generate percentile tables and charts from the CSV, never from memory

`perf_counter_ns()` at stage boundaries. Timings recorded in the response object, so the UI HUD and
the benchmark read the identical source and cannot disagree.

### What we explicitly do not do

- Report a warm-cache number as the headline
- Average `t_core` and `t_stt` into one figure
- Report only the best of several runs
- Exclude degraded requests from the percentiles
- Time with the client on the same machine as the server for `t_e2e` (that hides real RTT)
- Report P50 alone, which is where the brief's insistence on P100 comes from

---

## Reporting format

Committed to `bench/results/` and reproduced in the README.

### Headline table

| Metric | P50 | P70 | P90 | P95 | P99 | **P100** | mean | n |
|---|---|---|---|---|---|---|---|---|
| `t_core` (cold, c=1, T2) | | | | | | | | |
| `t_core` (cold, c=4, T2) | | | | | | | | |
| `t_core` (cold, c=16, T2) | | | | | | | | |
| `t_stt` Sarvam streaming | | | | | | | | |
| `t_stt` Sarvam non-streaming | | | | | | | | |
| `t_stt` ElevenLabs streaming | | | | | | | | |
| `t_e2e` (streaming, spec. retrieval on) | | | | | | | | |
| `t_e2e` (streaming, spec. off) | | | | | | | | |
| `t_e2e` (non-streaming baseline) | | | | | | | | |

P50 / P70 / P100 are the brief's ask; P90 / P95 / P99 are included because a jump from P99 to P100
reveals whether the tail is a systemic problem or one unlucky GC pause, and that distinction is the
actual engineering content.

### Supporting artefacts

1. **Stage breakdown** — stacked bar of mean cost per stage, plus a P95 variant showing which stage
   owns the tail
2. **Percentile curve** — `t_core` from P0 to P100, with the 200 ms line drawn
3. **Per-language table** — Tamil should not be quietly 3× slower than English; if it is, that is a
   tokenisation bug worth finding
4. **Scaling curve** — `t_core` P50/P95/P100 across T0→T3. Turns "we hit 200 ms" into "here is how
   it scales and where it breaks"
5. **Degradation report** — % of requests hitting each ladder rung, and the measured nDCG delta on
   degraded requests. The credibility exhibit.
6. **Concurrency curve** — P95 vs concurrency, to show whether we are latency-bound or
   throughput-bound
7. **Raw CSVs** — every query, every stage, every run. Auditable.

---

## Risk register

| Risk | Impact | Mitigation |
|---|---|---|
| Rerank P95 exceeds estimate on 4 vCPU | Blows the budget | Gate harder; smaller reranker; ColBERT late interaction; GPU as last resort |
| NLI groundedness slower than 25 ms | Blows the budget | Smaller NLI model; run it async and retract a bad answer; degrade to lexical overlap |
| ONNX int8 costs too much recall | Quality loss | Measure the recall delta; fall back to fp32 with quantised HNSW instead |
| T3 index exceeds RAM | Swapping, tail explosion | Binary quantisation, or cap at T2 and publish the ceiling |
| Fly.io noisy neighbour | Erratic P100 | Dedicated CPU tier; report the variance across runs |
| Streaming STT partials thrash | Speculation wasted, CPU burn | Stability window; cap speculative attempts; report hit rate |
| GC / allocator pauses at P100 | Single-query tail spikes | Preallocate buffers, avoid per-request large allocations, tune thread pools |
| Concurrency 16 queues on shared ONNX sessions | P95 collapse under load | Per-worker session pool, bounded queue, backpressure |

The two most likely to actually bite are the first two, and both have the same shape: a local model
is slower than estimated. Both are handled by the same mechanism — mark the stage optional and let
the degradation ladder shed it — which is the argument for having built the ladder before knowing
the numbers.
