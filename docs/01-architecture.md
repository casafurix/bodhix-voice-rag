# 01 — Architecture

> **MVP scope note:** the reranker (stage 6) and the abstractive "rich path" (stage 10) described
> below are **not built** — extractive-only, no reranker, satisfies the brief without either.
> Everything else on this page (deadline-propagated DAG, in-process retrieval, guardrail veto
> points) is running as designed. Live status: [docs/13-build-status.md](13-build-status.md).

## Design constraints, in priority order

1. **`t_core` P100 < 200 ms.** This is a hard constraint, not a goal. It vetoes any design that
   introduces a network hop between retrieval stages.
2. **Grounded or silent.** An ungrounded answer is worse than a refusal. The guardrail gate can
   veto any answer.
3. **Auditable.** Every graded number must be reproducible from a committed script.
4. **Demoable.** Internal state — stage timings, retrieval scores, guardrail verdicts — must be
   observable in the UI, or it cannot appear in the demo video.

Constraint 1 has one dominant consequence worth stating plainly: **every millisecond of network
round trip inside the retrieval loop is unaffordable.** A hosted vector DB at 25 ms/query, a
hosted embedding API at 60 ms, and a hosted reranker at 80 ms together consume 165 ms of a 200 ms
budget before any work is done. So the entire retrieval path — embeddings, ANN index, lexical
index, reranker — runs **in-process, on CPU, in one container**. This single decision is what
makes the target reachable.

---

## Service topology

Two deployables, both in Mumbai / `ap-south`.

```
┌─────────────────────────────────────────────────────────────────────┐
│  web/  ·  Next.js on Vercel                                         │
│  · mic capture → 16 kHz mono PCM via AudioWorklet                   │
│  · client-side VAD (silence detection) to mark end-of-speech         │
│  · WebSocket audio out, SSE answer stream in                        │
│  · renders: answer, citations, guardrail trace, latency HUD          │
└────────────────────────┬────────────────────────────────────────────┘
                         │  WSS (audio frames)  +  SSE (answer stream)
┌────────────────────────▼────────────────────────────────────────────┐
│  api/  ·  FastAPI on Fly.io (region bom), 1 machine, 4 vCPU / 8 GB  │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  Harness — typed stage DAG with deadline propagation          │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  in-process, no network:                                            │
│  · ONNX Runtime  — multilingual-e5-small (query encoder)            │
│  · ONNX Runtime  — jina-reranker-v2-m3 (gated cross-encoder)        │
│  · ONNX Runtime  — NLI entailment model (groundedness gate)         │
│  · bm25s         — sparse lexical index, memory-mapped              │
│  · Qdrant        — local/embedded mode, HNSW + int8, mmap'd volume   │
│                                                                     │
│  outbound network (outside t_core):                                 │
│  · Sarvam /speech-to-text  (primary STT)                            │
│  · ElevenLabs /v1/speech-to-text  (failover STT)                    │
│  · Groq or Cerebras  (rich abstractive path only)                   │
└─────────────────────────────────────────────────────────────────────┘
```

Why one machine and not a fleet: at this corpus size the index fits in RAM on a single 8 GB
machine, and a single machine has no inter-service latency. Horizontal scale is a problem we do
not have and should not pay for. Detail in [10-deployment.md](10-deployment.md).

---

## Request lifecycle

The end-to-end path, with the budget boundary marked explicitly.

```
  ─── t_e2e ────────────────────────────────────────────────────────────────────▶

  user speaks          end of speech
       │                    │
       ▼                    ▼
  ┌─────────────────────────┐
  │ 0. capture + stream     │   audio frames pushed to STT as they arrive,
  │    (concurrent w/ 1)    │   so 1 is already warm when speech ends
  └─────────────────────────┘
                            ┌──────────────────────────┐
                            │ 1. STT final commit      │   ← t_stt, measured
                            │    Sarvam saaras:v3      │     separately
                            └────────────┬─────────────┘
                                         │
       ═══════════ t_core starts here ═══╪══════════════════════════════════════
                                         ▼
                            ┌──────────────────────────┐
                            │ 2. normalise + guard-in  │   ~3 ms
                            │    lang id, PII, unsafe, │
                            │    injection, gibberish  │
                            └────────────┬─────────────┘
                                         ▼
                            ┌──────────────────────────┐
                            │ 3. encode query          │   ~5 ms
                            │    e5-small ONNX int8    │
                            └────────────┬─────────────┘
                                         ▼
                            ┌──────────────────────────┐
                            │ 4. hybrid retrieve       │   ~12 ms
                            │    dense ∥ BM25 → RRF    │   (run in parallel)
                            │    top-50 candidates     │
                            └────────────┬─────────────┘
                                         ▼
                            ┌──────────────────────────┐
                            │ 5. coverage gate         │   ~1 ms
                            │    score distribution →  │
                            │    OUT_OF_SCOPE?         │
                            └────────────┬─────────────┘
                                         ▼
                            ┌──────────────────────────┐
                            │ 6. rerank (gated)        │   ~45 ms, skippable
                            │    top-50 → top-5        │   if margin high or
                            │    cross-encoder int8    │   budget thin
                            └────────────┬─────────────┘
                                         ▼
                            ┌──────────────────────────┐
                            │ 7. context assembly      │   ~4 ms
                            │    parent-window expand, │
                            │    dedup, token budget   │
                            └────────────┬─────────────┘
                                         ▼
                            ┌──────────────────────────┐
                            │ 8. fast answer           │   ~20 ms
                            │    extractive span sel.  │
                            └────────────┬─────────────┘
                                         ▼
                            ┌──────────────────────────┐
                            │ 9. guard-out             │   ~25 ms
                            │    NLI entailment,       │
                            │    numeric check, lang   │
                            └────────────┬─────────────┘
       ═══════════ t_core ends here ═════╪══════════════════════════════════════
                                         ▼
                                   answer streamed to client  ◀── target < 200 ms
                                         │
                            ┌────────────▼─────────────┐
                            │ 10. rich answer (async)  │   +300–700 ms
                            │     LLM streams over the │   replaces fast answer
                            │     same context, then   │   in UI on success,
                            │     re-runs guard-out    │   silently dropped on
                            └──────────────────────────┘   failure
```

Nominal `t_core` sum: 3 + 5 + 12 + 1 + 45 + 4 + 20 + 25 = **115 ms**, leaving ~85 ms of headroom
for tail variance. Stage 6 is the expensive one and is the first thing the degradation ladder
sheds. Full budget analysis and what happens at the tail: [08-latency.md](08-latency.md).

---

## The two answer paths

Both paths consume the **identical** assembled context from stage 7. This matters: it means the
rich answer is never better-informed than the fast one, only better-worded, so an upgrade can
never contradict what the user already heard.

### Fast path — extractive, in budget

1. For each of the top-5 reranked chunks, score candidate answer spans. Span scoring uses the
   cross-encoder's token-level relevance signal already computed in stage 6, so the marginal cost
   is small.
2. Select the best span, expand to sentence boundaries for readability.
3. Emit with the chunk id, so the citation is exact.

Properties: ~20 ms, deterministic, cacheable, and **grounded by construction** — the answer is a
verbatim substring of a retrieved passage, so the failure mode is "wrong passage", never
"invented fact". The output guardrail still runs, because a correct span from an irrelevant
passage is still a wrong answer.

### Rich path — abstractive, out of budget, optional

A small fast model (Llama 3.1 8B / gpt-oss-20b class on Groq or Cerebras) generates a fluent
answer over the same context with a hard instruction to answer only from it. It streams to the
client, is re-checked by the output guardrail on completion, and is **discarded silently** if it
fails groundedness — the user keeps the extractive answer and never sees a failed generation.

This path is a product enhancement, not a requirement satisfier, and is behind a flag so the
graded benchmark can run with it off.

---

## Why not the obvious alternatives

Recording the roads not taken, with the actual reason.

| Alternative | Why not |
|---|---|
| Hosted vector DB (Pinecone, Weaviate Cloud) | 20–80 ms network round trip. Consumes 10–40 % of the total budget for zero functional gain at this corpus size. |
| Hosted embedding API (OpenAI, Cohere, Voyage) | 40–120 ms round trip to encode a 12-word query that a local int8 model encodes in 5 ms. |
| LangChain / LlamaIndex as the orchestrator | Framework overhead is measurable at this budget, the abstractions hide exactly the timings we need to expose, and the brief wants to see *our* harness. We use individual libraries, not a framework. |
| An agent loop (ReAct, tool-calling agent) | Each agent turn is an LLM call at 200 ms+. Structurally incompatible with the budget. The harness is a DAG for this reason — see [06-harness.md](06-harness.md). |
| GPU inference | A GPU would cut rerank from 45 ms to ~8 ms, but adds cost, cold-start risk and deployment complexity. We hit the target on CPU; GPU is a documented stretch option if the tail misbehaves. |
| Serverless (Lambda, Vercel Functions) for the API | Cold starts of seconds, and a multi-GB index that cannot live in a serverless filesystem. A long-lived machine with a warm mmap'd index is the only viable shape. |
| Full 55 GB corpus | Ingest cost dominates the timeline for no marks. We index a stratified subset and document the scaling curve instead — see [02-dataset.md](02-dataset.md). |

---

## Interfaces

### `POST /ask` — text in, for benchmarking and debugging

```jsonc
// request
{ "query": "मैनहट्टन प्रोजेक्ट की सफलता का तत्काल प्रभाव क्या था?",
  "budget_ms": 200,
  "lang_hint": "hi",
  "options": { "rerank": "auto", "rich_path": false } }

// response
{ "trace_id": "01J8…",
  "verdict": "ANSWERED",          // or REFUSED
  "refusal_code": null,           // e.g. "OUT_OF_SCOPE"
  "answer": { "text": "…", "mode": "extractive", "language": "hi" },
  "citations": [ { "chunk_id": "hi/1185869/p0", "score": 0.87,
                   "strategy": "parent_child", "span": [12, 96] } ],
  "guardrails": { "input": { … }, "output": { "entailment": 0.91,
                   "numeric_check": "pass", "language_match": true } },
  "timings_ms": { "guard_in": 2.8, "encode": 4.9, "retrieve": 11.6,
                  "coverage": 0.7, "rerank": 44.2, "assemble": 3.8,
                  "answer": 19.4, "guard_out": 24.1, "t_core": 111.5 },
  "degradations": [] }
```

The response is deliberately verbose. Everything the brief grades — timings, guardrail verdicts,
citations, degradations — is in the payload, which means the demo UI can render it without a
separate debug channel, and the benchmark harness can read it without instrumenting internals.

### `WS /listen` — the voice path

Client streams PCM frames; server relays to the STT provider, emits interim transcripts, and on
final commit runs the same harness as `/ask` and streams the answer over SSE. The voice endpoint
is a thin shell over the text endpoint, so there is exactly one code path to optimise and test.

### `GET /healthz`, `GET /metrics`

Readiness includes index-loaded and models-warm, so the load balancer never routes to a machine
that would serve a 3-second first request.
