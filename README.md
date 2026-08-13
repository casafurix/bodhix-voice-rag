# BodhiX Voice RAG

## Hacker House Goa 2026 — Open Trial 2

This repository is the **BodhiX** team submission for the second Hacker House Goa 2026 screening
task, **Build a Voice-Enabled RAG Model**.

Task 1 (`Frame in Goa`) lives in a separate repository. This repo is Task 2 only.

> **Status: planning.** Nothing is implemented yet. This repo currently contains the brief,
> the architecture plan, and the decision records in [`docs/`](docs/). Code lands next.

---

## The task, in one line

A user speaks a question. The pipeline transcribes it, retrieves grounded context from the
AI4Bharat **MSMARCO-XI** corpus, and returns an answer — end to end, under a 200 ms budget,
with a real harness and real guardrails around it.

```
Voice input → Speech-to-text → Chunking / Retrieval (vector DB) → Answer generation
```

### Graded requirements

| # | Requirement | Where it is designed |
|---|---|---|
| 1 | Speech-to-text via **Sarvam or ElevenLabs** | [docs/05-speech-to-text.md](docs/05-speech-to-text.md) |
| 2 | Chunking strategy must be **vast**, not one naive fixed-size split | [docs/03-chunking.md](docs/03-chunking.md) |
| 3 | Full process **under 200 ms** | [docs/08-latency.md](docs/08-latency.md) |
| 4 | **P50 / P70 / P100** latency across many queries | [docs/08-latency.md](docs/08-latency.md) |
| 5 | Model runs inside a proper **harness** | [docs/06-harness.md](docs/06-harness.md) |
| 6 | **Guardrails** — knows when *not* to answer | [docs/07-guardrails.md](docs/07-guardrails.md) |

---

## Dataset

[`ai4bharat/MSMARCO-XI`](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI) — MS MARCO
translated into 14 Indic languages, with the original English retained alongside every row.

- **11,451,314 rows**, **55.6 GB** of Parquet, `train` (10.1 M) + `validation` (1.37 M)
- Each row carries a query, an answer, and **10 candidate passages** with `is_selected`
  relevance labels — free ground truth for retrieval evaluation
- Each row is **bilingual**: the Indic translation *and* the source English are both present

Two consequences shape the whole build, and they are the reason this is more interesting than a
generic "chat with your PDF" RAG:

1. **`is_selected` is a labelled retrieval benchmark.** We do not have to guess whether our
   chunking is good. We can measure Recall@k, nDCG@10 and MRR for every chunking strategy and
   pick the winner from data. See [docs/09-evaluation.md](docs/09-evaluation.md).
2. **The corpus is natively cross-lingual and parallel.** A Hindi question can legitimately be
   answered from an English passage and vice versa. That unlocks a chunking and indexing
   strategy that simply does not exist for monolingual corpora.
   See [docs/03-chunking.md](docs/03-chunking.md).

Full analysis, subsetting and corpus-construction plan: [docs/02-dataset.md](docs/02-dataset.md).

---

## Headline design decisions

These are the calls that define the build. Each has a full rationale in `docs/`.

### 1. Dual-path answering, so 200 ms is a real number and not a fudge

A hosted LLM cannot produce a complete answer in 200 ms. Time-to-first-token alone is
typically 100–250 ms. Rather than quietly redefine the target, the pipeline has two answer
paths that run against the same retrieved context:

- **Fast path (extractive).** Answer is composed from spans inside the retrieved passages by a
  quantised cross-encoder and a span scorer. No LLM in the loop. This is the path that must
  and will land inside the 200 ms budget, and it is grounded by construction — the answer text
  *is* corpus text, so it cannot hallucinate.
- **Rich path (abstractive).** A small, fast LLM (Groq / Cerebras class) streams a fluent
  answer over the identical context, arriving 300–700 ms later, and replaces the fast answer
  in the UI once its groundedness check passes.

The user hears an answer inside budget and gets a better one a beat later. We report both
numbers, clearly labelled, and never present a single figure that hides a stage.
See [docs/08-latency.md](docs/08-latency.md).

### 2. STT is measured, reported, and excluded from the retrieval budget — honestly

Sarvam and ElevenLabs are both network APIs. A round trip from a browser to a hosted STT
endpoint is physically incapable of fitting in 200 ms alongside retrieval and generation. We
therefore publish a **stage-by-stage latency table** with three headline figures:

- `t_core` — transcript in → answer out (target: **< 200 ms**, P100)
- `t_stt` — STT round trip, measured separately per provider
- `t_e2e` — end of speech → answer rendered (measured, reported, not hidden)

Streaming STT closes most of the gap: both providers offer realtime WebSocket transcription, so
`t_stt` is measured from *end of speech*, not from start of upload, and we begin retrieval
speculatively on partial transcripts. See [docs/05-speech-to-text.md](docs/05-speech-to-text.md)
and [docs/08-latency.md](docs/08-latency.md).

### 3. Chunking is an experiment, not an opinion

The brief explicitly penalises a single naive fixed-size split. We implement **twelve** chunking
and indexing strategies behind one interface, benchmark all of them on the `is_selected` labels,
and ship the measured champion with the ablation table published in the repo. The strategies
include semantic drift splitting, late chunking, small-to-big parent/child, proposition
decomposition, doc2query multi-vector indexing, script-aware token sizing, and cross-lingual
parallel twin chunks. See [docs/03-chunking.md](docs/03-chunking.md).

### 4. The harness is a deadline-aware state machine, not an agent loop

An agentic while-loop is the wrong tool at a 200 ms budget. The harness is an explicit typed
DAG where every request carries a millisecond budget, every stage reports its cost, and stages
**downgrade themselves** when the remaining budget runs thin — skip the reranker, skip the
abstractive path, return the extractive answer — instead of overrunning. Retries, circuit
breakers, STT provider failover, schema-validated structured output and a replayable trace log
sit on top. See [docs/06-harness.md](docs/06-harness.md).

### 5. Guardrails are a gate with named refusal codes, and they are visible

Off-topic detection is done from the **retrieval score distribution**, not a vibe check: if the
corpus does not cover the question, the top-k similarity scores collapse and we abstain with
`OUT_OF_SCOPE`. Groundedness is enforced by an NLI entailment check plus a hard rule that every
numeric token in the answer must appear in the retrieved context. Every refusal has a machine
code and a human explanation, and the demo UI shows the guardrail trace live.
See [docs/07-guardrails.md](docs/07-guardrails.md).

---

## Planned architecture

```
                                       ┌──────────────────────────────┐
  Browser (Next.js, Vercel)            │  FastAPI harness (Fly.io bom)│
  ┌───────────────────────┐            │                              │
  │ mic → 16 kHz PCM      │──WS audio─▶│  stage: stt (Sarvam primary, │
  │ waveform + VAD        │◀─partials──│         ElevenLabs failover) │
  │                       │            │  stage: normalise + guard-in │
  │ answer stream         │◀───SSE─────│  stage: embed (e5-small ONNX)│
  │ citations             │            │  stage: hybrid retrieve      │
  │ guardrail trace       │            │         (dense + BM25 + RRF) │
  │ latency HUD           │            │  stage: rerank (gated)       │
  └───────────────────────┘            │  stage: answer — fast path   │
                                       │  stage: guard-out (NLI)      │
                                       │  stage: answer — rich path   │
                                       └──────────────┬───────────────┘
                                                      │
                                       ┌──────────────▼───────────────┐
                                       │ Qdrant, colocated, no network│
                                       │ HNSW + int8 SQ + payload idx │
                                       │ BM25 sidecar index           │
                                       └──────────────────────────────┘
```

Component-by-component detail and the reasoning for each choice:
[docs/01-architecture.md](docs/01-architecture.md).

### Stack

| Layer | Choice | Why |
|---|---|---|
| Retrieval + harness | Python 3.12, FastAPI, Pydantic v2 | The embedding / ANN / reranker ecosystem is Python-native. Pydantic gives typed stage contracts for free. |
| Frontend | Next.js on Vercel | Same stack as Task 1, zero-config deploy, `MediaRecorder` + WebSocket is trivial. |
| Vector DB | Qdrant, colocated with the API | A network hop to a hosted vector DB costs 20–80 ms we do not have. Full options matrix in [docs/04-retrieval.md](docs/04-retrieval.md). |
| Embeddings | `multilingual-e5-small`, ONNX int8, 384-d | Covers all 14 dataset languages, ~4–8 ms query encode on CPU. Alternatives evaluated in [docs/04-retrieval.md](docs/04-retrieval.md). |
| STT | Sarvam `saaras:v3` primary, ElevenLabs `scribe_v2` failover | Sarvam is Indic-native and India-hosted, which matches both the corpus and the users. [docs/05-speech-to-text.md](docs/05-speech-to-text.md) |
| Region | Everything in `ap-south` / Mumbai | Region choice buys more latency than any amount of micro-optimisation. |

---

## Repository layout (planned)

```
bodhix-voice-rag/
├── README.md
├── docs/                     ← the plan (this is what exists today)
├── api/                      ← FastAPI harness
│   ├── harness/              ← stages, budget, retries, tracing
│   ├── retrieval/            ← chunkers, indexers, hybrid search, rerank
│   ├── guardrails/           ← input + output gates
│   └── stt/                  ← Sarvam + ElevenLabs adapters
├── ingest/                   ← MSMARCO-XI download, dedup, chunk, embed, index
├── bench/                    ← latency + retrieval-quality harnesses
│   └── results/              ← committed CSVs and charts (the graded numbers)
└── web/                      ← Next.js frontend
```

---

## Documentation index

| Doc | Contents |
|---|---|
| [00-task-brief.md](docs/00-task-brief.md) | The brief as issued, plus our reading of each ambiguous clause |
| [01-architecture.md](docs/01-architecture.md) | System design, request lifecycle, service topology |
| [02-dataset.md](docs/02-dataset.md) | MSMARCO-XI analysis, subsetting, dedup, corpus construction |
| [03-chunking.md](docs/03-chunking.md) | The twelve chunking strategies and the ablation plan |
| [04-retrieval.md](docs/04-retrieval.md) | Vector DB options matrix, embeddings, hybrid search, reranking |
| [05-speech-to-text.md](docs/05-speech-to-text.md) | Sarvam vs ElevenLabs, streaming, failover, decision criteria |
| [06-harness.md](docs/06-harness.md) | Stage DAG, budgets, retries, degradation ladder, tracing |
| [07-guardrails.md](docs/07-guardrails.md) | Input gates, groundedness, abstention, refusal taxonomy |
| [08-latency.md](docs/08-latency.md) | Budget allocation, measurement methodology, reporting format |
| [09-evaluation.md](docs/09-evaluation.md) | Retrieval and answer-quality metrics using `is_selected` |
| [10-deployment.md](docs/10-deployment.md) | Hosting, regions, index artefacts, cold starts, cost |
| [11-roadmap.md](docs/11-roadmap.md) | Day-by-day plan from 13 Aug to the 22 Aug deadline |
| [12-submission.md](docs/12-submission.md) | Form, videos, promotion requirements, final checklist |

---

## Submission

- **Form:** https://forms.gle/MNvCjcv23Hn2Eeu58
- **Deadline:** 22 August 2026, 11:59 PM. **No resubmissions.**
- **Required:** GitHub repo link, live working link, 2 videos
  - Video 1 — 90 s team/process video
  - Video 2 — end-to-end demo video
- **Promotion:** both videos posted by **every** team member to Instagram, X and LinkedIn, every
  post tagged **`#RAGInGoa`**, at least one Instagram account public.

Tracked in [docs/12-submission.md](docs/12-submission.md).

---

## Team

Built by **BodhiX**.
