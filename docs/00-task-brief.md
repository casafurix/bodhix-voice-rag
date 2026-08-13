# 00 — Task brief and interpretation

## The brief as issued

**HH Goa 2026 Shortlisting Task 2: Build a Voice-Enabled RAG Model**

> **What to build.** A voice-enabled Retrieval-Augmented Generation (RAG) system — a user speaks
> a question, your pipeline transcribes it, retrieves relevant context from a provided dataset,
> and returns an answer, end to end.
>
> Pipeline shape: Voice input → Speech-to-text → Chunking/Retrieval (vector DB) → Answer generation
>
> **Dataset.** `https://huggingface.co/datasets/ai4bharat/MSMARCO-XI`
>
> **Technical requirements**
> 1. **Speech-to-text.** Use either Sarvam or ElevenLabs for voice-to-text. Pick one.
> 2. **Chunking.** Chunking strategy should be vast — don't submit a single naive fixed-size
>    chunking approach. We want to see real thought put into how the dataset is split, indexed,
>    and retrieved (e.g. multiple chunking strategies, overlap handling, semantic vs. fixed-size
>    splitting, metadata-aware chunking, etc.).
> 3. **Latency target.** The full process — chunking + vector DB retrieval + everything through
>    output — should complete in under 200 ms.
> 4. **Latency analytics.** Submit P50 / P70 / P100 latency numbers for your pipeline, measured
>    across a reasonable number of test queries — not a single best-case run.
> 5. **Harness your model.** Your model/pipeline should be run inside a proper harness —
>    structured orchestration around the model (tool calls, retries, structured input/output
>    handling, error recovery) rather than a single raw prompt-in, text-out call.
> 6. **Guardrail your model.** Add guardrails around your model — handling for off-topic queries,
>    unsafe/inappropriate inputs, hallucination checks, or answers not grounded in the retrieved
>    context. Show that your system knows when not to answer, not just how to answer.
>
> **Submission.** Form, GitHub repo link, live working link, 2 videos (90 s team/process video;
> end-to-end demo video). No resubmissions.
>
> **Promotion (mandatory).** Both videos uploaded to Instagram, X and LinkedIn by every individual
> team member. At least one Instagram account public. Every post on every platform by every member
> must include `#RAGInGoa`.
>
> **Timeline.** Launch 13 August 2026. Deadline 22 August 2026, 11:59 PM.

---

## Where the brief is ambiguous, and how we are reading it

Every ambiguity below is resolved in the direction of *measure it and publish the number*, never
in the direction of *pick the flattering interpretation and stay quiet*. Reviewers of a latency
task have seen every trick; being visibly honest is the differentiator.

### A. "The full process … under 200 ms"

The literal reading — microphone silence to rendered answer, including a hosted STT round trip
and a hosted LLM completion, under 200 ms — is not physically achievable. Rough floors:

| Unavoidable cost | Realistic floor |
|---|---|
| Browser → India-hosted API TLS round trip | 20–60 ms |
| Hosted STT round trip after end of speech (streaming, final commit) | 150–500 ms |
| Hosted LLM time-to-first-token | 100–250 ms |
| Hosted LLM full short completion | 300–800 ms |

Any submission claiming a sub-200 ms figure that includes a hosted STT call and a full LLM
completion is either measuring something else or not telling the truth.

**Our reading.** The clause "chunking + vector DB retrieval + everything through output"
describes the *retrieval-and-answer* pipeline: the part we build, control and optimise. STT is a
third-party dependency the brief itself mandates. So:

- **`t_core`** — normalised transcript in → grounded answer out. **Hard target: P100 < 200 ms.**
  This is the number the brief is asking for, and it covers chunk-time retrieval, hybrid search,
  reranking, answer construction and output guardrails.
- **`t_stt`** — measured separately, per provider, from *end of speech* to final transcript.
- **`t_e2e`** — the honest wall-clock number a user experiences. Reported prominently, not buried.

All three are published with P50 / P70 / P90 / P95 / P99 / P100, plus per-stage breakdowns.
See [08-latency.md](08-latency.md).

### B. "Answer generation" — does it require an LLM?

The brief says "returns an answer". It does not say "an LLM writes the answer". Extractive QA —
selecting and stitching the answer span out of retrieved passages — is a legitimate,
long-established form of answer generation, and it is the only form that fits a 200 ms budget
while being **grounded by construction**: the answer text is corpus text, so hallucination is
structurally impossible.

We therefore ship both, on the same retrieved context:

- **Fast extractive path** — inside the 200 ms `t_core` budget. Always produced.
- **Rich abstractive path** — a small fast LLM streams a fluent answer, arrives ~300–700 ms
  later, replaces the extractive answer once it passes the groundedness gate.

This is not a dodge around the requirement; it is a stronger answer to it. A voice assistant that
speaks a correct grounded answer in 200 ms and then refines the phrasing is better product design
than one that stalls for a second. Detail in [01-architecture.md](01-architecture.md).

### C. "Chunking … in the full process"

Read literally, chunking happens at *ingest* time over 55 GB of Parquet and obviously cannot
occur inside a 200 ms request. The brief means the chunking *strategy* is part of the graded
pipeline, and that whatever chunk-assembly work happens per request counts against the budget.

We honour both readings:

- **Offline:** corpus chunking, embedding and indexing. Timed and reported separately as an
  ingest benchmark, not against the 200 ms budget.
- **Online:** per-request chunk work *does* exist in our design and *is* counted — parent-window
  expansion for small-to-big retrieval, sentence-window stitching, and context assembly under a
  token budget all happen inside `t_core`.

See [03-chunking.md](03-chunking.md).

### D. "Pick one" STT provider

We integrate both, designate **Sarvam** the primary, and use **ElevenLabs** strictly as a circuit-
breaker failover. The graded pipeline uses one provider; the second exists so the live demo link
does not die if a provider has a bad five minutes on the day a judge clicks it. This satisfies
"pick one" while being the correct engineering decision for a system that has to stay up.
Rationale and the criteria that could flip the primary: [05-speech-to-text.md](05-speech-to-text.md).

### E. "A reasonable number of test queries"

We commit to **≥ 300 queries**, stratified across languages and `query_type`, run both cold and
warm, and both sequentially and at concurrency 1 / 4 / 16. Raw per-query CSVs are committed to
`bench/results/` so the numbers are auditable rather than asserted.
See [09-evaluation.md](09-evaluation.md).

### F. "A proper harness"

We read this as: no raw single API call, and no free-running agent loop either. At a 200 ms budget
an agentic while-loop is the wrong architecture. The harness is an explicit typed stage DAG with
deadline propagation, retries, circuit breakers, schema-validated structured I/O, tool-call
dispatch to retrieval, and a replayable trace per request. See [06-harness.md](06-harness.md).

### G. "Show that your system knows when not to answer"

The operative word is *show*. Guardrails that exist only in code are invisible in a demo video.
So the guardrail decision is a first-class part of the API response and is rendered live in the
UI as a trace panel, and the demo script deliberately includes adversarial inputs — an off-topic
question, an unsafe request, an injected instruction, and unintelligible audio — so the
abstention behaviour is on camera. See [07-guardrails.md](07-guardrails.md) and
[12-submission.md](12-submission.md).

---

## Definition of done

The build is submittable when all of the following are true:

- [ ] A live URL accepts microphone input and returns a grounded, cited answer
- [ ] `t_core` P100 < 200 ms over ≥ 300 stratified queries, with the raw CSV in the repo
- [ ] P50 / P70 / P100 published for `t_core`, `t_stt` and `t_e2e`
- [ ] ≥ 6 chunking strategies implemented, all benchmarked, ablation table published
- [ ] Retrieval quality (Recall@k, nDCG@10, MRR) reported against `is_selected` labels
- [ ] Harness demonstrates a retry, a provider failover, and a budget-driven degradation on camera
- [ ] Every refusal code in the taxonomy is reachable and demonstrated
- [ ] Both videos recorded, uploaded to all three platforms by every member, all tagged `#RAGInGoa`
- [ ] Form submitted once, final
