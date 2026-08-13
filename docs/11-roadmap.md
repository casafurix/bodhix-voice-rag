# 11 — Roadmap

**Launch:** 13 August 2026 · **Deadline:** 22 August 2026, 11:59 PM · **10 days, no resubmissions.**

## Team and ownership

Three people, three tracks that can run in parallel with minimal blocking. Clear ownership matters
more than usual on a 10-day build with a hard stop.

| Track | Owner | Scope |
|---|---|---|
| **A — Data & Retrieval** | Agnibha (`casafurix`) | Ingest, chunkers, embeddings, Qdrant, hybrid search, rerank, ablation |
| **B — Harness & Guardrails** | Smil (`smil-thakur`) | Stage DAG, budgets, retries, failover, guardrails, red-team suite |
| **C — Voice & Frontend** | Simran (`SIMRAN719`) | STT adapters, streaming, VAD, Next.js UI, guardrail trace panel, latency HUD |

Shared: benchmarking, videos, and the promotion posts (which every member must do individually —
see [12-submission.md](12-submission.md)).

**Integration contract fixed on day 1.** The `/ask` request/response schema in
[01-architecture.md](01-architecture.md) is frozen early so all three tracks can build against it
without waiting for each other. Track C mocks it, Track B implements it, Track A fills it.

---

## Phase plan

```
Day 1–2   Foundations       schema frozen, repo scaffolded, T0 index exists
Day 3–4   Core pipeline     end-to-end text→answer working, STT bake-off done
Day 5–6   Depth             all chunkers, guardrails, harness hardening
Day 7     Ablation          measure everything, pick the champion
Day 8     Scale + polish    T2 index, deploy, benchmark for real
Day 9     Videos            record, edit, post
Day 10    Buffer + submit   fix, verify, submit early
```

Day 10 is buffer on purpose. A 10-day plan that uses all 10 days ships nothing on day 10.

---

## Day-by-day

### Day 1 (13 Aug) — Foundations
- [ ] **All:** read the brief and `docs/` together; agree the plan; confirm the 200 ms interpretation
- [ ] **All:** freeze the `/ask` schema — this unblocks everything
- [ ] Repo scaffolded: `api/`, `ingest/`, `bench/`, `web/`; CI running lint + tests
- [ ] **A:** stream MSMARCO-XI, inspect real rows, confirm schema and the missing `teltrain` gap
- [ ] **B:** `Deadline`, `Stage`, `Context` primitives + a two-stage toy DAG
- [ ] **C:** Sarvam + ElevenLabs keys obtained; `curl` a real transcription from both
- [ ] **Risk check:** if API keys are slow to obtain, that surfaces today, not on day 6

### Day 2 (14 Aug) — T0 index and first light
- [ ] **A:** ingest → dedup → S2 `passage_native` → embed → Qdrant. **T0 (50 K) index exists.**
- [ ] **A:** `qrels.jsonl` derived from `is_selected`; `run_retrieval.py` prints Recall@k
- [ ] **B:** full DAG skeleton with all stages stubbed, timings flowing into the response
- [ ] **C:** mic capture → 16 kHz PCM → `POST` → transcript on screen. Ugly is fine.
- [ ] **Milestone:** retrieval metrics measurable; voice input captured

### Day 3 (15 Aug) — First end-to-end
- [ ] **A:** hybrid search (dense ∥ BM25 + RRF); Indic tokenisation for BM25
- [ ] **A:** reranker integrated, ONNX int8, timed
- [ ] **B:** `guard_in` checks 1–5; extractive `answer_fast`; `assemble`
- [ ] **C:** **STT bake-off** — 60 clips, 5 languages, both providers, WER/CER + retrieval-preserving accuracy + `t_stt`. **Publish and confirm or flip the primary.**
- [ ] **Milestone: 🎯 first end-to-end voice → answer.** Slow and rough, but complete. This is the single most important milestone in the plan — everything after is improvement, not integration risk.

### Day 4 (16 Aug) — Make it fast
- [ ] **A:** ONNX int8 everywhere; Qdrant `ef` sweep; scalar quantisation; concurrent arms
- [ ] **A:** first `t_core` measurement on T0 — find out how far off 200 ms we are
- [ ] **B:** deadline propagation live; degradation ladder rungs 1–4; adaptive rerank gate
- [ ] **C:** streaming STT over WebSocket; client-side VAD; end-of-speech detection
- [ ] **Decision point:** is the budget reachable on 4 vCPU? If `rerank` is far worse than estimated, choose the smaller reranker now rather than on day 8

### Day 5 (17 Aug) — Chunking depth
- [ ] **A:** S1, S3, S5, S9, S10 implemented behind the `Chunker` interface
- [ ] **A:** metadata payloads + payload indexes; `query_type` routing
- [ ] **B:** `coverage_gate` with threshold calibration; in-domain vs out-of-domain query sets built
- [ ] **B:** `guard_out` — NLI, numeric, citations, language consistency
- [ ] **C:** guardrail trace panel + latency HUD in the UI (reads the response, no new API)
- [ ] **C:** speculative retrieval on stable partials

### Day 6 (18 Aug) — Hardening
- [ ] **A:** S6 `late_chunking`; S11 `script_aware_sizing`; S4 if time permits
- [ ] **A:** quality filters incl. twin-cosine translation filtering, with before/after metrics
- [ ] **B:** retries, circuit breakers, STT failover; **fault-injection suite**
- [ ] **B:** `redteam.jsonl` — off-topic, injections (×5 languages), unsafe, audio pathologies, near-miss
- [ ] **B:** budget tests at `budget_ms` ∈ {200,150,100,60}
- [ ] **C:** citations UI, refusal rendering per code, iOS Safari testing on a real device
- [ ] **All:** start capturing process footage for Video 1 — it cannot be reconstructed later

### Day 7 (19 Aug) — Measure and decide
- [ ] **A:** **full chunking ablation** on T0/T1 → `chunking_ablation.csv`; per-language cuts
- [ ] **A:** cost-gate decision on S7/S8 — build them only if the sampled gain justifies it
- [ ] **A:** **pick the champion** by the pre-declared rule: best nDCG@10 with ≥ 60 ms headroom
- [ ] **B:** guardrail metrics — refusal precision/recall, over-refusal rate, coverage ROC
- [ ] **C:** UI feature-complete and frozen
- [ ] **Milestone:** champion config selected from data; every graded number has a script

### Day 8 (20 Aug) — Scale and deploy
- [ ] **A:** build **T1 and T2** indexes with the champion config; snapshot artefacts + manifests
- [ ] **All:** deploy — Fly `bom` + Vercel `bom1`; warmup gate; rate limits; spend caps
- [ ] **All:** **the real benchmark run** — 300–500 queries × cold/warm × c=1/4/16 × T0→T2, 3 repeats
- [ ] **All:** `report.py` → headline table, percentile curve, stage breakdown, scaling curve, degradation report
- [ ] **All:** verify the live link from a phone on mobile data, and cold-start behaviour
- [ ] **Milestone: 🎯 live link working, numbers published in the repo**
- [ ] **Hard gate:** if `t_core` P100 > 200 ms, today is the day to fix it. Levers in priority order: harder rerank gating, smaller reranker, lower `ef`, cap the tier, GPU.

### Day 9 (21 Aug) — Videos and promotion
- [ ] **Video 1 (90 s, process):** the team working — whiteboard, the ablation table being read, a failing test, the STT bake-off, the deadline debate. Process, not product.
- [ ] **Video 2 (demo):** the script in [12-submission.md](12-submission.md), including all seven guardrail refusals on camera
- [ ] Both videos edited and exported
- [ ] **All three members** post both videos to **Instagram, X and LinkedIn**, each with `#RAGInGoa`
- [ ] Confirm at least one Instagram account is **public**
- [ ] Collect all 9 post URLs into `docs/12-submission.md`
- [ ] README updated with the real measured numbers

### Day 10 (22 Aug) — Buffer and submit
- [ ] Final pass on README and docs — every number matches the committed CSVs
- [ ] Full pre-submission checklist ([10-deployment.md](10-deployment.md) + [12-submission.md](12-submission.md))
- [ ] Live link re-verified; API keys re-verified against quota
- [ ] **Submit the form by 18:00**, not 23:58. No resubmissions means no room for a form that fails to load.

---

## Milestones and hard gates

| Day | Milestone | If missed |
|---|---|---|
| 2 | T0 index queryable | Drop to English-only; unblock retrieval first |
| **3** | **End-to-end voice → answer** | **Critical.** Cut all optional chunkers, ship the vertical slice |
| 4 | Know the real `t_core` | Choose the smaller reranker now, not later |
| 7 | Champion chosen from data | Ship S5 on prior reasoning; publish a partial ablation |
| **8** | **Live link + published numbers** | **Critical.** Cut T2, submit on T1 |
| 9 | Videos posted by all 3 members | Blocks submission entirely — promotion is mandatory |

---

## Descoping order

Decided now, so it is not debated at 2 am on day 9.

1. S4, S7, S8, S12 chunking strategies (highest cost, most likely cut)
2. T3 stretch index
3. Rich abstractive path (extractive alone satisfies the brief)
4. Languages beyond English + Hindi + one more
5. Speculative retrieval (a latency nicety, not a requirement)
6. `/diagnose` endpoint
7. Semantic cache

**Never cut:** the ≥ 300-query benchmark, the chunking ablation table, the guardrail refusal
taxonomy, the live link, the videos. These are the graded requirements. Everything else is upside.

---

## Standing risks

| Risk | Watch | Mitigation |
|---|---|---|
| Ingest takes far longer than expected | Day 2, 8 | Tiered indexes; rent a GPU; drop languages |
| `t_core` cannot reach 200 ms on CPU | Day 4 | Documented lever list; GPU as last resort |
| Sarvam rate limits block the voice benchmark | Day 3 | Cache STT results; benchmark `t_core` from text |
| A track blocks on another | Daily | Schema frozen day 1; mocks on both sides |
| Videos left to the last day | Day 6 | Start capturing process footage from day 6 |
| Promotion posts forgotten by one member | Day 9 | Explicit checklist with 9 URL slots — it is mandatory and it is a single point of failure |
| Someone gets sick / has exams | Any | Tracks are independent; the descope order is pre-agreed |

The promotion risk is worth flagging twice: it is the only requirement that can be perfectly
satisfied by the code and still fail the submission because one person forgot to post on LinkedIn.
