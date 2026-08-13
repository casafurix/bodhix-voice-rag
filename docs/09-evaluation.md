# 09 — Evaluation

Latency without quality is meaningless: an empty string returns in 0 ms. This doc covers the
quality side, and the point that the dataset makes it nearly free.

## The advantage we have

MSMARCO-XI ships `is_selected` binary relevance labels for ~10 candidate passages per query. That
is a **qrels file**, which means retrieval quality is directly measurable with no annotation work.
Most submissions will not use it. See [02-dataset.md](02-dataset.md) §1.

Stated caveat, up front: MS MARCO labels are **sparse and incomplete** — an unjudged passage is
labelled `0` even when relevant, so absolute scores understate true performance. We therefore use
these metrics **comparatively**, to rank our own configurations against each other, which is exactly
what the chunking ablation needs. We do not claim leaderboard-comparable absolute numbers.

---

## Retrieval metrics

Computed against `is_selected` for every chunking strategy and retrieval configuration.

| Metric | Why it is here |
|---|---|
| **Recall@5 / @20 / @50** | Recall@50 is the ceiling the reranker can work within — if the gold passage is not in the candidate set, nothing downstream can recover it |
| **nDCG@10** | Primary ranking metric; the chunking champion is selected on this |
| **MRR@10** | Directly proxies "is the answer in the first thing the user hears" |
| **MAP** | Stability check across the whole ranking |
| **Hit@1** | The extractive path leans hardest on rank 1 |

Chunk-level labels are derived by inheritance: a chunk is relevant if its parent passage has
`is_selected == 1`. For strategies producing derived units (S9 doc2query, S8 propositions) the label
follows the source passage. This inheritance rule is written down and applied identically across all
strategies, because inconsistent label mapping is the easiest way to produce an ablation table that
means nothing.

### Reported cuts

- Per chunking strategy (the [03](03-chunking.md) ablation)
- Per language — a strategy winning on English and losing on Tamil is a finding, and on a
  machine-translated corpus it is a likely one
- Per `query_type` — we expect the sparse arm to carry NUMERIC/ENTITY and dense to carry DESCRIPTION;
  if the data disagrees, the fusion weights are wrong
- Per query length
- Dense-only vs sparse-only vs hybrid — proves the hybrid arm earns its milliseconds
- With and without rerank — quantifies what the adaptive gate costs when it fires

---

## Answer quality

Harder, because there is no single correct string. But the dataset gives us a reference answer
(`Answer` / `Eng_Answer`), which is more than most RAG projects have.

| Metric | Method | Note |
|---|---|---|
| **Answer F1 / EM** | Token overlap with the reference `Answer` | Standard MS MARCO-style; weak but cheap and unbiased |
| **Semantic similarity** | Cosine between generated and reference answer embeddings | Tolerates paraphrase, which F1 punishes unfairly |
| **Groundedness rate** | % of answers passing the NLI gate | Our own guardrail metric, from [07](07-guardrails.md) |
| **Citation precision** | Are the cited chunks actually the labelled-relevant ones? | Uses `is_selected` again |
| **LLM-as-judge** | A strong model scores correctness/groundedness/fluency 1–5 against reference + context | Run offline on a 150-query sample. Biased and we say so |
| **Human spot-check** | Team reviews 50 answers per language, blind | The only check that catches what automated metrics miss, e.g. fluent-but-wrong Tamil |

**Extractive vs abstractive comparison** is a headline result: the extractive path is 15× faster and
grounded by construction. If it also scores comparably on answer F1, that is a genuinely interesting
finding about this workload and it justifies the dual-path architecture with evidence rather than
assertion. If it scores much worse, we report that too and the rich path becomes more important.

**Cross-lingual consistency** — the same question in all five languages against the same corpus.
Do we get the same answer? A high variance here means either the translations are bad or our
cross-lingual retrieval is. Almost free to compute, and no monolingual submission can produce it.

---

## Guardrail metrics

From [07-guardrails.md](07-guardrails.md), because a guardrail without a false-positive rate is not
characterised.

| Metric | Definition | Target |
|---|---|---|
| **Refusal precision** | Of refusals, how many *should* have been refused | > 0.9 |
| **Refusal recall** | Of things that should be refused, how many were | > 0.85 |
| **Over-refusal rate** | Answerable queries wrongly refused | **< 3 %** |
| **Coverage-gate ROC / AUC** | In-domain vs out-of-domain separability | Published curve |
| **Injection block rate** | Per language | 100 % on the committed set |
| **Unsafe block rate** | Per category | 100 % on the committed set |
| **Hallucination catch rate** | Deliberately ungrounded generations vetoed | > 0.9 |

Over-refusal has the tightest target on the list, deliberately. A system that refuses everything
scores perfectly on every other guardrail metric and is useless. The near-miss cohort in
`bench/redteam.jsonl` exists to keep that honest.

---

## Ingest benchmarks

Reported separately from `t_core` — chunking the corpus is not in the request path
([00](00-task-brief.md) §C) — but it is real engineering and belongs in the record.

| Metric | Per strategy |
|---|---|
| Wall-clock ingest time | |
| Chunks produced | |
| Index size on disk / in RAM | |
| Embedding throughput (chunks/s) | |
| Dedup ratio achieved | |
| Quality-filter rejection counts, by filter | |
| LLM cost (S7, S8, S9) | |

The LLM-cost column is what makes the cost-gating decision in [03](03-chunking.md) concrete: if
`contextual_prefix` costs $40 to build and adds 0.4 nDCG points, it does not ship, and the table
shows why.

---

## Harness tests as evidence

The fault-injection and budget suites from [06](06-harness.md) produce reportable numbers, not just
pass/fail:

- Degradation ladder rung distribution under normal load
- Recovery success rate per injected fault class
- Failover latency: Sarvam failure → ElevenLabs answer
- Deadline-adherence rate at `budget_ms` ∈ {200, 150, 100, 60}
- Structured-output repair rate on the rich path

That last set is the direct evidence for the harness requirement. "We have retries" is a claim;
"forced 500s on STT, 100 % recovered via failover at P95 +340 ms" is a measurement.

---

## Reproducibility

Every number in the submission traces to a committed script.

```
bench/
├── queries.jsonl              # fixed 300–500 query benchmark set
├── redteam.jsonl              # adversarial + near-miss set
├── qrels.jsonl                # derived from is_selected
├── run_latency.py             # → results/latency_*.csv
├── run_retrieval.py           # → results/retrieval_*.csv
├── run_chunking_ablation.py   # → results/chunking_ablation.csv
├── run_guardrails.py          # → results/guardrails_*.csv
├── run_ingest_bench.py        # → results/ingest_*.csv
├── replay.py                  # re-run a historical trace against current code
├── report.py                  # CSVs → percentile tables + charts
└── results/                   # committed: raw CSVs, charts, and a manifest
```

Each results file carries a manifest header: git commit, index artefact id, model revisions, machine
spec, timestamp. A reviewer can tell exactly what produced a number, and `replay.py` means we can
tell a real regression from noise when we change the chunker on day 7.

---

## CI

On every push: unit + contract tests, golden retrieval set, the red-team suite, and a fast latency
smoke test on T0. A guardrail regression or a latency regression fails the build. The full T2
benchmark runs on demand, since it takes too long for a pre-commit loop.

The point of CI here is narrow but important: nine days is long enough to accidentally break the
coverage gate on day 6 while tuning the chunker, and not notice until the demo.
