# Benchmark report

## Latency (`POST /ask`, full pipeline)

320 successful requests, budget 200 ms

| metric | avg | p50 | p70 | p95 | p99 | p100 |
|---|---|---|---|---|---|---|
| t_core | 64.2 | 42.1 | 44.6 | 63.5 | 152.3 | 5975.8 |
| t_e2e | 66.7 | 44.4 | 46.9 | 66.3 | 155.3 | 5987.9 |

- **Degradation rate:** 1/320 (0.3%) over budget
- Answered: 295 · refused out-of-domain (correct): 15 · over-refused in-domain: 10
- Over-refusal rate: 3.3% of in-domain queries

## Chunking-strategy ablation

Recall@10 / nDCG@10 / MRR against the dataset's own `is_selected` labels.

| arm | recall@10 | nDCG@10 | MRR |
|---|---|---|---|
| s5_parent_child **(champion)** | 0.6000 | 0.2309 | 0.2715 |
| s3_sentence_window | 0.5091 | 0.2161 | 0.2588 |
| ENSEMBLE_rrf | 0.9091 | 0.2051 | 0.3209 |
| sparse_bm25 | 0.9091 | 0.1890 | 0.3392 |
| s9_doc2query | 0.9455 | 0.1861 | 0.3207 |
| s1_fixed | 0.4909 | 0.1639 | 0.2595 |
| s2_passage_native | 0.4909 | 0.1639 | 0.2595 |
| s10_crosslingual_twin | 0.3818 | 0.1577 | 0.1774 |

Champion by nDCG@10: **s5_parent_child**
