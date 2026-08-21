"""Calibration sweep for the coverage gate thresholds (TAU_*). See
docs/07-guardrails.md and api/guardrails/coverage_gate.py's TODO(calibration).

Method: run the exact production retrieve+fuse path over two query sets —

- IN_DOMAIN: queries derived from real corpus chunks (the corpus provably
  covers these), so their fused-score distribution is what "answerable"
  looks like.
- OUT_OF_DOMAIN: hand-written questions on topics absent from the corpus
  (verified by keyword scan), whose distribution is what "must refuse" looks
  like.

Prints both distributions and suggests TAU_ABSOLUTE / TAU_MEAN values that
separate them (highest threshold that keeps every in-domain query, lowest
that rejects every out-of-domain query).

Usage:
    uv run --extra dev python -m bench.run_guardrails_calibration [n_pairs]
"""

from __future__ import annotations

import random
import statistics
import sys
from pathlib import Path

import polars as pl

from api.retrieval.embed import embed_query
from api.retrieval.qdrant_store import search_dense_grouped

OUT_OF_DOMAIN_QUERIES = [
    "what is the capital of India",
    "how do I bake a chocolate cake",
    "who won the football world cup final",
    "what is the boiling point of water",
    "how does photosynthesis work in plants",
    "best restaurants in Paris",
    "how to invest in the stock market",
    "what causes earthquakes",
    "symptoms of vitamin d deficiency",
    "how to learn japanese quickly",
]

TOP_K = 50


def coverage_scores(query: str) -> list[float]:
    """The exact production retrieve shape (pipeline.py stage 4), returning
    raw dense cosine scores — the signal coverage_gate actually gates on."""
    vector = embed_query(query)
    _, global_scores = search_dense_grouped(vector, per_arm_k=TOP_K)
    return global_scores


def fused_stats(scores: list[float]) -> tuple[float, float]:
    """(top1, mean5) — the two statistics TAU_ABSOLUTE/TAU_MEAN gate on."""
    return scores[0], statistics.mean(scores[:5])


def derive_queries(n: int) -> list[str]:
    """One query per randomly-sampled corpus chunk: the chunk's first
    sentence, truncated. Not elegant prose, but it IS text the corpus
    demonstrably contains — which is what the in-domain arm needs."""
    shards = sorted(Path("ingest/embeddings_cache").glob("shard_*.parquet"))
    df = pl.concat([pl.read_parquet(s, columns=["text"]) for s in shards])
    rng = random.Random(42)
    rows = df.sample(n=n, shuffle=True, seed=42)["text"].to_list()
    del rng
    queries = []
    for t in rows:
        q = t.replace("\n", " ").strip()
        sentences = q.split(". ")
        queries.append((sentences[0] if sentences else q)[:120])
    return queries


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) else 30

    print(f"Embedding + retrieving for {n} in-domain + {len(OUT_OF_DOMAIN_QUERIES)} "
          f"out-of-domain queries...")
    embed_query("warmup")

    in_stats = [fused_stats(coverage_scores(q)) for q in derive_queries(n)]
    out_stats = [fused_stats(coverage_scores(q)) for q in OUT_OF_DOMAIN_QUERIES]

    print("\nIN-DOMAIN (corpus covers these):")
    for name, idx in [("top1", 0), ("mean5", 1)]:
        vals = sorted(s[idx] for s in in_stats)
        print(f"  {name:<6} min={vals[0]:.4f}  p10={vals[len(vals)//10]:.4f}  "
              f"median={statistics.median(vals):.4f}  max={vals[-1]:.4f}")

    print("\nOUT-OF-DOMAIN (must refuse):")
    for name, idx in [("top1", 0), ("mean5", 1)]:
        vals = sorted(s[idx] for s in out_stats)
        print(f"  {name:<6} min={vals[0]:.4f}  median={statistics.median(vals):.4f}  "
              f"max={vals[-1]:.4f}")

    in_top1 = min(s[0] for s in in_stats)
    in_mean5 = min(s[1] for s in in_stats)
    out_top1 = max(s[0] for s in out_stats)
    out_mean5 = max(s[1] for s in out_stats)

    print("\nSUGGESTED THRESHOLDS:")
    if in_top1 > out_top1:
        tau_abs = round((in_top1 + out_top1) / 2, 4)
        print(f"  TAU_ABSOLUTE = {tau_abs}  (separates: in-domain min top1="
              f"{in_top1:.4f} > out-domain max top1={out_top1:.4f})")
    else:
        print(f"  !! top1 distributions OVERLAP (in-min={in_top1:.4f} <= out-max="
              f"{out_top1:.4f}) — top1 alone cannot separate; inspect per-query")
        for q, s in zip(OUT_OF_DOMAIN_QUERIES, out_stats):
            print(f"     out: top1={s[0]:.4f} mean5={s[1]:.4f}  {q!r}")
    if in_mean5 > out_mean5:
        tau_mean = round((in_mean5 + out_mean5) / 2, 4)
        print(f"  TAU_MEAN     = {tau_mean}  (in-min mean5={in_mean5:.4f} > out-max "
              f"mean5={out_mean5:.4f})")
    else:
        print(f"  !! mean5 distributions OVERLAP (in-min={in_mean5:.4f} <= out-max="
              f"{out_mean5:.4f})")
        for q, s in zip(OUT_OF_DOMAIN_QUERIES, out_stats):
            print(f"     out: top1={s[0]:.4f} mean5={s[1]:.4f}  {q!r}")


if __name__ == "__main__":
    main()
