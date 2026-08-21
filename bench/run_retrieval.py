"""Chunking-strategy ablation — the graded "vast, decided by data" table.

For every query in bench/queries.jsonl with relevance labels (the dataset's
own is_selected flags), retrieve per chunking strategy and score:

- Recall@10: fraction of queries whose top-10 hits include >=1 relevant doc
- nDCG@10: graded ranking quality against the relevant set
- MRR: reciprocal rank of the first relevant hit

Arms: one per strategy (S1, S2, S3, S5, S9, S10) plus the production
ensemble (all strategies + sparse, RRF-fused — what /ask actually serves).
Relevance is cross-lingual by construction: a query's relevant docs are
both its {lang} and en doc-id prefixes (docs/02-dataset.md).

Writes bench/results/retrieval_ablation.csv. See docs/09-evaluation.md.

Usage:
    uv run --extra dev python -m bench.run_retrieval
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from api.retrieval.embed import embed_query
from api.retrieval.fuse import ScoredChunk, reciprocal_rank_fusion
from api.retrieval.qdrant_store import search_dense_grouped
from api.retrieval.sparse import search_sparse

RESULTS = Path("bench/results")
TOP_K = 10


def percentile(values: list[float], pct: float) -> float:
    values = sorted(values)
    k = (len(values) - 1) * (pct / 100)
    f, c = int(k), min(int(k) + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (k - f) * (values[c] - values[f])


def load_relevance() -> list[dict]:
    """query -> set of relevant doc_ids, from the index payloads themselves."""
    from qdrant_client import models  # noqa: F401
    from api.retrieval.qdrant_store import COLLECTION_NAME, get_client

    client = get_client()
    # doc_id prefix -> is_selected, via scroll over all points
    selected: dict[str, bool] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            COLLECTION_NAME, limit=500, offset=offset,
            with_payload=["doc_id", "chunk_id"], with_vectors=False,
        )
        for p in points:
            payload = p.payload or {}
            selected[payload["chunk_id"]] = bool(payload.get("is_selected"))
        if offset is None:
            break

    # chunk_id = f"{doc_id}/cN"; doc_id = f"{lang}/{query_id}/p{i}"
    query_relevant: dict[str, set[str]] = defaultdict(set)
    for chunk_id, sel in selected.items():
        doc_id = chunk_id.rsplit("/c", 1)[0]
        lang, qid = doc_id.split("/")[:2]
        if sel:
            query_relevant[qid].add(doc_id)

    queries = [json.loads(l) for l in Path("bench/queries.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    out = []
    for rec in queries:
        if rec["domain"] != "in":
            continue
        relevant = query_relevant.get(rec["qid"], set())
        if not relevant:
            continue
        out.append({**rec, "key": f"{rec['lang']}::{rec['qid']}", "relevant": relevant})
    return out


def evaluate(ranked_doc_lists: dict[str, dict[str, list[str]]], queries: list[dict]) -> dict:
    """ranked_doc_lists: arm -> {key -> [doc_id...]} per query."""
    rows = {}
    for arm in ranked_doc_lists:
        recalls, ndcgs, mrrs = [], [], []
        for q in queries:
            ranked = ranked_doc_lists[arm][q["key"]]
            relevant = q["relevant"]
            hits = [(i + 1, d) for i, d in enumerate(ranked[:TOP_K]) if d in relevant]

            recalls.append(1.0 if hits else 0.0)

            dcg = sum(1.0 / math.log2(rank + 1) for rank, _ in hits)
            idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), TOP_K)))
            ndcgs.append(dcg / idcg if idcg else 0.0)

            mrrs.append(1.0 / hits[0][0] if hits else 0.0)

        rows[arm] = {
            "recall@10": statistics.mean(recalls),
            "ndcg@10": statistics.mean(ndcgs),
            "mrr": statistics.mean(mrrs),
        }
    return rows


def main() -> None:
    queries = load_relevance()
    print(f"[ablation] {len(queries)} labelled queries")

    arms: dict[str, dict[str, list[ScoredChunk]]] = defaultdict(lambda: defaultdict(list))
    ensemble: dict[str, list[ScoredChunk]] = {}

    for i, q in enumerate(queries):
        vector = embed_query(q["query"])
        grouped, _ = search_dense_grouped(vector, per_arm_k=TOP_K)
        sparse_hits = search_sparse(q["query"], top_k=TOP_K * 3)

        ranked_lists: list[list[ScoredChunk]] = []

        def to_arm(hits):
            return [
                ScoredChunk(chunk_id=(h.payload or {}).get("chunk_id", str(h.id)), score=h.score)
                for h in hits
            ]

        for hits in grouped:
            strategy = (hits[0].payload or {}).get("strategy", "unknown") if hits else "unknown"
            scored = to_arm(hits)
            arms[strategy][q["key"]] = scored
            ranked_lists.append(scored)
        sparse_scored = [ScoredChunk(chunk_id=cid, score=s) for cid, s in sparse_hits]
        arms["sparse_bm25"][q["key"]] = sparse_scored
        ranked_lists.append(sparse_scored)

        fused = reciprocal_rank_fusion(ranked_lists)
        # map back to doc ids; RRF returns ScoredChunk with chunk_id
        ensemble[q["key"]] = [
            ScoredChunk(chunk_id=fused_chunk.chunk_id, score=fused_chunk.score)
            for fused_chunk in fused
        ]

        if (i + 1) % 50 == 0:
            print(f"[ablation] {i + 1}/{len(queries)} queries retrieved")

    ranked_docs: dict[str, dict[str, list[str]]] = {}
    for arm, per_query in arms.items():
        ranked_docs[arm] = {
            key: [c.chunk_id.rsplit("/c", 1)[0] for c in chunks]
            for key, chunks in per_query.items()
        }
    ranked_docs["ENSEMBLE_rrf"] = {
        key: [c.chunk_id.rsplit("/c", 1)[0] for c in chunks]
        for key, chunks in ensemble.items()
    }

    table = evaluate(ranked_docs, queries)

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "retrieval_ablation.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arm", "recall@10", "ndcg@10", "mrr"])
        best_ndcg = max(table, key=lambda a: table[a]["ndcg@10"])
        for arm in sorted(table):
            r = table[arm]
            marker = " <- champion" if arm == best_ndcg else ""
            w.writerow([arm + marker, f"{r['recall@10']:.4f}", f"{r['ndcg@10']:.4f}", f"{r['mrr']:.4f}"])

    print(f"\n{'arm':<22}{'recall@10':>11}{'ndcg@10':>10}{'mrr':>9}")
    for arm in sorted(table, key=lambda a: -table[a]["ndcg@10"]):
        r = table[arm]
        star = " *" if arm == best_ndcg else ""
        print(f"{arm:<22}{r['recall@10']:>11.4f}{r['ndcg@10']:>10.4f}{r['mrr']:>9.4f}{star}")
    print(f"\nchampion (nDCG@10): {best_ndcg}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
