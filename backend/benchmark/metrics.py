from __future__ import annotations
import math
from statistics import mean, median
from typing import Iterable

def recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    return float(any(cid in relevant_ids for cid in ranked_ids[:k]))

def reciprocal_rank(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, cid in enumerate(ranked_ids, start=1):
        if cid in relevant_ids:
            return 1.0 / rank
    return 0.0

def ndcg_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    dcg = 0.0
    for rank, cid in enumerate(ranked_ids[:k], start=1):
        if cid in relevant_ids:
            dcg += 1.0 / math.log2(rank + 1)
    ideal = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal + 1))
    return dcg / idcg if idcg else 0.0

def percentile(values: Iterable[float], p: float) -> float:
    xs = sorted(float(v) for v in values)
    if not xs:
        return 0.0
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] + (xs[hi] - xs[lo]) * frac

def per_query_metrics(ranked_ids: list[str], relevant_ids: set[str]) -> dict[str, float]:
    return {
        "recall@1": recall_at_k(ranked_ids, relevant_ids, 1),
        "recall@5": recall_at_k(ranked_ids, relevant_ids, 5),
        "recall@10": recall_at_k(ranked_ids, relevant_ids, 10),
        "mrr": reciprocal_rank(ranked_ids, relevant_ids),
        "ndcg@10": ndcg_at_k(ranked_ids, relevant_ids, 10),
    }

def aggregate_query_rows(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {}
    latencies = [float(row["total_retrieval_ms"]) for row in rows]
    return {
        "queries": len(rows),
        "recall@1": round(mean(row["recall@1"] for row in rows), 6),
        "recall@5": round(mean(row["recall@5"] for row in rows), 6),
        "recall@10": round(mean(row["recall@10"] for row in rows), 6),
        "mrr": round(mean(row["mrr"] for row in rows), 6),
        "ndcg@10": round(mean(row["ndcg@10"] for row in rows), 6),
        "p50_retrieval_ms": round(median(latencies), 2),
        "p95_retrieval_ms": round(percentile(latencies, 0.95), 2),
    }
