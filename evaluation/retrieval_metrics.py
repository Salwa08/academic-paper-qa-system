"""Retrieval metrics for labeled chunk-id evaluation."""

from math import log2
from typing import Iterable, Sequence, Set


def _relevant_set(relevant_ids: Iterable[str]) -> Set[str]:
    return {chunk_id for chunk_id in relevant_ids if chunk_id}


def hit_rate_at_k(ranked_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    relevant = _relevant_set(relevant_ids)
    if not relevant:
        return 0.0
    return 1.0 if any(chunk_id in relevant for chunk_id in ranked_ids[:k]) else 0.0


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    relevant = _relevant_set(relevant_ids)
    if not relevant:
        return 0.0
    found = sum(1 for chunk_id in ranked_ids[:k] if chunk_id in relevant)
    return found / len(relevant)


def mrr_at_k(ranked_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    relevant = _relevant_set(relevant_ids)
    if not relevant:
        return 0.0
    for rank, chunk_id in enumerate(ranked_ids[:k], 1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    relevant = _relevant_set(relevant_ids)
    if not relevant:
        return 0.0
    dcg = 0.0
    for rank, chunk_id in enumerate(ranked_ids[:k], 1):
        if chunk_id in relevant:
            dcg += 1.0 / log2(rank + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def metrics_at_k(ranked_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> dict:
    return {
        f"HitRate@{k}": hit_rate_at_k(ranked_ids, relevant_ids, k),
        f"Recall@{k}": recall_at_k(ranked_ids, relevant_ids, k),
        f"MRR@{k}": mrr_at_k(ranked_ids, relevant_ids, k),
        f"nDCG@{k}": ndcg_at_k(ranked_ids, relevant_ids, k),
    }
