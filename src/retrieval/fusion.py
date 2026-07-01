"""Rank fusion utilities."""

from typing import Dict, Iterable, List

from src.schemas import SearchResult


def reciprocal_rank_fusion(result_lists: Iterable[List[SearchResult]], top_k: int = 5, rrf_k: int = 60) -> List[SearchResult]:
    fused_scores: Dict[str, float] = {}
    exemplars: Dict[str, SearchResult] = {}

    for results in result_lists:
        for result in results:
            fused_scores[result.chunk_id] = fused_scores.get(result.chunk_id, 0.0) + (1.0 / (rrf_k + result.rank))
            exemplars.setdefault(result.chunk_id, result)

    ranked_ids = sorted(fused_scores, key=lambda chunk_id: fused_scores[chunk_id], reverse=True)[:top_k]
    fused: List[SearchResult] = []
    for rank, chunk_id in enumerate(ranked_ids, 1):
        base = exemplars[chunk_id]
        metadata = dict(base.metadata)
        metadata["source_retriever"] = base.retriever
        fused.append(SearchResult(
            chunk_id=base.chunk_id,
            paper_id=base.paper_id,
            text=base.text,
            rank=rank,
            score=fused_scores[chunk_id],
            retriever="hybrid_rrf",
            metadata=metadata,
        ))
    return fused
