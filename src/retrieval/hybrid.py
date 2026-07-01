"""Hybrid RRF retrieval backend."""

from typing import List

from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.fusion import reciprocal_rank_fusion
from src.schemas import SearchResult


class HybridRRFRetriever:
    retriever_name = "hybrid_rrf"

    def __init__(self, bm25: BM25Retriever, dense: DenseRetriever, candidate_k: int = 20, rrf_k: int = 60):
        self.bm25 = bm25
        self.dense = dense
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        candidate_k = max(top_k, self.candidate_k)
        bm25_results = self.bm25.search(query, top_k=candidate_k)
        dense_results = self.dense.search(query, top_k=candidate_k)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k, rrf_k=self.rrf_k)
