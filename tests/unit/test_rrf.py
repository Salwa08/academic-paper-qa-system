from src.retrieval.fusion import reciprocal_rank_fusion
from src.schemas import SearchResult


def result(chunk_id, rank, retriever):
    return SearchResult(chunk_id=chunk_id, paper_id="p", text=chunk_id, rank=rank, score=1.0, retriever=retriever)


def test_rrf_prefers_shared_high_rank_chunks():
    fused = reciprocal_rank_fusion([
        [result("shared", 1, "bm25"), result("bm25_only", 2, "bm25")],
        [result("shared", 1, "dense"), result("dense_only", 2, "dense")],
    ], top_k=3, rrf_k=60)

    assert fused[0].chunk_id == "shared"
    assert fused[0].retriever == "hybrid_rrf"
