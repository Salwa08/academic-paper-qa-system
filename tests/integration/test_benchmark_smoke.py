from src.benchmarking.corpus import build_chunks, load_manifest
from src.retrieval.bm25 import BM25Retriever


def test_benchmark_smoke_builds_sample_chunks_and_runs_bm25():
    papers = load_manifest()
    chunks = build_chunks("simple", paper_ids=[papers[0].paper_id])
    retriever = BM25Retriever(chunks)
    results = retriever.search("attention mechanism", top_k=3)

    assert results
    assert results[0].chunk_id
    assert results[0].paper_id == papers[0].paper_id
    assert results[0].rank == 1
    assert isinstance(results[0].score, float)
