from src.schemas import Chunk, Paper, SearchResult


def test_schema_serialization_preserves_ids():
    paper = Paper(paper_id="p1", title="Paper", source_path="paper.md")
    chunk = Chunk(chunk_id="p1:simple:0000", paper_id="p1", text="hello", chunk_index=0, chunker="simple")
    result = SearchResult(chunk_id=chunk.chunk_id, paper_id="p1", text="hello", rank=1, score=1.0, retriever="bm25")

    assert paper.to_dict()["paper_id"] == "p1"
    assert chunk.to_dict()["chunk_id"] == "p1:simple:0000"
    assert result.to_dict()["chunk_id"] == chunk.chunk_id
