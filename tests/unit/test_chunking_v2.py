from src.data.chunking import SectionAwareV2Chunker, SimpleChunker


def test_simple_chunker_generates_stable_chunk_ids():
    text = " ".join(f"token{i}" for i in range(90))
    chunks = SimpleChunker(chunk_size=40, overlap=10, min_tokens=10).chunk_paper("p1", text)

    assert [chunk.chunk_id for chunk in chunks] == ["p1:simple:0000", "p1:simple:0001", "p1:simple:0002"]
    assert all(chunk.text for chunk in chunks)


def test_section_aware_v2_preserves_metadata_with_token_windows():
    text = "# Abstract\n" + "alpha " * 45 + "\n# Method\n" + "model " * 45
    chunks = SectionAwareV2Chunker(chunk_size=30, overlap=5, min_tokens=10).chunk_paper("p2", text)

    assert chunks[0].chunk_id == "p2:section_aware_v2:0000"
    assert {chunk.section_type for chunk in chunks} >= {"abstract", "method"}
    assert all(len(chunk.text.split()) >= 10 for chunk in chunks)
