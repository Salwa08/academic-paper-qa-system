"""Paper processor for academic papers - handles BM25 indexing and retrieval."""
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys

# Add src to path FIRST (before importing from src)
src_path = Path(__file__).resolve().parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from rank_bm25 import BM25Okapi
import re
import json
from .chunking import SimpleChunker, SectionAwareChunker
from .text_preprocessor import clean_text, simple_tokenize

# Use absolute paths relative to project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
CHUNKS_JSON = DATA_DIR / "processed_chunks.json"
INDEX_DIR = DATA_DIR / "bm25_index"

def process_papers(force_reprocess: bool = False, chunker = None) -> Dict[str, Any]:
    """
    Main function:
    - Reads raw paper texts from DATA_DIR
    - Cleans and chunks the texts
    - Saves processed chunks to CHUNKS_JSON
    - Builds BM25 index and saves to INDEX_DIR
    - Returns list of processed chunks 
    """
    if chunker is None:
        chunker = SimpleChunker()
    
    if CHUNKS_JSON.exists() and not force_reprocess:
        print("Loading existing chunks...")
        with open(CHUNKS_JSON, 'r', encoding='utf-8') as f:
            data = json.load(f)
        chunks = data["chunks"]
        tokenized_corpus = [simple_tokenize(chunk["chunk"]) for chunk in chunks]
        bm25 = BM25Okapi(tokenized_corpus)
        return {
            "chunks": chunks,
            "bm25": bm25
        }
    
    print("Processing papers...")
    chunks = []
    # Look for markdown files recursively in DATA_DIR
    md_files = list(DATA_DIR.rglob("*.md"))
    
    if not md_files:
        print("No markdown files found!")
        return {"chunks": [], "bm25": None}
    
    for paper_file in md_files:
        print(f"Processing {paper_file.name}...")
        with open(paper_file, 'r', encoding='utf-8') as f:
            raw = f.read()
        
        # For section-aware chunking, keep raw markdown (headers needed for splitting)
        # For simple chunking, use clean_text
        if isinstance(chunker, SectionAwareChunker):
            text_to_chunk = raw
        else:
            text_to_chunk = clean_text(raw)
        
        paper_chunks = chunker.chunk_text(text_to_chunk)
        for chunk in paper_chunks:
            if isinstance(chunk, str): # Phase 1: simple string chunks
                chunks.append({
                    "paper_id": paper_file.stem,  # Use file stem as paper ID
                    "chunk": chunk,
                    "section_type": "other",
                })
            else: # Phase 2: section-aware chunks
                chunks.append({
                    "paper_id": paper_file.stem,
                    "chunk": chunk["content"],
                    "section_type": chunk["section_type"]
                })
    
    print(f"Processed {len(chunks)} chunks from {len(md_files)} papers")

    with open(CHUNKS_JSON, 'w', encoding='utf-8') as f:
        json.dump({"chunks": chunks}, f, ensure_ascii=False, indent=4)

    # --- Build BM25 Index ---
    tokenized_corpus = [simple_tokenize(chunk["chunk"]) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    print(f"-> Build index over {len(chunks)} chunks")

    return {
        "chunks": chunks,
        "bm25": bm25
    }

def retrieve(query: str, index_data: Dict[str, Any], top_k: int = 5, return_metadata: bool = False, section_boosts: Optional[Dict[str, Any]] = None) -> List[Any]:
    """
    Retrieve relevant chunks for a query using BM25.
    
    Args:
        query: Search query
        index_data: BM25 index data
        top_k: Number of results to return
        return_metadata: If True, return dicts with metadata; if False, return just text
        section_boosts: Dict mapping section types to boost factors (optional)
        
    Returns:
        List of chunks (strings if return_metadata=False, dicts if True)
    """
    tokenized_query = simple_tokenize(query)

    bm25 = index_data["bm25"]
    scores = bm25.get_scores(tokenized_query)
    if section_boosts:
        for i, chunk_data in enumerate(index_data["chunks"]):
            section_type = chunk_data.get("section_type", "other")
            boost = section_boosts.get(section_type, 1.0)
            scores[i] *= boost

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    
    if return_metadata:
        # Return full chunk objects with metadata
        retrieved_chunks = [
            {
                "chunk": index_data["chunks"][i]["chunk"],
                "paper_id": index_data["chunks"][i].get("paper_id", "unknown"),
                "score": float(scores[i])
            }
            for i in top_indices
        ]
    else:
        # Return just the text (backward compatible)
        retrieved_chunks = [index_data["chunks"][i]["chunk"] for i in top_indices]

    return retrieved_chunks


# Quick testing
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Phase 2: Section-Aware Chunking + Score Boosting")
    print("=" * 60)
    
    # Process papers with section-aware chunking
    index_data = process_papers(force_reprocess=True, chunker=SectionAwareChunker())
    
    # Test query about architecture (should boost "method" sections)
    query = "What is the BERT architecture?"
    print(f"\nQuery: {query}")
    
    # Test WITHOUT section boosting
    print("\n--- WITHOUT section boosting ---")
    results_no_boost = retrieve(query, index_data, top_k=3, return_metadata=True)
    for i, res in enumerate(results_no_boost, 1):
        print(f"\nResult {i}:")
        print(f"  Paper: {res['paper_id']}")
        print(f"  Score: {res['score']:.2f}")
        print(f"  Preview: {res['chunk'][:150]}...")
    
    # Test WITH section boosting (boost method sections for architecture questions)
    print("\n\n--- WITH section boosting (method sections x1.5) ---")
    section_boosts = {"method": 1.5, "introduction": 0.8}
    results_with_boost = retrieve(query, index_data, top_k=3, return_metadata=True, section_boosts=section_boosts)
    for i, res in enumerate(results_with_boost, 1):
        print(f"\nResult {i}:")
        print(f"  Paper: {res['paper_id']}")
        print(f"  Score: {res['score']:.2f}")
        print(f"  Preview: {res['chunk'][:150]}...")