"""Paper processor for academic papers - handles BM25 indexing and retrieval."""
import os
from pathlib import Path
from typing import List, Dict, Any
import sys

from rank_bm25 import BM25Okapi
import re
import json

# Add src to path
src_path = Path(__file__).resolve().parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from data.text_preprocessor import clean_text, simple_tokenize

# Use absolute paths relative to project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
CHUNKS_JSON = DATA_DIR / "processed_chunks.json"
INDEX_DIR = DATA_DIR / "bm25_index"

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    words = simple_tokenize(text)
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk_words = words[i:i + chunk_size]
        if len(chunk_words) < 30:
            continue
        chunks.append(" ".join(chunk_words))
    return chunks

def process_papers(force_reprocess: bool = False) -> Dict[str, Any]:
    """
    Main function:
    - Reads raw paper texts from DATA_DIR
    - Cleans and chunks the texts
    - Saves processed chunks to CHUNKS_JSON
    - Builds BM25 index and saves to INDEX_DIR
    - Returns list of processed chunks 
    """
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
        cleaned_text = clean_text(raw)
        paper_chunks = chunk_text(cleaned_text)
        for chunk in paper_chunks:
            chunks.append({
                "paper_id": paper_file.stem,  # Use file stem as paper ID
                "chunk": chunk
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

def retrieve(query: str, index_data: Dict[str, Any], top_k: int = 5, return_metadata: bool = False):
    """
    Retrieve relevant chunks for a query using BM25.
    
    Args:
        query: Search query
        index_data: BM25 index data
        top_k: Number of results to return
        return_metadata: If True, return dicts with metadata; if False, return just text
        
    Returns:
        List of chunks (strings if return_metadata=False, dicts if True)
    """
    tokenized_query = simple_tokenize(query)

    bm25 = index_data["bm25"]
    scores = bm25.get_scores(tokenized_query)
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
    index_data = process_papers(force_reprocess=True)
    query = "What problem does the attention mechanism solve?"
    results = retrieve(query, index_data, top_k=3)
    for i, res in enumerate(results):
        print(f"Result {i+1}:\n{res}\n")