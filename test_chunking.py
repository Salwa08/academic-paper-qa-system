"""Test script for section-aware chunking."""

from src.data.chunking import SectionAwareChunker
from pathlib import Path

def test_section_aware_chunking():
    # Load one of your processed markdown papers
    paper_path = Path("data/processed/1810.04805/1810.04805/1810.04805.md")
    
    if not paper_path.exists():
        print(f"Error: {paper_path} not found")
        return
    
    with open(paper_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Testing with paper: {paper_path}")
    print(f"Text length: {len(text)} characters\n")
    
    # Create chunker
    chunker = SectionAwareChunker(chunk_size=500, overlap=50)
    
    # Get chunks
    chunks = chunker.chunk_text(text)
    
    print(f"Total chunks: {len(chunks)}\n")
    
    # Analyze section types
    section_counts = {}
    for chunk in chunks:
        section_type = chunk['section_type']
        section_counts[section_type] = section_counts.get(section_type, 0) + 1
    
    print("Section distribution:")
    for section_type, count in sorted(section_counts.items()):
        print(f"  {section_type}: {count} chunks")
    
    # Show first 10 chunks in detail
    print("\nFirst 10 chunks (detailed):")
    for i, chunk in enumerate(chunks[:10], 1):
        print(f"\n--- Chunk {i} ---")
        print(f"Section type: {chunk['section_type']}")
        print(f"Metadata: {chunk['metadata']}")
        print(f"Content preview: {chunk['content'][:150]}...")
    
    # Analyze "other" chunks to see what sections they're from
    print("\n\nSample 'other' chunks (first 5):")
    other_chunks = [c for c in chunks if c['section_type'] == 'other']
    for i, chunk in enumerate(other_chunks[:5], 1):
        # Debug: extract what section name was detected
        section_name = chunker._extract_section_name_from_metadata(chunk['metadata'])
        print(f"\n--- Other Chunk {i} ---")
        print(f"Extracted section name: '{section_name}'")
        print(f"Metadata: {chunk['metadata']}")
        print(f"Content preview: {chunk['content'][:100]}...")

if __name__ == "__main__":
    test_section_aware_chunking()
