"""
Main entry point for the Academic Paper Q&A System.
Phase 2: Hybrid BM25 + Dense retrieval with section-aware chunking.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(src_path))

# Import approaches
sys.path.insert(0, str(src_path / "approaches" / "keyword_search"))
sys.path.insert(0, str(src_path / "approaches" / "hybrid_search"))
from keyword_qa import KeywordSearchQA
from hybrid_qa import HybridQA
from src.data.chunking import SimpleChunker, SectionAwareChunker


def main():
    """
    Run the Academic Paper Q&A System.
    
    Phase 2: Support for both BM25 and Hybrid (BM25 + Dense) retrieval
    with choice of chunking strategies.
    """
    print("\n" + "=" * 70)
    print("📚 ACADEMIC PAPER Q&A SYSTEM - PHASE 2 COMPLETE")
    print("=" * 70)
    print("Available Approaches:")
    print("  1. BM25 Keyword Search (Phase 1)")
    print("  2. Hybrid Search (BM25 + Dense embeddings)")
    print("=" * 70 + "\n")
    
    # Choose approach
    while True:
        try:
            approach_choice = input("Choose approach (1 or 2): ").strip()
            if approach_choice == "1":
                approach_name = "BM25 Keyword Search"
                qa_system = KeywordSearchQA()
                break
            elif approach_choice == "2":
                approach_name = "Hybrid BM25 + Dense"
                qa_system = HybridQA()
                break
            else:
                print("Please enter 1 or 2.")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            return
    
    print(f"\nSelected: {approach_name}")
    print("\n" + "=" * 70)
    print("Available Chunking Strategies:")
    print("  1. Simple Chunking (500 tokens, 50 overlap)")
    print("  2. Section-Aware Chunking (preserves markdown structure)")
    print("=" * 70 + "\n")
    
    # Choose chunking
    while True:
        try:
            chunking_choice = input("Choose chunking (1 or 2): ").strip()
            if chunking_choice == "1":
                chunker = SimpleChunker()
                chunking_name = "Simple Chunking"
                break
            elif chunking_choice == "2":
                chunker = SectionAwareChunker()
                chunking_name = "Section-Aware Chunking"
                break
            else:
                print("Please enter 1 or 2.")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            return
    
    print(f"\nSelected: {chunking_name}")
    print("Dataset: 3 academic papers (BERT, Attention Is All You Need, GraphIE)")
    print("\n🔧 Initializing system...")
    qa_system.initialize(top_k=3, chunker=chunker)
    
    print("\n✅ System ready! You can now ask questions about the academic papers.\n")
    print("Example questions:")
    print("  - What is BERT?")
    print("  - How does masked language modeling work?")
    print("  - What is the attention mechanism?")
    print("  - What datasets were used for evaluation?")
    print("\nType 'quit' or 'exit' to end the session.\n")
    
    # Interactive Q&A loop
    while True:
        try:
            # Get user question
            question = input("❓ Your question: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit', 'q']:
                break
            
            # Get answer
            print("\n🔍 Searching...")
            result = qa_system.answer_with_timing(question, top_k=3)
            
            if result.get("success", False):
                print("\n" + "=" * 70)
                print("📝 ANSWER")
                print("=" * 70)
                
                # Show the top retrieved chunk
                answer = result["answer"]
                if len(answer) > 600:
                    print(answer[:600] + "...\n")
                else:
                    print(answer + "\n")
                
                # Show metadata
                print("-" * 70)
                print(f"📊 Retrieved {result['num_chunks']} relevant chunks")
                print(f"⏱️  Response time: {result['response_time']:.3f}s")
                print("=" * 70 + "\n")
            else:
                print(f"\n❌ Error: {result.get('error', 'Unknown error')}\n")
                
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user.")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
    
    # Final statistics
    print("\n" + "=" * 70)
    print("📊 SESSION SUMMARY")
    print("=" * 70)
    stats = qa_system.get_performance_stats()
    config = qa_system.get_model_config()
    
    print(f"Approach: {approach_name}")
    print(f"Chunking: {chunking_name}")
    print(f"Total questions asked: {qa_system.total_queries}")
    print(f"Average response time: {stats.get('avg_response_time', 0):.3f}s")
    if 'num_chunks_indexed' in config:
        print(f"Chunks indexed: {config['num_chunks_indexed']}")
    if 'embedding_model' in config:
        print(f"Embedding model: {config['embedding_model']}")
    print("=" * 70)
    print("\n✨ Thanks for using the Q&A system! Phase 2 complete.\n")


if __name__ == "__main__":
    main()
