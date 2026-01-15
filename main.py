"""
Main entry point for the Academic Paper Q&A System.
Ugly MVP - Phase 1: BM25-based retrieval on BERT paper.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(src_path))

# Import directly - note folder is 01_keyword_search
sys.path.insert(0, str(src_path / "approaches" / "01_keyword_search"))
from keyword_qa import KeywordSearchQA


def main():
    """
    Run the ugly MVP Q&A system.
    
    Phase 1 Goal: Demonstrate BM25-based retrieval working end-to-end
    on the BERT paper with a simple interactive Q&A interface.
    """
    print("\n" + "=" * 70)
    print("📚 ACADEMIC PAPER Q&A SYSTEM - UGLY MVP (Phase 1)")
    print("=" * 70)
    print("Approach: BM25 Keyword Search")
    print("Dataset: BERT paper (1810.04805)")
    print("=" * 70 + "\n")
    
    # Initialize the BM25 Q&A system
    print("🔧 Initializing BM25 retrieval system...")
    qa_system = KeywordSearchQA()
    qa_system.initialize(top_k=3)
    
    print("\n✅ System ready! You can now ask questions about the BERT paper.\n")
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
    
    print(f"Total questions asked: {qa_system.total_queries}")
    print(f"Average response time: {stats.get('avg_response_time', 0):.3f}s")
    print(f"Chunks indexed: {config['num_chunks_indexed']}")
    print(f"Retrieval algorithm: {config['algorithm']}")
    print("=" * 70)
    print("\n✨ Thanks for using the Q&A system! MVP Phase 1 complete.\n")


if __name__ == "__main__":
    main()
