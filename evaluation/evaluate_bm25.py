import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json

# Add project paths for imports
approaches_path = PROJECT_ROOT / "src" / "approaches" / "keyword_search"
sys.path.insert(0, str(approaches_path))

from metrics_bm25 import evaluate_bm25
from keyword_qa import KeywordSearchQA

def evaluate_bm25_system(section_boosts: dict = None, chunker=None, force_reprocess: bool = False) -> dict:
    """Evaluate the BM25 Q&A system on test questions."""
    # Load configuration
    test_questions_path = Path(__file__).parent / "test_questions.json"

    # Load test questions
    with open(test_questions_path, 'r') as f:
        test_data = json.load(f)

    test_questions = [item['question'] for item in test_data]
    relevant_keywords = {item['question']: item.get('relevant_keywords', []) for item in test_data}
    # Initialize BM25 Q&A system
    qa_system = KeywordSearchQA()
    qa_system.initialize(top_k=3, chunker=chunker, force_reprocess=force_reprocess)
    # Evaluate
    metrics = evaluate_bm25(qa_system, test_questions, relevant_keywords, section_boosts=section_boosts)
    # Print summary
    print("BM25 Evaluation Results:")
    print("=" * 24)
    print(f"Questions evaluated: {metrics['total_questions']}")
    print(f"Hit Rate: {metrics['hit_rate']:.2f}")
    print(f"Avg Position: {metrics['avg_position']:.2f}")
    print(f"Avg response time: {metrics['avg_response_time']:.3f}s")
    return metrics

if __name__ == "__main__":
    from keyword_qa import KeywordSearchQA
    from src.data.chunking import SimpleChunker, SectionAwareChunker
    
    results = {}
    
    print("="*60)
    print("Phase 1: Simple Chunking (Baseline)")
    print("="*60)
    results['simple_chunking'] = evaluate_bm25_system(section_boosts=None, chunker=SimpleChunker(), force_reprocess=True)
    
    print("\n" + "="*60)
    print("Phase 2a: Section-Aware Chunking (NO boost)")
    print("="*60)
    results['section_aware_no_boost'] = evaluate_bm25_system(section_boosts=None, chunker=SectionAwareChunker(), force_reprocess=True)
    
    print("\n" + "="*60)
    print("Phase 2a: Section-Aware + Boosting")
    print("="*60)
    section_boosts = {"method": 1.5, "experiments": 1.3, "introduction": 0.8}
    results['section_aware_with_boost'] = evaluate_bm25_system(section_boosts=section_boosts, chunker=SectionAwareChunker())
    
    # Save combined results
    output_path = Path(__file__).parent / "results" / "bm25_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
