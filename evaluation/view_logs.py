"""
View and analyze query logs.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(src_path))

from utils.query_logger import QueryLogger


def main():
    """Display query log statistics."""
    logger = QueryLogger()
    
    # Get stats
    stats = logger.get_stats()
    
    print("\n" + "="*50)
    print("QUERY LOG STATISTICS")
    print("="*50)
    
    print(f"\nTotal queries: {stats['total_queries']}")
    
    if stats['total_queries'] > 0:
        print(f"Successful queries: {stats['successful_queries']}")
        print(f"Success rate: {stats['success_rate']:.2%}")
        print(f"Average response time: {stats['avg_response_time']:.4f}s")
        
        print("\n📚 Paper Retrieval Distribution:")
        for paper_id, count in sorted(stats['paper_distribution'].items(), 
                                      key=lambda x: x[1], reverse=True):
            print(f"  {paper_id}: {count} queries")
        
        print(f"\nMost retrieved paper: {stats['most_retrieved_paper']}")
        
        # Show recent queries
        print("\n" + "="*50)
        print("RECENT QUERIES (last 5)")
        print("="*50)
        
        recent_logs = logger.get_logs(limit=5)
        for i, log in enumerate(recent_logs, 1):
            print(f"\n{i}. {log['question']}")
            print(f"   Papers: {', '.join(log['retrieved_papers'])}")
            print(f"   Time: {log['response_time']}s | Retrieved: {log['num_retrieved']} chunks")
    else:
        print("\nNo queries logged yet. Run some queries to see stats!")
    
    print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    main()
