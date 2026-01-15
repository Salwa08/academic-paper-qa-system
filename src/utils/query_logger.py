"""
Query logging utility for tracking Q&A system usage and performance.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any


class QueryLogger:
    """Logs queries and retrieval results for analysis."""
    
    def __init__(self, log_file: str = "logs/query_log.jsonl"):
        """
        Initialize query logger.
        
        Args:
            log_file: Path to log file (JSONL format)
        """
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
    def log_query(self, 
                  question: str,
                  retrieved_chunks: List[Dict[str, Any]],
                  response_time: float,
                  top_k: int,
                  success: bool = True,
                  metadata: Dict[str, Any] = None):
        """
        Log a query and its results.
        
        Args:
            question: The user's question
            retrieved_chunks: List of retrieved chunks with paper_id and chunk text
            response_time: Time taken to retrieve (seconds)
            top_k: Number of chunks requested
            success: Whether retrieval was successful
            metadata: Additional metadata to log
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "top_k": top_k,
            "num_retrieved": len(retrieved_chunks),
            "retrieved_papers": self._extract_paper_ids(retrieved_chunks),
            "response_time": round(response_time, 4),
            "success": success
        }
        
        # Add optional metadata
        if metadata:
            log_entry["metadata"] = metadata
            
        # Append to JSONL file
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def _extract_paper_ids(self, chunks: List[Any]) -> List[str]:
        """Extract unique paper IDs from retrieved chunks."""
        paper_ids = []
        for chunk in chunks:
            # Handle both dict format and string format
            if isinstance(chunk, dict):
                paper_id = chunk.get('paper_id', 'unknown')
            else:
                # For string chunks, we can't extract paper_id
                paper_id = 'unknown'
            if paper_id not in paper_ids:
                paper_ids.append(paper_id)
        return paper_ids
    
    def get_logs(self, limit: int = None) -> List[Dict[str, Any]]:
        """
        Read logs from file.
        
        Args:
            limit: Maximum number of recent logs to return
            
        Returns:
            List of log entries
        """
        if not self.log_file.exists():
            return []
        
        logs = []
        with open(self.log_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    logs.append(json.loads(line))
        
        if limit:
            return logs[-limit:]
        return logs
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get aggregate statistics from logs.
        
        Returns:
            Dictionary with query statistics
        """
        logs = self.get_logs()
        
        if not logs:
            return {"total_queries": 0}
        
        total_queries = len(logs)
        successful_queries = sum(1 for log in logs if log.get('success', True))
        avg_response_time = sum(log['response_time'] for log in logs) / total_queries
        
        # Count paper distribution
        paper_counts = {}
        for log in logs:
            for paper_id in log.get('retrieved_papers', []):
                paper_counts[paper_id] = paper_counts.get(paper_id, 0) + 1
        
        return {
            "total_queries": total_queries,
            "successful_queries": successful_queries,
            "success_rate": successful_queries / total_queries,
            "avg_response_time": round(avg_response_time, 4),
            "paper_distribution": paper_counts,
            "most_retrieved_paper": max(paper_counts.items(), key=lambda x: x[1])[0] if paper_counts else None
        }
