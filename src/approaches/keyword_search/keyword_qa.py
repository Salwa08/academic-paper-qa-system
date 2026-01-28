"""
Keyword-based Q&A using BM25 retrieval.
Approach 1: Traditional sparse retrieval using BM25 algorithm.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import sys

# Add src to path for imports
src_path = Path(__file__).resolve().parent.parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from utils.base_qa_model import BaseQAModel
from data.paper_processor import process_papers, retrieve
from utils.query_logger import QueryLogger


class KeywordSearchQA(BaseQAModel):
    """BM25-based keyword search Q&A system."""
    
    def __init__(self, enable_logging: bool = True):
        super().__init__(
            name="BM25_Keyword_Search",
            description="Traditional information retrieval using BM25 ranking algorithm"
        )
        self.index_data = None
        self.top_k = 5
        self.query_logger = QueryLogger() if enable_logging else None
    
    def initialize(self, top_k: int = 5, force_reprocess: bool = False, chunker=None, **kwargs) -> None:
        """
        Initialize the BM25 index.
        
        Args:
            top_k: Number of top results to retrieve
            force_reprocess: Whether to force reprocessing of papers
            chunker: Optional chunker instance to use for processing papers
        """
        self.logger.info("Initializing BM25 Keyword Search model...")
        self.top_k = top_k
        
        try:
            # Load or build the BM25 index
            self.index_data = process_papers(force_reprocess=force_reprocess, chunker=chunker)
            
            if not self.index_data or not self.index_data.get("chunks"):
                raise ValueError("No chunks available in the index")
            
            num_chunks = len(self.index_data["chunks"])
            self.logger.info(f"✓ BM25 index loaded with {num_chunks} chunks")
            self.is_initialized = True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize BM25 index: {e}")
            raise
    
    def answer_question(self, question: str, top_k: Optional[int] = None, section_boosts: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """
        Answer a question using BM25 retrieval.
        
        Args:
            question: The question to answer
            top_k: Number of results to return (overrides default)
            section_boosts: Optional dictionary mapping section types to boost factors
        Returns:
            Dictionary with answer, retrieved chunks, and metadata
        """
        if not self.is_initialized:
            raise RuntimeError("Model not initialized. Call initialize() first.")
        
        k = top_k if top_k is not None else self.top_k
        
        try:
            # Retrieve relevant chunks with metadata for logging
            retrieved_chunks = retrieve(question, self.index_data, top_k=k, return_metadata=True, section_boosts=section_boosts)
            
            # Extract just text for formatting answer
            chunk_texts = [chunk["chunk"] if isinstance(chunk, dict) else chunk for chunk in retrieved_chunks]
            
            # For MVP, we just concatenate top chunks as the "answer"
            # In production, you'd use an extractive model or LLM here
            answer = self._format_answer(chunk_texts)
            
            self.total_queries += 1
            
            result = {
                "question": question,
                "answer": answer,
                "retrieved_chunks": retrieved_chunks,
                "num_chunks": len(retrieved_chunks),
                "approach": self.name,
                "success": True
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error answering question: {e}")
            return {
                "question": question,
                "answer": "",
                "retrieved_chunks": [],
                "num_chunks": 0,
                "approach": self.name,
                "success": False,
                "error": str(e)
            }
    
    def _format_answer(self, chunks: List[str]) -> str:
        """
        Format retrieved chunks into a simple answer.
        
        For the ugly MVP, we just return the top chunk.
        In production, you'd do proper answer extraction.
        """
        if not chunks:
            return "No relevant information found."
        
        # Return the most relevant chunk as the answer
        return chunks[0]
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            "name": self.name,
            "description": self.description,
            "top_k": self.top_k,
            "num_chunks_indexed": len(self.index_data["chunks"]) if self.index_data else 0,
            "algorithm": "BM25Okapi"
        }


def main():
    """Test the keyword search Q&A system."""
    print("=" * 60)
    print("BM25 Keyword Search Q&A - Ugly MVP Test")
    print("=" * 60)
    
    # Initialize model
    qa_model = KeywordSearchQA()
    qa_model.initialize(top_k=3)
    
    # Test questions
    test_questions = [
        "What is BERT?",
        "How does the masked language model work?",
        "What is the attention mechanism?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'=' * 60}")
        print(f"Question {i}: {question}")
        print("=" * 60)
        
        result = qa_model.answer_with_timing(question, top_k=3)
        
        print(f"\nAnswer (top chunk):")
        print("-" * 60)
        print(result["answer"][:500] + "...")
        
        print(f"\n Retrieved {result['num_chunks']} chunks in {result['response_time']:.3f}s")
    
    # Print stats
    print("\n" + "=" * 60)
    print("Performance Summary")
    print("=" * 60)
    stats = qa_model.get_performance_stats()
    print(f"Total queries: {qa_model.total_queries}")
    print(f"Average response time: {stats.get('avg_response_time', 0):.3f}s")
    print(f"Model config: {qa_model.get_model_config()}")


if __name__ == "__main__":
    main()
