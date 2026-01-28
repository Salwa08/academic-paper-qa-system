from typing import Any, Dict, List, Optional
from sentence_transformers import SentenceTransformer
import chromadb
from pathlib import Path
import json
from src.data.paper_processor import process_papers, retrieve
from src.data.chunking import SimpleChunker , SectionAwareChunker
from utils.base_qa_model import BaseQAModel
from utils.query_logger import QueryLogger


class HybridQA(BaseQAModel):
    def __init__(self, enable_logging: bool = True):
        super().__init__(name="Hybrid_BM25_Dense", description="Hybrid search combining BM25 and Sentence-BERT embeddings")
        self.model = None
        self.collection = None
        self.chunks_data = None
        self.bm25_data = None
        self.chunk_to_metadata = None
        self.query_logger = QueryLogger() if enable_logging else None

    def initialize(self, top_k: int = 5, chroma_path: str = "./chroma_ai_papers", chunker=None, force_reprocess: bool = False, **kwargs):
        self.top_k = top_k
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        
        if chunker is None:
            chunker = SectionAwareChunker()
        chunker_type = "section_aware" if isinstance(chunker, SectionAwareChunker) else "simple"
        
        chroma_path = f"{chroma_path}_{chunker_type}"
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name="ai_papers",
            metadata={"hnsw:space": "cosine"}  
        )
        
        self.bm25_data = process_papers(force_reprocess=force_reprocess, chunker=chunker)
        
        chunks_path = Path(f"data/processed/processed_chunks_{chunker_type}.json")
        if not chunks_path.exists():
            raise FileNotFoundError(f"Processed chunks file not found at {chunks_path}. Please run the paper processing step first.")
        
        with open(chunks_path, 'r', encoding="utf-8") as f:
            data = json.load(f)
        self.chunks_data = data['chunks']
        self.chunk_to_metadata = {chunk['chunk']: {'paper_id': chunk['paper_id'], 'section_type': chunk.get('section_type', 'other')} for chunk in self.chunks_data}
        if self.collection.count() == 0:
            documents = [chunk['chunk'] for chunk in self.chunks_data]
            metadatas = [{'paper_id': chunk['paper_id'], 'section_type': chunk.get('section_type', 'other')} for chunk in self.chunks_data]
            ids = [f"chunk_{i}" for i in range(len(self.chunks_data))]

            self.logger.info(f"Generating embeddings for {len(documents)} chunks...")
            embeddings = self.model.encode(documents).tolist()
            self.collection.add(documents=documents, metadatas=metadatas, ids=ids, embeddings=embeddings)

        self.is_initialized = True
    

    def answer_question(self, question: str, top_k: Optional[int] = None, alpha: float = 0.4, beta: float = 0.6, section_boosts: Optional[Dict[str, float]] = None, **kwargs):
        k = top_k or self.top_k
        
        # Retrieve more candidates for better fusion
        retrieve_k = max(k * 3, 15)
        
        # Dense Retrieval
        query_embedding = self.model.encode([question]).tolist()[0]
        results = self.collection.query(query_embeddings=[query_embedding], n_results=retrieve_k)
        
        dense_chunks = [
            {'chunk': doc, 'metadata': meta, 'distance': dist} 
            for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0], results['distances'][0])
        ]
        dense_scores = {chunk['chunk']: 1 - chunk['distance'] for chunk in dense_chunks}
        
        # BM25 Retrieval
        bm25_results = retrieve(question, self.bm25_data, top_k=retrieve_k, return_metadata=True)
        bm25_scores = {res['chunk']: res['score'] for res in bm25_results}
        
        # Fuse scores
        fused_results = []
        all_chunks = set(dense_scores.keys()) | set(bm25_scores.keys())
        for chunk_text in all_chunks:
            dense_score = dense_scores.get(chunk_text, 0.0)
            bm25_score = bm25_scores.get(chunk_text, 0.0)
            fused_score = alpha * bm25_score + beta * dense_score
            
            # Apply section boosting if provided
            if section_boosts:
                metadata = self.chunk_to_metadata.get(chunk_text, {'paper_id': 'unknown', 'section_type': 'other'})
                section_type = metadata.get('section_type', 'other')
                boost = section_boosts.get(section_type, 1.0)
                fused_score *= boost
            
            metadata = self.chunk_to_metadata.get(chunk_text, {'paper_id': 'unknown', 'section_type': 'other'})
            fused_results.append({
                'chunk': chunk_text,
                'metadata': metadata,
                'fused_score': fused_score
            })
        
        fused_results.sort(key=lambda x: x['fused_score'], reverse=True)
        top_results = fused_results[:k]
        
        answer = self._format_answer([res['chunk'] for res in top_results])
        
        if self.query_logger:
            self.query_logger.log_query(
                question=question,
                retrieved_chunks=top_results,
                response_time=0.0,
                top_k=k,
                success=True
            )
        
        return {
            "question": question,
            "answer": answer,
            "retrieved_chunks": top_results,
            "num_chunks": len(top_results),
            "approach": self.name,
            "success": True
        }



    def _format_answer(self, chunks: List[str]) -> str:
        return chunks[0] if chunks else "No relevant information found."
    
    def get_model_config(self):
        return {"name": self.name, "top_k": self.top_k, "embedding_model": "all-MiniLM-L6-v2"}



def main():
    """Test the Hybrid search Q&A system."""
    print("=" * 60)

    # Initialize
    qa_model = HybridQA()
    qa_model.initialize(top_k=3, chroma_path="./chroma_ai_papers", chunker_type="section_aware")

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

    # print stats
    print("\n" + "=" * 60)
    print("Performance Summary")
    print("=" * 60)
    stats = qa_model.get_performance_stats()
    print(f"Total queries: {qa_model.total_queries}")
    print(f"Average response time: {stats.get('avg_response_time', 0):.3f}s")
    print(f"Model config: {qa_model.get_model_config()}")

if __name__ == "__main__":
    main()






