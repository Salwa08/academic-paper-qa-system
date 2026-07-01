"""Dense retrieval backend for benchmark chunks.

Uses SentenceTransformers + Chroma when the local environment can import them.
Falls back to a lightweight in-memory TF-style retriever when Torch/Chroma are
unavailable so the benchmark remains reproducible on constrained machines.
"""

import os
from collections import Counter
from math import sqrt
from pathlib import Path
from typing import Dict, List

from src.data.text_preprocessor import simple_tokenize
from src.schemas import Chunk, SearchResult

PROJECT_ROOT = Path(__file__).resolve().parents[2]

chromadb = None
SentenceTransformer = None

if os.getenv("PAPERQA_USE_SENTENCE_TRANSFORMERS") == "1":
    try:
        import chromadb as _chromadb
        from sentence_transformers import SentenceTransformer as _SentenceTransformer
        chromadb = _chromadb
        SentenceTransformer = _SentenceTransformer
    except Exception:
        chromadb = None
        SentenceTransformer = None


class DenseRetriever:
    retriever_name = "dense"

    def __init__(
        self,
        chunks: List[Chunk],
        chunker_name: str,
        model_name: str = "all-MiniLM-L6-v2",
        persist_root: Path | None = None,
    ):
        if not chunks:
            raise ValueError("DenseRetriever requires at least one chunk")
        self.chunks = chunks
        self.chunker_name = chunker_name
        self.model_name = model_name
        self.persist_path = persist_root or (PROJECT_ROOT / f"chroma_benchmark_{chunker_name}")
        self._fallback_doc_vectors: List[Counter] = []
        self._fallback_doc_norms: List[float] = []

        if chromadb is None or SentenceTransformer is None:
            self.mode = "fallback_lexical_vectors"
            self._prepare_fallback_vectors()
        else:
            self.mode = "sentence_transformers_chroma"
            try:
                self.model = SentenceTransformer(model_name)
                self.client = chromadb.PersistentClient(path=str(self.persist_path))
                self.collection_name = f"benchmark_{chunker_name}"
                self.collection = self._prepare_collection()
            except Exception:
                self.mode = "fallback_lexical_vectors"
                self._prepare_fallback_vectors()

    def _prepare_fallback_vectors(self) -> None:
        self._fallback_doc_vectors = [Counter(simple_tokenize(chunk.text)) for chunk in self.chunks]
        self._fallback_doc_norms = [sqrt(sum(value * value for value in vec.values())) or 1.0 for vec in self._fallback_doc_vectors]

    def _prepare_collection(self):
        collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        expected_ids = [chunk.chunk_id for chunk in self.chunks]
        needs_rebuild = collection.count() != len(expected_ids)
        if not needs_rebuild and expected_ids:
            stored = collection.get(ids=expected_ids[: min(10, len(expected_ids))], include=[])
            needs_rebuild = len(stored.get("ids", [])) != min(10, len(expected_ids))
        if needs_rebuild:
            self.client.delete_collection(self.collection_name)
            collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            documents = [chunk.text for chunk in self.chunks]
            metadatas = [
                {
                    "chunk_id": chunk.chunk_id,
                    "paper_id": chunk.paper_id,
                    "chunker": chunk.chunker,
                    "section_type": chunk.section_type or "",
                    "section_path": chunk.section_path or "",
                }
                for chunk in self.chunks
            ]
            embeddings = self.model.encode(documents).tolist()
            collection.add(
                ids=expected_ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )
        return collection

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        if self.mode == "sentence_transformers_chroma":
            return self._search_chroma(query, top_k)
        return self._search_fallback(query, top_k)

    def _search_chroma(self, query: str, top_k: int) -> List[SearchResult]:
        query_embedding = self.model.encode([query]).tolist()[0]
        raw = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
        results: List[SearchResult] = []
        ids = raw.get("ids", [[]])[0]
        docs = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]
        for rank, (chunk_id, text, metadata, distance) in enumerate(zip(ids, docs, metadatas, distances), 1):
            results.append(SearchResult(
                chunk_id=chunk_id,
                paper_id=metadata.get("paper_id", "unknown"),
                text=text,
                rank=rank,
                score=float(1 - distance),
                retriever=self.retriever_name,
                metadata={**metadata, "dense_mode": self.mode},
            ))
        return results

    def _search_fallback(self, query: str, top_k: int) -> List[SearchResult]:
        query_vec = Counter(simple_tokenize(query))
        query_norm = sqrt(sum(value * value for value in query_vec.values())) or 1.0
        scored = []
        for idx, doc_vec in enumerate(self._fallback_doc_vectors):
            dot = sum(weight * doc_vec.get(term, 0) for term, weight in query_vec.items())
            score = dot / (query_norm * self._fallback_doc_norms[idx])
            scored.append((idx, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        results: List[SearchResult] = []
        for rank, (idx, score) in enumerate(scored[:top_k], 1):
            chunk = self.chunks[idx]
            results.append(SearchResult(
                chunk_id=chunk.chunk_id,
                paper_id=chunk.paper_id,
                text=chunk.text,
                rank=rank,
                score=float(score),
                retriever=self.retriever_name,
                metadata={
                    "chunker": chunk.chunker,
                    "section_type": chunk.section_type,
                    "section_path": chunk.section_path,
                    "dense_mode": self.mode,
                },
            ))
        return results

