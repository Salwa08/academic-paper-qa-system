"""BM25 retrieval backend for benchmark chunks."""

from collections import Counter
from math import log
from typing import List

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

from src.data.text_preprocessor import simple_tokenize
from src.schemas import Chunk, SearchResult


class _FallbackBM25Okapi:
    """Small BM25 fallback for smoke tests when rank-bm25 is not installed."""

    def __init__(self, tokenized_corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus = tokenized_corpus
        self.k1 = k1
        self.b = b
        self.doc_freqs = [Counter(doc) for doc in tokenized_corpus]
        self.doc_lens = [len(doc) for doc in tokenized_corpus]
        self.avgdl = sum(self.doc_lens) / len(self.doc_lens) if self.doc_lens else 0.0
        document_frequency = Counter()
        for doc in tokenized_corpus:
            document_frequency.update(set(doc))
        total_docs = len(tokenized_corpus)
        self.idf = {
            term: log(1 + (total_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in document_frequency.items()
        }

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        scores = []
        for freqs, doc_len in zip(self.doc_freqs, self.doc_lens):
            score = 0.0
            for term in query_tokens:
                tf = freqs.get(term, 0)
                if tf == 0:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * doc_len / (self.avgdl or 1.0))
                score += self.idf.get(term, 0.0) * (tf * (self.k1 + 1)) / denom
            scores.append(score)
        return scores


class BM25Retriever:
    retriever_name = "bm25"

    def __init__(self, chunks: List[Chunk]):
        if not chunks:
            raise ValueError("BM25Retriever requires at least one chunk")
        self.chunks = chunks
        self._tokenized_corpus = [simple_tokenize(chunk.text) for chunk in chunks]
        bm25_cls = BM25Okapi or _FallbackBM25Okapi
        self._bm25 = bm25_cls(self._tokenized_corpus)

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        tokenized_query = simple_tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results: List[SearchResult] = []
        for rank, idx in enumerate(top_indices, 1):
            chunk = self.chunks[idx]
            results.append(SearchResult(
                chunk_id=chunk.chunk_id,
                paper_id=chunk.paper_id,
                text=chunk.text,
                rank=rank,
                score=float(scores[idx]),
                retriever=self.retriever_name,
                metadata={
                    "chunker": chunk.chunker,
                    "section_type": chunk.section_type,
                    "section_path": chunk.section_path,
                },
            ))
        return results
