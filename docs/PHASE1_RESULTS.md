# Phase 1 Complete: BM25 Baseline Results

**Date:** January 15, 2026  
**Status:** ✅ Complete

---

## Phase 1 Goals

Build a functional BM25-based keyword search system as a baseline for the Q&A project.

**Objectives:**

- ✅ Implement BM25 retrieval algorithm
- ✅ Process 3 research papers into searchable chunks
- ✅ Build interactive CLI interface
- ✅ Create evaluation framework
- ✅ Implement query logging

---

## Implementation Details

### Papers Processed

1. **1810.04805** - BERT: Pre-training of Deep Bidirectional Transformers
2. **1706.03762** - Attention Is All You Need
3. **1810.13083** - GraphIE: A Graph-Based Framework for Information Extraction

**Total chunks:** 49  
**Chunk size:** 500 tokens with 50 token overlap

### Technical Stack

- `rank-bm25` - BM25Okapi algorithm
- `marker-pdf` - PDF to Markdown extraction
- Python 3.12.4

### Architecture

```
Query → Tokenization → BM25 Scoring → Top-K Retrieval → Format Answer
                          ↓
                    Query Logger
```

---

## Evaluation Results

**Test Set:** 10 questions (4 BERT, 4 Attention, 2 GraphIE)

### Metrics

| Metric                | Value  | Notes                                           |
| --------------------- | ------ | ----------------------------------------------- |
| **Hit Rate@3**        | 70%    | 7/10 questions found relevant keywords in top-3 |
| **Avg Position**      | 1.43   | When found, relevant info ranked 1st or 2nd     |
| **Avg Response Time** | 0.001s | ~1 millisecond per query                        |
| **Papers Retrieved**  | All 3  | Good coverage across dataset                    |

### Per-Paper Performance

| Paper                  | Questions | Hits | Hit Rate |
| ---------------------- | --------- | ---- | -------- |
| BERT (1810.04805)      | 4         | 3    | 75%      |
| Attention (1706.03762) | 4         | 3    | 75%      |
| GraphIE (1810.13083)   | 2         | 1    | 50%      |

---

## Key Learnings

### ✅ What Worked Well

1. **Fast retrieval** - Sub-millisecond response times make it practical
2. **Good keyword matching** - 70% hit rate shows BM25 is effective for term-based queries
3. **Early ranking** - Position 1.43 means relevant chunks rank at the top when found
4. **Simple but functional** - Minimal dependencies, easy to run and debug

### ⚠️ Limitations Identified

1. **Semantic gap** - 30% miss rate highlights keyword-only limitations
2. **Paraphrase sensitivity** - Questions worded differently than paper text struggle
3. **No context awareness** - Can't handle "What does it solve?" without explicit keywords
4. **Binary relevance** - Uses keyword proxy instead of proper ground truth labels

### 🎯 Insights for Phase 2

- **Semantic search needed** - Dense embeddings will help with paraphrased questions
- **Hybrid approach** - Combine BM25 (good precision) with semantic search (better recall)
- **Re-ranking** - A cross-encoder could improve top-3 ordering
- **Better chunking** - Section-aware chunks could preserve more context

---

## Code Highlights

### Core Components

1. **[keyword_qa.py](../src/approaches/01_keyword_search/keyword_qa.py)**

   - BM25 Q&A implementation
   - Inherits from BaseQAModel
   - Automatic query logging

2. **[paper_processor.py](../src/data/paper_processor.py)**

   - Chunk creation (500 tokens, 50 overlap)
   - BM25 index building
   - Metadata extraction

3. **[query_logger.py](../src/utils/query_logger.py)**

   - JSONL logging format
   - Aggregate statistics
   - Paper distribution tracking

4. **[evaluate_bm25.py](../evaluation/evaluate_bm25.py)**
   - Automated evaluation pipeline
   - Keyword-based relevance proxy
   - Results saved to JSON

### Example Usage

```python
from src.approaches.keyword_search.keyword_qa import KeywordSearchQA

qa = KeywordSearchQA()
qa.initialize(top_k=3)

result = qa.answer_with_timing("What is BERT?")
# Returns top 3 chunks in ~1ms
# Automatically logged to logs/query_log.jsonl
```

---

## Files Added in Phase 1

**Core Implementation:**

- `src/approaches/01_keyword_search/keyword_qa.py`
- `src/data/paper_processor.py` (enhanced)
- `src/data/text_preprocessor.py` (enhanced)
- `src/data/download_and_extract.py` (enhanced)
- `src/utils/query_logger.py` (new)

**Evaluation:**

- `evaluation/metrics_bm25.py`
- `evaluation/evaluate_bm25.py`
- `evaluation/test_questions.json`
- `evaluation/view_logs.py`
- `evaluation/results/bm25_results.json` (generated)

**Documentation:**

- `docs/MVP_PHASE1_COMPLETE.md` (this file)
- `QUICKSTART.md`
- `README.md` (updated with roadmap)

---

## Query Log Sample

```json
{
  "timestamp": "2026-01-15T20:26:00",
  "question": "What is BERT?",
  "top_k": 3,
  "num_retrieved": 3,
  "retrieved_papers": ["1810.04805"],
  "response_time": 0.001,
  "success": true
}
```

**Current Stats (4 queries logged):**

- Success rate: 100%
- Most retrieved paper: BERT (1810.04805)
- Avg response time: 0.8ms

---

## Next Steps: Phase 2

### Planned Improvements

1. **Section-aware chunking**

   - Preserve markdown structure from marker-pdf
   - Include section headers in chunk metadata
   - Better citation with section references

2. **Dense embeddings**

   - Add Sentence-BERT (all-MiniLM-L6-v2)
   - Build vector store (ChromaDB or FAISS)
   - Semantic similarity search

3. **Hybrid retrieval**

   - BM25 + semantic search
   - Score fusion (weighted average or RRF)
   - Compare vs BM25-only baseline

4. **Expand dataset**
   - Add 7-10 more papers
   - More diverse test questions
   - Better evaluation metrics

### Expected Phase 2 Results

- Hit Rate: 85-95% (up from 70%)
- Better semantic matching
- Improved handling of paraphrased questions

---

## Conclusion

Phase 1 achieved its goal: a working, evaluated baseline system. The 70% hit rate and 1ms response time prove BM25 is a solid foundation. The 30% miss rate and semantic limitations provide clear motivation for Phase 2's hybrid approach.

**Phase 1: Mission accomplished.** ✅  
**Next: Phase 2 - Hybrid retrieval with semantic search.** 🚀

---

**Built by:** Salwa Khattami  
**Build log:** [X/Twitter](https://x.com/your-handle)  
**Repo:** [github.com/Salwa08/academic-paper-qa-system](https://github.com/Salwa08/academic-paper-qa-system)
