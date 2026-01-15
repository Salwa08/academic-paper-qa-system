# Academic Paper Q&A System

A progressively smarter Question & Answer system for AI research papers, building from traditional sparse retrieval to advanced RAG systems. Learning and building in public.

**Public build log:** [X/Twitter threads](https://x.com/your-handle)

---

## 🎯 Project Vision

Building a production-quality Q&A system over research papers through iterative improvement:

- Start with simple, working baseline (BM25)
- Add semantic understanding (embeddings + hybrid search)
- Incorporate LLMs with grounding (RAG with citations)
- Optimize for production (re-ranking, agentic flows, multimodal)
- Deploy and monitor real usage

## 📍 Current Status: Phase 2 Starting

**Phase 1 Complete** ✅ | **Phase 2 In Progress** 🚧

---

## 🗺️ Roadmap & Progress

### Phase 1 – Ugly MVP: BM25 Keyword Search ✅ (Completed Jan 2026)

**Goal:** Traditional sparse retrieval baseline — fast and functional.

**Implementation:**

- BM25 scoring algorithm (rank-bm25)
- Simple text chunking (500 tokens, 50 overlap)
- Basic preprocessing (lowercase, tokenization)
- 3 papers indexed: BERT, Attention Is All You Need, GraphIE
- 49 chunks total

**Evaluation Results** (10 test questions, keyword-proxy relevance):

- **Hit Rate@3:** 70% — 7/10 questions found relevant keywords in top-3 chunks
- **Avg Position:** 1.43 — When found, relevant info appeared very early
- **Avg Response Time:** ~1ms — Blazing fast retrieval

**Key Files:**

- [src/approaches/01_keyword_search/keyword_qa.py](src/approaches/01_keyword_search/keyword_qa.py)
- [evaluation/metrics_bm25.py](evaluation/metrics_bm25.py)
- [evaluation/test_questions.json](evaluation/test_questions.json)

**Outcome:** Fast, decent baseline. Relevant chunks usually rank early when hit. 30% misses highlight need for semantic understanding.

**How to Run:**

```bash
# Interactive CLI
python main.py

# Run evaluation
python evaluation/evaluate_bm25.py

# View query logs
python evaluation/view_logs.py
```

---

### Phase 2 – Sectioned & Hybrid RAG 🚧 (Starting Now)

**Goal:** Smarter chunking + hybrid retrieval for better semantic matching.

**Planned Improvements:**

- **Section-aware chunking:** Preserve markdown structure from marker-pdf extraction
- **Dense embeddings:** Sentence-BERT (all-MiniLM-L6-v2) for semantic search
- **Hybrid retrieval:** BM25 + semantic search with score fusion
- **Basic re-ranking:** Optional cross-encoder for top results
- **Better citations:** Chunk metadata with section/page references

**Expected Outcomes:**

- Hit Rate: 85-95% (up from 70%)
- Better handling of semantic/paraphrased questions
- Improved answer grounding with section context

**Status:** In progress — starting with section-aware chunking + dense embedding integration

**Branch:** `phase-2-hybrid-sectioned`

---

### Phase 3 – Advanced RAG + Agentic Extraction (Planned)

**Goal:** Production-quality optimizations with multimodal support.

**Key Techniques:**

- Agentic parsing (tables, figures, equations — using DeepLearning.AI Document AI patterns)
- Advanced re-ranking (Cohere Rerank or cross-encoder)
- Context optimization (query rewriting, compression)
- Full citation tracking (section + page + figure refs)
- Simple agentic flow (LangGraph for query decomposition)

**Expected Outcomes:**

- Drastically lower hallucinations
- Grounded, cited answers
- Multimodal support (handle figures/tables/equations)

**Status:** Planned after Phase 2 evaluation

---

### Phase 4 – Extractive + Fine-tuned (Longer Term)

- Extractive Q&A with BERT-style span extraction
- Domain-specific fine-tuning on paper Q&A pairs
- **Status:** Future exploration

---

### Phase 5 – Full Production RAG (End Goal)

- Complete pipeline: Simple RAG → Advanced RAG → Fine-tuned
- Deployment (HuggingFace Spaces / FastAPI / mobile?)
- Comprehensive evaluation (RAGAS, faithfulness, human feedback)
- Real-time paper ingestion and updates
- **Status:** Vision for production system

---

## 📊 Dataset

**Current Papers (Phase 1):**

- 1810.04805 - BERT: Pre-training of Deep Bidirectional Transformers
- 1706.03762 - Attention Is All You Need
- 1810.13083 - GraphIE: A Graph-Based Framework for Information Extraction
- **Total chunks:** 49 (500 tokens each, 50 overlap)

**Source:** ArXiv API (open access papers)

---

## 🛠️ Technical Stack

**Current (Phase 1):**

- `rank-bm25` - BM25 algorithm for sparse retrieval
- `marker-pdf` - High-quality PDF to Markdown extraction
- `pdfplumber` - Fallback PDF text extraction
- `arxiv` - ArXiv API client

**Planned (Phase 2+):**

- `sentence-transformers` - Dense embeddings
- `chromadb` / `faiss` - Vector database
- `langchain` - RAG orchestration
- `transformers` - LLM integration
- `cohere` - Re-ranking (optional)

---

## � Quick Start

### Installation

```bash
git clone https://github.com/Salwa08/academic-paper-qa-system.git
cd q&a-system
pip install -r requirements.txt
```

### Phase 1 Usage (BM25 System)

```python
from src.approaches.keyword_search.keyword_qa import KeywordSearchQA

# Initialize BM25 system
qa_system = KeywordSearchQA()
qa_system.initialize(top_k=3)

# Ask a question
result = qa_system.answer_with_timing("What is BERT?")

print(f"Answer: {result['answer'][:200]}...")
print(f"Retrieved {result['num_chunks']} chunks in {result['response_time']:.3f}s")
```

### Run Interactive CLI

```bash
python main.py
```

### Run Evaluation

```bash
python evaluation/evaluate_bm25.py
python evaluation/view_logs.py  # View query statistics
```

---

## 📁 Project Structure

```
q&a-system/
├── src/
│   ├── approaches/
│   │   └── 01_keyword_search/    # ✅ BM25 implementation
│   │       └── keyword_qa.py
│   ├── data/
│   │   ├── arxiv_scraper.py      # Paper downloads
│   │   ├── download_and_extract.py
│   │   ├── paper_processor.py    # BM25 indexing
│   │   └── text_preprocessor.py
│   └── utils/
│       ├── base_qa_model.py      # Base class for all approaches
│       ├── config.py
│       └── query_logger.py       # Query tracking
├── data/
│   ├── processed/
│   │   └── processed_chunks.json # 49 chunks from 3 papers
│   └── raw/pdfs/
├── evaluation/
│   ├── metrics_bm25.py           # Hit rate, avg position metrics
│   ├── evaluate_bm25.py          # Evaluation script
│   ├── test_questions.json       # 10 test questions
│   ├── view_logs.py              # Query log viewer
│   └── results/
│       └── bm25_results.json
├── logs/
│   └── query_log.jsonl           # All queries tracked
├── main.py                       # Interactive CLI
├── config.yaml
└── requirements.txt
```

---

## 🔍 Key Learnings (Phase 1)

1. **BM25 is fast and effective** — 1ms retrieval, 70% hit rate for keyword-rich questions
2. **Semantic gap matters** — 30% of questions need semantic understanding beyond keywords
3. **Early ranking is good** — When BM25 finds relevant chunks, they rank in top 2 positions
4. **Query logging is essential** — Understanding usage patterns guides next improvements
5. **Good chunking matters** — 500 tokens with 50 overlap provides good context windows

---

## 📈 Future Improvements

**Phase 2 (Next):**

- [ ] Section-aware chunking with markdown structure
- [ ] Dense embeddings + hybrid search
- [ ] Expand to 10+ papers for better evaluation

**Phase 3+:**

- [ ] Agentic parsing (tables, figures, equations)
- [ ] Advanced re-ranking (Cohere/cross-encoder)
- [ ] Multi-modal Q&A
- [ ] Conversational memory
- [ ] Web interface (Streamlit/FastAPI)
- [ ] Real-time paper updates

---

## 🤝 Contributing

This is a learning project built in public. Feedback, suggestions, and contributions welcome!

---

## 📝 License

MIT License - See LICENSE file for details

---

**Part of:** [AI Journey 365](https://github.com/Salwa08/ai-journey-365)  
**Author:** Salwa Khattami  
**Started:** January 2026  
**Status:** Phase 2 In Progress 🚧

**Current Focus:** Finishing Phase 1 documentation → Starting Phase 2 (section-aware chunking + hybrid retrieval)
