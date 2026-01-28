# Academic Paper Q&A System

A progressively smarter Question & Answer system for AI research papers, building from traditional sparse retrieval to advanced RAG systems. Learning and building in public.

**Public build log:** [X/Twitter threads](https://x.com/Salma70330363)

---

## 🎯 Project Vision

Building a production-quality Q&A system over research papers through iterative improvement:

- Start with simple, working baseline (BM25)
- Add semantic understanding (embeddings + hybrid search)
- Incorporate LLMs with grounding (RAG with citations)
- Optimize for production (re-ranking, agentic flows, multimodal)
- Deploy and monitor real usage

## 📍 Current Status: Phase 2 Complete, Phase 3 Starting

**Phase 1 Complete** ✅ | **Phase 2 Complete** ✅ | **Phase 3 In Progress** 🚧

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

- [src/approaches/keyword_search/keyword_qa.py](src/approaches/keyword_search/keyword_qa.py)
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

### Phase 2 – Sectioned & Hybrid RAG ✅ (Completed Jan 2026)

**Goal:** Smarter chunking + hybrid retrieval for better semantic matching.

**Implementation:**

- **Section-aware chunking:** Preserve markdown structure from marker-pdf extraction (444 chunks vs 49 simple)
- **Dense embeddings:** Sentence-BERT (all-MiniLM-L6-v2) for semantic search with ChromaDB
- **Hybrid retrieval:** BM25 + semantic search with score fusion (α=0.4, β=0.6)
- **Dynamic chunking:** Support for SimpleChunker and SectionAwareChunker
- **Better metadata:** Section types and paper IDs for improved grounding

**Evaluation Results** (13 test questions, keyword-proxy relevance):

**BM25 Results:**

- **Simple Chunking:** 92% hit rate, avg position 1.25
- **Section-Aware (No Boost):** 46% hit rate, avg position 1.17
- **Section-Aware (With Boost):** 46% hit rate, avg position 1.17

**Hybrid Results:**

- **Simple Chunking:** 92% hit rate, avg position 1.33
- **Section-Aware (No Boost):** 62% hit rate, avg position 1.50
- **Section-Aware (With Boost):** 62% hit rate, avg position 1.88

**Key Files:**

- [src/approaches/hybrid_search/hybrid_qa.py](src/approaches/hybrid_search/hybrid_qa.py)
- [src/data/chunking.py](src/data/chunking.py)
- [evaluation/evaluate_hybrid.py](evaluation/evaluate_hybrid.py)
- [evaluation/results/hybrid_results.json](evaluation/results/hybrid_results.json)

**Outcome:** Hybrid search improves semantic matching (62% vs 46% on section-aware), but chunking strategy significantly impacts performance. Simple chunking achieves 92% hit rate for both approaches. Fusion weight tuning improved section-aware performance by 15%.

**Optional Phase 2 Enhancements:**

- Implement cross-encoder re-ranking for top results

**How to Run:**

```bash
# Run hybrid evaluation
python evaluation/evaluate_hybrid.py

# Interactive CLI (supports both BM25 and hybrid)
python main.py
```

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
- Deployment (HuggingFace Spaces / FastAPI )
- Comprehensive evaluation (RAGAS, faithfulness, human feedback)
- Real-time paper ingestion and updates
- **Status:** Vision for production system

---

## 📊 Dataset

**Current Papers (Phase 1-2):**

- 1810.04805 - BERT: Pre-training of Deep Bidirectional Transformers
- 1706.03762 - Attention Is All You Need
- 1810.13083 - GraphIE: A Graph-Based Framework for Information Extraction
- **Simple chunks:** 49 (500 tokens each, 50 overlap)
- **Section-aware chunks:** 444 (preserves markdown structure)

**Source:** ArXiv API (open access papers)

---

## 🛠️ Technical Stack

**Current (Phase 1-2):**

- `rank-bm25` - BM25 algorithm for sparse retrieval
- `sentence-transformers` - Dense embeddings (all-MiniLM-L6-v2)
- `chromadb` - Vector database for semantic search
- `marker-pdf` - High-quality PDF to Markdown extraction
- `pdfplumber` - Fallback PDF text extraction
- `arxiv` - ArXiv API client

---

## � Quick Start

### Installation

```bash
git clone https://github.com/Salwa08/academic-paper-qa-system.git
cd q&a-system
pip install -r requirements.txt
```

### Phase 1-2 Usage

```python
# BM25 system
from src.approaches.keyword_search.keyword_qa import KeywordSearchQA
qa_system = KeywordSearchQA()
qa_system.initialize(top_k=3)

# Hybrid system
from src.approaches.hybrid_search.hybrid_qa import HybridQA
qa_system = HybridQA()
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

**Features:**

- Choose between BM25 or Hybrid search approaches
- Select chunking strategy (Simple or Section-Aware)
- Interactive Q&A with performance metrics

### Run Evaluation

```bash
# BM25 evaluation
python evaluation/evaluate_bm25.py

# Hybrid evaluation
python evaluation/evaluate_hybrid.py

# View query logs
python evaluation/view_logs.py
```

```

---

## 📁 Project Structure

```

q&a-system/
├── src/
│ ├── approaches/
│ │ ├── keyword_search/ # ✅ BM25 implementation
│ │ │ └── keyword_qa.py
│ │ └── hybrid_search/ # ✅ Hybrid BM25 + dense
│ │ └── hybrid_qa.py
│ ├── data/
│ │ ├── arxiv_scraper.py # Paper downloads
│ │ ├── chunking.py # ✅ Simple & section-aware chunkers
│ │ ├── download_and_extract.py
│ │ ├── paper_processor.py # BM25 indexing
│ │ └── text_preprocessor.py
│ └── utils/
│ ├── base_qa_model.py # Base class for all approaches
│ ├── config.py
│ └── query_logger.py # Query tracking
├── data/
│ ├── processed/
│ │ ├── processed_chunks_simple.json # 49 chunks
│ │ └── processed_chunks_section_aware.json # 444 chunks
│ └── raw/pdfs/
├── evaluation/
│ ├── metrics_bm25.py # Hit rate, avg position metrics
│ ├── evaluate_bm25.py # ✅ BM25 evaluation
│ ├── evaluate_hybrid.py # ✅ Hybrid evaluation
│ ├── test_questions.json # 13 test questions
│ ├── view_logs.py # Query log viewer
│ └── results/
│ ├── bm25_results.json # ✅ All phases
│ └── hybrid_results.json # ✅ All phases
├── logs/
│ └── query_log.jsonl # All queries tracked
├── main.py # Interactive CLI
├── config.yaml
└── requirements.txt

```

---

## 🔍 Key Learnings (Phase 1-2)

1. **BM25 is fast and effective** — 1ms retrieval, 70-92% hit rate depending on chunking
2. **Chunking strategy matters hugely** — Simple chunking (92%) vs section-aware (46-54%), highlighting the importance of domain-specific chunking
3. **Hybrid search adds semantic value** — 54% vs 46% on section-aware chunks shows benefit of combining sparse + dense retrieval
4. **Early ranking is good** — When systems find relevant chunks, they rank in top 2 positions
5. **Query logging is essential** — Understanding usage patterns guides improvements
6. **Good chunking matters** — 500 tokens with 50 overlap provides good context, but structure preservation is key for academic papers
7. **Score fusion needs tuning** — Equal weighting (α=0.5, β=0.5) works but optimizing to α=0.4, β=0.6 improved section-aware performance by 15%

---

## 📈 Future Improvements

**Phase 3 (Next):**

- [ ] LLM integration with RAG (GPT-4 + grounding)
- [ ] Advanced re-ranking (cross-encoder models)
- [ ] Agentic parsing (tables, figures, equations)
- [ ] Expand to 10+ papers for robust evaluation
- [ ] Query rewriting and decomposition

**Phase 4+:**

- [ ] Extractive Q&A with BERT-style span extraction
- [ ] Domain-specific fine-tuning on paper Q&A pairs
- [ ] Multi-modal Q&A (handle figures/tables/equations)
- [ ] Conversational memory and follow-up questions
- [ ] Web interface (Streamlit/FastAPI)
- [ ] Real-time paper updates and ingestion

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
**Status:** Phase 2 Complete ✅, Phase 3 Starting 🚧

**Current Focus:** Phase 2 documentation complete → Starting Phase 3 (LLM integration + advanced RAG)
```
