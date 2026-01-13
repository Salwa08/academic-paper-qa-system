# Academic Paper Q&A System

A comprehensive Question & Answer system for academic papers demonstrating multiple NLP approaches from basic keyword search to advanced RAG (Retrieval-Augmented Generation) systems.

## 🎯 Objectives

- [ ] Compare 6 different Q&A approaches on academic papers
- [ ] Demonstrate progression from traditional IR to modern RAG systems
- [ ] Build a comprehensive evaluation framework for Q&A systems
- [ ] Create an interactive web interface showcasing all approaches
- [ ] Deploy a production-ready system with performance monitoring

## � Current Status

**Ugly MVP Complete** ✅  
BM25 retrieval pipeline working on 1 paper (BERT). 22 chunks indexed with caching support. Next steps: full evaluation framework + expand to more papers.

## �📊 Dataset

**ArXiv Academic Papers**

- Description: Academic papers from computer science domains (AI, ML, NLP)
- Source: ArXiv API (cs.AI, cs.LG, cs.CL categories)
- Size: ~1,000-5,000 papers with abstracts and full text
- Format: PDF text extraction, JSON metadata
- License: Open access academic papers

## 🛠️ Implementation Approaches

### Approach 1: Keyword Search (BM25)

- Traditional information retrieval with BM25 ranking
- Key techniques: BM25 scoring, document chunking, text preprocessing
- Expected outcomes: Fast retrieval, good baseline performance
- **Status**: Implemented (ugly MVP)

### Approach 2: Extractive Q&A (BERT)

- Neural extractive question answering using pre-trained BERT
- Key techniques: BERT, transformer attention, span extraction
- Expected outcomes: Precise answer extraction from context

### Approach 3: Similarity-based (Semantic Search)

- Semantic similarity using sentence embeddings
- Key techniques: Sentence-BERT, semantic embeddings, vector similarity
- Expected outcomes: Better semantic understanding than keyword search

### Approach 4: Simple RAG

- Basic retrieval-augmented generation with LLM
- Key techniques: Document retrieval + GPT generation, prompt engineering
- Expected outcomes: Fluent, contextual answers with citations

### Approach 5: Advanced RAG

- Optimized RAG with re-ranking and context optimization
- Key techniques: Dense retrieval, re-ranking, chunk optimization, citation tracking
- Expected outcomes: Production-quality Q&A with high accuracy

### Approach 6: Fine-tuned Model

- Domain-specific fine-tuning on academic paper Q&A
- Key techniques: Model fine-tuning, domain adaptation, transfer learning
- Expected outcomes: Specialized performance on academic content

## 📈 Results

| Approach         | Accuracy | Response Time | BLEU Score | User Rating | Deployment Cost |
| ---------------- | -------- | ------------- | ---------- | ----------- | --------------- |
| Keyword Search   | 0.XX     | 0.1s          | 0.XX       | X.X/5       | $0/month        |
| Extractive Q&A   | 0.XX     | 0.5s          | 0.XX       | X.X/5       | $5/month        |
| Similarity-based | 0.XX     | 0.3s          | 0.XX       | X.X/5       | $10/month       |
| Simple RAG       | 0.XX     | 2.0s          | 0.XX       | X.X/5       | $25/month       |
| Advanced RAG     | 0.XX     | 3.5s          | 0.XX       | X.X/5       | $50/month       |
| Fine-tuned       | 0.XX     | 1.0s          | 0.XX       | X.X/5       | $30/month       |

## 🚀 Usage

```python
from src.approaches.advanced_rag import AdvancedRAGSystem

# Initialize the best performing system
qa_system = AdvancedRAGSystem()

# Load academic papers
qa_system.load_documents('data/processed/arxiv_papers.json')

# Ask a question
question = "What are the main advantages of transformer models over RNNs?"
answer = qa_system.answer(question)

print(f"Answer: {answer.text}")
print(f"Sources: {answer.sources}")
print(f"Confidence: {answer.confidence}")
```

## 📁 Project Structure

```
q&a-system/
├── src/                           # Source code
│   ├── approaches/                # Different Q&A approaches
│   │   ├── 01_keyword_search/     # TF-IDF based search
│   │   ├── 02_extractive_qa/      # BERT extractive Q&A
│   │   ├── 03_similarity_based/   # Semantic similarity search
│   │   ├── 04_simple_rag/         # Basic RAG implementation
│   │   ├── 05_advanced_rag/       # Optimized RAG system
│   │   └── 06_fine_tuned/         # Fine-tuned model approach
│   ├── data/                      # Data processing utilities
│   │   ├── arxiv_scraper.py       # ArXiv API integration
│   │   ├── pdf_processor.py       # PDF text extraction
│   │   └── data_preprocessor.py   # Text cleaning & chunking
│   ├── utils/                     # Utility functions
│   │   ├── evaluation_metrics.py  # Q&A evaluation metrics
│   │   ├── vector_store.py        # Vector database management
│   │   └── model_utils.py         # Model loading utilities
│   └── web_interface/             # Web application
│       ├── streamlit_app.py       # Main web interface
│       ├── api.py                 # FastAPI backend
│       └── components/            # UI components
├── data/                          # Dataset files
│   ├── raw/                       # Original ArXiv papers
│   ├── processed/                 # Processed & chunked text
│   └── external/                  # External datasets for evaluation
├── notebooks/                     # Jupyter notebooks
│   ├── exploratory/               # Data exploration
│   │   ├── 01_arxiv_data_analysis.ipynb
│   │   └── 02_text_preprocessing_analysis.ipynb
│   ├── experiments/               # Approach experiments
│   │   ├── 03_keyword_search_experiments.ipynb
│   │   ├── 04_extractive_qa_experiments.ipynb
│   │   ├── 05_similarity_experiments.ipynb
│   │   ├── 06_rag_experiments.ipynb
│   │   └── 07_fine_tuning_experiments.ipynb
│   └── reports/                   # Final analysis
│       └── 08_comparative_analysis.ipynb
├── models/                        # Saved models & embeddings
│   ├── saved_models/              # Trained/fine-tuned models
│   ├── checkpoints/               # Training checkpoints
│   └── exports/                   # Exported model files
├── evaluation/                    # Evaluation framework
│   ├── test_datasets/             # Evaluation datasets
│   ├── metrics/                   # Evaluation scripts
│   └── benchmarks/                # Performance benchmarks
├── tests/                         # Tests
│   ├── unit/                      # Unit tests
│   └── integration/               # Integration tests
├── docs/                          # Documentation
│   ├── images/                    # Documentation images
│   ├── reports/                   # Generated reports
│   └── deployment/                # Deployment guides
├── requirements.txt               # Dependencies
├── config.yaml                    # Configuration
├── .gitignore                     # Git ignore rules
└── README.md                      # This file
```

## 🔍 Key Learnings

1. **Traditional vs Modern NLP**: Evolution from TF-IDF to transformer-based approaches
2. **RAG Architecture**: Understanding retrieval-augmented generation pipelines
3. **Evaluation Challenges**: Difficulty in evaluating open-ended Q&A systems
4. **Performance Trade-offs**: Accuracy vs speed vs cost considerations
5. **Domain Adaptation**: Benefits of fine-tuning for specific domains

## 🎯 Future Improvements

- [ ] Multi-modal Q&A (handling figures, tables, equations)
- [ ] Conversational Q&A with context memory
- [ ] Real-time paper ingestion pipeline
- [ ] Advanced citation and source tracking
- [ ] Integration with academic search engines
- [ ] Mobile-friendly interface

## 📚 Technical Stack

**Core NLP Libraries:**

- `transformers` - Hugging Face transformers for BERT, GPT models
- `sentence-transformers` - Sentence embeddings
- `langchain` - RAG pipeline orchestration
- `faiss-cpu` - Efficient vector similarity search
- `chromadb` - Vector database for embeddings

**Data Processing:**

- `arxiv` - ArXiv API client
- `PyPDF2` - PDF text extraction
- `beautifulsoup4` - HTML parsing
- `pandas` - Data manipulation
- `numpy` - Numerical operations

**Web Interface:**

- `streamlit` - Interactive web interface
- `fastapi` - REST API backend
- `plotly` - Interactive visualizations
- `gradio` - ML model interfaces

**Evaluation:**

- `nltk` - Text processing and BLEU scores
- `rouge-score` - ROUGE evaluation metrics
- `scikit-learn` - Classification metrics
- `seaborn` - Statistical visualizations

## 📊 Evaluation Metrics

- **Accuracy**: Manual evaluation on test questions
- **BLEU Score**: N-gram overlap with reference answers
- **ROUGE Score**: Recall-oriented evaluation
- **Semantic Similarity**: Cosine similarity with reference answers
- **Response Time**: Average query processing time
- **Citation Quality**: Accuracy of source attribution

## 🚀 Quick Start

1. **Clone and setup:**

```bash
git clone <repository-url>
cd q&a-system
pip install -r requirements.txt
```

2. **Download and process data:**

```bash
python src/data/arxiv_scraper.py --categories cs.AI cs.LG cs.CL --max_papers 1000
python src/data/data_preprocessor.py --input data/raw --output data/processed
```

3. **Run experiments:**

```bash
# Start with keyword search
python src/approaches/01_keyword_search/keyword_search.py

# Try advanced RAG
python src/approaches/05_advanced_rag/advanced_rag.py
```

4. **Launch web interface:**

```bash
streamlit run src/web_interface/streamlit_app.py
```

## 📈 Performance Benchmarks

System tested on:

- **Dataset**: 5,000 ArXiv papers (CS domains)
- **Hardware**: Intel i7, 16GB RAM, GPU optional
- **Test Questions**: 100 manually curated academic questions
- **Evaluation**: Human experts + automated metrics

---

**Part of:** [AI Journey 365](https://github.com/Salwa08/ai-journey-365)  
**Author:** Salwa Khattami  
**Date:** August 18, 2025  
**Status:** In Development 🚧
