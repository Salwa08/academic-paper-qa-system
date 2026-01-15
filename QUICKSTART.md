# Quick Start Guide - Phase 1 (BM25 System)

Get the Q&A system running in 5 minutes.

## Prerequisites

- Python 3.8+
- pip

## Installation

```bash
# Clone repository
git clone https://github.com/Salwa08/academic-paper-qa-system.git
cd q&a-system

# Install dependencies
pip install -r requirements.txt
```

## Quick Test

### 1. Interactive CLI

```bash
python main.py
```

Then ask questions like:

- "What is BERT?"
- "How does attention work?"
- "What is GraphIE?"

### 2. Run Evaluation

```bash
# Run evaluation on 10 test questions
python evaluation/evaluate_bm25.py

# View query statistics
python evaluation/view_logs.py
```

### 3. Use in Code

```python
from src.approaches.keyword_search.keyword_qa import KeywordSearchQA

# Initialize
qa = KeywordSearchQA()
qa.initialize(top_k=3)

# Ask question
result = qa.answer_with_timing("What is BERT?")
print(result['answer'][:200])
print(f"Found in {result['response_time']:.3f}s")
```

## What's Included

- **3 papers indexed:** BERT, Attention Is All You Need, GraphIE
- **49 chunks** ready for search
- **BM25 retrieval** with ~1ms response time
- **Query logging** in `logs/query_log.jsonl`
- **Evaluation framework** with 10 test questions

## Results (Phase 1)

- Hit Rate@3: 70%
- Avg Position: 1.43
- Avg Response Time: 1ms

## Adding More Papers

```bash
# Edit src/data/download_and_extract.py to add paper IDs
# Then run:
cd src/data
python download_and_extract.py
python paper_processor.py
```

## Troubleshooting

**Import errors?**

```bash
# Make sure you're in the project root
python main.py
```

**No papers found?**

```bash
# Check if processed chunks exist
ls data/processed/processed_chunks.json
```

**Want to rebuild index?**

```bash
cd src/data
python paper_processor.py
```

## Next Steps

- Check [README.md](README.md) for full roadmap
- See [docs/MVP_PHASE1_COMPLETE.md](docs/MVP_PHASE1_COMPLETE.md) for Phase 1 details
- Phase 2: Hybrid retrieval with semantic search (coming soon)

---

Questions? Open an issue or reach out on X/Twitter!
