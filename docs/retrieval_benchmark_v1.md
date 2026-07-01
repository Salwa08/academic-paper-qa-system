# Retrieval Benchmark v1

| Chunker | Retriever | Questions | Latency (s) | Hit@3 | Recall@3 | MRR@3 | nDCG@3 | Hit@5 | Recall@5 | MRR@5 | nDCG@5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| simple | bm25 | 13 | 0.0003 | 0.3846 | 0.1410 | 0.2821 | 0.1555 | 0.5385 | 0.2436 | 0.3167 | 0.2145 |
| simple | dense | 13 | 0.0516 | 0.7692 | 0.2949 | 0.3974 | 0.2544 | 0.9231 | 0.4487 | 0.4321 | 0.3413 |
| simple | hybrid_rrf | 13 | 0.0254 | 0.6154 | 0.2436 | 0.3077 | 0.2064 | 0.8462 | 0.4487 | 0.3654 | 0.3276 |
| section_aware_v2 | bm25 | 13 | 0.0005 | 0.4615 | 0.1667 | 0.2051 | 0.1374 | 0.6923 | 0.3590 | 0.2628 | 0.2478 |
| section_aware_v2 | dense | 13 | 0.0285 | 0.2308 | 0.0897 | 0.1026 | 0.0706 | 0.3846 | 0.1410 | 0.1410 | 0.1017 |
| section_aware_v2 | hybrid_rrf | 13 | 0.0312 | 0.3846 | 0.1410 | 0.2436 | 0.1421 | 0.5385 | 0.2179 | 0.2821 | 0.1888 |

Dense backend: `sentence_transformers_chroma`.

Scoring uses finalized `relevant_chunk_ids` from `evaluation/relevance_labels.json`; keyword hints are not used by the benchmark runner.
