"""Run retrieval benchmark v1 across retrievers and chunkers."""

import csv
import json
import os
import time
from pathlib import Path
import sys
from statistics import mean
from typing import Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.retrieval_metrics import metrics_at_k
from src.benchmarking.corpus import load_chunks, save_chunks
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRRFRetriever

LABELS_PATH = PROJECT_ROOT / "evaluation" / "relevance_labels.json"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
DOCS_DIR = PROJECT_ROOT / "docs"

CHUNKERS = ["simple", "section_aware_v2"]
RETRIEVERS = ["bm25", "dense", "hybrid_rrf"]
KS = [3, 5]


def _load_labels() -> List[dict]:
    if not LABELS_PATH.exists():
        raise FileNotFoundError(
            f"Missing finalized labels at {LABELS_PATH}. Run evaluation/build_relevance_labels.py and finalize labels first."
        )
    return json.loads(LABELS_PATH.read_text(encoding="utf-8-sig"))


def _labels_for_chunker(item: dict, chunker_name: str) -> List[str]:
    labels = item.get("relevant_chunk_ids", [])
    if isinstance(labels, dict):
        return labels.get(chunker_name, [])
    return labels


def _average(rows: Iterable[Dict[str, float]], key: str) -> float:
    values = [row[key] for row in rows]
    return mean(values) if values else 0.0


def evaluate_retriever(retriever, labels: List[dict], chunker_name: str) -> dict:
    per_query = []
    for item in labels:
        relevant_ids = _labels_for_chunker(item, chunker_name)
        if not relevant_ids:
            continue
        start = time.perf_counter()
        results = retriever.search(item["question"], top_k=max(KS))
        latency = time.perf_counter() - start
        ranked_ids = [result.chunk_id for result in results]
        row = {
            "question_id": item["question_id"],
            "latency_seconds": latency,
        }
        for k in KS:
            row.update(metrics_at_k(ranked_ids, relevant_ids, k))
        per_query.append(row)

    aggregate = {
        "questions_evaluated": len(per_query),
        "avg_latency_seconds": _average(per_query, "latency_seconds"),
    }
    for k in KS:
        for metric in ("HitRate", "Recall", "MRR", "nDCG"):
            metric_key = f"{metric}@{k}"
            aggregate[metric_key] = _average(per_query, metric_key)
    return {"aggregate": aggregate, "per_query": per_query}


def _write_csv(rows: List[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "chunker", "retriever", "questions_evaluated", "avg_latency_seconds",
        "HitRate@3", "Recall@3", "MRR@3", "nDCG@3",
        "HitRate@5", "Recall@5", "MRR@5", "nDCG@5",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_float(value: float) -> str:
    return f"{value:.4f}"


def _write_markdown(rows: List[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "Chunker", "Retriever", "Questions", "Latency (s)",
        "Hit@3", "Recall@3", "MRR@3", "nDCG@3",
        "Hit@5", "Recall@5", "MRR@5", "nDCG@5",
    ]
    lines = ["# Retrieval Benchmark v1", "", "| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        values = [
            row["chunker"], row["retriever"], str(row["questions_evaluated"]), _format_float(row["avg_latency_seconds"]),
            _format_float(row["HitRate@3"]), _format_float(row["Recall@3"]), _format_float(row["MRR@3"]), _format_float(row["nDCG@3"]),
            _format_float(row["HitRate@5"]), _format_float(row["Recall@5"]), _format_float(row["MRR@5"]), _format_float(row["nDCG@5"]),
        ]
        lines.append("| " + " | ".join(values) + " |")
    dense_backend = os.getenv(
        "PAPERQA_DENSE_BACKEND_NOTE",
        "sentence_transformers_chroma" if os.getenv("PAPERQA_USE_SENTENCE_TRANSFORMERS") == "1" else "fallback_lexical_vectors",
    )
    lines.extend([
        "",
        f"Dense backend: `{dense_backend}`.",
        "",
        "Scoring uses finalized `relevant_chunk_ids` from `evaluation/relevance_labels.json`; keyword hints are not used by the benchmark runner.",
    ])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def main() -> None:
    labels = _load_labels()
    results = {"runs": []}
    summary_rows: List[dict] = []

    for chunker_name in CHUNKERS:
        chunks = load_chunks(chunker_name, rebuild=True)
        save_chunks(chunks, chunker_name)
        bm25 = BM25Retriever(chunks)
        dense = DenseRetriever(chunks, chunker_name=chunker_name)
        retrievers = {
            "bm25": bm25,
            "dense": dense,
            "hybrid_rrf": HybridRRFRetriever(bm25=bm25, dense=dense),
        }
        for retriever_name in RETRIEVERS:
            print(f"Evaluating {retriever_name} on {chunker_name}...")
            evaluation = evaluate_retriever(retrievers[retriever_name], labels, chunker_name)
            aggregate = evaluation["aggregate"]
            summary_row = {"chunker": chunker_name, "retriever": retriever_name, **aggregate}
            summary_rows.append(summary_row)
            results["runs"].append({
                "chunker": chunker_name,
                "retriever": retriever_name,
                **evaluation,
            })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "retrieval_benchmark_v1.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(summary_rows, RESULTS_DIR / "retrieval_benchmark_v1.csv")
    _write_markdown(summary_rows, DOCS_DIR / "retrieval_benchmark_v1.md")
    print("Wrote retrieval benchmark v1 results.")


if __name__ == "__main__":
    main()




