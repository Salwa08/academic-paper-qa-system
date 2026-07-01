"""Draft relevance-label helper for benchmark v1.

This helper is intentionally separate from scoring. It can suggest labels from the old
keyword hints, but the benchmark runner only reads finalized relevant_chunk_ids.
"""

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarking.corpus import load_chunks

TEST_QUESTIONS = PROJECT_ROOT / "evaluation" / "test_questions.json"
OUTPUT = PROJECT_ROOT / "evaluation" / "relevance_labels.draft.json"


def main() -> None:
    test_data = json.loads(TEST_QUESTIONS.read_text(encoding="utf-8-sig"))
    chunks_by_chunker = {
        "simple": load_chunks("simple", rebuild=True),
        "section_aware_v2": load_chunks("section_aware_v2", rebuild=True),
    }
    labels = []
    for idx, item in enumerate(test_data, 1):
        question_labels = {}
        keywords = [keyword.lower() for keyword in item.get("relevant_keywords", [])]
        for chunker_name, chunks in chunks_by_chunker.items():
            matches = []
            for chunk in chunks:
                if chunk.paper_id != item.get("paper_id"):
                    continue
                text = chunk.text.lower()
                if any(keyword in text for keyword in keywords):
                    matches.append(chunk.chunk_id)
            question_labels[chunker_name] = matches[:3]
        labels.append({
            "question_id": f"q{idx:02d}",
            "question": item["question"],
            "paper_id": item.get("paper_id"),
            "relevant_chunk_ids": question_labels,
        })
    OUTPUT.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote draft labels to {OUTPUT}")


if __name__ == "__main__":
    main()

