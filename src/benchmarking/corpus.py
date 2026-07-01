"""Corpus loading and chunk generation for retrieval benchmark v1."""

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from src.data.chunking import SectionAwareV2Chunker, SimpleChunker
from src.schemas import Chunk, Paper

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "sample" / "sample_corpus_manifest.json"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_manifest(manifest_path: Path = DEFAULT_MANIFEST) -> List[Paper]:
    with open(manifest_path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    papers = []
    for item in data.get("papers", []):
        papers.append(Paper(
            paper_id=item["paper_id"],
            title=item["title"],
            source_path=item["source_path"],
            authors=item.get("authors"),
            year=item.get("year"),
            metadata=item.get("metadata", {}),
        ))
    return papers


def get_chunker(chunker_name: str, chunk_size: int = 500, overlap: int = 50):
    if chunker_name == "simple":
        return SimpleChunker(chunk_size=chunk_size, overlap=overlap)
    if chunker_name == "section_aware_v2":
        return SectionAwareV2Chunker(chunk_size=chunk_size, overlap=overlap)
    raise ValueError(f"Unknown benchmark chunker: {chunker_name}")


def build_chunks(
    chunker_name: str,
    manifest_path: Path = DEFAULT_MANIFEST,
    chunk_size: int = 500,
    overlap: int = 50,
    paper_ids: Sequence[str] | None = None,
) -> List[Chunk]:
    papers = load_manifest(manifest_path)
    selected = set(paper_ids or [])
    chunker = get_chunker(chunker_name, chunk_size=chunk_size, overlap=overlap)
    chunks: List[Chunk] = []

    for paper in papers:
        if selected and paper.paper_id not in selected:
            continue
        source_path = PROJECT_ROOT / paper.source_path
        if not source_path.exists():
            raise FileNotFoundError(f"Manifest source file not found: {source_path}")
        text = source_path.read_text(encoding="utf-8")
        chunks.extend(chunker.chunk_paper(paper.paper_id, text))
    return chunks


def chunks_output_path(chunker_name: str) -> Path:
    return PROCESSED_DIR / f"benchmark_chunks_{chunker_name}.json"


def save_chunks(chunks: Iterable[Chunk], chunker_name: str) -> Path:
    output_path = chunks_output_path(chunker_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"chunker": chunker_name, "chunks": [chunk.to_dict() for chunk in chunks]}
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def load_chunks(chunker_name: str, rebuild: bool = False) -> List[Chunk]:
    output_path = chunks_output_path(chunker_name)
    if rebuild or not output_path.exists():
        chunks = build_chunks(chunker_name)
        save_chunks(chunks, chunker_name)
        return chunks

    data = json.loads(output_path.read_text(encoding="utf-8-sig"))
    return [Chunk(**item) for item in data.get("chunks", [])]


if __name__ == "__main__":
    for name in ("simple", "section_aware_v2"):
        built = build_chunks(name)
        path = save_chunks(built, name)
        print(f"Wrote {len(built)} {name} chunks to {path}")

