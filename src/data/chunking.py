"""Chunking strategies for processing academic papers."""

from typing import Any, Dict, List, Optional, Tuple, Union
import re

from src.schemas import Chunk
from .text_preprocessor import clean_text, simple_tokenize

MarkdownHeaderTextSplitter = None
RecursiveCharacterTextSplitter = None


class BaseChunker:
    """Base class for all chunking strategies."""

    name = "base"

    def __init__(self, chunk_size: int = 500, overlap: int = 50, min_tokens: int = 30):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_tokens = min_tokens

    def chunk_text(self, text: str) -> Union[List[str], List[Dict[str, Any]]]:
        raise NotImplementedError("chunk_text method must be implemented by subclasses.")

    def chunk_paper(self, paper_id: str, text: str) -> List[Chunk]:
        raise NotImplementedError("chunk_paper method must be implemented by subclasses.")

    def _make_chunk_id(self, paper_id: str, chunk_index: int) -> str:
        return f"{paper_id}:{self.name}:{chunk_index:04d}"


class SimpleChunker(BaseChunker):
    """Simple token-window chunking."""

    name = "simple"

    def chunk_text(self, text: str) -> List[str]:
        words = simple_tokenize(text)
        chunks = []
        step = max(1, self.chunk_size - self.overlap)

        for i in range(0, len(words), step):
            chunk_words = words[i:i + self.chunk_size]
            if len(chunk_words) < self.min_tokens:
                continue
            chunks.append(" ".join(chunk_words))
        return chunks

    def chunk_paper(self, paper_id: str, text: str) -> List[Chunk]:
        cleaned = clean_text(text)
        words = simple_tokenize(cleaned)
        chunks: List[Chunk] = []
        step = max(1, self.chunk_size - self.overlap)

        for start in range(0, len(words), step):
            chunk_words = words[start:start + self.chunk_size]
            if len(chunk_words) < self.min_tokens:
                continue
            chunk_index = len(chunks)
            chunks.append(Chunk(
                chunk_id=self._make_chunk_id(paper_id, chunk_index),
                paper_id=paper_id,
                text=" ".join(chunk_words),
                chunk_index=chunk_index,
                chunker=self.name,
                section_type="full_text",
                start_token=start,
                end_token=start + len(chunk_words),
            ))
        return chunks


class SectionAwareChunker(BaseChunker):
    """Original section-aware chunking, preserved for backward compatibility."""

    name = "section_aware"

    def __init__(self, chunk_size: int = 500, overlap: int = 50, min_tokens: int = 30):
        super().__init__(chunk_size, overlap, min_tokens)
        if MarkdownHeaderTextSplitter is None or RecursiveCharacterTextSplitter is None:
            self.splitter = None
            self.chunker = None
        else:
            self.splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
                ("####", "Header 4"),
            ])
            self.chunker = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.overlap,
                separators=["\n\n", "\n", " ", ""],
            )
        self.section_patterns = _section_patterns()

    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        if self.splitter is None or self.chunker is None:
            return [
                {
                    "section_type": chunk.section_type,
                    "content": chunk.text,
                    "metadata": chunk.metadata,
                }
                for chunk in SectionAwareV2Chunker(
                    self.chunk_size, self.overlap, self.min_tokens
                ).chunk_paper("unknown", text)
            ]

        text = self.clean_md(text)
        sections = self.splitter.split_text(text)
        chunks = self.chunker.split_documents(sections)
        chunked_sections = []

        for chunk in chunks:
            metadata = chunk.metadata
            section_type = self._get_section_type_from_metadata(metadata)
            chunked_sections.append({
                "section_type": section_type,
                "content": chunk.page_content,
                "metadata": metadata,
            })
        return chunked_sections

    def chunk_paper(self, paper_id: str, text: str) -> List[Chunk]:
        chunks = []
        for raw in self.chunk_text(text):
            chunk_text = raw.get("content") or raw.get("text") or ""
            tokens = simple_tokenize(chunk_text)
            if len(tokens) < self.min_tokens:
                continue
            chunk_index = len(chunks)
            chunks.append(Chunk(
                chunk_id=self._make_chunk_id(paper_id, chunk_index),
                paper_id=paper_id,
                text=chunk_text,
                chunk_index=chunk_index,
                chunker=self.name,
                section_type=raw.get("section_type", "other"),
                metadata=raw.get("metadata", {}),
            ))
        return chunks

    def _extract_section_name_from_metadata(self, metadata: dict) -> str:
        for level in range(4, 0, -1):
            key = f"Header {level}"
            if key in metadata:
                return metadata[key].strip().lower()
        for key, value in metadata.items():
            if "header" in key.lower():
                return value.strip().lower()
        return "unknown"

    def _map_section_name_to_type(self, section_name: str) -> str:
        return map_section_name_to_type(section_name, self.section_patterns)

    def _get_section_type_from_metadata(self, metadata: dict) -> str:
        for level in range(4, 0, -1):
            key = f"Header {level}"
            if key in metadata:
                section_type = self._map_section_name_to_type(metadata[key].strip().lower())
                if section_type != "other":
                    return section_type
        return "other"

    def clean_md(self, text: str) -> str:
        return clean_markdown_artifacts(text)


class SectionAwareV2Chunker(BaseChunker):
    """Section-aware metadata with stable token-window chunk bodies."""

    name = "section_aware_v2"

    def __init__(self, chunk_size: int = 500, overlap: int = 50, min_tokens: int = 30):
        super().__init__(chunk_size, overlap, min_tokens)
        self.section_patterns = _section_patterns()

    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        return [chunk.to_dict() for chunk in self.chunk_paper("unknown", text)]

    def chunk_paper(self, paper_id: str, text: str) -> List[Chunk]:
        sections = self._extract_sections(clean_markdown_artifacts(text))
        chunks: List[Chunk] = []
        for section_path, section_text in sections:
            words = simple_tokenize(clean_text(section_text))
            if len(words) < self.min_tokens:
                continue
            section_type = map_section_name_to_type(section_path, self.section_patterns)
            step = max(1, self.chunk_size - self.overlap)
            for start in range(0, len(words), step):
                chunk_words = words[start:start + self.chunk_size]
                if len(chunk_words) < self.min_tokens:
                    continue
                chunk_index = len(chunks)
                chunks.append(Chunk(
                    chunk_id=self._make_chunk_id(paper_id, chunk_index),
                    paper_id=paper_id,
                    text=" ".join(chunk_words),
                    chunk_index=chunk_index,
                    chunker=self.name,
                    section_path=section_path or "unknown",
                    section_type=section_type,
                    start_token=start,
                    end_token=start + len(chunk_words),
                    metadata={"section_path": section_path or "unknown"},
                ))
        return chunks

    def _extract_sections(self, text: str) -> List[Tuple[str, str]]:
        sections: List[Tuple[str, List[str]]] = []
        current_headers: List[str] = []
        current_lines: List[str] = []

        def flush() -> None:
            body = "\n".join(current_lines).strip()
            if body:
                sections.append((" > ".join(current_headers), current_lines.copy()))

        for line in text.splitlines():
            match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
            if match:
                flush()
                level = len(match.group(1))
                header = re.sub(r"\s+", " ", match.group(2)).strip()
                current_headers = current_headers[:level - 1] + [header]
                current_lines = []
            else:
                current_lines.append(line)
        flush()

        if not sections:
            return [("full_text", text)]
        return [(path, "\n".join(lines)) for path, lines in sections]


def clean_markdown_artifacts(text: str) -> str:
    text = re.sub(r'<span id="page-\d+-\d+"></span>', '', text)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    return text.strip()


def _section_patterns() -> Dict[str, List[str]]:
    return {
        "abstract": ["abstract", "summary", "resume", "résumé"],
        "introduction": ["introduction", "intro", "motivation", "background", "overview", "problem statement"],
        "related_work": ["related work", "prior work", "previous work", "literature review", "related studies", "survey", "state of the art"],
        "method": ["method", "methods", "approach", "proposed method", "model", "architecture", "algorithm", "framework", "pre-training", "training", "fine-tuning", "system", "design", "bert", "gpt", "transformer"],
        "experiments": ["experiment", "experiments", "results", "evaluation", "empirical study", "ablation", "analysis", "performance", "benchmark", "case study", "glue", "squad", "swag"],
        "conclusion": ["conclusion", "conclusions", "discussion", "summary", "final remarks", "limitations", "future work", "outlook"],
        "appendix": ["appendix", "supplementary", "supplement", "additional material"],
    }


def map_section_name_to_type(section_name: Optional[str], patterns: Optional[Dict[str, List[str]]] = None) -> str:
    name = (section_name or "").lower()
    patterns = patterns or _section_patterns()
    for section_type, needles in patterns.items():
        if any(needle in name for needle in needles):
            return section_type
    return "other"



