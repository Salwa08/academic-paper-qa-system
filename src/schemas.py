"""Core data schemas for retrieval benchmarking."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Paper:
    paper_id: str
    title: str
    source_path: str
    authors: Optional[List[str]] = None
    year: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Chunk:
    chunk_id: str
    paper_id: str
    text: str
    chunk_index: int
    chunker: str
    section_path: Optional[str] = None
    section_type: Optional[str] = None
    start_token: Optional[int] = None
    end_token: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SearchResult:
    chunk_id: str
    paper_id: str
    text: str
    rank: int
    score: float
    retriever: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
