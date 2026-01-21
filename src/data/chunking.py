"""
Chunking strategies for processing academic papers.
Supports simple token-based and section-aware chunking.
"""

from typing import List, Union, Dict, Any
import re

from text_preprocessor import simple_tokenize
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

class BaseChunker:
    """Base class for all chunking strategies."""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str) -> Union[List[str], List[Dict[str, Any]]]:
        """Chunk text into smaller pieces. To be implemented by subclasses."""
        raise NotImplementedError("chunk_text method must be implemented by subclasses.")

class SimpleChunker(BaseChunker):
    """Simple token-based chunking (Phase 1 approach)."""
    
    def chunk_text(self, text: str) -> List[str]:
        words = simple_tokenize(text)
        chunks = []
        step = self.chunk_size - self.overlap
        
        for i in range(0, len(words), step):
            chunk_words = words[i:i + self.chunk_size]
            if len(chunk_words) < 30:
                continue
            chunks.append(" ".join(chunk_words))
        return chunks

class SectionAwareChunker(BaseChunker):
    """Section-aware chunking that preserves markdown structure."""
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        super().__init__(chunk_size, overlap)
        self.splitter = MarkdownHeaderTextSplitter(headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
        ])

        self.chunker = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap,
            separators=["\n\n", "\n", " ", ""]
        )

        self.section_patterns = {
            'abstract': ['abstract', 'summary', 'résumé'],
            'introduction': [
                'introduction', 'intro', 'motivation', 'background', 'overview', 'problem statement',
                'preamble', 'context'
            ],
            'related_work': [
                'related work', 'prior work', 'previous work', 'literature review', 'related studies',
                'background', 'survey', 'state of the art'
            ],
            'method': [
                'method', 'methods', 'approach', 'proposed method', 'model', 'architecture',
                'algorithm', 'framework', 'pre-training', 'training', 'fine-tuning', 'our approach',
                'system', 'design', 'bert', 'gpt', 'transformer'
            ],
            'experiments': [
                'experiment', 'experiments', 'results', 'evaluation', 'empirical study',
                'ablation', 'analysis', 'performance', 'benchmark', 'case study',
                'qualitative', 'quantitative', 'glue', 'squad', 'swag'
            ],
            'conclusion': [
                'conclusion', 'conclusions', 'discussion', 'summary', 'final remarks',
                'limitations', 'future work', 'outlook', 'broader impact'
            ],
            'appendix': ['appendix', 'supplementary', 'supplement', 'additional material']
        }
        
        
        # Note: RecursiveCharacterTextSplitter needs a single chunk_size value
        # For section-specific sizes, would need to create separate splitters per section
        # For now, use the base chunk_size for all sections

    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        # Clean markdown artifacts before processing
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
                "metadata": metadata
            })
        
        return chunked_sections
    
    def _extract_section_name_from_metadata(self, metadata: dict) -> str:
        # Check from deepest to shallowest header (most specific to least specific)
        for level in range(4, 0, -1):
            key = f"Header {level}" 
            if key in metadata:
                return metadata[key].strip().lower()
        
        # Fallback: check any key containing 'header'
        for k, v in metadata.items():
            if 'header' in k.lower():
                return v.strip().lower()
        return "unknown"
    
    def _map_section_name_to_type(self, section_name: str) -> str:
        # First try exact match with deepest section name
        for section_type, patterns in self.section_patterns.items():
                if any(p in section_name.lower() for p in patterns):
                    return section_type
        return "other"
    
    def _get_section_type_from_metadata(self, metadata: dict) -> str:
        """Try to classify using all available headers, from deepest to shallowest."""
        # Try headers from most specific (Header 4) to least specific (Header 1)
        for level in range(4, 0, -1):
            key = f"Header {level}"
            if key in metadata:
                section_name = metadata[key].strip().lower()
                section_type = self._map_section_name_to_type(section_name)
                if section_type != "other":
                    return section_type
        return "other"
    
    def clean_md(self, text: str) -> str:
        text = re.sub(r'<span id="page-\d+-\d+"></span>', '', text)
        text = re.sub(r'!\[.*?\]\(.*?\)', '', text)  # remove images
        return text.strip()