"""
Multi-Approach Q&A System for Academic Papers
=============================================

A comprehensive question-answering system that implements and compares 
multiple approaches from keyword search to advanced RAG systems.

Author: Salwa Khattami
"""

__version__ = "1.0.0"
__author__ = "Salwa Khattami"

# Import main components
from .src.utils.config import Config
from .src.data.paper_processor import PaperProcessor
from .src.utils.evaluation import QAEvaluator

__all__ = [
    "Config",
    "PaperProcessor", 
    "QAEvaluator",
]
