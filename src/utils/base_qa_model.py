"""
Base Q&A model class for all approaches.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import time
import logging
from pathlib import Path
import json

from ..utils.config import config


class BaseQAModel(ABC):
    """Abstract base class for all Q&A models."""
    
    def __init__(self, name: str, description: str = ""):
        """
        Initialize base Q&A model.
        
        Args:
            name: Name of the approach
            description: Description of the approach
        """
        self.name = name
        self.description = description
        self.is_initialized = False
        self.logger = self._setup_logger()
        self.config = config
        
        # Performance tracking
        self.response_times = []
        self.total_queries = 0
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for the model."""
        logger = logging.getLogger(f"qa_system.{self.name}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(getattr(logging, config.logging.get('level', 'INFO')))
        
        return logger
    
    @abstractmethod
    def initialize(self, **kwargs) -> None:
        """Initialize the model with necessary components."""
        pass
    
    @abstractmethod
    def answer_question(self, question: str, **kwargs) -> Dict[str, Any]:
        """
        Answer a question and return structured response.
        
        Args:
            question: The question to answer
            **kwargs: Additional parameters specific to the approach
            
        Returns:
            Dictionary containing:
            - answer: The generated answer
            - confidence: Confidence score (if available)
            - sources: Source documents/passages used
            - response_time: Time taken to generate answer
            - metadata: Additional metadata
        """
        pass
    
    def answer_with_timing(self, question: str, **kwargs) -> Dict[str, Any]:
        """Answer question with automatic timing."""
        if not self.is_initialized:
            raise RuntimeError(f"Model {self.name} not initialized. Call initialize() first.")
        
        start_time = time.time()
        
        try:
            result = self.answer_question(question, **kwargs)
            response_time = time.time() - start_time
            
            # Add timing information
            result['response_time'] = response_time
            
            # Track performance
            self.response_times.append(response_time)
            self.total_queries += 1
            
            self.logger.info(
                f"Question answered in {response_time:.2f}s using {self.name}"
            )
            
            return result
            
        except Exception as e:
            response_time = time.time() - start_time
            self.logger.error(f"Error answering question: {str(e)}")
            return {
                'answer': f"Error: {str(e)}",
                'confidence': 0.0,
                'sources': [],
                'response_time': response_time,
                'metadata': {'error': str(e)},
                'success': False
            }
    
    def batch_answer(self, questions: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Answer multiple questions."""
        results = []
        for i, question in enumerate(questions):
            self.logger.info(f"Processing question {i+1}/{len(questions)}")
            result = self.answer_with_timing(question, **kwargs)
            results.append(result)
        
        return results
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        if not self.response_times:
            return {}
        
        import numpy as np
        return {
            'avg_response_time': np.mean(self.response_times),
            'median_response_time': np.median(self.response_times),
            'min_response_time': np.min(self.response_times),
            'max_response_time': np.max(self.response_times),
            'total_queries': self.total_queries,
            'std_response_time': np.std(self.response_times)
        }
    
    def save_model_info(self, save_path: Optional[str] = None) -> None:
        """Save model information and performance stats."""
        if save_path is None:
            save_path = f"models/saved_models/{self.name}_info.json"
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        info = {
            'name': self.name,
            'description': self.description,
            'performance_stats': self.get_performance_stats(),
            'config': self.get_model_config(),
            'is_initialized': self.is_initialized
        }
        
        with open(save_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        self.logger.info(f"Model info saved to {save_path}")
    
    @abstractmethod
    def get_model_config(self) -> Dict[str, Any]:
        """Return model-specific configuration."""
        pass
    
    def reset_stats(self) -> None:
        """Reset performance statistics."""
        self.response_times = []
        self.total_queries = 0
        self.logger.info("Performance statistics reset")
    
    def __repr__(self) -> str:
        """String representation of the model."""
        status = "initialized" if self.is_initialized else "not initialized"
        return f"{self.__class__.__name__}(name='{self.name}', status='{status}')"


class QAResult:
    """Structured result from Q&A model."""
    
    def __init__(
        self,
        question: str,
        answer: str,
        confidence: float = 0.0,
        sources: List[Dict[str, Any]] = None,
        response_time: float = 0.0,
        metadata: Dict[str, Any] = None,
        success: bool = True
    ):
        """
        Initialize Q&A result.
        
        Args:
            question: Original question
            answer: Generated answer
            confidence: Confidence score (0-1)
            sources: List of source documents/passages
            response_time: Time taken to generate answer
            metadata: Additional metadata
            success: Whether the query was successful
        """
        self.question = question
        self.answer = answer
        self.confidence = confidence
        self.sources = sources or []
        self.response_time = response_time
        self.metadata = metadata or {}
        self.success = success
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'question': self.question,
            'answer': self.answer,
            'confidence': self.confidence,
            'sources': self.sources,
            'response_time': self.response_time,
            'metadata': self.metadata,
            'success': self.success
        }
    
    def __repr__(self) -> str:
        """String representation of result."""
        return f"QAResult(question='{self.question[:50]}...', confidence={self.confidence:.2f})"
