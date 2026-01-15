"""
Configuration management for Q&A system.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import os
from dataclasses import dataclass


@dataclass
class DataConfig:
    """Data configuration settings."""
    raw_path: str = "data/raw/"
    raw_data_path: str = "data/raw/"  # Alias for compatibility
    processed_path: str = "data/processed/"
    processed_data_path: str = "data/processed/"  # Alias for compatibility
    external_data_path: str = "data/external/"
    vector_store_path: str = "data/vector_store/"
    arxiv_categories: list = None
    max_papers: int = 1000
    chunk_size: int = 512
    chunk_overlap: int = 50

    def __post_init__(self):
        if self.arxiv_categories is None:
            self.arxiv_categories = ["cs.AI", "cs.LG", "cs.CL"]
        # Sync aliases
        if self.raw_data_path and not self.raw_path:
            self.raw_path = self.raw_data_path
        if self.processed_data_path and not self.processed_path:
            self.processed_path = self.processed_data_path


@dataclass
class ModelConfig:
    """Model configuration settings."""
    embedding_model: str = "all-MiniLM-L6-v2"
    qa_model: str = "deepset/roberta-base-squad2"
    llm_model: str = "gpt-3.5-turbo"
    max_length: int = 512
    temperature: float = 0.7
    top_k: int = 5


@dataclass
class EvaluationConfig:
    """Evaluation configuration settings."""
    test_questions_path: str = "evaluation/test_questions.json"
    test_size: float = 0.2
    metrics: list = None
    output_path: str = "evaluation/results/"
    human_eval: dict = None  # Optional human eval config
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ["rouge", "bleu", "bert_score", "semantic_similarity"]
        if self.human_eval is None:
            self.human_eval = {}


class Config:
    """Main configuration manager for Q&A system."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize configuration from YAML file."""
        self.config_path = Path(config_path)
        self._config = self._load_config()
        
        # Initialize sub-configurations
        self.data = DataConfig(**self._config.get('data', {}))
        self.model = ModelConfig(**self._config.get('model', {}))
        self.evaluation = EvaluationConfig(**self._config.get('evaluation', {}))
        
        # Other settings
        self.project = self._config.get('project', {})
        self.logging = self._config.get('logging', {})
        self.api = self._config.get('api', {})
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            # Return default config if file doesn't exist
            return self._get_default_config()
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            'project': {
                'name': 'multi-approach-qa-system',
                'version': '1.0.0',
                'description': 'Multi-approach Q&A system for academic papers',
                'author': 'Salwa Khattami'
            },
            'data': {},
            'model': {},
            'evaluation': {},
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
            'api': {
                'host': 'localhost',
                'port': 8000
            }
        }
    
    def get_openai_key(self) -> Optional[str]:
        """Get OpenAI API key from environment or config."""
        # Try environment variable first
        key = os.getenv('OPENAI_API_KEY')
        if key:
            return key
        
        # Try config file
        return self._config.get('api_keys', {}).get('openai')
    
    def get_data_path(self, data_type: str = 'raw') -> Path:
        """Get data path for specified type."""
        if data_type == 'raw':
            return Path(self.data.raw_path)
        elif data_type == 'processed':
            return Path(self.data.processed_path)
        elif data_type == 'vector_store':
            return Path(self.data.vector_store_path)
        else:
            raise ValueError(f"Unknown data type: {data_type}")
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        def deep_update(d: dict, u: dict) -> dict:
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = deep_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        self._config = deep_update(self._config, updates)
        self._save_config()
    
    def _save_config(self) -> None:
        """Save current configuration to file."""
        with open(self.config_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self._config.copy()
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"Config(project='{self.project.get('name', 'unknown')}')"


# Global config instance
config = Config()
