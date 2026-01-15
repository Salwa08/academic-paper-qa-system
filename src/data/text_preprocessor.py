"""Text preprocessing utilities for tokenization and cleaning."""
import re
from typing import List

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text.strip()


def simple_tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    return text.split()