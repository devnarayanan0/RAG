"""
Chunking strategy implementations for document text segmentation.

Strategies:
- SEMANTIC: Groups sentences by meaning similarity (0.7 threshold)
- RECURSIVE: Splits by hierarchical delimiters (sentence → line → word)
- SIMPLE: Fixed-size chunks with word-based splitting
"""

import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import List
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Lazy-load embedding model
_embedding_model = None

def get_embedding_model():
    """Lazy-load sentence transformer model."""
    global _embedding_model
    if _embedding_model is None:
        try:
            _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            return None
    return _embedding_model


class ChunkingStrategy(ABC):
    """Base class for chunking strategies."""
    
    @abstractmethod
    def chunk(self, text: str) -> List[str]:
        """Split text into chunks."""
        pass
    
    @staticmethod
    def _validate_text(text: str) -> str:
        """Validate and normalize input text."""
        if not text or not isinstance(text, str):
            return ""
        return text.strip()


class SemanticChunker(ChunkingStrategy):
    """
    Groups sentences by semantic similarity.
    
    Configuration:
    - SIMILARITY_THRESHOLD: 0.7 (group similar sentences)
    - MIN_CHUNK_WORDS: 50
    - MAX_CHUNK_WORDS: 1000
    """
    
    SIMILARITY_THRESHOLD = 0.7
    MIN_CHUNK_WORDS = 50
    MAX_CHUNK_WORDS = 1000
    
    def chunk(self, text: str) -> List[str]:
        """Chunk text based on semantic similarity."""
        text = self._validate_text(text)
        if not text:
            return []
        
        sentences = self._split_sentences(text)
        if not sentences:
            return []
        
        model = get_embedding_model()
        if not model:
            # Fallback to simple chunker
            return SimpleChunker().chunk(text)
        
        embeddings = self._get_embeddings(sentences, model)
        if not embeddings:
            return SimpleChunker().chunk(text)
        
        grouped = self._group_by_similarity(sentences, embeddings)
        chunks = self._build_chunks(grouped)
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences, handling abbreviations."""
        sentences = []
        current = ""
        
        for i, char in enumerate(text):
            current += char
            
            if char in ".!?":
                # Check if followed by space or newline
                if i + 1 < len(text) and text[i + 1] not in " \n":
                    continue
                
                sent = current.strip()
                if sent and len(sent) > 3:
                    sentences.append(sent)
                    current = ""
            elif char == "\n" and current.strip():
                sent = current.strip()
                if sent and len(sent) > 3:
                    sentences.append(sent)
                    current = ""
        
        if current.strip() and len(current.strip()) > 3:
            sentences.append(current.strip())
        
        return sentences
    
    def _get_embeddings(self, sentences: List[str], model) -> List[np.ndarray]:
        """Generate embeddings for sentences."""
        try:
            embeddings = model.encode(sentences, convert_to_numpy=True)
            if isinstance(embeddings, np.ndarray) and len(embeddings.shape) == 1:
                return [embeddings]
            return list(embeddings) if isinstance(embeddings, np.ndarray) else embeddings
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return []
    
    def _calculate_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity."""
        if vec1 is None or vec2 is None:
            return 0.0
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    def _group_by_similarity(self, sentences: List[str], embeddings: List) -> List[List[str]]:
        """Group sentences by semantic similarity."""
        if not sentences:
            return []
        
        groups = [[sentences[0]]]
        
        for i in range(1, len(sentences)):
            if i < len(embeddings) and i - 1 < len(embeddings):
                sim = self._calculate_similarity(embeddings[i - 1], embeddings[i])
                
                if sim >= self.SIMILARITY_THRESHOLD:
                    groups[-1].append(sentences[i])
                else:
                    groups.append([sentences[i]])
            else:
                groups.append([sentences[i]])
        
        return groups
    
    def _build_chunks(self, groups: List[List[str]]) -> List[str]:
        """Build final chunks from groups, respecting size limits."""
        chunks = []
        current_chunk = []
        current_words = 0
        
        for group in groups:
            group_text = " ".join(group)
            group_words = len(group_text.split())
            
            # If adding this group exceeds max, finalize current chunk
            if current_words + group_words > self.MAX_CHUNK_WORDS and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [group_text]
                current_words = group_words
            else:
                current_chunk.append(group_text)
                current_words += group_words
        
        # Add remaining
        if current_chunk:
            final_text = " ".join(current_chunk)
            if len(final_text.split()) >= self.MIN_CHUNK_WORDS:
                chunks.append(final_text)
        
        return chunks if chunks else [" ".join(g) for g in groups]


class RecursiveTextSplitter(ChunkingStrategy):
    """
    Recursively splits by delimiters (sentence → line → word).
    
    Configuration:
    - CHUNK_SIZE: 500 characters
    - OVERLAP: 50 characters
    """
    
    CHUNK_SIZE = 500
    OVERLAP = 50
    
    def chunk(self, text: str) -> List[str]:
        """Chunk text recursively by delimiters."""
        text = self._validate_text(text)
        if not text:
            return []
        
        # Try splitting by sentences first
        chunks = self._split_recursive(text, ["\n\n", "\n", ". ", " "])
        return chunks
    
    def _split_recursive(self, text: str, delimiters: List[str]) -> List[str]:
        """Recursively split by delimiters."""
        final_chunks = []
        separator = delimiters[-1]
        
        for i, delim in enumerate(delimiters):
            if delim in text:
                separator = delim
                break
        
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)
        
        good_splits = [s for s in splits if len(s) > 0]
        
        # Merge small splits
        merged = self._merge_splits(good_splits, separator)
        
        # Chunk by size
        for chunk in merged:
            if len(chunk) > self.CHUNK_SIZE:
                if delimiters:
                    sub_chunks = self._split_recursive(chunk, delimiters[delimiters.index(separator) + 1:])
                    final_chunks.extend(sub_chunks)
                else:
                    final_chunks.append(chunk)
            else:
                final_chunks.append(chunk)
        
        return [c for c in final_chunks if c.strip()]
    
    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """Merge small splits to reach minimum size."""
        separator_len = len(separator)
        good_splits = []
        current = ""
        
        for split in splits:
            if not split.strip():
                continue
            
            if len(current) + len(split) + separator_len <= self.CHUNK_SIZE:
                current += separator + split if current else split
            else:
                if current:
                    good_splits.append(current)
                current = split
        
        if current:
            good_splits.append(current)
        
        return good_splits


class SimpleChunker(ChunkingStrategy):
    """
    Fixed-size chunks by word count.
    
    Configuration:
    - CHUNK_SIZE: 300 words
    - OVERLAP: 50 words
    """
    
    CHUNK_SIZE = 300
    OVERLAP = 50
    
    def chunk(self, text: str) -> List[str]:
        """Chunk text into fixed-size chunks by words."""
        text = self._validate_text(text)
        if not text:
            return []
        
        words = text.split()
        if not words:
            return []
        
        chunks = []
        step = self.CHUNK_SIZE - self.OVERLAP
        
        for i in range(0, len(words), step):
            chunk_words = words[i : i + self.CHUNK_SIZE]
            chunk_text = " ".join(chunk_words)
            if chunk_text.strip():
                chunks.append(chunk_text)
        
        return chunks


def get_chunker(strategy: str = "semantic") -> ChunkingStrategy:
    """
    Factory function to get chunking strategy.
    
    Args:
        strategy: One of "semantic", "recursive", "simple"
    
    Returns:
        ChunkingStrategy instance
    """
    strategy_lower = (strategy or "semantic").lower().strip()
    
    if strategy_lower == "recursive":
        return RecursiveTextSplitter()
    elif strategy_lower == "simple":
        return SimpleChunker()
    else:
        return SemanticChunker()
