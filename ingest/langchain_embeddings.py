"""
LangChain HuggingFace Embeddings wrapper.

Uses BERT all-MiniLM-L6-v2 model (384-dimensional embeddings)
Same model as before, just wrapped by LangChain for consistency.
"""

import logging
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

# Lazy-load embeddings
_embedding_model = None


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Lazy-load HuggingFace embeddings.
    
    Model: all-MiniLM-L6-v2
    - Dimensions: 384
    - Size: ~100MB
    - Speed: Fast inference
    - Quality: Excellent semantic similarity
    
    Returns:
        HuggingFaceEmbeddings instance
    """
    global _embedding_model
    
    if _embedding_model is None:
        try:
            logger.info("Loading HuggingFace embeddings (all-MiniLM-L6-v2)...")
            _embedding_model = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},  # Use CPU for Railway compatibility
                encode_kwargs={"normalize_embeddings": True}
            )
            logger.info("✓ Embeddings loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            raise
    
    return _embedding_model


def embed_text(text: str) -> List[float]:
    """
    Embed a single text using LangChain.
    
    Args:
        text: Text to embed
    
    Returns:
        384-dimensional embedding vector
    """
    try:
        model = get_embedding_model()
        embeddings = model.embed_query(text)
        return embeddings
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        return None


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed multiple texts.
    
    Args:
        texts: List of texts to embed
    
    Returns:
        List of embedding vectors
    """
    try:
        model = get_embedding_model()
        embeddings = model.embed_documents(texts)
        return embeddings
    except Exception as e:
        logger.error(f"Batch embedding failed: {e}")
        return []
