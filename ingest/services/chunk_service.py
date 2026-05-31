import logging
from ingest.services.semantic_chunk import chunk_document
from ingest.services.chunking_strategies import get_chunker

logger = logging.getLogger(__name__)


def chunk_text(text: str, strategy: str = "semantic") -> list[str]:
    """
    Split text into chunks using specified strategy.
    
    Strategies:
    - "semantic": Groups by meaning similarity (recommended for quality)
    - "recursive": Recursive split by delimiters (balanced)
    - "simple": Fixed-size word chunks (fastest)
    
    Args:
        text: Input text to chunk
        strategy: Chunking strategy ("semantic", "recursive", "simple")
    
    Returns:
        List of chunks
    """
    if not text or not text.strip():
        return []

    try:
        chunker = get_chunker(strategy)
        chunks = chunker.chunk(text)
        
        if chunks:
            logger.info(f"Generated {len(chunks)} chunks using {strategy} strategy")
        
        return chunks
    except Exception as e:
        logger.error(f"Chunking failed with {strategy}: {e}. Falling back to semantic.")
        # Fallback
        chunks = chunk_document(text)
        return chunks if chunks else []
