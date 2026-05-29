import logging
from ingest.services.semantic_chunk import chunk_document

logger = logging.getLogger(__name__)


def chunk_text(text: str) -> list[str]:
    """
    Split text into semantic chunks based on meaning and context.
    
    Uses sentence embeddings to group related content together,
    ensuring chunks represent complete topics or ideas.
    
    Args:
        text: Input text to chunk
    
    Returns:
        List of semantic chunks
    """
    if not text or not text.strip():
        return []

    chunks = chunk_document(text)
    
    if chunks:
        logger.info(f"Generated {len(chunks)} semantic chunks")
    
    return chunks
