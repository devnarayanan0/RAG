from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging

logger = logging.getLogger(__name__)

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

def chunk_text(text: str) -> list[str]:
    """Split text into chunks with overlap."""
    if not text or not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_text(text)
    
    # Filter empty chunks
    chunks = [c.strip() for c in chunks if c.strip()]

    logger.info(f"Generated {len(chunks)} chunks")
    return chunks
