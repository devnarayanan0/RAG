import logging
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

SIMILARITY_THRESHOLD = 0.7
MIN_CHUNK_WORDS = 50
MAX_CHUNK_WORDS = 1000
SENTENCE_SPLIT_CHARS = [".!", "?", "\n"]


def split_sentences(text: str) -> List[str]:
    """Split text into sentences, handling abbreviations."""
    if not text or not text.strip():
        return []
    
    text = text.strip()
    sentences = []
    current = ""
    
    for i, char in enumerate(text):
        current += char
        
        if char in ".!?":
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
    
    if current.strip():
        sent = current.strip()
        if sent and len(sent) > 3:
            sentences.append(sent)
    
    return sentences


def create_sentence_embeddings(sentences: List[str]) -> List[np.ndarray]:
    """Generate embedding vectors for sentences."""
    if not sentences:
        return []
    
    try:
        embeddings = embedding_model.encode(sentences, convert_to_numpy=True)
        return embeddings if isinstance(embeddings, list) else [embeddings]
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        return []


def calculate_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    if vec1 is None or vec2 is None:
        return 0.0
    
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def group_semantic_chunks(sentences: List[str], embeddings: List[np.ndarray]) -> List[List[str]]:
    """Group sentences into semantic chunks based on similarity."""
    if not sentences:
        return []
    
    if len(sentences) == 1:
        return [sentences]
    
    chunks = [[sentences[0]]]
    
    for i in range(1, len(sentences)):
        last_sent = sentences[i - 1]
        curr_sent = sentences[i]
        
        similarity = calculate_similarity(embeddings[i - 1], embeddings[i])
        
        chunk_word_count = sum(len(s.split()) for s in chunks[-1])
        curr_word_count = len(curr_sent.split())
        
        should_merge = (
            similarity >= SIMILARITY_THRESHOLD and
            chunk_word_count + curr_word_count <= MAX_CHUNK_WORDS
        )
        
        if should_merge:
            chunks[-1].append(curr_sent)
        else:
            chunks.append([curr_sent])
    
    return chunks


def merge_small_chunks(chunks: List[List[str]]) -> List[List[str]]:
    """Merge chunks that are too small with neighbors."""
    if not chunks:
        return chunks
    
    merged = []
    i = 0
    
    while i < len(chunks):
        current_chunk = chunks[i]
        word_count = sum(len(s.split()) for s in current_chunk)
        
        if word_count < MIN_CHUNK_WORDS and i + 1 < len(chunks):
            current_chunk.extend(chunks[i + 1])
            i += 2
        else:
            merged.append(current_chunk)
            i += 1
    
    return merged


def chunk_document(text: str) -> List[str]:
    """
    Semantically chunk document into meaningful segments.
    
    Args:
        text: Input document text
    
    Returns:
        List of semantic chunks as strings
    """
    if not text or not text.strip():
        return []
    
    # Split into sentences
    sentences = split_sentences(text)
    
    if not sentences:
        return [text]
    
    if len(sentences) == 1:
        return sentences
    
    # Generate embeddings for sentences
    embeddings = create_sentence_embeddings(sentences)
    
    if not embeddings or len(embeddings) != len(sentences):
        fallback_chunks = [s for s in sentences if s.strip()]
        return fallback_chunks if fallback_chunks else [text]
    
    # Group into semantic chunks
    sentence_groups = group_semantic_chunks(sentences, embeddings)
    
    # Merge small chunks
    merged_groups = merge_small_chunks(sentence_groups)
    
    # Join sentences within each chunk
    chunks = [" ".join(group) for group in merged_groups]
    
    logger.info(f"Chunked document: {len(text.split())} words → {len(chunks)} semantic chunks")
    
    return chunks
