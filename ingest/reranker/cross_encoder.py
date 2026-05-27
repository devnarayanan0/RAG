import logging
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

CROSS_ENCODER_MODEL = "cross-encoder/mmarco-MiniLMv2-L12-H384-v1"

try:
    reranker_model = CrossEncoder(CROSS_ENCODER_MODEL)
    logger.info(f"✓ Cross Encoder loaded: {CROSS_ENCODER_MODEL}")
except Exception as e:
    logger.error(f"Failed to load Cross Encoder: {e}")
    reranker_model = None


def rerank_chunks(query: str, chunks: list) -> dict:
    """
    Rerank retrieved chunks using BERT cross-encoder.
    
    Computes relevance scores for (query, chunk) pairs and sorts by score.
    
    Args:
        query: User query string
        chunks: List of chunks with format {id, text, vector_score, metadata}
    
    Returns:
        {
            original_count: Number of input chunks,
            reranked_count: Number of reranked chunks,
            chunks: Reranked list with added relevance_score,
            error: Error message if any
        }
    """
    if not reranker_model:
        logger.error("Cross Encoder not initialized")
        return {
            "error": "Cross Encoder unavailable",
            "original_count": len(chunks),
            "reranked_count": 0,
            "chunks": []
        }
    
    if not chunks:
        return {
            "original_count": 0,
            "reranked_count": 0,
            "chunks": [],
            "error": None
        }
    
    try:
        query_chunk_pairs = [[query, chunk.get("text", "")] for chunk in chunks]
        relevance_scores = reranker_model.predict(query_chunk_pairs)
        
        reranked_chunks = []
        for chunk, score in zip(chunks, relevance_scores):
            reranked_chunks.append({
                "id": chunk.get("id", chunk.get("chunk_id", "")),
                "text": chunk.get("text", ""),
                "vector_score": chunk.get("vector_score", 0.0),
                "relevance_score": float(score),
                "metadata": chunk.get("metadata", {})
            })
        
        reranked_chunks = sorted(
            reranked_chunks,
            key=lambda x: x["relevance_score"],
            reverse=True
        )
        
        top_score = reranked_chunks[0]["relevance_score"] if reranked_chunks else 0
        logger.info(f"Reranked {len(reranked_chunks)} chunks - Top score: {top_score:.3f}")
        
        return {
            "original_count": len(chunks),
            "reranked_count": len(reranked_chunks),
            "chunks": reranked_chunks,
            "error": None
        }
    
    except Exception as e:
        logger.error(f"Reranking failed: {e}")
        return {
            "original_count": len(chunks),
            "reranked_count": 0,
            "chunks": [],
            "error": str(e)
        }
