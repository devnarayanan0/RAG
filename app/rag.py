import logging
import os
from typing import Optional
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from groq import Groq
from dotenv import load_dotenv
from ingest.reranker.cross_encoder import rerank_chunks

logger = logging.getLogger(__name__)
load_dotenv()

# Configuration
TOP_K_RETRIEVAL = 10
TOP_K_FINAL = 5
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 500
MIN_CONTEXT_LENGTH = 20

SYSTEM_PROMPT = """Answer only from the given context.

Format for chat:
- Short lines
- Blank lines
- Bullet points
- No huge paragraphs

If answer missing say exactly:
I could not find that in the retrieved documents."""

# Lazy initialization for Railway deployment
_embedding_model: Optional[SentenceTransformer] = None
_pinecone_index = None
_groq_client: Optional[Groq] = None


def get_embedding_model() -> Optional[SentenceTransformer]:
    """Get embedding model with lazy initialization."""
    global _embedding_model
    if _embedding_model is None:
        try:
            _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("✓ Embedding model loaded")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
    return _embedding_model


def get_pinecone_index():
    """Get Pinecone index with lazy initialization."""
    global _pinecone_index
    if _pinecone_index is None:
        try:
            api_key = os.getenv("PINECONE_API_KEY")
            index_name = os.getenv("PINECONE_INDEX")
            if not api_key or not index_name:
                raise ValueError("PINECONE_API_KEY or PINECONE_INDEX not set")
            pc = Pinecone(api_key=api_key)
            _pinecone_index = pc.Index(index_name)
            logger.info(f"✓ Pinecone connected to {index_name}")
        except Exception as e:
            logger.error(f"Failed to connect Pinecone: {e}")
    return _pinecone_index


def get_groq_client() -> Optional[Groq]:
    """Get Groq client with lazy initialization."""
    global _groq_client
    if _groq_client is None:
        try:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY not set")
            _groq_client = Groq(api_key=api_key)
            logger.info("✓ Groq client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Groq: {e}")
    return _groq_client


def embed_question(question: str) -> Optional[list]:
    """Generate embedding vector for question."""
    if not question or not isinstance(question, str):
        logger.warning("Invalid question input")
        return None
    
    try:
        model = get_embedding_model()
        if not model:
            return None
        embedding = model.encode([question.strip()])[0]
        return embedding.tolist()
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        return None


def get_chunks(vector: list) -> list:
    """Retrieve top-K chunks from Pinecone."""
    if not vector or len(vector) != 384:
        logger.warning("Invalid vector input")
        return []
    
    try:
        index = get_pinecone_index()
        if not index:
            return []
        result = index.query(vector=vector, top_k=TOP_K_RETRIEVAL, include_metadata=True)
        matches = result.get("matches", [])
        return matches
    except Exception as e:
        logger.error(f"Pinecone query failed: {e}")
        return []


def get_text(chunk: dict) -> str:
    """Safely extract text from chunk metadata."""
    if not isinstance(chunk, dict):
        return ""
    metadata = chunk.get("metadata", {})
    text = metadata.get("text", "")
    return text.strip() if isinstance(text, str) else ""


def build_context(chunks: list) -> str:
    """Build context string from chunks."""
    if not chunks:
        return ""
    texts = [get_text(chunk) for chunk in chunks]
    texts = [t for t in texts if t]
    return "\n\n".join(texts)


def ask_groq(question: str, context: str) -> str:
    """Generate LLM response."""
    if not question or not context:
        return "Insufficient context or question"
    
    try:
        client = get_groq_client()
        if not client:
            return "LLM service unavailable"
        
        prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion:\n{question}"
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Groq request failed: {e}")
        return "Unable to generate response"


def ask(question: str) -> str:
    """Retrieve and answer without reranking."""
    question = question.strip() if question else ""
    if not question:
        return "Empty question"
    
    vector = embed_question(question)
    if not vector:
        return "Embedding failed"
    
    chunks = get_chunks(vector)
    if not chunks:
        return "No relevant context found"
    
    context = build_context(chunks)
    if not context or len(context) < MIN_CONTEXT_LENGTH:
        return "Insufficient context"
    
    return ask_groq(question, context)


def ask_with_rerank(question: str) -> dict:
    """Retrieve, rerank, and answer with stats."""
    question = question.strip() if question else ""
    if not question:
        return {"response": "Empty question", "retrieval": {"retrieved": 0, "reranked": 0, "scores": []}}
    
    vector = embed_question(question)
    if not vector:
        return {"response": "Embedding failed", "retrieval": {"retrieved": 0, "reranked": 0, "scores": []}}
    
    chunks = get_chunks(vector)
    if not chunks:
        return {"response": "No relevant context found", "retrieval": {"retrieved": 0, "reranked": 0, "scores": []}}
    
    # Convert to format expected by reranker
    chunks_for_rerank = []
    for chunk in chunks:
        chunks_for_rerank.append({
            "id": chunk.get("id", ""),
            "chunk_id": chunk.get("metadata", {}).get("chunk_id", ""),
            "text": get_text(chunk),
            "vector_score": chunk.get("score", 0.0),
            "metadata": chunk.get("metadata", {})
        })
    
    # Rerank
    try:
        rerank_result = rerank_chunks(question, chunks_for_rerank)
        if rerank_result.get("error"):
            logger.warning(f"Reranking failed: {rerank_result.get('error')}")
            reranked = chunks_for_rerank
        else:
            reranked = rerank_result.get("chunks", [])[:TOP_K_FINAL]
    except Exception as e:
        logger.error(f"Reranking exception: {e}")
        reranked = chunks_for_rerank[:TOP_K_FINAL]
    
    # Build context from reranked
    context = "\n\n".join([chunk.get("text", "") for chunk in reranked])
    if not context or len(context) < MIN_CONTEXT_LENGTH:
        return {"response": "Insufficient context after reranking", "retrieval": {"retrieved": len(chunks), "reranked": len(reranked), "scores": []}}
    
    # Extract scores for response
    scores = []
    for chunk in reranked:
        scores.append({
            "id": chunk.get("id", chunk.get("chunk_id", "")),
            "vector_score": chunk.get("vector_score", 0.0),
            "relevance_score": chunk.get("relevance_score", 0.0),
            "text_preview": chunk.get("text", "")[:150]
        })
    
    response = ask_groq(question, context)
    
    return {
        "response": response,
        "retrieval": {
            "retrieved": len(chunks),
            "reranked": len(reranked),
            "scores": scores
        }
    }
