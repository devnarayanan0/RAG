import logging
import os
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from groq import Groq
from dotenv import load_dotenv
from ingest.reranker.cross_encoder import rerank_chunks

load_dotenv()

# Initialize clients and models
pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
vector_index = pinecone_client.Index(os.getenv("PINECONE_INDEX"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

logger = logging.getLogger(__name__)

# Configuration
TOP_K_RETRIEVAL = 10
TOP_K_FINAL = 5
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 500
MIN_TEXT_LENGTH = 50

SYSTEM_PROMPT = """Answer only from the given context.

Format for chat:
- Short lines
- Blank lines
- Bullet points
- No huge paragraphs

If answer missing say exactly:
I could not find that in the retrieved documents."""


def embed_question(question: str) -> list:
    """Generate embedding vector for question."""
    embedding = embedding_model.encode([question])[0]
    return embedding.tolist()


def retrieve_similar_chunks(vector: list) -> list:
    """Retrieve top-K similar chunks from vector database."""
    result = vector_index.query(
        vector=vector,
        top_k=TOP_K_RETRIEVAL,
        include_metadata=True
    )
    return result.get("matches", [])


def extract_text_from_matches(matches: list) -> str:
    """Extract and concatenate text from retrieved matches."""
    texts = []
    for match in matches:
        metadata = match.get("metadata", {})
        text = metadata.get("text", "")
        if text:
            texts.append(text)
    return "\n\n".join(texts)


def format_prompt(question: str, context: str) -> str:
    """Create LLM prompt from question and context."""
    return f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion:\n{question}"


def convert_matches_to_chunks(matches: list) -> list:
    """Convert Pinecone matches to internal chunk format."""
    chunks = []
    for match in matches:
        metadata = match.get("metadata", {})
        chunks.append({
            "id": match.get("id", ""),
            "chunk_id": metadata.get("chunk_id", ""),
            "text": metadata.get("text", ""),
            "vector_score": match.get("score", 0.0),
            "metadata": metadata
        })
    return chunks


def extract_top_scores(reranked_chunks: list) -> list:
    """Extract and format retrieval scores from reranked chunks."""
    scores = []
    for chunk in reranked_chunks:
        scores.append({
            "id": chunk.get("id", chunk.get("chunk_id", "")),
            "vector_score": chunk.get("vector_score", 0.0),
            "relevance_score": chunk.get("relevance_score", 0.0),
            "text_preview": chunk.get("text", "")[:200]
        })
    return scores


def rerank_and_select_best(question: str, chunks: list) -> tuple:
    """Rerank chunks and select top final results."""
    rerank_result = rerank_chunks(question, chunks)
    
    if rerank_result.get("error"):
        return None, rerank_result
    
    reranked = rerank_result.get("chunks", [])[:TOP_K_FINAL]
    stats = {
        "retrieved": rerank_result.get("original_count", 0),
        "reranked": len(reranked),
        "scores": extract_top_scores(reranked)
    }
    
    return reranked, stats


def generate_answer(question: str, context: str) -> str:
    """Generate LLM response from question and context."""
    prompt = format_prompt(question, context)
    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS
    )
    return response.choices[0].message.content


def ask_question(question: str) -> str:
    """Answer question using retrieval and generation (no reranking)."""
    query_vector = embed_question(question)
    matches = retrieve_similar_chunks(query_vector)
    
    if not matches:
        return "No relevant context found"
    
    context = extract_text_from_matches(matches)
    return generate_answer(question, context)


def ask_question_with_reranking(question: str) -> dict:
    """Answer question with cross-encoder reranking and retrieval stats."""
    query_vector = embed_question(question)
    matches = retrieve_similar_chunks(query_vector)
    
    if not matches:
        return {
            "response": "No relevant context found",
            "retrieval": {
                "retrieved": 0,
                "reranked": 0,
                "scores": []
            }
        }
    
    chunks = convert_matches_to_chunks(matches)
    reranked, stats = rerank_and_select_best(question, chunks)
    
    if reranked is None:
        context = extract_text_from_matches(matches)
    else:
        context = "\n\n".join([chunk.get("text", "") for chunk in reranked])
    
    answer = generate_answer(question, context)
    
    return {
        "response": answer,
        "retrieval": stats
    }
