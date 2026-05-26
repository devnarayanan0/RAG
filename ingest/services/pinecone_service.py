import uuid
from datetime import datetime
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

model = SentenceTransformer("all-MiniLM-L6-v2")

def generate_embeddings(chunks: list[str]) -> list[dict]:
    """Generate embeddings for chunks."""
    if not chunks:
        return []

    embeddings = model.encode(chunks)
    return [emb.tolist() for emb in embeddings]

def build_vectors(
    chunks: list[str],
    metadata_template: dict,
    chunk_hashes: list[str]
) -> list[dict]:
    """Build vector records for Pinecone."""
    embeddings = generate_embeddings(chunks)
    vectors = []

    for i, (chunk, emb, chunk_hash) in enumerate(zip(chunks, embeddings, chunk_hashes)):
        vector = {
            "id": str(uuid.uuid4()),
            "values": emb,
            "metadata": {
                **metadata_template,
                "chunk_id": str(uuid.uuid4()),
                "chunk_index": i,
                "hash": chunk_hash,
                "text": chunk,
            }
        }
        vectors.append(vector)

    logger.info(f"Built {len(vectors)} vectors for Pinecone")
    return vectors

def upsert_vectors(pc_index, vectors: list[dict]) -> int:
    """Upsert vectors to Pinecone. Returns count of vectors inserted."""
    if not vectors:
        return 0

    pc_index.upsert(vectors=vectors)
    logger.info(f"Upserted {len(vectors)} vectors to Pinecone")
    return len(vectors)
