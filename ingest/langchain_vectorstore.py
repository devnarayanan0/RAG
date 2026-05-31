"""
LangChain Pinecone VectorStore wrapper.

Wraps Pinecone with LangChain for consistent API.
Handles document storage, retrieval, and deletion.
"""

import logging
from typing import List
from langchain.schema import Document
from langchain_pinecone import PineconeVectorStore
from ingest.langchain_embeddings import get_embedding_model
from pinecone import Pinecone
import os

logger = logging.getLogger(__name__)

_vector_store = None


def get_vector_store() -> PineconeVectorStore:
    """
    Get or initialize Pinecone vector store.
    
    Returns:
        PineconeVectorStore instance
    """
    global _vector_store
    
    if _vector_store is None:
        try:
            logger.info("Initializing Pinecone vector store...")
            
            # Initialize Pinecone
            pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            index = pc.Index(os.getenv("PINECONE_INDEX"))
            
            # Get embeddings
            embeddings = get_embedding_model()
            
            # Create vector store
            _vector_store = PineconeVectorStore(
                index=index,
                embedding=embeddings,
                text_key="text"
            )
            
            logger.info("✓ Pinecone vector store initialized")
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise
    
    return _vector_store


def add_documents(documents: List[Document]) -> int:
    """
    Add documents to vector store.
    
    Args:
        documents: List of LangChain Documents to store
    
    Returns:
        Number of documents added
    """
    try:
        vector_store = get_vector_store()
        vector_store.add_documents(documents)
        logger.info(f"Added {len(documents)} documents to vector store")
        return len(documents)
    except Exception as e:
        logger.error(f"Failed to add documents: {e}")
        return 0


def delete_documents(document_ids: List[str]) -> int:
    """
    Delete documents from vector store.
    
    Args:
        document_ids: List of document IDs to delete
    
    Returns:
        Number of documents deleted
    """
    try:
        vector_store = get_vector_store()
        vector_store.delete(ids=document_ids)
        logger.info(f"Deleted {len(document_ids)} documents from vector store")
        return len(document_ids)
    except Exception as e:
        logger.error(f"Failed to delete documents: {e}")
        return 0


def similarity_search(query: str, k: int = 5) -> List[Document]:
    """
    Search for similar documents.
    
    Args:
        query: Search query
        k: Number of results to return
    
    Returns:
        List of relevant documents
    """
    try:
        vector_store = get_vector_store()
        results = vector_store.similarity_search(query, k=k)
        logger.info(f"Found {len(results)} similar documents")
        return results
    except Exception as e:
        logger.error(f"Similarity search failed: {e}")
        return []


def as_retriever(search_kwargs: dict = None):
    """
    Get vector store as LangChain retriever.
    
    Args:
        search_kwargs: Retriever search parameters
    
    Returns:
        LangChain Retriever object
    """
    try:
        vector_store = get_vector_store()
        if search_kwargs is None:
            search_kwargs = {"k": 5}
        return vector_store.as_retriever(search_kwargs=search_kwargs)
    except Exception as e:
        logger.error(f"Failed to get retriever: {e}")
        return None
