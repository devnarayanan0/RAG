"""
LangChain Retrieval Pipeline.

Uses RetrievalQA chain with Groq LLM for RAG queries.

Features:
- Semantic search in Pinecone
- Groq LLM integration
- Retrieved source tracking
- Error handling & fallbacks
"""

import logging
import os
from typing import Dict
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from ingest.langchain_vectorstore import as_retriever, similarity_search
from langchain.schema import Document

logger = logging.getLogger(__name__)

_llm = None


def get_groq_llm() -> ChatGroq:
    """
    Get or initialize Groq LLM.
    
    Model: llama-3.3-70b-versatile
    Config:
    - Temperature: 0.1 (deterministic)
    - Max tokens: 500
    
    Returns:
        ChatGroq instance
    """
    global _llm
    
    if _llm is None:
        try:
            logger.info("Initializing Groq LLM...")
            _llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0.1,
                max_tokens=500,
                api_key=os.getenv("GROQ_API_KEY")
            )
            logger.info("✓ Groq LLM initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Groq: {e}")
            raise
    
    return _llm


def ask(question: str) -> str:
    """
    Ask a question using LangChain RetrievalQA.
    
    Pipeline:
    1. Retrieve relevant documents
    2. Build context from retrieved docs
    3. Generate answer using Groq LLM
    
    Args:
        question: User question
    
    Returns:
        Answer string
    """
    try:
        logger.info(f"Processing question: {question}")
        
        llm = get_groq_llm()
        retriever = as_retriever(search_kwargs={"k": 5})
        
        if not retriever:
            return "Error: Vector store not available"
        
        # Custom prompt
        prompt_template = """Use the following pieces of context to answer the question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer:"""
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create RetrievalQA chain
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=False
        )
        
        result = qa.run(question)
        logger.info(f"Generated answer: {result[:100]}...")
        
        return result
    
    except Exception as e:
        logger.error(f"Question answering failed: {e}")
        return f"Error: {str(e)}"


def ask_with_sources(question: str) -> Dict:
    """
    Ask a question and return answer with source documents.
    
    Returns:
        {
            "answer": str,
            "sources": [
                {
                    "content": str,
                    "source": str,
                    "relevance": float
                }
            ]
        }
    """
    try:
        logger.info(f"Processing question with sources: {question}")
        
        llm = get_groq_llm()
        retriever = as_retriever(search_kwargs={"k": 5})
        
        if not retriever:
            return {"answer": "Error: Vector store not available", "sources": []}
        
        # Retrieve documents
        retrieved_docs = retriever.get_relevant_documents(question)
        
        # Custom prompt
        prompt_template = """Use the following pieces of context to answer the question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer:"""
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create chain
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        
        result = qa({"query": question})
        
        # Format sources
        sources = []
        for doc in result.get("source_documents", []):
            sources.append({
                "content": doc.page_content[:200],  # First 200 chars
                "source": doc.metadata.get("source", "Unknown"),
                "type": doc.metadata.get("source_type", "unknown")
            })
        
        return {
            "answer": result.get("result", ""),
            "sources": sources
        }
    
    except Exception as e:
        logger.error(f"Question answering with sources failed: {e}")
        return {
            "answer": f"Error: {str(e)}",
            "sources": []
        }


def semantic_search_docs(query: str, k: int = 5) -> list:
    """
    Search for relevant documents without LLM.
    
    Args:
        query: Search query
        k: Number of results
    
    Returns:
        List of relevant Document objects
    """
    try:
        logger.info(f"Semantic search for: {query}")
        docs = similarity_search(query, k=k)
        logger.info(f"Found {len(docs)} relevant documents")
        return docs
    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        return []
