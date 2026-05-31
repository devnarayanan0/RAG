"""
LangChain Text Splitters with strategy selection.

Wraps LangChain splitters and custom chunking strategies
while preserving semantic chunking quality.
"""

import logging
from typing import List
from langchain.schema import Document
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter
)

# Import custom semantic chunker (preserved feature)
from ingest.services.semantic_chunk import chunk_document
from ingest.services.chunking_strategies import SemanticChunker

logger = logging.getLogger(__name__)


def get_langchain_splitter(strategy: str = "semantic"):
    """
    Get text splitter based on strategy.
    
    Strategies:
    - "semantic": Custom semantic chunker (best quality)
    - "recursive": LangChain recursive splitter (balanced)
    - "simple": LangChain simple splitter (fastest)
    
    Args:
        strategy: Chunking strategy name
    
    Returns:
        Splitter object with split_documents() method
    """
    strategy_lower = (strategy or "semantic").lower().strip()
    
    if strategy_lower == "recursive":
        return LangChainRecursiveSplitter()
    elif strategy_lower == "simple":
        return LangChainSimpleSplitter()
    else:
        return SemanticChunkSplitter()


class SemanticChunkSplitter:
    """
    Custom semantic chunker using sentence similarity.
    
    Configuration:
    - Min: 50 words, Max: 1000 words
    - Similarity threshold: 0.7
    - Preserves topic coherence
    """
    
    def __init__(self):
        self.name = "semantic"
        self.semantic_chunker = SemanticChunker()
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents using semantic similarity."""
        split_docs = []
        
        for doc in documents:
            if not doc.page_content or not doc.page_content.strip():
                continue
            
            # Use custom semantic chunking
            chunks = self.semantic_chunker.chunk(doc.page_content)
            
            for i, chunk in enumerate(chunks):
                new_doc = Document(
                    page_content=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk_id": i,
                        "chunking_strategy": "semantic",
                        "total_chunks": len(chunks)
                    }
                )
                split_docs.append(new_doc)
        
        logger.info(f"Generated {len(split_docs)} semantic chunks from {len(documents)} documents")
        return split_docs


class LangChainRecursiveSplitter:
    """
    LangChain RecursiveCharacterTextSplitter.
    
    Hierarchical splitting: sentence → line → word
    
    Configuration:
    - Chunk size: 500 characters
    - Overlap: 50 characters
    - Separators: ["\n\n", "\n", ". ", " ", ""]
    """
    
    def __init__(self):
        self.name = "recursive"
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents using recursive character splitting."""
        split_docs = self.splitter.split_documents(documents)
        
        # Add metadata
        for i, doc in enumerate(split_docs):
            doc.metadata["chunk_id"] = i
            doc.metadata["chunking_strategy"] = "recursive"
        
        logger.info(f"Generated {len(split_docs)} recursive chunks from {len(documents)} documents")
        return split_docs


class LangChainSimpleSplitter:
    """
    LangChain CharacterTextSplitter (simple).
    
    Fixed-size character chunks.
    
    Configuration:
    - Chunk size: 1000 characters
    - Overlap: 100 characters
    - Separator: "\n\n"
    """
    
    def __init__(self):
        self.name = "simple"
        self.splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separator="\n\n",
            length_function=len
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents using simple character splitting."""
        split_docs = self.splitter.split_documents(documents)
        
        # Add metadata
        for i, doc in enumerate(split_docs):
            doc.metadata["chunk_id"] = i
            doc.metadata["chunking_strategy"] = "simple"
        
        logger.info(f"Generated {len(split_docs)} simple chunks from {len(documents)} documents")
        return split_docs
