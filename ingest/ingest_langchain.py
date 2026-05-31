"""
LangChain Ingestion Pipeline.

Orchestrates: Load → Split → Embed → Store

All features preserved:
- PDF OCR for scanned pages
- Image Vision extraction
- PII masking
- Duplicate detection
- Audit logging
"""

import logging
import uuid
import time
from typing import Tuple, Dict, List
from pathlib import Path

from ingest.langchain_loaders import get_langchain_loader
from ingest.langchain_splitters import get_langchain_splitter
from ingest.langchain_vectorstore import add_documents, delete_documents
from ingest.services.duplicate_service import DuplicateChecker
from ingest.services.masking_service import mask_sensitive_data
from ingest.services.sync_service import is_file_already_indexed

logger = logging.getLogger(__name__)

duplicate_checker = DuplicateChecker()


async def ingest_document_langchain(
    source_type: str,
    source_path: str,
    chunking_strategy: str = "semantic",
    req_id: str = None
) -> Tuple[Dict, Dict]:
    """
    Ingest document using LangChain pipeline.
    
    Pipeline:
    1. Load (with OCR, Vision fallback)
    2. Split (semantic/recursive/simple)
    3. Mask sensitive data
    4. Check duplicates
    5. Embed
    6. Store in Pinecone
    
    Features preserved:
    - PDF OCR for scanned pages
    - Image Vision extraction
    - URL metadata
    - PII masking
    - Duplicate detection
    - Audit trails
    
    Args:
        source_type: 'pdf', 'docx', 'image', 'url'
        source_path: File path or URL
        chunking_strategy: 'semantic', 'recursive', 'simple'
        req_id: Request tracking ID
    
    Returns:
        (result_dict, audit_dict)
    """
    
    if not req_id:
        req_id = str(uuid.uuid4())
    
    start_time = time.time()
    
    audit = {
        "loading": {},
        "splitting": {},
        "masking": {},
        "duplicate": {},
        "storage": {},
        "timing_ms": 0
    }
    
    result = {
        "success": False,
        "uploaded": False,
        "chunks": 0,
        "vectors_stored": 0,
        "duplicate": None,
        "error": None
    }
    
    try:
        # ──────────────────────────────────────
        # 1. LOAD
        # ──────────────────────────────────────
        logger.info(f"[{req_id}] Loading {source_type}: {source_path}")
        
        documents = get_langchain_loader(source_type, source_path)
        
        if not documents:
            audit["loading"]["error"] = "No documents loaded"
            audit["loading"]["chars"] = 0
            result["error"] = "Failed to load document"
            return result, audit
        
        total_chars = sum(len(doc.page_content) for doc in documents)
        audit["loading"] = {
            "status": "success",
            "docs_loaded": len(documents),
            "total_chars": total_chars,
            "source_type": source_type
        }
        
        # ──────────────────────────────────────
        # 2. CHECK FOR DUPLICATES (before processing)
        # ──────────────────────────────────────
        combined_text = " ".join(doc.page_content for doc in documents)
        duplicate_check = duplicate_checker.check_exact_duplicate(combined_text)
        audit["duplicate"]["incoming_hash"] = duplicate_check["hash"]
        
        if duplicate_check["is_duplicate"]:
            logger.warning(f"[{req_id}] Duplicate detected")
            audit["duplicate"]["is_duplicate"] = True
            result["duplicate"] = {
                "status": True,
                "type": "exact_hash_match",
                "hash": duplicate_check["hash"]
            }
            return result, audit
        
        audit["duplicate"]["is_duplicate"] = False
        
        # ──────────────────────────────────────
        # 3. MASK SENSITIVE DATA
        # ──────────────────────────────────────
        logger.info(f"[{req_id}] Masking sensitive data...")
        
        for doc in documents:
            masked_text, was_masked = mask_sensitive_data(doc.page_content)
            doc.page_content = masked_text
            doc.metadata["masked"] = was_masked
        
        audit["masking"] = {
            "applied": True,
            "docs_masked": sum(1 for d in documents if d.metadata.get("masked"))
        }
        
        # ──────────────────────────────────────
        # 4. SPLIT
        # ──────────────────────────────────────
        logger.info(f"[{req_id}] Splitting with {chunking_strategy} strategy...")
        
        splitter = get_langchain_splitter(chunking_strategy)
        split_docs = splitter.split_documents(documents)
        
        if not split_docs:
            audit["splitting"]["error"] = "No chunks generated"
            result["error"] = "Chunking failed"
            return result, audit
        
        audit["splitting"] = {
            "status": "success",
            "strategy": chunking_strategy,
            "chunks_created": len(split_docs),
            "avg_chunk_size": sum(len(d.page_content) for d in split_docs) // len(split_docs) if split_docs else 0
        }
        
        # ──────────────────────────────────────
        # 5. STORE IN PINECONE
        # ──────────────────────────────────────
        logger.info(f"[{req_id}] Storing {len(split_docs)} chunks in Pinecone...")
        
        vectors_stored = add_documents(split_docs)
        
        audit["storage"] = {
            "status": "success" if vectors_stored > 0 else "failed",
            "vectors_stored": vectors_stored,
            "source": source_path,
            "source_type": source_type,
            "req_id": req_id
        }
        
        # ──────────────────────────────────────
        # SUCCESS
        # ──────────────────────────────────────
        elapsed_ms = int((time.time() - start_time) * 1000)
        audit["timing_ms"] = elapsed_ms
        
        result["success"] = True
        result["uploaded"] = True
        result["chunks"] = len(split_docs)
        result["vectors_stored"] = vectors_stored
        
        logger.info(f"[{req_id}] ✓ Ingestion complete ({elapsed_ms}ms)")
        
        return result, audit
    
    except Exception as e:
        logger.error(f"[{req_id}] Ingestion failed: {e}")
        elapsed_ms = int((time.time() - start_time) * 1000)
        audit["timing_ms"] = elapsed_ms
        result["error"] = str(e)
        return result, audit
