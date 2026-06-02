import os
import logging
import uuid
import time
from datetime import datetime
from collections import deque
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from pinecone import Pinecone

from ingest.ingestRouter import (
    ingest_document, 
    ingest_image_objects,
    create_extraction_session,
    step_encode_image,
    step_get_vision_description,
    step_preview_chunks,
    step_generate_embeddings,
    step_store_in_pinecone,
    get_session_state,
    extract_pdf_live_with_vision
)
from ingest.services.sync_service import (
    sync_folder_to_vector_db,
    scan_local_files,
    fetch_existing_sources,
    detect_new_files,
    detect_deleted_files,
    delete_missing_vectors,
)
from ingest.services.chunk_service import chunk_text
from ingest.services.duplicate_service import DuplicateChecker
from ingest.services.pinecone_service import build_vectors, upsert_vectors

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))

# Audit log storage
audit_logs = deque(maxlen=100)
audit_index = {}
BASE_DATA = "data"

# Chunking strategy storage (global preference)
chunking_config = {"strategy": "semantic"}  # "semantic", "recursive", "simple"

# Extracted text session storage (for chunk preview before upload)
extracted_text_sessions = {}  # session_id -> {"text": str, "filename": str, "upload_time": str}

def query_pinecone_for_hash(hash_val: str) -> dict:
    """Query Pinecone for vectors matching a hash."""
    try:
        results = index.query(vector=[0]*384, top_k=100, filter={"hash": {"$eq": hash_val}})
        if results.get("matches"):
            match = results["matches"][0]
            return {
                "found": True,
                "vector_id": match.get("id"),
                "metadata": match.get("metadata", {}),
                "score": match.get("score")
            }
    except Exception as e:
        logger.debug(f"Pinecone query error: {e}")
    return {"found": False}


app = FastAPI(title="Promantus Ingestion API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
#
@app.get("/", tags=["System"])
async def root():
    """Serve admin frontend."""
    path = "ingest/frontend/admin.html"
    if os.path.exists(path):
        return FileResponse(path, media_type="text/html")
    return {"message": "Ingestion API"}


@app.get("/health", tags=["System"])
def health():
    return {"status": "healthy"}


@app.get("/config/chunking", tags=["Configuration"])
def get_chunking_config():
    """Get current chunking strategy configuration."""
    return {
        "current_strategy": chunking_config.get("strategy", "semantic"),
        "available_strategies": {
            "semantic": {
                "name": "Semantic Chunking",
                "description": "Groups sentences by semantic similarity (meaning). Best for quality.",
                "min_chunk_size": 50,
                "max_chunk_size": 1000,
                "threshold": 0.7
            },
            "recursive": {
                "name": "Recursive Text Splitter",
                "description": "Hierarchical splitting by delimiters (sentence → line → word). Balanced approach.",
                "chunk_size": 500,
                "overlap": 50
            },
            "simple": {
                "name": "Simple Chunker",
                "description": "Fixed-size word-based chunks. Fastest option.",
                "chunk_size": 300,
                "overlap": 50
            }
        }
    }


@app.get("/audit/{req_id}", tags=["Audit"])
def get_audit(req_id: str):
    """Get audit log for request."""
    if req_id in audit_index:
        return audit_index[req_id]
    return {"error": "Not found", "request_id": req_id}


@app.get("/audit/list/recent", tags=["Audit"])
def list_audits():
    """List recent audit logs."""
    return list(audit_logs)


@app.post("/upload", tags=["Ingestion"])
async def upload(
    source_type: str = Form(...),
    extraction_type: str = Form(default="text"),
    chunking_strategy: str = Form(default="semantic"),
    url: str | None = Form(None),
    file: UploadFile | None = File(None),
):
    """Upload and save document (does not ingest or sync)."""
    # Store chunking strategy preference
    chunking_config["strategy"] = chunking_strategy
    req_id = str(uuid.uuid4())
    start_time = time.time()
    
    log = {
        "request_id": req_id,
        "source_type": source_type,
        "extraction_type": extraction_type,
        "upload": {},
        "extraction": {},
        "chunking": {},
        "duplicate": {},
        "pinecone": {},
        "metadata_sample": None,
        "processing_time_ms": 0,
        "status": "pending"
    }

    if source_type == "url":
        if not url:
            log["status"] = "error"
            log["error"] = "URL required"
            audit_logs.append(log)
            audit_index[req_id] = log
            return {"error": "URL required", "request_id": req_id}

        log["upload"] = {"type": "url", "url": url}
        result, audit = await ingest_document(source_type, url, index, req_id)
        log.update(audit)

    elif source_type in ["pdf", "docx", "image"]:
        if not file:
            log["status"] = "error"
            log["error"] = "File required"
            audit_logs.append(log)
            audit_index[req_id] = log
            return {"error": "File required", "request_id": req_id}

        folder = os.path.join(BASE_DATA, source_type)
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, file.filename)

        content = await file.read()
        with open(path, "wb") as f:
            f.write(content)

        logger.info(f"File saved: {path}")
        log["upload"] = {
            "filename": file.filename,
            "path": path,
            "saved": True
        }

        # Handle image object extraction
        if source_type == "image" and extraction_type == "object":
            result, audit = await ingest_image_objects(path, index, req_id)
        else:
            result, audit = await ingest_document(source_type, path, index, req_id)
        
        log.update(audit)

    else:
        log["status"] = "error"
        log["error"] = "Invalid source type"
        audit_logs.append(log)
        audit_index[req_id] = log
        return {"error": "Invalid source type", "request_id": req_id}

    log["processing_time_ms"] = int((time.time() - start_time) * 1000)
    log["status"] = "success" if result.get("uploaded") else "rejected"
    
    # Enrich duplicate response with Pinecone metadata
    if not result.get("uploaded") and result.get("duplicate", {}).get("status"):
        hash_val = result.get("duplicate", {}).get("incoming_hash")
        if hash_val:
            match = query_pinecone_for_hash(hash_val)
            if match.get("found"):
                meta = match.get("metadata", {})
                result["duplicate"]["matched_file"] = meta.get("filename", "Unknown")
                result["duplicate"]["matched_chunk"] = meta.get("chunk_index")
                result["duplicate"]["matched_upload_time"] = meta.get("upload_time")
                result["duplicate"]["matched_source"] = meta.get("source")
                result["duplicate"]["existing_vector_id"] = match.get("vector_id")
    
    audit_logs.append(log)
    audit_index[req_id] = log

    return {**result, "request_id": req_id}


@app.post("/pdf/live-extract", tags=["PDF Extraction"])
async def pdf_live_extract(file: UploadFile = File(...)):
    """Live PDF extraction with embedded images and vision descriptions."""
    try:
        folder = os.path.join(BASE_DATA, "pdf")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, file.filename)
        
        content = await file.read()
        with open(path, "wb") as f:
            f.write(content)
        
        result = await extract_pdf_live_with_vision(path)
        
        return {
            "success": result.get("success", False),
            "filename": result.get("filename"),
            "total_pages": result.get("total_pages", 0),
            "total_images": result.get("total_images", 0),
            "pages": result.get("pages", []),
            "error": result.get("error")
        }
    except Exception as e:
        logger.error(f"PDF live extract error: {e}")
        return {
            "success": False,
            "error": str(e),
            "filename": file.filename if file else None
        }


#
@app.post("/image/start", tags=["Image Extraction"])
async def start_image(file: UploadFile = File(...)):
    """Save image and create extraction session."""
    try:
        folder = os.path.join(BASE_DATA, "image")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, file.filename)
        
        content = await file.read()
        with open(path, "wb") as f:
            f.write(content)
        
        session_id = create_extraction_session(path)
        
        return {
            "success": True,
            "session_id": session_id,
            "filename": file.filename,
            "path": path,
            "size_bytes": len(content)
        }
    except Exception as e:
        logger.error(f"Image start error: {e}")
        return {"success": False, "error": str(e)}


@app.post("/image/encode", tags=["Image Extraction"])
async def encode_image(session_id: str = Form(...)):
    """Encode image to base64."""
    try:
        result = await step_encode_image(session_id)
        return result
    except Exception as e:
        logger.error(f"Image encode error: {e}")
        return {"success": False, "error": str(e)}


@app.post("/image/describe", tags=["Image Extraction"])
async def describe_image(session_id: str = Form(...)):
    """Get vision description."""
    try:
        result = await step_get_vision_description(session_id)
        return result
    except Exception as e:
        logger.error(f"Image describe error: {e}")
        return {"success": False, "error": str(e)}


@app.post("/image/preview-chunks", tags=["Image Extraction"])
async def preview_chunks(session_id: str = Form(...)):
    """Preview text chunks."""
    try:
        result = await step_preview_chunks(session_id)
        return result
    except Exception as e:
        logger.error(f"Preview chunks error: {e}")
        return {"success": False, "error": str(e)}


@app.post("/image/embed", tags=["Image Extraction"])
async def embed_chunks(session_id: str = Form(...)):
    """Generate embeddings."""
    try:
        result = await step_generate_embeddings(session_id)
        return result
    except Exception as e:
        logger.error(f"Embed error: {e}")
        return {"success": False, "error": str(e)}


@app.post("/image/store", tags=["Image Extraction"])
async def store_image(session_id: str = Form(...)):
    """Store in Pinecone."""
    try:
        result = await step_store_in_pinecone(session_id, index)
        return result
    except Exception as e:
        logger.error(f"Store error: {e}")
        return {"success": False, "error": str(e)}


@app.get("/image/session/{session_id}", tags=["Image Extraction"])
async def get_image_session(session_id: str):
    """Get extraction session state."""
    try:
        result = await get_session_state(session_id)
        return result
    except Exception as e:
        logger.error(f"Session error: {e}")
        return {"success": False, "error": str(e)}


@app.post("/preview-chunks", tags=["Chunking"])
async def preview_chunks_endpoint(
    extracted_text: str = Form(...),
    chunking_strategy: str = Form(default="semantic"),
    filename: str = Form(default="document")
):
    """
    Preview chunks before uploading to Pinecone.
    
    Args:
        extracted_text: Text extracted from document
        chunking_strategy: Strategy to use ("semantic", "recursive", "simple")
        filename: Original filename for reference
    """
    try:
        logger.info(f"[preview-chunks] strategy={chunking_strategy}, filename={filename}, text_len={len(extracted_text)}")
        # Generate chunks with selected strategy
        chunks = chunk_text(extracted_text, strategy=chunking_strategy)
        
        if not chunks:
            logger.warning(f"[preview-chunks] No chunks generated for strategy={chunking_strategy}")
            return {
                "success": False,
                "error": "No chunks generated",
                "strategy": chunking_strategy
            }
        
        # Store for later upload
        session_id = str(uuid.uuid4())
        extracted_text_sessions[session_id] = {
            "text": extracted_text,
            "filename": filename,
            "chunks": chunks,
            "strategy": chunking_strategy,
            "created_at": datetime.utcnow().isoformat()
        }
        logger.info(f"[preview-chunks] Created session: session_id={session_id}, total_chunks={len(chunks)}")
        
        # Return preview (first 3 chunks + summary)
        preview_chunks = chunks[:3]
        return {
            "success": True,
            "session_id": session_id,
            "total_chunks": len(chunks),
            "strategy": chunking_strategy,
            "preview": preview_chunks,
            "stats": {
                "text_length": len(extracted_text),
                "avg_chunk_size": sum(len(c) for c in chunks) // len(chunks) if chunks else 0,
                "min_chunk_size": min(len(c) for c in chunks) if chunks else 0,
                "max_chunk_size": max(len(c) for c in chunks) if chunks else 0
            }
        }
    except Exception as e:
        logger.error(f"Preview chunks error: {e}")
        return {"success": False, "error": str(e)}


@app.post("/upload-extracted", tags=["Ingestion"])
async def upload_extracted(
    session_id: str = Form(...),
    source_type: str = Form(default="extracted"),
    filename: str = Form(default="document")
):
    """
    Upload previewed chunks to Pinecone vector database.
    
    Args:
        session_id: Session ID from preview-chunks
        source_type: Source type (pdf, docx, image, etc.)
        filename: Original filename
    """
    try:
        logger.info(f"[upload-extracted] session_id={session_id}, filename={filename}, source_type={source_type}")
        # Retrieve session
        if session_id not in extracted_text_sessions:
            logger.error(f"[upload-extracted] Invalid session ID: {session_id}")
            return {"success": False, "error": "Invalid session ID"}
        
        session = extracted_text_sessions[session_id]
        chunks = session.get("chunks", [])
        strategy = session.get("strategy", "semantic")
        
        if not chunks:
            logger.error(f"[upload-extracted] No chunks found in session {session_id}")
            return {"success": False, "error": "No chunks in session"}
        
        # Generate metadata and hashes
        duplicate_checker = DuplicateChecker()
        chunk_hashes = [duplicate_checker.generate_hash(c) for c in chunks]
        
        base_metadata = {
            "source": f"{source_type}/{filename}",
            "source_type": source_type,
            "filename": filename,
            "upload_time": datetime.utcnow().isoformat(),
            "ocr_used": False,
            "masked": False,
            "extraction_type": "extracted",
            "chunking_strategy": strategy,
            "description_generated": False,
            "llm": "none"
        }
        
        # Build vectors
        logger.info(f"[upload-extracted] Building {len(chunks)} vectors...")
        embedding_vectors = build_vectors(chunks, base_metadata, chunk_hashes)
        
        if not embedding_vectors:
            logger.error("[upload-extracted] Failed to build vectors.")
            return {"success": False, "error": "Failed to build vectors"}
        
        # Upsert to Pinecone
        logger.info(f"[upload-extracted] Upserting vectors to Pinecone...")
        stored_count = upsert_vectors(index, embedding_vectors)
        logger.info(f"[upload-extracted] Upsert success. Written: {stored_count}")
        
        # Clean up session
        del extracted_text_sessions[session_id]
        
        return {
            "success": True,
            "vectors_written": stored_count,
            "total_chunks": len(chunks),
            "filename": filename,
            "strategy": strategy,
            "processing_time_ms": 0,
            "metadata_sample": embedding_vectors[0].get("metadata") if embedding_vectors else None
        }
    except Exception as e:
        logger.error(f"Upload extracted error: {e}")
        return {"success": False, "error": str(e)}


@app.get("/sync/compare", tags=["Synchronization"])
async def sync_compare():
    """
    Compare local files with Pinecone index to show pending changes (VCS-style diff).
    """
    try:
        local_files = scan_local_files()
        existing_sources = fetch_existing_sources(index)
        
        new_files = sorted(list(detect_new_files(local_files, existing_sources)))
        deleted_files = sorted(list(detect_deleted_files(local_files, existing_sources)))
        in_sync_files = sorted(list(local_files - set(new_files) - set(deleted_files)))
        
        logger.info(f"[sync-compare] new_files={len(new_files)}, deleted_files={len(deleted_files)}, in_sync_files={len(in_sync_files)}")
        return {
            "success": True,
            "new_files": new_files,
            "deleted_files": deleted_files,
            "in_sync_files": in_sync_files
        }
    except Exception as e:
        logger.error(f"Sync compare error: {e}")
        return {"success": False, "error": str(e)}


@app.post("/sync", tags=["Synchronization"])
async def sync(chunking_strategy: str = Form(default=None)):
    """
    Synchronize local files with Pinecone.
    
    Scans data folders, detects new/deleted files,
    ingests new files, and removes deleted files from Pinecone.
    
    Args:
        chunking_strategy: Override global strategy ("semantic", "recursive", "simple")
    """
    # Use provided strategy or fall back to stored preference
    strategy = (chunking_strategy or chunking_config.get("strategy", "semantic")).lower().strip()
    sync_id = str(uuid.uuid4())
    start_time = time.time()
    
    logger.info(f"[{sync_id}] Starting sync...")
    
    try:
        # Scan and detect
        local_files = scan_local_files()
        existing_sources = fetch_existing_sources(index)
        
        new_files = detect_new_files(local_files, existing_sources)
        deleted_files = detect_deleted_files(local_files, existing_sources)
        existing_files = local_files - new_files
        
        logger.info(f"[{sync_id}] New: {len(new_files)}, Existing: {len(existing_files)}, Deleted: {len(deleted_files)}")
        
        # Process deletions
        deleted_count = delete_missing_vectors(index, deleted_files) if deleted_files else 0
        
        # Process new files
        chunks_added = 0
        vectors_added = 0
        ingestion_errors = []
        
        for source_path in new_files:
            try:
                parts = source_path.split("/")
                source_type = parts[0]
                
                # Prepend BASE_DATA to locate the local file correctly
                full_source_path = os.path.join(BASE_DATA, source_path)
                logger.info(f"[{sync_id}] Ingesting: {full_source_path} (from relative source_path: {source_path})")
                
                result, audit = await ingest_document(source_type, full_source_path, index, sync_id, chunking_strategy=strategy)
                
                if audit.get("pinecone", {}).get("status") == "success":
                    chunks_added += audit.get("chunking", {}).get("total_chunks", 0)
                    vectors_added += audit.get("pinecone", {}).get("vectors_written", 0)
                else:
                    ingestion_errors.append(f"{source_path}: {result}")
            
            except Exception as e:
                logger.error(f"[{sync_id}] Ingestion failed for {source_path}: {e}")
                ingestion_errors.append(f"{source_path}: {str(e)}")
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        return {
            "success": True,
            "sync_id": sync_id,
            "new_files_ingested": len(new_files),
            "existing_files_skipped": len(existing_files),
            "deleted_files_removed": deleted_count,
            "chunks_added": chunks_added,
            "vectors_added": vectors_added,
            "errors": ingestion_errors if ingestion_errors else None,
            "processing_time_ms": elapsed_ms
        }
    
    except Exception as e:
        logger.error(f"[{sync_id}] Sync failed: {e}")
        return {
            "success": False,
            "sync_id": sync_id,
            "error": str(e),
            "processing_time_ms": int((time.time() - start_time) * 1000)
        }
