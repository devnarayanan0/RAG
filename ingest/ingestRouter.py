import logging
from datetime import datetime
from pathlib import Path
import uuid
import os

from ingest.extractor.pdf import process_pdf, extract_pdf_live
from ingest.extractor.docx import process_docx
from ingest.extractor.image import process_image
from ingest.extractor.url import process_url
from ingest.services.chunk_service import chunk_text
from ingest.services.masking_service import mask_sensitive_data
from ingest.services.duplicate_service import DuplicateChecker
from ingest.services.pinecone_service import build_vectors, upsert_vectors
from ingest.services.image_encoder import encode_image_base64
from ingest.services.groq_vision import describe_image
from ingest.services.sync_service import sync_folder_to_vector_db, is_file_already_indexed

logger = logging.getLogger(__name__)
BASE_DATA_DIR = "ingest/data"

duplicate_checker = DuplicateChecker()
extraction_sessions = {}


def get_relative_source_path(absolute_path: str) -> str:
    """Convert absolute file path to relative source path from data/."""
    try:
        rel = Path(absolute_path).relative_to(BASE_DATA_DIR)
        return str(rel)
    except (ValueError, TypeError):
        return absolute_path


def get_extraction_session(session_id: str) -> dict:
    """Retrieve extraction session or raise error."""
    if session_id not in extraction_sessions:
        raise ValueError("Invalid session")
    return extraction_sessions[session_id]


def create_extraction_session(image_path: str) -> str:
    """Create extraction session."""
    session_id = str(uuid.uuid4())
    extraction_sessions[session_id] = {
        "image_path": image_path,
        "filename": Path(image_path).name,
        "base64": None,
        "mime_type": None,
        "description": None,
        "chunks": [],
        "vectors": [],
        "upload_time": datetime.utcnow().isoformat()
    }
    return session_id


async def step_encode_image(session_id: str) -> dict:
    """Encode image file to base64 format."""
    try:
        session = get_extraction_session(session_id)
    except ValueError as e:
        return {"success": False, "error": str(e)}
    
    encoding_result = encode_image_base64(session["image_path"])
    if not encoding_result.get("success"):
        return {"success": False, "error": encoding_result.get("error")}
    
    session["base64"] = encoding_result["base64"]
    session["mime_type"] = encoding_result["mime_type"]
    
    return {
        "success": True,
        "base64_length": len(encoding_result["base64"]),
        "filename": session["filename"],
        "path": session["image_path"]
    }


async def step_get_vision_description(session_id: str) -> dict:
    """Analyze image using Groq Vision API."""
    try:
        session = get_extraction_session(session_id)
    except ValueError as e:
        return {"success": False, "error": str(e)}
    
    if not session["base64"]:
        return {"success": False, "error": "Image not encoded - run step_encode_image first"}
    
    vision_result = describe_image(session["base64"], session["mime_type"])
    if not vision_result.get("success"):
        return {"success": False, "error": vision_result.get("error")}
    
    session["description"] = vision_result["description"]
    return {
        "success": True,
        "description": vision_result["description"],
        "char_count": len(vision_result["description"]),
        "model": vision_result.get("model", "groq"),
        "processing_time_ms": vision_result.get("processing_time_ms", 0)
    }


async def step_preview_chunks(session_id: str) -> dict:
    """Split description into chunks and preview."""
    try:
        session = get_extraction_session(session_id)
    except ValueError as e:
        return {"success": False, "error": str(e)}
    
    if not session["description"]:
        return {"success": False, "error": "No description available"}
    
    chunks = chunk_text(session["description"])
    session["chunks"] = chunks
    
    return {
        "success": True,
        "chunks": chunks,
        "total_chunks": len(chunks),
        "chunk_size": 800,
        "overlap": 150
    }


async def step_generate_embeddings(session_id: str) -> dict:
    """Generate embedding vectors for all chunks."""
    try:
        session = get_extraction_session(session_id)
    except ValueError as e:
        return {"success": False, "error": str(e)}
    
    if not session["chunks"]:
        return {"success": False, "error": "No chunks to embed"}
    
    chunk_hashes = [duplicate_checker.generate_hash(c) for c in session["chunks"]]
    metadata = {
        "source": "image",
        "extraction_type": "object",
        "filename": session["filename"],
        "upload_time": session["upload_time"],
        "description_generated": True,
        "ocr_used": False
    }
    
    embedding_vectors = build_vectors(session["chunks"], metadata, chunk_hashes)
    session["vectors"] = embedding_vectors
    
    return {
        "success": True,
        "vectors_created": len(embedding_vectors),
        "sample_metadata": embedding_vectors[0]["metadata"] if embedding_vectors else None
    }


async def step_store_in_pinecone(session_id: str, vector_index) -> dict:
    """Persist embedding vectors to Pinecone index."""
    try:
        session = get_extraction_session(session_id)
    except ValueError as e:
        return {"success": False, "error": str(e)}
    
    if not session["vectors"]:
        return {"success": False, "error": "No vectors to store"}
    
    stored_count = upsert_vectors(vector_index, session["vectors"])
    del extraction_sessions[session_id]
    
    return {
        "success": True,
        "vectors_written": stored_count
    }


async def get_session_state(session_id: str) -> dict:
    """Get current extraction session progress and state."""
    try:
        session = get_extraction_session(session_id)
    except ValueError as e:
        return {"success": False, "error": str(e)}
    
    return {
        "success": True,
        "filename": session["filename"],
        "has_base64": session["base64"] is not None,
        "has_description": session["description"] is not None,
        "chunk_count": len(session["chunks"]),
        "vector_count": len(session["vectors"]),
        "upload_time": session["upload_time"]
    }


async def ingest_document(source_type: str, source_path: str, pc_index, req_id: str = None, chunking_strategy: str = "semantic") -> tuple:
    """Ingest document through extraction, chunking, embedding pipeline."""
    audit = {
        "extraction": {},
        "chunking": {},
        "duplicate": {},
        "pinecone": {},
        "metadata_sample": None,
        "sync": {}
    }

    source_rel = get_relative_source_path(source_path)

    if source_type != 'url' and is_file_already_indexed(source_rel, pc_index):
        logger.info(f"File already indexed: {source_rel}")
        audit["duplicate"]["is_duplicate"] = True
        audit["duplicate"]["type"] = "already_indexed"
        audit["duplicate"]["source"] = source_rel
        return {
            "uploaded": False,
            "duplicate": {
                "status": True,
                "type": "already_indexed",
                "source": source_rel
            },
            "chunks": 0,
            "vectors_written": 0
        }, audit

    # extract
    if source_type == 'pdf':
        res = process_pdf(source_path)
    elif source_type == 'docx':
        res = process_docx(source_path)
    elif source_type == 'image':
        res = process_image(source_path)
    elif source_type == 'url':
        res = process_url(source_path)
    else:
        raise ValueError(f"Unsupported source: {source_type}")

    text = res['text']
    meta = res['metadata']
    
    audit["extraction"] = {
        "chars": len(text),
        "ocr_used": meta.get('ocr_used', False),
        "filename": meta.get('filename', source_path)
    }

    if not text or not text.strip():
        audit["error"] = "No content extracted"
        return {
            "uploaded": False,
            "error": "No content extracted",
            "chunks": 0,
            "vectors_written": 0
        }, audit

    # Check for exact duplicates by hash
    duplicate_check = duplicate_checker.check_exact_duplicate(text)
    audit["duplicate"]["incoming_hash"] = duplicate_check['hash']
    
    if duplicate_check['is_duplicate']:
        logger.warning(f"Duplicate rejected: {meta.get('filename', source_path)}")
        audit["duplicate"]["is_duplicate"] = True
        audit["duplicate"]["type"] = "exact_hash_match"
        audit["duplicate"]["matched_hash"] = duplicate_check['hash']
        return {
            "uploaded": False,
            "duplicate": {
                "status": True,
                "type": "exact_hash_match",
                "incoming_hash": duplicate_check['hash'],
                "existing_hash": duplicate_check['hash']
            },
            "chunks": 0,
            "vectors_written": 0
        }, audit

    audit["duplicate"]["is_duplicate"] = False

    # mask
    text, was_masked = mask_sensitive_data(text)
    meta['masked'] = was_masked
    audit["extraction"]["masked"] = was_masked

    # chunk
    chunks = chunk_text(text, strategy=chunking_strategy)
    logger.info(f"Generated {len(chunks)} chunks using {chunking_strategy} strategy")

    audit["chunking"] = {
        "count": len(chunks),
        "strategy": chunking_strategy,
        "samples": chunks[:3] if chunks else []
    }

    if not chunks:
        audit["error"] = "No chunks generated"
        return {
            "uploaded": False,
            "error": "No chunks generated",
            "chunks": 0,
            "vectors_written": 0
        }, audit

    # embed
    chunk_hashes = [duplicate_checker.generate_hash(c) for c in chunks]
    base_metadata = {
        "source": source_rel,
        "source_type": meta.get('source_type', source_type),
        "filename": meta.get('filename', source_path),
        "upload_time": datetime.utcnow().isoformat(),
        "ocr_used": meta.get('ocr_used', False),
        "masked": was_masked,
    }

    embedding_vectors = build_vectors(chunks, base_metadata, chunk_hashes)
    logger.info(f"Built {len(embedding_vectors)} vectors")

    # store
    stored_count = upsert_vectors(pc_index, embedding_vectors)
    logger.info(f"Upserted {stored_count} vectors")

    audit["pinecone"] = {
        "vectors_written": stored_count,
        "status": "success" if stored_count > 0 else "failed"
    }

    if embedding_vectors:
        audit["metadata_sample"] = embedding_vectors[0]["metadata"]

    return {
        "uploaded": True,
        "duplicate": {"status": False, "type": None},
        "upload": {
            "saved": True,
            "location": source_path,
            "filename": meta.get('filename', source_path)
        },
        "extraction": {
            "chars": len(text),
            "ocr_used": meta.get('ocr_used', False)
        },
        "chunking": {
            "total_chunks": len(chunks),
            "sample_chunk_ids": [embedding_vectors[i]["metadata"].get("chunk_id") for i in range(min(3, len(embedding_vectors)))]
        },
        "metadata": {
            "filename": meta.get('filename', source_path),
            "source": meta.get('source_type', source_type),
            "upload_time": datetime.utcnow().isoformat(),
            "masked": was_masked,
            "hash": duplicate_check['hash'],
            "chunk_ids": [v["metadata"].get("chunk_id") for v in embedding_vectors],
            "chunk_indexes": [v["metadata"].get("chunk_index") for v in embedding_vectors]
        },
        "pinecone": {
            "inserted": True,
            "vectors_written": stored_count,
            "namespace": "default",
            "vector_ids": [v["id"] for v in embedding_vectors]
        },
        "chunks": len(chunks),
        "vectors_written": stored_count
    }, audit


async def ingest_image_objects(image_path: str, pc_index, req_id: str = None) -> tuple:
    """Ingest image via vision LLM."""
    audit = {
        "extraction": {},
        "chunking": {},
        "pinecone": {},
        "metadata_sample": None,
        "vision": {}
    }

    filename = Path(image_path).name

    encoding_result = encode_image_base64(image_path)
    if not encoding_result.get("success"):
        audit["error"] = encoding_result.get("error", "Encoding failed")
        return {
            "uploaded": False,
            "error": "Image conversion failed",
            "vectors_written": 0,
            "extraction_type": "object"
        }, audit

    base64_str = encoding_result["base64"]
    mime_type = encoding_result["mime_type"]
    audit["vision"]["base64_length"] = len(base64_str)
    logger.info(f"Image encoded: {len(base64_str)} chars")

    vision_result = describe_image(base64_str, mime_type)
    if not vision_result.get("success"):
        audit["error"] = vision_result.get("error", "Vision analysis failed")
        return {
            "uploaded": False,
            "error": "Vision description failed",
            "vectors_written": 0,
            "extraction_type": "object"
        }, audit

    description = vision_result["description"]
    audit["vision"]["processing_time_ms"] = vision_result.get("processing_time_ms", 0)
    audit["vision"]["model"] = vision_result.get("model", "unknown")
    audit["extraction"]["chars"] = len(description)
    logger.info(f"Description: {len(description)} chars")

    chunks = chunk_text(description)
    logger.info(f"Generated {len(chunks)} chunks")

    audit["chunking"] = {
        "count": len(chunks),
        "chunk_size": 800,
        "overlap": 150,
        "samples": chunks[:3] if chunks else []
    }

    if not chunks:
        audit["error"] = "No chunks generated"
        return {
            "uploaded": False,
            "error": "Chunking failed",
            "vectors_written": 0,
            "extraction_type": "object"
        }, audit

    chunk_hashes = [duplicate_checker.generate_hash(c) for c in chunks]
    upload_time = datetime.utcnow().isoformat()
    base_metadata = {
        "source": "image",
        "extraction_type": "object",
        "filename": filename,
        "upload_time": upload_time,
        "description_generated": True,
        "llm": vision_result.get("model", "groq"),
        "ocr_used": False
    }

    embedding_vectors = build_vectors(chunks, base_metadata, chunk_hashes)
    logger.info(f"Built {len(embedding_vectors)} vectors")

    stored_count = upsert_vectors(pc_index, embedding_vectors)
    logger.info(f"Upserted {stored_count} vectors")

    audit["pinecone"] = {
        "vectors_written": stored_count,
        "status": "success" if stored_count > 0 else "failed"
    }

    if embedding_vectors:
        audit["metadata_sample"] = embedding_vectors[0]["metadata"]

    description_preview = description[:200] + "…" if len(description) > 200 else description
    return {
        "uploaded": True,
        "extraction_type": "object",
        "description_generated": True,
        "description_preview": description_preview,
        "chunks": len(chunks),
        "vectors_written": stored_count,
        "processing_time_ms": audit["vision"].get("processing_time_ms", 0),
        "upload": {"saved": True, "location": image_path, "filename": filename},
        "extraction": {"chars": len(description), "ocr_used": False},
        "vision": audit["vision"],
        "chunking": audit["chunking"],
        "pinecone": {
            "inserted": True,
            "vectors_written": stored_count,
            "namespace": "default"
        }
    }, audit


async def extract_pdf_live_with_vision(pdf_path: str) -> dict:
    """Live PDF extraction with Groq Vision analysis of embedded images."""
    extracted_pages = []
    total_page_count = 0
    analyzed_image_count = 0
    
    try:
        for page_data in extract_pdf_live(pdf_path):
            total_page_count = page_data.get("total_pages", 0)
            page_number = page_data.get("page", 0)
            page_text = page_data.get("text", "")
            page_images = page_data.get("images", [])
            
            page_result = {
                "page": page_number,
                "text": page_text,
                "images": [],
                "total_pages": total_page_count
            }
            
            for image_data in page_images:
                try:
                    base64_content = image_data.get("base64", "")
                    if not (base64_content and len(base64_content) > 500):
                        continue
                    
                    vision_response = await describe_image(
                        base64_content,
                        image_data.get("mime_type", "image/png")
                    )
                    
                    if not vision_response.get("success"):
                        logger.debug(f"Vision analysis failed: {vision_response.get('error', 'Unknown')}")
                        continue
                    
                    analyzed_image_count += 1
                    image_width = image_data.get('width', 0)
                    image_height = image_data.get('height', 0)
                    
                    page_result["images"].append({
                        "index": len(page_result["images"]) + 1,
                        "dimensions": f"{image_width}x{image_height}",
                        "description": vision_response.get("description", ""),
                        "model": vision_response.get("model", "groq"),
                        "success": True
                    })
                    
                except Exception as e:
                    logger.debug(f"Image processing error on page {page_number}: {e}")
                    continue
            
            extracted_pages.append(page_result)
            logger.info(
                f"Extracted page {page_number}/{total_page_count}: "
                f"{len(page_text)} text chars, {len(page_result['images'])} images analyzed"
            )
        
        filename = Path(pdf_path).name
        
        return {
            "success": True,
            "filename": filename,
            "pages": extracted_pages,
            "total_pages": total_page_count,
            "total_images": analyzed_image_count,
            "upload_time": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"PDF live extraction error: {e}")
        return {
            "success": False,
            "error": str(e),
            "filename": Path(pdf_path).name if pdf_path else None
        }
