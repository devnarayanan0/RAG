import logging
from datetime import datetime
from pathlib import Path
import uuid

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

logger = logging.getLogger(__name__)

dup_checker = DuplicateChecker()
extraction_sessions = {}


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
    """Encode image to base64."""
    if session_id not in extraction_sessions:
        return {"success": False, "error": "Invalid session"}
    
    sess = extraction_sessions[session_id]
    result = encode_image_base64(sess["image_path"])
    
    if not result.get("success"):
        return {"success": False, "error": result.get("error")}
    
    sess["base64"] = result["base64"]
    sess["mime_type"] = result["mime_type"]
    
    return {
        "success": True,
        "base64_length": len(result["base64"]),
        "filename": sess["filename"],
        "path": sess["image_path"]
    }


async def step_get_vision_description(session_id: str) -> dict:
    """Get vision description from Groq."""
    if session_id not in extraction_sessions:
        return {"success": False, "error": "Invalid session"}
    
    sess = extraction_sessions[session_id]
    
    if not sess["base64"]:
        return {
            "success": False,
            "error": "Image not encoded yet",
            "debug": {"base64_length": 0, "mime_type": sess.get("mime_type")}
        }
    
    result = describe_image(sess["base64"], sess["mime_type"])
    if not result.get("success"):
        return {
            "success": False,
            "error": result.get("error"),
            "debug": result.get("debug", {})
        }
    
    sess["description"] = result["description"]
    return {
        "success": True,
        "description": result["description"],
        "chars": len(result["description"]),
        "model": result.get("model", "groq"),
        "processing_time_ms": result.get("processing_time_ms", 0),
        "debug": result.get("debug", {})
    }


async def step_preview_chunks(session_id: str) -> dict:
    """Generate and preview chunks."""
    if session_id not in extraction_sessions:
        return {"success": False, "error": "Invalid session"}
    
    sess = extraction_sessions[session_id]
    
    if not sess["description"]:
        return {"success": False, "error": "No description available"}
    
    chunks = chunk_text(sess["description"])
    sess["chunks"] = chunks
    
    return {
        "success": True,
        "chunks": chunks,
        "total_chunks": len(chunks),
        "chunk_size": 800,
        "overlap": 150
    }


async def step_generate_embeddings(session_id: str) -> dict:
    """Generate embeddings for chunks."""
    if session_id not in extraction_sessions:
        return {"success": False, "error": "Invalid session"}
    
    sess = extraction_sessions[session_id]
    
    if not sess["chunks"]:
        return {"success": False, "error": "No chunks to embed"}
    
    hashes = [dup_checker.generate_hash(c) for c in sess["chunks"]]
    meta = {
        "source": "image",
        "extraction_type": "object",
        "filename": sess["filename"],
        "upload_time": sess["upload_time"],
        "description_generated": True,
        "ocr_used": False
    }
    
    vectors = build_vectors(sess["chunks"], meta, hashes)
    sess["vectors"] = vectors
    
    return {
        "success": True,
        "vectors_created": len(vectors),
        "sample_metadata": vectors[0]["metadata"] if vectors else None
    }


async def step_store_in_pinecone(session_id: str, pc_index) -> dict:
    """Store vectors in Pinecone."""
    if session_id not in extraction_sessions:
        return {"success": False, "error": "Invalid session"}
    
    sess = extraction_sessions[session_id]
    
    if not sess["vectors"]:
        return {"success": False, "error": "No vectors to store"}
    
    written = upsert_vectors(pc_index, sess["vectors"])
    del extraction_sessions[session_id]
    
    return {
        "success": True,
        "vectors_written": written,
        "message": f"Stored {written} vectors"
    }


async def get_session_state(session_id: str) -> dict:
    """Get extraction session state."""
    if session_id not in extraction_sessions:
        return {"success": False, "error": "Invalid session"}
    
    sess = extraction_sessions[session_id]
    return {
        "success": True,
        "filename": sess["filename"],
        "has_base64": sess["base64"] is not None,
        "has_description": sess["description"] is not None,
        "chunk_count": len(sess["chunks"]),
        "vector_count": len(sess["vectors"]),
        "upload_time": sess["upload_time"]
    }


async def ingest_document(source_type: str, source_path: str, pc_index, req_id: str = None) -> tuple:
    """Ingest document through extraction, chunking, embedding pipeline."""
    audit = {
        "extraction": {},
        "chunking": {},
        "duplicate": {},
        "pinecone": {},
        "metadata_sample": None
    }

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

    # check duplicates
    dup = dup_checker.check_exact_duplicate(text)
    audit["duplicate"]["incoming_hash"] = dup['hash']
    
    if dup['is_duplicate']:
        logger.warning(f"Duplicate rejected: {meta.get('filename', source_path)}")
        audit["duplicate"]["is_duplicate"] = True
        audit["duplicate"]["type"] = "exact_hash_match"
        audit["duplicate"]["matched_hash"] = dup['hash']
        return {
            "uploaded": False,
            "duplicate": {
                "status": True,
                "type": "exact_hash_match",
                "incoming_hash": dup['hash'],
                "existing_hash": dup['hash']
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
    chunks = chunk_text(text)
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
            "error": "No chunks generated",
            "chunks": 0,
            "vectors_written": 0
        }, audit

    # embed
    hashes = [dup_checker.generate_hash(c) for c in chunks]
    base_meta = {
        "source": meta['source_type'],
        "filename": meta.get('filename', source_path),
        "upload_time": datetime.utcnow().isoformat(),
        "ocr_used": meta.get('ocr_used', False),
        "masked": was_masked,
    }

    vectors = build_vectors(chunks, base_meta, hashes)
    logger.info(f"Built {len(vectors)} vectors")

    # store
    written = upsert_vectors(pc_index, vectors)
    logger.info(f"Upserted {written} vectors")

    audit["pinecone"] = {
        "vectors_written": written,
        "status": "success" if written > 0 else "failed"
    }

    if vectors:
        audit["metadata_sample"] = vectors[0]["metadata"]

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
            "sample_chunk_ids": [vectors[i]["metadata"].get("chunk_id") for i in range(min(3, len(vectors)))]
        },
        "metadata": {
            "filename": meta.get('filename', source_path),
            "source": meta.get('source_type', source_type),
            "upload_time": datetime.utcnow().isoformat(),
            "masked": was_masked,
            "hash": dup['hash'],
            "chunk_ids": [v["metadata"].get("chunk_id") for v in vectors],
            "chunk_indexes": [v["metadata"].get("chunk_index") for v in vectors]
        },
        "pinecone": {
            "inserted": True,
            "vectors_written": written,
            "namespace": "default",
            "vector_ids": [v["id"] for v in vectors]
        },
        "chunks": len(chunks),
        "vectors_written": written
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

    # encode
    res = encode_image_base64(image_path)
    if not res.get("success"):
        audit["error"] = res.get("error", "Encoding failed")
        return {
            "uploaded": False,
            "error": "Image conversion failed",
            "vectors_written": 0,
            "extraction_type": "object"
        }, audit

    b64 = res["base64"]
    mime = res["mime_type"]
    audit["vision"]["base64_length"] = len(b64)
    logger.info(f"Image encoded: {len(b64)} chars")

    # describe
    res = describe_image(b64, mime)
    if not res.get("success"):
        audit["error"] = res.get("error", "Vision analysis failed")
        return {
            "uploaded": False,
            "error": "Vision description failed",
            "vectors_written": 0,
            "extraction_type": "object"
        }, audit

    desc = res["description"]
    audit["vision"]["processing_time_ms"] = res.get("processing_time_ms", 0)
    audit["vision"]["model"] = res.get("model", "unknown")
    audit["extraction"]["chars"] = len(desc)
    logger.info(f"Description: {len(desc)} chars")

    # chunk
    chunks = chunk_text(desc)
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

    # embed
    hashes = [dup_checker.generate_hash(c) for c in chunks]
    upload_time = datetime.utcnow().isoformat()
    base_meta = {
        "source": "image",
        "extraction_type": "object",
        "filename": filename,
        "upload_time": upload_time,
        "description_generated": True,
        "llm": res.get("model", "groq"),
        "ocr_used": False
    }

    vectors = build_vectors(chunks, base_meta, hashes)
    logger.info(f"Built {len(vectors)} vectors")

    # store
    written = upsert_vectors(pc_index, vectors)
    logger.info(f"Upserted {written} vectors")

    audit["pinecone"] = {
        "vectors_written": written,
        "status": "success" if written > 0 else "failed"
    }

    if vectors:
        audit["metadata_sample"] = vectors[0]["metadata"]

    return {
        "uploaded": True,
        "extraction_type": "object",
        "description_generated": True,
        "description_preview": desc[:200] + "…" if len(desc) > 200 else desc,
        "chunks": len(chunks),
        "vectors_written": written,
        "processing_time_ms": audit["vision"].get("processing_time_ms", 0),
        "upload": {"saved": True, "location": image_path, "filename": filename},
        "extraction": {"chars": len(desc), "ocr_used": False},
        "vision": audit["vision"],
        "chunking": audit["chunking"],
        "pinecone": {
            "inserted": True,
            "vectors_written": written,
            "namespace": "default"
        }
    }, audit


async def extract_pdf_live_with_vision(pdf_path: str) -> dict:
    """Live PDF extraction with image descriptions via Groq Vision."""
    pages_data = []
    total_pages = 0
    images_count = 0
    
    try:
        for page_result in extract_pdf_live(pdf_path):
            total_pages = page_result.get("total_pages", 0)
            page_num = page_result.get("page", 0)
            text = page_result.get("text", "")
            images = page_result.get("images", [])
            
            page_obj = {
                "page": page_num,
                "text": text,
                "images": [],
                "total_pages": total_pages
            }
            
            # Process images with Vision API
            for img in images:
                try:
                    base64_str = img.get("base64", "")
                    if base64_str and len(base64_str) > 500:
                        img_desc = await describe_image(
                            base64_str,
                            img.get("mime_type", "image/png")
                        )
                        
                        if img_desc.get("success"):
                            images_count += 1
                            page_obj["images"].append({
                                "index": len(page_obj["images"]) + 1,
                                "dimensions": f"{img.get('width', 0)}x{img.get('height', 0)}",
                                "description": img_desc.get("description", ""),
                                "model": img_desc.get("model", "groq"),
                                "success": True
                            })
                        else:
                            logger.debug(f"Vision API failed for page {page_num}: {img_desc.get('error', 'Unknown')}")
                    
                except Exception as e:
                    logger.debug(f"Image processing error on page {page_num}: {e}")
            
            pages_data.append(page_obj)
            logger.info(f"Extracted page {page_num}/{total_pages} - text:{len(text)} chars, images:{len(page_obj['images'])}")
        
        filename = Path(pdf_path).name
        
        return {
            "success": True,
            "filename": filename,
            "pages": pages_data,
            "total_pages": total_pages,
            "total_images": images_count,
            "upload_time": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"PDF live extraction error: {e}")
        return {
            "success": False,
            "error": str(e),
            "filename": Path(pdf_path).name if pdf_path else None
        }
