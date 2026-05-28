"""
FOLDER SYNCHRONIZATION - USAGE EXAMPLES
========================================

This file demonstrates the three-case sync logic with real scenarios.

SETUP:
------
Local: ingest/data/pdf/report.pdf, ingest/data/image/photo.jpg
Pinecone: Contains vectors with:
  - source="pdf/bank.pdf" (exists locally)
  - source="docx/old.docx" (DELETED locally)
"""

from ingest.services.sync_service import (
    sync_folder_to_vector_db,
    is_file_already_indexed,
    scan_local_files,
    fetch_existing_sources,
    detect_new_files,
    detect_deleted_files
)


def example_sync_workflow(vector_index):
    """Example of full sync workflow."""
    
    # STEP 1: Run sync
    sync_result = sync_folder_to_vector_db(vector_index)
    print(f"""
    Sync Results:
    - Local files: {sync_result['local_files_count']}
    - Indexed in Pinecone: {sync_result['existing_sources_count']}
    - Deleted from Pinecone: {sync_result['deleted_count']}
    - New files to ingest: {sync_result['new_files_count']}
    - New files: {sync_result['new_files']}
    """)
    
    # STEP 2: New files are returned for ingestion
    new_files = sync_result['new_files']
    # These would be: ["pdf/report.pdf", "image/photo.jpg"]
    # They're ready for ingest_document() to process
    
    return sync_result


def example_case_1_already_indexed(vector_index):
    """
    CASE 1: File exists locally + already indexed in Pinecone
    
    Local:   ingest/data/pdf/bank.pdf
    Pinecone: source="pdf/bank.pdf" ✓ exists
    
    Behavior: Skip ingestion, return early
    """
    
    source_path = "ingest/data/pdf/bank.pdf"
    source_rel = "pdf/bank.pdf"
    
    # Check if already indexed
    is_indexed = is_file_already_indexed(source_rel, vector_index)
    
    if is_indexed:
        print(f"""
        ✓ CASE 1: File already indexed
        - Source: {source_rel}
        - Action: Skip ingestion
        - Reason: No duplicate vectors
        - Result: Return 'already_indexed' response
        """)
        return {
            "uploaded": False,
            "duplicate": {
                "status": True,
                "type": "already_indexed",
                "source": source_rel
            }
        }


def example_case_2_new_file(vector_index):
    """
    CASE 2: File exists locally + NOT indexed in Pinecone
    
    Local:   ingest/data/pdf/report.pdf
    Pinecone: source="pdf/report.pdf" ✗ NOT found
    
    Behavior: Ingest normally
    """
    
    source_path = "ingest/data/pdf/report.pdf"
    source_rel = "pdf/report.pdf"
    
    is_indexed = is_file_already_indexed(source_rel, vector_index)
    
    if not is_indexed:
        print(f"""
        ✓ CASE 2: New file - ingest normally
        - Source: {source_rel}
        - Action: Extract → Chunk → Embed → Store
        - Metadata: {{'source': '{source_rel}', ...}}
        - Result: Vectors stored with source metadata
        """)
        # Would call ingest_document() here
        return {
            "uploaded": True,
            "chunks": 5,
            "vectors_written": 5
        }


def example_case_3_deleted_file(vector_index):
    """
    CASE 3: File deleted locally + vectors exist in Pinecone
    
    Local:   ingest/data/docx/ (empty)
    Pinecone: source="docx/old.docx" ✓ exists
    
    Behavior: Delete vectors automatically
    """
    
    local_files = scan_local_files()
    existing_sources = fetch_existing_sources(vector_index)
    
    # Detect deleted
    deleted_sources = detect_deleted_files(local_files, existing_sources)
    
    if deleted_sources:
        print(f"""
        ✓ CASE 3: File deleted locally - cleanup Pinecone
        - Deleted sources: {deleted_sources}
        - Action: index.delete(filter={{'source': {{'$eq': source}}}})
        - Result: Orphaned vectors removed automatically
        """)
        return {
            "deleted_count": len(deleted_sources),
            "sources_deleted": list(deleted_sources)
        }


def example_metadata_structure():
    """Example of metadata stored in Pinecone vectors."""
    
    metadata_example = {
        "source": "pdf/bank.pdf",          # Relative path for sync
        "source_type": "pdf",               # Type
        "filename": "bank.pdf",             # Original filename
        "chunk_id": "chunk_0001",           # Sequential ID
        "chunk_index": 0,                   # Position in document
        "upload_time": "2026-05-28T10:30:45Z",  # When ingested
        "ocr_used": False,                  # Whether OCR was needed
        "masked": False,                    # Sensitive data masked?
    }
    
    print(f"""
    Metadata stored in each vector:
    {metadata_example}
    
    Key points:
    - 'source' = relative path for sync purposes
    - 'filename' = original file name
    - 'chunk_index' = order within document
    - Other fields = audit trail
    """)


if __name__ == "__main__":
    print("SYNC LOGIC EXAMPLES")
    print("=" * 50)
    print("\nNote: These are pseudo-code examples.")
    print("To run actual sync, use sync_folder_to_vector_db(vector_index)")
    print("\nImplementation Details:")
    print("- Location: ingest/services/sync_service.py")
    print("- Integration: Called in ingest_document()")
    print("- Called at: Start of POST /ingest endpoint")
    print("\nNo breaking changes to existing ingestion.")
