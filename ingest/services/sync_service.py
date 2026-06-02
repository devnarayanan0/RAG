import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Set, Dict, List

logger = logging.getLogger(__name__)

BASE_DATA_DIR = "data"
FOLDERS = ["pdf", "image", "docx", "urls"]


def scan_local_files() -> Set[str]:
    """Scan local data folders and return relative paths from data/."""
    local_files = set()

    for folder in FOLDERS:
        folder_path = Path(BASE_DATA_DIR) / folder
        if folder_path.exists():
            for file_path in folder_path.rglob("*"):
                if file_path.is_file():
                    rel_path = str(file_path.relative_to(BASE_DATA_DIR))
                    local_files.add(rel_path)

    return local_files


def fetch_existing_sources(vector_index) -> Set[str]:
    """Query Pinecone for all unique source values in metadata."""
    existing_sources = set()

    try:
        results = vector_index.query(
            vector=[0.0] * 384,
            top_k=10000,
            include_metadata=True
        )

        for match in results.get("matches", []):
            metadata = match.get("metadata", {})
            source = metadata.get("source")
            if source:
                existing_sources.add(source)

    except Exception as e:
        logger.warning(f"Failed to fetch existing sources: {e}")

    return existing_sources


def detect_deleted_files(local_files: Set[str], existing_sources: Set[str]) -> Set[str]:
    """Find files in Pinecone but not in local folders."""
    return existing_sources - local_files


def detect_new_files(local_files: Set[str], existing_sources: Set[str]) -> Set[str]:
    """Find files locally but not indexed in Pinecone."""
    return local_files - existing_sources


def delete_missing_vectors(vector_index, deleted_sources: Set[str]) -> int:
    """Delete all vectors for files that no longer exist locally."""
    deleted_count = 0

    for source in deleted_sources:
        try:
            vector_index.delete(filter={"source": {"$eq": source}})
            deleted_count += 1
            logger.info(f"Deleted vectors for removed file: {source}")
        except Exception as e:
            logger.error(f"Failed to delete vectors for {source}: {e}")

    return deleted_count


def sync_folder_to_vector_db(vector_index) -> Dict:
    """
    Synchronize local data folders with Pinecone.

    Returns:
        {
            "local_files_count": int,
            "existing_sources_count": int,
            "deleted_count": int,
            "new_files": [relative_paths],
            "timestamp": ISO timestamp
        }
    """
    logger.info("Starting folder-to-Pinecone synchronization...")

    local_files = scan_local_files()
    existing_sources = fetch_existing_sources(vector_index)

    deleted_sources = detect_deleted_files(local_files, existing_sources)
    deleted_count = delete_missing_vectors(vector_index, deleted_sources) if deleted_sources else 0

    new_files = detect_new_files(local_files, existing_sources)

    result = {
        "local_files_count": len(local_files),
        "existing_sources_count": len(existing_sources),
        "deleted_count": deleted_count,
        "new_files_count": len(new_files),
        "new_files": sorted(list(new_files)),
        "timestamp": datetime.utcnow().isoformat()
    }

    logger.info(
        f"Sync complete: {len(local_files)} local | "
        f"{len(existing_sources)} indexed | "
        f"{deleted_count} deleted | "
        f"{len(new_files)} new"
    )

    return result


def is_file_already_indexed(file_source: str, vector_index) -> bool:
    """Check if a file is already indexed by source path."""
    try:
        results = vector_index.query(
            vector=[0.0] * 384,
            top_k=1,
            filter={"source": {"$eq": file_source}},
            include_metadata=True
        )
        return len(results.get("matches", [])) > 0
    except Exception as e:
        logger.debug(f"Failed to check if indexed: {e}")
        return False
