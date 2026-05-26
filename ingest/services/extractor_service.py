import logging
from pathlib import Path
from ingest.extractor.pdf import process_pdf
from ingest.extractor.docx import process_docx
from ingest.extractor.image import process_image
from ingest.extractor.url import process_url

logger = logging.getLogger(__name__)

SUPPORTED_TYPES = ['pdf', 'docx', 'image', 'url']

async def extract_content(source_type: str, source_path: str) -> dict:
    """
    Extract content from source. Returns:
    {
        'text': str,
        'metadata': {
            'source_type': str,
            'filename': str,
            'ocr_used': bool
        }
    }
    """
    if source_type not in SUPPORTED_TYPES:
        raise ValueError(f"Unsupported source type: {source_type}")

    logger.info(f"{source_type.upper()} extraction started")

    if source_type == 'pdf':
        result = process_pdf(source_path)
    elif source_type == 'docx':
        result = process_docx(source_path)
    elif source_type == 'image':
        result = process_image(source_path)
    elif source_type == 'url':
        result = process_url(source_path)

    logger.info(f"{source_type.upper()} extraction complete")
    return result
