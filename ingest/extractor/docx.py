import logging
from pathlib import Path
from docx import Document

logger = logging.getLogger(__name__)

def extract_docx(path: str) -> str:
    """Extract text from DOCX preserving structure."""
    document = Document(path)
    paragraphs = []

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if text:
            paragraphs.append(text)

    return "\n".join(paragraphs)

def process_docx(path: str) -> dict:
    """Extract and return text with metadata."""
    text = extract_docx(path)
    filename = Path(path).name

    return {
        "text": text,
        "metadata": {
            "source_type": "docx",
            "filename": filename,
            "ocr_used": False,
            "path": path
        }
    }

    text = extract_docx(
        path
    )

    chunks = chunk_text(
        text
    )

    source = (
        path
        .split("/")[-1]
        .split(".")[0]
    )

    return create_vectors(
        chunks,
        source
    )