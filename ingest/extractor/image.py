
import logging
from pathlib import Path
from PIL import Image
import pytesseract

logger = logging.getLogger(__name__)

def extract_image(path: str) -> str:
    """Extract text from image using OCR."""
    image = Image.open(path)
    #gray
    image = image.convert('L')
    text = pytesseract.image_to_string(image)
    return text

def process_image(path: str) -> dict:
    """Extract and return text with metadata. OCR only."""
    text = extract_image(path)
    filename = Path(path).name

    logger.info("OCR extraction complete")

    return {
        "text": text,
        "metadata": {
            "source_type": "image",
            "filename": filename,
            "ocr_used": True,
            "path": path
        }
    }