import logging
import io
import base64
from pathlib import Path
from pypdf import PdfReader
from PIL import Image
import pytesseract

logger = logging.getLogger(__name__)

# Constants
MIN_TEXT_LENGTH_FOR_NATIVE = 50
MIN_IMAGE_DIMENSION = 20
MIN_IMAGE_DATA_SIZE = 100
MIN_BASE64_LENGTH = 500

# Image extraction constraints
MIN_TEXT_LENGTH_BEFORE_OCR = 50


def extract_text_from_page(page):
    """Extract native text from PDF page."""
    if not hasattr(page, "extract_text"):
        return ""
    return page.extract_text().strip()


def apply_ocr_to_page(page):
    """Extract text from page using OCR."""
    try:
        page_image_bytes = page.to_image(fmt="png")
        image = Image.open(io.BytesIO(page_image_bytes))
        ocr_text = pytesseract.image_to_string(image)
        return ocr_text.strip()
    except Exception as e:
        logger.debug(f"OCR failed: {e}")
        return ""


def extract_text_from_pdf(path: str) -> tuple[str, bool]:
    """
    Extract text from PDF with OCR fallback for scanned pages.
    
    Returns:
        (text, ocr_used): Extracted text and whether OCR was applied
    """
    reader = PdfReader(path)
    extracted_pages = []
    ocr_used = False

    for page in reader.pages:
        text = extract_text_from_page(page)

        if text and len(text) > MIN_TEXT_LENGTH_FOR_NATIVE:
            extracted_pages.append(text)
        else:
            ocr_text = apply_ocr_to_page(page)
            if ocr_text:
                extracted_pages.append(ocr_text)
                ocr_used = True
                logger.info("OCR fallback applied to page")

    return "\n".join(extracted_pages), ocr_used


def process_pdf(path: str) -> dict:
    """Extract text and metadata from PDF."""
    text, ocr_used = extract_text_from_pdf(path)
    filename = Path(path).name

    return {
        "text": text,
        "metadata": {
            "source_type": "pdf",
            "filename": filename,
            "ocr_used": ocr_used,
            "path": path
        }
    }


def determine_mime_type(pdf_object: dict) -> str:
    """Determine MIME type from PDF image encoding."""
    filters = pdf_object.get("/Filter", [])
    
    if "/DCTDecode" in filters:
        return "image/jpeg"
    return "image/png"


def get_image_dimensions(pdf_object: dict) -> tuple:
    """Extract width and height from PDF image object."""
    width = int(pdf_object.get("/Width", 0))
    height = int(pdf_object.get("/Height", 0))
    return width, height


def extract_image_data(pdf_object: dict) -> bytes:
    """Extract raw image data from PDF object."""
    try:
        return pdf_object.get_data()
    except Exception as e:
        logger.debug(f"Failed to extract image data: {e}")
        return b""


def encode_image_to_base64(image_bytes: bytes) -> str:
    """Encode image bytes to base64 string."""
    return base64.b64encode(image_bytes).decode("utf-8")


def is_valid_embedded_image(pdf_object: dict) -> bool:
    """Check if PDF object is a valid embedded image."""
    if pdf_object.get("/Subtype") != "/Image":
        return False
    
    if "/Width" not in pdf_object or "/Height" not in pdf_object:
        return False
    
    width, height = get_image_dimensions(pdf_object)
    return width >= MIN_IMAGE_DIMENSION and height >= MIN_IMAGE_DIMENSION


def extract_images_from_page(page_index: int, page) -> list:
    """Extract embedded images from PDF page as base64."""
    images = []
    
    try:
        if not hasattr(page, "resources"):
            return images
        
        resources = page["/Resources"]
        if not resources or "/XObject" not in resources:
            return images
        
        x_objects = resources["/XObject"].get_object()
        
        for obj_name in x_objects:
            try:
                x_obj = x_objects[obj_name].get_object()
                
                if not is_valid_embedded_image(x_obj):
                    continue
                
                image_data = extract_image_data(x_obj)
                if len(image_data) < MIN_IMAGE_DATA_SIZE:
                    continue
                
                base64_str = encode_image_to_base64(image_data)
                if len(base64_str) < MIN_BASE64_LENGTH:
                    continue
                
                width, height = get_image_dimensions(x_obj)
                mime_type = determine_mime_type(x_obj)
                
                images.append({
                    "page": page_index,
                    "base64": base64_str,
                    "width": width,
                    "height": height,
                    "mime_type": mime_type,
                    "size_bytes": len(image_data)
                })
                
                logger.debug(f"Extracted image: {width}x{height}, {mime_type}")
                
            except Exception as e:
                logger.debug(f"Error processing PDF object on page {page_index}: {e}")
                continue
    
    except Exception as e:
        logger.debug(f"Error accessing page resources: {e}")
    
    return images


def extract_pdf_live(path: str):
    """
    Live PDF extraction generator.
    
    Yields page data with text and embedded images as they are processed.
    """
    try:
        reader = PdfReader(path)
        total_pages = len(reader.pages)
        
        for page_index, page in enumerate(reader.pages, 1):
            page_text = extract_text_from_page(page)
            
            if not page_text or len(page_text) < MIN_TEXT_LENGTH_BEFORE_OCR:
                ocr_text = apply_ocr_to_page(page)
                if ocr_text:
                    page_text = ocr_text
                    logger.debug(f"OCR applied to page {page_index}")
            
            page_images = extract_images_from_page(page_index, page)
            
            yield {
                "page": page_index,
                "text": page_text,
                "images": page_images,
                "total_pages": total_pages
            }
            
            logger.info(
                f"Extracted page {page_index}/{total_pages}: "
                f"{len(page_text)} chars, {len(page_images)} images"
            )
    
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        raise
