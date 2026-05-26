import logging
from pathlib import Path
from pypdf import PdfReader
from PIL import Image
import pytesseract
import io
import base64

logger = logging.getLogger(__name__)

def extract_pdf(path: str) -> tuple[str, bool]:
    """Extract text from PDF. Returns (text, ocr_used)."""
    reader = PdfReader(path)
    pages = []
    ocr_used = False

    for page in reader.pages:
        text = page.extract_text().strip()

        if text and len(text) > 50:
            pages.append(text)
        else:
            # OCR fallback for scanned pages
            pix = page.to_image(fmt="png")
            image = Image.open(io.BytesIO(pix))
            ocr_text = pytesseract.image_to_string(image)

            if ocr_text.strip():
                pages.append(ocr_text)
                ocr_used = True
                logger.info("OCR fallback enabled")

    return "\n".join(pages), ocr_used

def process_pdf(path: str) -> dict:
    """Extract and return text with metadata."""
    text, ocr_used = extract_pdf(path)

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


def extract_images_from_page(page_index: int, page):
    """Safely extract embedded images from PDF page as base64."""
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
                
                # Check if it's an image
                if x_obj.get("/Subtype") != "/Image":
                    continue
                
                # Validate required properties
                if "/Width" not in x_obj or "/Height" not in x_obj:
                    logger.debug(f"Skipping XObject: missing dimensions")
                    continue
                
                width = int(x_obj["/Width"])
                height = int(x_obj["/Height"])
                
                # Skip tiny/invalid dimensions
                if width < 20 or height < 20:
                    logger.debug(f"Skipping XObject: dimensions too small ({width}x{height})")
                    continue
                
                # Extract image data
                color_space = x_obj.get("/ColorSpace", "/DeviceRGB")
                if isinstance(color_space, list):
                    color_space = color_space[0]
                
                # Determine MIME type
                if "/FlateDecode" in x_obj.get("/Filter", []):
                    mime_type = "image/png"
                elif "/DCTDecode" in x_obj.get("/Filter", []):
                    mime_type = "image/jpeg"
                else:
                    mime_type = "image/png"
                
                # Get image data
                if "/Filter" in x_obj:
                    try:
                        image_data = x_obj.get_data()
                    except Exception as e:
                        logger.debug(f"Failed to get XObject data: {e}")
                        continue
                else:
                    image_data = x_obj.get_data()
                
                # Skip if data is too small
                if len(image_data) < 100:
                    logger.debug(f"Skipping XObject: image data too small ({len(image_data)} bytes)")
                    continue
                
                # Convert to base64
                base64_str = base64.b64encode(image_data).decode('utf-8')
                
                # Skip if base64 encoding failed
                if len(base64_str) < 500:
                    logger.debug(f"Skipping XObject: base64 too small ({len(base64_str)} bytes)")
                    continue
                
                images.append({
                    "page": page_index,
                    "base64": base64_str,
                    "width": width,
                    "height": height,
                    "mime_type": mime_type,
                    "size_bytes": len(image_data)
                })
                
                logger.debug(f"Extracted image from page {page_index}: {width}x{height}, {len(base64_str)} base64 bytes")
                
            except Exception as e:
                logger.debug(f"Error processing XObject on page {page_index}: {e}")
                continue
    
    except Exception as e:
        logger.debug(f"Error accessing page resources on page {page_index}: {e}")
    
    return images


def extract_pdf_live(path: str):
    """Generator: yield page data with text and images as extracted."""
    try:
        reader = PdfReader(path)
        total_pages = len(reader.pages)
        
        for page_idx, page in enumerate(reader.pages, 1):
            # Extract text
            text = page.extract_text().strip() if hasattr(page, "extract_text") else ""
            
            # OCR fallback for low-text pages
            if not text or len(text) < 50:
                try:
                    pix = page.to_image(fmt="png")
                    image = Image.open(io.BytesIO(pix))
                    ocr_text = pytesseract.image_to_string(image)
                    if ocr_text.strip():
                        text = ocr_text
                        logger.debug(f"OCR applied to page {page_idx}")
                except Exception as e:
                    logger.debug(f"OCR failed on page {page_idx}: {e}")
            
            # Extract embedded images
            images = extract_images_from_page(page_idx, page)
            
            yield {
                "page": page_idx,
                "text": text,
                "images": images,
                "total_pages": total_pages
            }
            
            logger.info(f"Live extraction: page {page_idx}/{total_pages} - text:{len(text)} chars, images:{len(images)}")
    
    except Exception as e:
        logger.error(f"Error in PDF live extraction: {e}")
        raise