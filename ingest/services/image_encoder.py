import base64
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def encode_image_base64(path: str) -> dict:
    """Encode image to base64 with MIME type detection and validation."""
    try:
        # Convert to absolute path to handle relative paths
        file_path = Path(path).resolve()
        
        # Verify file exists
        if not file_path.exists():
            error_msg = f"Image file not found: {path} (resolved to {file_path})"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "path": str(file_path),
                "exists": False
            }
        
        # Determine MIME type from extension
        ext = file_path.suffix.lower()
        mime_map = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        mime_type = mime_map.get(ext, 'image/jpeg')
        
        # Read and encode
        with open(file_path, 'rb') as f:
            image_data = f.read()
        
        if not image_data:
            error_msg = f"Image file is empty: {file_path}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "size": 0
            }
        
        base64_str = base64.b64encode(image_data).decode('utf-8')
        
        # Validate base64 encoding
        if not base64_str:
            error_msg = "Base64 encoding resulted in empty string"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "base64_length": 0
            }
        
        if len(base64_str) < 100:
            error_msg = f"Base64 too short ({len(base64_str)} chars). Image file may be invalid or too small."
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "base64_length": len(base64_str),
                "file_size": len(image_data)
            }
        
        return {
            "success": True,
            "base64": base64_str,
            "mime_type": mime_type,
            "size": len(image_data),
            "base64_length": len(base64_str),
            "filename": file_path.name,
            "path": str(file_path)
        }
    except FileNotFoundError as e:
        error_msg = f"Image file not found: {path}"
        logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "path": path
        }
    except Exception as e:
        error_msg = f"Image conversion failed: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "exception": str(e)
        }
