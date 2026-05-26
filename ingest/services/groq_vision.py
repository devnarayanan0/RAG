import logging
import os
import time
import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_VISION_MODEL = os.getenv("GROQ_VISION_MODEL") or "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

VISION_PROMPT = """Analyze this image in detail.

Return your analysis in exactly 2 paragraphs:

Paragraph 1: Main subject and objects visible
- What is the primary subject?
- What objects are visible?
- How are they arranged?

Paragraph 2: Scene context and details
- Scene description and context
- Any visible text or signage
- Overall visual interpretation

If the image is unclear or unreadable, explain why instead of providing generic descriptions."""


def describe_image(base64_image: str, mime_type: str = "image/jpeg") -> dict:
    """
    Generate description of image using Groq Vision API.
    
    Uses multimodal message format with image_url data URI.
    Validates inputs before sending to avoid silent failures.
    """
    
    # ============ INPUT VALIDATION ============
    
    if not GROQ_API_KEY:
        error_msg = "GROQ_API_KEY not configured in environment"
        logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "debug": {
                "api_key_present": False,
                "base64_length": 0,
                "mime_type": mime_type
            }
        }
    
    if not base64_image:
        error_msg = "base64_image is empty or null"
        logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "debug": {
                "base64_length": 0,
                "mime_type": mime_type
            }
        }
    
    if not mime_type:
        error_msg = "mime_type is empty or null"
        logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "debug": {
                "base64_length": len(base64_image),
                "mime_type": None
            }
        }
    
    if len(base64_image) < 1000:
        error_msg = f"base64_image too short ({len(base64_image)} chars). Expected > 1000 chars for valid image."
        logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "debug": {
                "base64_length": len(base64_image),
                "mime_type": mime_type,
                "expected_minimum": 1000
            }
        }
    
    try:
        start_time = time.time()
        
        # ============ BUILD MULTIMODAL MESSAGE ============
        
        # Format image as data URI: data:mime;base64,encoded
        image_data_uri = f"data:{mime_type};base64,{base64_image}"
        
        # Multimodal content: text + image_url
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": VISION_PROMPT
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_uri
                        }
                    }
                ]
            }
        ]
        
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": GROQ_VISION_MODEL,
            "messages": messages,
            "max_tokens": 1024,
            "temperature": 0.3
        }
        
        logger.info(f"Sending vision request to Groq: {GROQ_VISION_MODEL}")
        logger.debug(f"Image data URI length: {len(image_data_uri)}")
        
        # ============ SEND TO GROQ API ============
        
        response = requests.post(
            GROQ_API_URL,
            json=payload,
            headers=headers,
            timeout=30
        )
        
        elapsed = time.time() - start_time
        
        # ============ HANDLE RESPONSE ============
        
        if response.status_code != 200:
            error_msg = f"Groq API error: {response.status_code} - {response.text}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "debug": {
                    "base64_length": len(base64_image),
                    "mime_type": mime_type,
                    "model": GROQ_VISION_MODEL,
                    "status_code": response.status_code,
                    "response_body": response.text[:500],
                    "processing_time_ms": int(elapsed * 1000)
                }
            }
        
        data = response.json()
        
        # Extract description from response
        if "choices" not in data or not data["choices"]:
            error_msg = "Groq returned empty choices in response"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "debug": {
                    "base64_length": len(base64_image),
                    "mime_type": mime_type,
                    "model": GROQ_VISION_MODEL,
                    "response": data,
                    "processing_time_ms": int(elapsed * 1000)
                }
            }
        
        choice = data["choices"][0]
        if "message" not in choice or "content" not in choice["message"]:
            error_msg = "Groq response missing message content"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "debug": {
                    "base64_length": len(base64_image),
                    "mime_type": mime_type,
                    "model": GROQ_VISION_MODEL,
                    "response": data,
                    "processing_time_ms": int(elapsed * 1000)
                }
            }
        
        description = choice["message"]["content"].strip()
        
        # ============ VALIDATE DESCRIPTION ============
        
        if not description or description.startswith("Image analyzed"):
            error_msg = f"Groq returned generic placeholder response (length: {len(description)})"
            logger.warning(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "debug": {
                    "base64_length": len(base64_image),
                    "mime_type": mime_type,
                    "model": GROQ_VISION_MODEL,
                    "raw_response": description[:200],
                    "processing_time_ms": int(elapsed * 1000)
                }
            }
        
        logger.info(f"Vision description generated: {len(description)} chars in {elapsed:.2f}s")
        
        return {
            "success": True,
            "description": description,
            "model": GROQ_VISION_MODEL,
            "processing_time_ms": int(elapsed * 1000),
            "debug": {
                "base64_length": len(base64_image),
                "mime_type": mime_type,
                "image_uri_length": len(image_data_uri),
                "description_length": len(description),
                "api_status": response.status_code,
                "tokens_used": data.get("usage", {}).get("total_tokens", 0)
            }
        }
        
    except requests.exceptions.Timeout:
        error_msg = "Groq API request timed out (30s)"
        logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "debug": {
                "base64_length": len(base64_image) if base64_image else 0,
                "mime_type": mime_type
            }
        }
    
    except requests.exceptions.ConnectionError as e:
        error_msg = f"Failed to connect to Groq API: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "debug": {
                "base64_length": len(base64_image) if base64_image else 0,
                "mime_type": mime_type
            }
        }
    
    except Exception as e:
        error_msg = f"Vision analysis failed: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "debug": {
                "base64_length": len(base64_image) if base64_image else 0,
                "mime_type": mime_type,
                "exception": str(e)
            }
        }

