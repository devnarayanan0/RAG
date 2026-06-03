import logging
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

def extract_url(url: str) -> str:
    """Fetch and parse webpage, remove noise."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers, timeout=15)
    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(["script", "style", "noscript", "nav", "footer"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    return text

def process_url(url: str) -> dict:
    """Extract and return text with metadata."""
    text = extract_url(url)

    # Clean up URL for filename
    filename = url.replace("https://", "").replace("http://", "").split("/")[0]

    logger.info(f"URL extraction complete: {filename}")

    return {
        "text": text,
        "metadata": {
            "source_type": "url",
            "filename": filename,
            "ocr_used": False,
            "path": url
        }
    }