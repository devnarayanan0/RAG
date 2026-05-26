import re
import logging

logger = logging.getLogger(__name__)

# Patterns for sensitive data
PATTERNS = {
    'account': r'\b\d{8,}\b',  # 8+ consecutive digits
    'credit_card': r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
    'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
}

def mask_sensitive_data(text: str) -> tuple[str, bool]:
    """Mask sensitive data in text. Returns (masked_text, was_masked)."""
    original = text
    masked = False

    # Mask generic long numbers
    def replace_number(match):
        nonlocal masked
        masked = True
        digits = match.group(0)
        return 'X' * len(digits)

    masked_text = re.sub(PATTERNS['account'], replace_number, text)
    masked_text = re.sub(PATTERNS['ssn'], lambda m: 'XXX-XX-XXXX', masked_text)
    masked_text = re.sub(PATTERNS['credit_card'], lambda m: 'XXXX-XXXX-XXXX-XXXX', masked_text)

    if masked_text != original:
        masked = True
        logger.info("Data masking applied")

    return masked_text, masked
