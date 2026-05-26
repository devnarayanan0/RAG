import hashlib
import logging

logger = logging.getLogger(__name__)

class DuplicateChecker:
    def __init__(self):
        self.seen_hashes = {}
        self.embedding_threshold = 0.95

    def generate_hash(self, text: str) -> str:
        """Generate SHA256 hash of text."""
        return hashlib.sha256(text.encode()).hexdigest()

    def check_exact_duplicate(self, text: str) -> dict:
        """Check for exact duplicates by hash."""
        text_hash = self.generate_hash(text)

        if text_hash in self.seen_hashes:
            return {
                'is_duplicate': True,
                'reason': 'exact_hash_match',
                'hash': text_hash
            }

        self.seen_hashes[text_hash] = True
        return {
            'is_duplicate': False,
            'reason': None,
            'hash': text_hash
        }

    def check_near_duplicate(self, embedding1, embedding2, threshold=None) -> bool:
        """Check similarity between embeddings. Returns True if duplicate (similarity > threshold)."""
        if threshold is None:
            threshold = self.embedding_threshold

        # Cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarity = cosine_similarity([embedding1], [embedding2])[0][0]

        return similarity > threshold
