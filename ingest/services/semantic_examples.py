"""
SEMANTIC CHUNKING - USAGE EXAMPLES
===================================

This file demonstrates the semantic chunking approach compared to character-based splitting.
"""

from ingest.services.semantic_chunk import (
    chunk_document,
    split_sentences,
    create_sentence_embeddings,
    group_semantic_chunks,
    calculate_similarity
)


def example_comparison():
    """Compare character-based vs semantic chunking."""
    
    text = """
    Account Number Masking Requirements
    
    To protect user privacy, account numbers must be masked in all outputs.
    The system should detect patterns like 16-digit numbers and replace them
    with asterisks. Example: 1234-5678-9012-3456 becomes ****-****-****-3456.
    
    Implementation Details
    
    The masking service uses regex patterns to identify sensitive data. It checks
    for credit card formats, SSN patterns, and bank account numbers. Once identified,
    the entire number is replaced with asterisks except for the last 4 digits for
    user identification purposes.
    
    Testing Guidelines
    
    Test cases should include valid and invalid patterns. Edge cases include
    numbers with spaces, dashes, and consecutive digits. The masking must be
    consistent across all API responses and logged data to ensure compliance.
    """
    
    print("=" * 70)
    print("CHARACTER-BASED CHUNKING (Old - RecursiveCharacterTextSplitter)")
    print("=" * 70)
    print("""
    Problem:
    - Splits at 800 characters regardless of meaning
    - May cut paragraphs or sentences mid-way
    - Creates orphaned context without complete ideas
    
    Example chunks:
    [0] "Account Number Masking Requirements\n\nTo protect user privacy, account..."
    [1] "...in all outputs. The system should detect patterns like 16-digit numbers..."
    [2] "...identifies, the entire number is replaced with asterisks except for..."
    
    Issue: Chunk [0] ends mid-explanation, [1] starts with continuation
    """)
    
    print("\n" + "=" * 70)
    print("SEMANTIC CHUNKING (New - Meaning-Based)")
    print("=" * 70)
    
    chunks = chunk_document(text)
    
    print(f"""
    Solution:
    - Splits based on semantic similarity between sentences
    - Each chunk represents a complete topic or idea
    - Maintains paragraph and concept boundaries
    
    Generated chunks ({len(chunks)} total):
    """)
    
    for i, chunk in enumerate(chunks):
        words = len(chunk.split())
        print(f"\n    [{i}] ({words} words)")
        print(f"        {chunk[:80]}...")


def example_similarity_detection():
    """Show how semantic similarity drives chunking."""
    
    sentences = [
        "Account numbers must be masked for privacy.",
        "User privacy is critical in financial systems.",
        "The masking service uses regex patterns.",
        "Regex identifies credit cards and SSNs.",
        "Testing should include edge cases."
    ]
    
    embeddings = create_sentence_embeddings(sentences)
    
    print("\n" + "=" * 70)
    print("SIMILARITY DETECTION")
    print("=" * 70)
    print(f"\nSentences: {len(sentences)}")
    print(f"Embeddings: {len(embeddings)}")
    
    print("\nSimilarity between adjacent sentences:")
    for i in range(len(sentences) - 1):
        sim = calculate_similarity(embeddings[i], embeddings[i + 1])
        status = "MERGE ✓" if sim >= 0.7 else "SPLIT ✗"
        print(f"  [{i}→{i+1}] {sim:.3f}  {status}")
        print(f"         '{sentences[i][:45]}...'")
        print(f"         '{sentences[i+1][:45]}...'")


def example_chunk_size_control():
    """Demonstrate chunk size constraints."""
    
    print("\n" + "=" * 70)
    print("CHUNK SIZE CONTROL")
    print("=" * 70)
    print("""
    Target Range: 300-1000 words per chunk
    
    Constraints:
    - MIN_CHUNK_WORDS = 50: Avoid tiny chunks
    - MAX_CHUNK_WORDS = 1000: Prevent huge chunks
    - SIMILARITY_THRESHOLD = 0.7: Merge if ≥70% similar
    
    Behavior:
    1. Group similar sentences together
    2. Don't exceed MAX_CHUNK_WORDS
    3. Merge small chunks with neighbors
    4. Result: Balanced, semantic chunks
    
    Example:
    - Small chunk (30 words) → merged with next
    - Medium chunk (400 words) → kept as-is
    - Large chunk (1500 words) → would split if similarity drops
    """)


def example_all_sources():
    """Show that all ingestion sources use semantic chunking."""
    
    print("\n" + "=" * 70)
    print("SEMANTIC CHUNKING ACROSS ALL SOURCES")
    print("=" * 70)
    print("""
    Ingestion Flow (Unchanged):
    
    Document (any source)
        ↓
    Extraction
        ├─ PDF: extract_text_from_pdf() → plain text
        ├─ DOCX: extract_docx() → plain text
        ├─ Image: process_image() → extracted text
        └─ URL: process_url() → web content
        ↓
    Chunking (NOW SEMANTIC!)
        └─ chunk_text() → chunk_document()
        ↓
    Embeddings
        └─ SentenceTransformer (unchanged)
        ↓
    Pinecone Storage
        └─ Vectors + metadata (unchanged)
    
    Key Point:
    All sources automatically use semantic chunking
    via the unified chunk_text() wrapper in chunk_service.py
    """)


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "SEMANTIC CHUNKING IMPLEMENTATION" + " " * 21 + "║")
    print("╚" + "=" * 68 + "╝")
    
    example_comparison()
    example_similarity_detection()
    example_chunk_size_control()
    example_all_sources()
    
    print("\n" + "=" * 70)
    print("IMPLEMENTATION MODULES")
    print("=" * 70)
    print("""
    New: ingest/services/semantic_chunk.py
    - split_sentences() → sentence segmentation
    - create_sentence_embeddings() → embedding generation
    - calculate_similarity() → cosine similarity
    - group_semantic_chunks() → semantic grouping
    - merge_small_chunks() → size control
    - chunk_document() → main API
    
    Updated: ingest/services/chunk_service.py
    - chunk_text() now calls chunk_document()
    - Transparent replacement for existing code
    
    All ingestion sources work without modification!
    """)
