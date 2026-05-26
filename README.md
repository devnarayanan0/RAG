
## PROJECT ARCHITECTURE
```
Promantus RAG Project
│
├─── CHAT SYSTEM
│    ├─ app/main.py (FastAPI server + CORS + chat endpoint)
│    │  └─ app/rag.py (RAG pipeline orchestration)
│    │     ├─ Query embedding (all-MiniLM-L6-v2)
│    │     ├─ Pinecone retrieval (TOP_K=5)
│    │     ├─ Context building (join TOP_K matches)
│    │     └─ Groq LLM response (llama-3.3-70b-versatile, temp=0.1)
│    └─ app/models.py (Pydantic Query model)
│
├─── INGESTION SYSTEM (8001 port)
│    ├─ ingest/ingestServer.py (FastAPI server + CORS + middleware)
│    │  ├─ Routes: / /health /audit/{req_id} /audit/list/recent /upload
│    │  └─ Image steps: /image/start /image/encode /image/describe 
│    │     /image/preview-chunks /image/embed /image/store /image/session/{id}
│    │
│    ├─ ingest/ingestRouter.py (Pipeline orchestration)
│    │  ├─ ingest_document() — PDF/DOCX/Image OCR/URL extraction pipeline
│    │  ├─ ingest_image_objects() — Vision LLM description pipeline
│    │  └─ 7 step functions for image object extraction state management
│    │
│    ├─ ingest/extractor/
│    │  ├─ pdf.py (pypdf + pytesseract OCR fallback)
│    │  ├─ docx.py (python-docx text extraction)
│    │  ├─ image.py (pytesseract OCR text extraction)
│    │  └─ url.py (requests + BeautifulSoup parsing)
│    │
│    ├─ ingest/services/
│    │  ├─ chunk_service.py (RecursiveCharacterTextSplitter, 800 tokens)
│    │  ├─ duplicate_service.py (SHA256 hash + cosine similarity 0.95 threshold)
│    │  ├─ masking_service.py (Regex: account, SSN, credit card patterns)
│    │  ├─ pinecone_service.py (Vector building + upsert)
│    │  ├─ groq_vision.py (Vision LLM: meta-llama/llama-4-scout-17b-16e-instruct)
│    │  ├─ image_encoder.py (Base64 encoding + MIME detection)
│    │  └─ extractor_service.py (DEAD CODE — not imported)
│    │
│    └─ ingest/frontend/
│         ├─ admin.html (Upload UI + audit dashboard)
│         └─ index.html (Chat frontend)
│
└─── DATA STORAGE
     ├─ Pinecone (Vector DB, 384-dim embeddings)
     │  └─ Index: ragintern1
     └─ ingest/data/ (Temporary file storage)
```
### Data Pipeline

```
Document → Extract → Deduplicate → Mask → Chunk → Embed → Pinecone
  ↑                                                            ↓
  ├────────────────── Audit Log (100 recent) ←────────────────┤
                                                               ↓
Chat Query → Embed → Retrieve (TOP_K=5) → Build Context → LLM → Answer
```

### Metrics

| Metric | Value |
|--------|-------|
| Total Files | 20+ |
| Total Lines | ~2000 |
| Endpoints | 15 (8 ingestion + 4 audit + 3 chat/health) |
| Extractors | 4 (PDF, DOCX, Image, URL) |
| Services | 8 (chunk, mask, dup, pinecone, groq_vision, image_encoder, extractor, +1 unused) |
| Data Fields | 15+ per vector |
| TOP_K | 5 |
| Chunk Size | 800 tokens |
| Embedding Dim | 384 |
| Audit History | 100 |
| Temperature | 0.1 (chat), 0.3 (vision) |

---