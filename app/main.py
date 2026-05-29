import os
import logging
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.models import Query as QueryModel
from app.rag import ask, ask_with_rerank

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Promantus RAG API",
    version="1.0.0",
    description="Retrieval Augmented Generation with Cross-Encoder Reranking"
)

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_PATH = "app/frontend/index.html"


@app.get("/", tags=["System"])
async def root():
    """Serve chat frontend."""
    if os.path.exists(FRONTEND_PATH):
        return FileResponse(FRONTEND_PATH, media_type="text/html")
    return {"message": "RAG Server Running"}


@app.get("/health", tags=["System"])
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/chat", tags=["Chat"])
def chat(query: QueryModel, rerank: bool = Query(False)):
    """
    Answer user query using RAG with optional reranking.
    
    Parameters:
        query: User question
        rerank: Enable cross-encoder reranking (shows retrieval stats)
    """
    try:
        if rerank:
            result = ask_with_rerank(query.question)
            return {
                "question": query.question,
                "answer": result.get("response", "Unable to generate response"),
                "retrieval": result.get("retrieval", {})
            }
        
        answer = ask(query.question)
        return {
            "question": query.question,
            "answer": answer
        }
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        return {
            "question": query.question,
            "answer": "Service error - please try again",
            "error": "internal_error"
        }

