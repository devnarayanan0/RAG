import os
import logging
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.models import Query as QueryModel
from app.rag import ask_question, ask_question_with_reranking

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
    Answer user query using RAG.
    
    Parameters:
        query: User question
        rerank: Enable cross-encoder reranking (shows retrieval stats)
    """
    if rerank:
        result = ask_question_with_reranking(query.question)
        return {
            "question": query.question,
            "answer": result["response"],
            "retrieval": result["retrieval"]
        }
    
    answer = ask_question(query.question)
    return {
        "question": query.question,
        "answer": answer,
    }

