import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.models import Query
from app.rag import ask_question

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Promantus RAG API", version="1.0.0")
#
#
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
#
#
@app.get("/", tags=["System"])
async def root():
    """Serve chat frontend."""
    path = "app/frontend/index.html"
    if os.path.exists(path):
        return FileResponse(path, media_type="text/html")
    return {"message": "RAG Server Running"}

@app.get("/health", tags=["System"])
def health_check():
    return {"status": "healthy"}

@app.post("/chat", tags=["Chat"])
def chat(query: Query):
    """Process chat query and return answer."""
    answer = ask_question(query.question)
    return {
        "question": query.question,
        "answer": answer,
    }
