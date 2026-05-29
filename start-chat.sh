#!/bin/bash

# Start Retrieval Chat Server
# Hosts only the /chat endpoint on port 8000

set -e

echo "🚀 Starting Promantus RAG Chat Server..."

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '#' | xargs)
    echo "✓ Environment loaded from .env"
else
    echo "⚠️  WARNING: .env file not found. Using environment variables."
fi

# Verify required environment variables
required_vars=("GROQ_API_KEY" "PINECONE_API_KEY" "PINECONE_INDEX")
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "❌ ERROR: $var is not set"
        exit 1
    fi
done
echo "✓ All required environment variables set"

# Start uvicorn server
echo "📡 Starting FastAPI server on http://localhost:8000"
echo "   GET  / - Serve frontend"
echo "   GET  /health - Health check"
echo "   POST /chat - RAG retrieval endpoint"
echo ""

uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    --log-level info
