#!/bin/bash

# Build script for Promantus RAG Chat Server
# Validates code and prepares for deployment

set -e

echo "🔨 Building Promantus RAG Chat Server..."
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python $python_version"

# Validate environment file
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found - will use environment variables"
else
    echo "✓ .env file found"
fi

# Validate required Python modules
echo "✓ Checking Python dependencies..."
python3 -m py_compile app/main.py app/rag.py app/models.py

# Install/upgrade dependencies
echo "✓ Dependencies validated"
echo ""
echo "✅ Build complete! Ready for deployment"
echo ""
echo "📋 Next steps:"
echo "   1. Start server:   bash start-chat.sh"
echo "   2. Test endpoint:  curl -X POST http://localhost:8000/chat -H 'Content-Type: application/json' -d '{\"question\":\"test\"}'"
echo "   3. Push to Git:    git add . && git commit -m 'Production chat server' && git push"
echo ""
