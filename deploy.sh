#!/bin/bash
# deploy.sh - Chạy script này trên EC2

set -e  # Exit on any error

echo "🚀 DEPLOYING RAG CHATBOT WITH UPDATED DEPENDENCIES..."

# 1. Backup existing files
echo "=== 1. BACKUP ==="
if [ -f "app/main.py" ]; then
    cp app/main.py app/main.py.backup.$(date +%Y%m%d_%H%M%S)
    echo "✅ Backed up main.py"
fi

if [ -f "docker-compose.yml" ]; then
    cp docker-compose.yml docker-compose.yml.backup.$(date +%Y%m%d_%H%M%S)
    echo "✅ Backed up docker-compose.yml"
fi

# 2. Verify files exist
echo "=== 2. VERIFY FILES ==="
required_files=("app/main.py" "requirements.txt" "Dockerfile" "docker-compose.yml" ".env")
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Missing required file: $file"
        echo "Please create this file first!"
        exit 1
    fi
    echo "✅ Found: $file"
done

# 3. Stop existing containers
echo "=== 3. STOP EXISTING CONTAINERS ==="
docker-compose down 2>/dev/null || echo "No containers to stop"

# 4. Clean up Docker
echo "=== 4. DOCKER CLEANUP ==="
docker system prune -f
docker volume prune -f

# 5. Build and start services
echo "=== 5. BUILD AND START ==="
echo "Building with updated dependencies..."
docker-compose build --no-cache
docker-compose up -d

# 6. Wait for services to start
echo "=== 6. WAITING FOR SERVICES ==="
echo "Waiting 60 seconds for services to initialize..."
sleep 60

# 7. Check container status
echo "=== 7. CHECK CONTAINERS ==="
docker ps

# 8. Check logs
echo "=== 8. CHECK LOGS ==="
echo "--- API Logs ---"
docker logs rag-api --tail 20

echo "--- Postgres Logs ---"
docker logs rag-postgres --tail 10

# 9. Test API health
echo "=== 9. TEST API HEALTH ==="
max_attempts=30
attempt=1

while [ $attempt -le $max_attempts ]; do
    echo "Health check attempt $attempt/$max_attempts..."
    
    if curl -f http://localhost:8000/health 2>/dev/null; then
        echo "✅ API is healthy!"
        echo ""
        echo "Health response:"
        curl -s http://localhost:8000/health | python3 -m json.tool || curl -s http://localhost:8000/health
        break
    else
        echo "⏳ API not ready yet, waiting 10s..."
        sleep 10
        ((attempt++))
    fi
    
    if [ $attempt -gt $max_attempts ]; then
        echo "❌ API failed to start after $max_attempts attempts"
        echo "Check logs above for errors"
        exit 1
    fi
done

# 10. Test database
echo ""
echo "=== 10. TEST DATABASE ==="
if docker exec rag-postgres psql -U postgres -d rag_db -c "SELECT COUNT(*) FROM document_chunks;" 2>/dev/null; then
    echo "✅ Database connection OK"
else
    echo "⚠️ Database test failed, but continuing..."
fi

# 11. Setup embedding service
echo ""
echo "=== 11. SETUP EMBEDDING SERVICE ==="
echo "Installing embedding service dependencies..."

# Check if embedding service exists
if [ ! -f "embedding_service.py" ]; then
    echo "⚠️ embedding_service.py not found, creating it..."
    # Code will be in separate file
fi

# Install embedding dependencies
pip3 install --user sentence-transformers torch transformers

# Start embedding service
echo "Starting embedding service on port 8001..."
nohup python3 embedding_service.py > embedding.log 2>&1 &
EMBED_PID=$!

# Wait for embedding service
sleep 20

# Test embedding service
if curl -f http://localhost:8001/health 2>/dev/null; then
    echo "✅ Embedding service is running!"
    curl -s http://localhost:8001/health | python3 -m json.tool
else
    echo "⚠️ Embedding service failed to start"
    echo "Check embedding.log for details"
    tail -20 embedding.log 2>/dev/null || echo "No embedding.log found"
fi

# 12. Final summary
echo ""
echo "=== 🎉 DEPLOYMENT SUMMARY ==="
echo "✅ API Service: http://localhost:8000"
echo "✅ API Health: http://localhost:8000/health"
echo "✅ API Docs: http://localhost:8000/docs"
echo "⚙️ Embedding Service: http://localhost:8001"
echo "⚙️ Database: PostgreSQL on port 5432"
echo ""
echo "🌐 Public URLs (update Security Group first):"
echo "📍 API: http://13.210.146.91:8000"
echo "📍 Embedding: http://13.210.146.91:8001"
echo ""
echo "🔧 Next steps:"
echo "1. Add Security Group rule for port 8001"
echo "2. Test from Colab using public IP"
echo "3. Add some documents to test RAG"
echo ""
echo "📋 Useful commands:"
echo "  View logs: docker logs rag-api"
echo "  Restart: docker-compose restart"
echo "  Stop: docker-compose down"