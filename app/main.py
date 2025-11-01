import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from datetime import datetime

from app.config.settings import settings
from app.core.database import init_database, get_document_count
from app.services.embedding_service import check_gemini_embedding_health
from app.services.gemini_service import gemini_service
from app.api import api_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag-api")

# Create FastAPI app
app = FastAPI(
    title=settings.APP_TITLE,
    version=settings.APP_VERSION,
    description="RAG API for contractor recommendations"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allow_origins,
    allow_credentials=settings.allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router)

# Add direct /chat endpoint for frontend compatibility
from app.services.chat_service import chat_service
from app.models.schemas import ChatRequest, EnhancedChatResponse

@app.post("/chat", response_model=EnhancedChatResponse)
def chat_direct(req: ChatRequest):
    """Direct chat endpoint for frontend"""
    return chat_service.process_chat(req)

# Startup event
@app.on_event("startup")
def on_startup():
    """Initialize application on startup"""
    try:
        init_database()
        logger.info("Application started successfully")
    except Exception as e:
        logger.error(f"Startup error: {e}")

# Health check endpoint
@app.get("/health")
def health_check():
    """Health check endpoint"""
    try:
        doc_count = get_document_count()
        db_status = "connected"
    except Exception as e:
        doc_count = 0
        db_status = f"error: {e}"
    
    return {
        "status": "healthy" if db_status == "connected" else "degraded",
        "database": db_status,
        "document_count": doc_count,
        "embedding_service": check_gemini_embedding_health(),
        "gemini_configured": gemini_service.is_configured(),
        "embed_dim": settings.EMBED_DIM,
        "timestamp": datetime.now().isoformat()
    }

# Root endpoint
@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": f"{settings.APP_TITLE} is running!",
        "version": settings.APP_VERSION,
        "docs": "/docs"
    }

# Streaming chat endpoint (kept for compatibility)
@app.get("/chat/stream")
def chat_stream(q: str, k: int = 5):
    """Server-Sent Events: stream response from LLM"""
    if not gemini_service.is_configured():
        raise HTTPException(500, "GEMINI_API_KEY chưa cấu hình")

    def event_gen():
        try:
            # Send sources first
            yield "event: sources\n"
            yield "data: []\n\n"  # Simplified for now
            
            # Stream from Gemini
            prompt = f"Trả lời câu hỏi: {q}"
            for chunk in gemini_service.generate_streaming_response(prompt):
                txt = getattr(chunk, "text", "") or ""
                if txt:
                    yield "data: " + txt + "\n\n"
        except Exception as e:
            yield f"event: error\ndata: {str(e)}\n\n"
        
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
