import logging
from typing import List
from app.config.settings import settings

logger = logging.getLogger("rag-api")

def embed_via_gemini(texts: List[str]) -> List[List[float]]:
    """Get embeddings from Gemini API"""
    if not settings.GEMINI_API_KEY or settings.GEMINI_API_KEY == "your-gemini-key-here":
        raise Exception("GEMINI_API_KEY chưa cấu hình")
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=settings.GEMINI_API_KEY)
        
        # Use Gemini's text embedding model
        model = 'models/text-embedding-004'
        
        embeddings = []
        for text in texts:
            # Generate embedding for each text
            result = genai.embed_content(
                model=model,
                content=text,
                task_type="retrieval_document"
            )
            embedding = result['embedding']
            embeddings.append(embedding)
        
        return embeddings
        
    except Exception as e:
        logger.error(f"Gemini embedding error: {e}")
        raise Exception(f"Gemini embedding service error: {e}")

def check_gemini_embedding_health() -> str:
    """Check if Gemini embedding is available"""
    if not settings.GEMINI_API_KEY or settings.GEMINI_API_KEY == "your-gemini-key-here":
        return "not_configured"
    try:
        import google.generativeai as genai
        genai.configure(api_key=settings.GEMINI_API_KEY)
        # Test with a simple embedding
        genai.embed_content(
            model='models/text-embedding-004',
            content="test",
            task_type="retrieval_document"
        )
        return "connected"
    except Exception as e:
        logger.error(f"Gemini embedding health check failed: {e}")
        return "disconnected"
