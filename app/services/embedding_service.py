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
        model_name = 'models/text-embedding-004'
        
        embeddings = []
        for text in texts:
            try:
                # Ensure genai.embed_content exists and is callable
                if not hasattr(genai, 'embed_content'):
                    raise AttributeError("genai.embed_content not available. Update google-generativeai: pip install --upgrade google-generativeai")
                
                # Call embed_content as a function
                result = genai.embed_content(
                    model=model_name,
                    content=text,
                    task_type="retrieval_document"
                )
                
                # Extract embedding from result - handle various formats
                embedding = None
                
                if isinstance(result, dict):
                    embedding = result.get('embedding') or result.get('embeddings', [None])[0]
                elif hasattr(result, 'embedding'):
                    embedding = result.embedding
                elif hasattr(result, 'embeddings'):
                    # If result has embeddings (plural), take first
                    embeds = result.embeddings
                    embedding = embeds[0] if isinstance(embeds, (list, tuple)) and len(embeds) > 0 else embeds
                else:
                    # Try to get embedding attribute or use result directly
                    embedding = getattr(result, 'embedding', None) or result
                
                # Ensure embedding is a list
                if embedding is None:
                    raise ValueError("Embedding result is None")
                    
                if not isinstance(embedding, list):
                    if hasattr(embedding, '__iter__') and not isinstance(embedding, str):
                        embedding = list(embedding)
                    else:
                        raise ValueError(f"Invalid embedding format: {type(embedding)}")
                
                embeddings.append(embedding)
                
            except (AttributeError, TypeError) as e:
                error_msg = str(e)
                logger.error(f"Embedding API error: {error_msg}")
                
                # More specific error message
                if "'GenerativeModel'" in error_msg or "embed_content" in error_msg.lower():
                    raise Exception(
                        f"Gemini embedding API error: {error_msg}. "
                        "Please update google-generativeai: pip install --upgrade google-generativeai"
                    )
                raise Exception(f"Gemini embedding failed: {error_msg}")
        
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
