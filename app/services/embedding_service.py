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
                # Call embed_content as a function (NOT as method)
                result = genai.embed_content(
                    model=model_name,
                    content=text,
                    task_type="retrieval_document"
                )
                
                # Extract embedding - result is a dict with 'embedding' key
                embedding = None
                
                if isinstance(result, dict):
                    # Result is dict like: {'embedding': [0.1, 0.2, ...]}
                    embedding = result.get('embedding')
                elif hasattr(result, 'embedding'):
                    embedding = result.embedding
                elif hasattr(result, '__getitem__'):
                    # Try to access as dict-like
                    try:
                        embedding = result['embedding']
                    except (KeyError, TypeError):
                        pass
                
                if embedding is None:
                    raise ValueError(f"No embedding found in result: {result}")
                
                # Ensure embedding is a list
                if not isinstance(embedding, list):
                    if hasattr(embedding, '__iter__') and not isinstance(embedding, str):
                        embedding = list(embedding)
                    else:
                        raise ValueError(f"Invalid embedding format: {type(embedding)}")
                
                embeddings.append(embedding)
                
            except Exception as e:
                logger.error(f"Embedding error for text: {e}")
                raise Exception(f"Gemini embedding failed: {e}")
        
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
        
        # Test with a simple embedding - CALL AS FUNCTION, not method
        try:
            result = genai.embed_content(
                model='models/text-embedding-004',
                content="test",
                task_type="retrieval_document"
            )
            
            # Result is a dict with 'embedding' key
            if result is not None:
                if isinstance(result, dict) and 'embedding' in result:
                    return "connected"
                elif hasattr(result, 'embedding'):
                    return "connected"
                elif hasattr(result, '__getitem__'):
                    try:
                        if result['embedding']:
                            return "connected"
                    except (KeyError, TypeError):
                        pass
            
            logger.warning(f"Unexpected result format: {type(result)}")
            return "disconnected"
            
        except Exception as e:
            logger.error(f"Gemini embedding health check error: {e}")
            return "disconnected"
            
    except Exception as e:
        logger.error(f"Gemini embedding health check failed: {e}")
        return "disconnected"