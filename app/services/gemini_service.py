import requests
import logging
from typing import Optional
from app.config.settings import settings

logger = logging.getLogger("rag-api")

class GeminiService:
    """Service for interacting with Gemini API"""
    
    def __init__(self):
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize Gemini model"""
        if settings.GEMINI_API_KEY and settings.GEMINI_API_KEY != "your-gemini-key-here":
            try:
                import google.generativeai as genai
                genai.configure(api_key=settings.GEMINI_API_KEY)
                self.model = genai.GenerativeModel("gemini-1.5-flash")
                logger.info("Gemini model initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini model: {e}")
                self.model = None
        else:
            logger.warning("GEMINI_API_KEY not configured")
    
    def generate_response(self, prompt: str) -> Optional[str]:
        """Generate response using Gemini API"""
        if not self.model:
            return None
        
        try:
            response = self.model.generate_content(prompt)
            answer = getattr(response, "text", None) or (
                response.candidates[0].content.parts[0].text 
                if getattr(response, "candidates", None) else ""
            )
            return answer
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            return None
    
    def generate_streaming_response(self, prompt: str):
        """Generate streaming response using Gemini API"""
        if not self.model:
            return None
        
        try:
            return self.model.generate_content(prompt, stream=True)
        except Exception as e:
            logger.error(f"Gemini streaming error: {e}")
            return None
    
    def is_configured(self) -> bool:
        """Check if Gemini is properly configured"""
        return self.model is not None

# Global instance
gemini_service = GeminiService()
