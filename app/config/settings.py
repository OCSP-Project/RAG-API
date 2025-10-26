import os
from typing import Optional

class Settings:
    """Application configuration settings"""
    
    # App Info
    APP_TITLE: str = "OCSP RAG API"
    APP_VERSION: str = "1.0.0"
    
    # Database
    DATABASE_URL: str = os.getenv("URL") or os.getenv("DATABASE_URL") or ""
    
    # External APIs
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    EMBED_DIM: int = int(os.getenv("EMBED_DIM", "768"))
    
    # Frontend
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:3000")
    FRONTEND_ORIGIN: str = os.getenv("FRONTEND_ORIGIN", "")
    
    # CORS
    CORS_ALLOW_CREDENTIALS: bool = bool(os.getenv("CORS_ALLOW_CREDENTIALS", "true").lower() == "true")
    
    @property
    def allow_origins(self) -> list:
        """Get CORS allowed origins"""
        return ["*"] if not self.FRONTEND_ORIGIN else [self.FRONTEND_ORIGIN]
    
    @property
    def allow_credentials(self) -> bool:
        """Get CORS credentials setting"""
        if self.CORS_ALLOW_CREDENTIALS and self.allow_origins == ["*"]:
            return False
        return self.CORS_ALLOW_CREDENTIALS

# Global settings instance
settings = Settings()
