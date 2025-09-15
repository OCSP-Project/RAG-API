#!/usr/bin/env python3
# embedding_service.py - Tạo file này trong thư mục gốc

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Lazy import để tránh crash nếu chưa cài
try:
    from sentence_transformers import SentenceTransformer
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    print("WARNING: sentence-transformers not installed")
    print("Run: pip install sentence-transformers torch transformers")

MODEL_ID = os.environ.get("MODEL_ID", "dangvantuan/vietnamese-document-embedding")

app = FastAPI(title="Vietnamese Embedding Service", version="1.0.0")

# Global model variable
model = None

def load_model():
    global model
    if not MODEL_AVAILABLE:
        raise HTTPException(500, "sentence-transformers not installed")
    
    if model is None:
        print(f"Loading model: {MODEL_ID}")
        try:
            model = SentenceTransformer(MODEL_ID, trust_remote_code=True)
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            raise HTTPException(500, f"Model loading failed: {e}")
    
    return model

class EmbedRequest(BaseModel):
    texts: Optional[List[str]] = None
    text: Optional[str] = None
    inputs: Optional[List[str]] = None

@app.get("/")
def root():
    return {
        "service": "Vietnamese Embedding Service",
        "model": MODEL_ID,
        "status": "ready" if model else "not_loaded"
    }

@app.get("/health")
def health():
    try:
        m = load_model()
        # Test với một text đơn giản
        test_emb = m.encode(["ping"], normalize_embeddings=True)
        return {
            "status": "healthy",
            "model_id": MODEL_ID,
            "embedding_dim": len(test_emb[0]) if len(test_emb) > 0 else 0,
            "model_loaded": True
        }
    except Exception as e:
        return {
            "status": "error", 
            "error": str(e),
            "model_loaded": False
        }

@app.post("/embed")
def embed(req: EmbedRequest):
    try:
        # Parse input texts
        texts = req.texts or req.inputs or ([req.text] if req.text else [])
        
        if not texts:
            return {"embeddings": []}
        
        # Load model if not loaded
        m = load_model()
        
        # Generate embeddings
        embeddings = m.encode(
            texts,
            normalize_embeddings=True,
            convert_to_tensor=False,
            batch_size=32
        )
        
        # Convert to list if needed
        if hasattr(embeddings, 'tolist'):
            embeddings = embeddings.tolist()
        
        return {"embeddings": embeddings}
        
    except Exception as e:
        raise HTTPException(500, f"Embedding error: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8001"))
    print(f"Starting embedding service on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")