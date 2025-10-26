from fastapi import APIRouter, HTTPException
from app.models.schemas import (
    AddDocsRequest, QueryRequest, StoreChunkRequest, 
    IngestURLReq, DocumentUploadRequest, ChatRequest
)
from app.services.embedding_service import embed_via_gemini, check_gemini_embedding_health
from app.services.document_service import document_processor
from app.services.chat_service import chat_service
from app.core.database import store_chunks, search_similar_vectors, get_document_count
from app.config.settings import settings
import json

router = APIRouter()

@router.post("/add_docs")
def add_docs(req: AddDocsRequest):
    """Add documents to the knowledge base"""
    if not req.docs:
        raise HTTPException(400, "docs rỗng")
    
    texts = [d.content for d in req.docs]
    vecs = embed_via_gemini(texts)
    
    chunks_data = []
    for d, v in zip(req.docs, vecs):
        chunks_data.append({
            "content": d.content,
            "embedding": v,
            "metadata": json.dumps(d.metadata)
        })
    
    ids = store_chunks(chunks_data)
    return {"added": len(ids), "ids": ids}

@router.post("/store-chunk")
def store_chunk(request: StoreChunkRequest):
    """Store a single chunk with embedding"""
    if len(request.embedding) != settings.EMBED_DIM:
        raise HTTPException(400, f"Embedding phải có {settings.EMBED_DIM} chiều.")
    
    chunks_data = [{
        "content": request.content,
        "embedding": request.embedding,
        "metadata": json.dumps(request.metadata)
    }]
    
    ids = store_chunks(chunks_data)
    return {"message": "Chunk stored successfully", "id": ids[0]}

@router.post("/search-similar")
def search_similar(req: QueryRequest):
    """Search for similar documents"""
    return chat_service.search_similar(req.query, req.k)

@router.post("/upload_document")
async def upload_document(req: DocumentUploadRequest):
    """Upload and process document from URL or file path"""
    try:
        result = document_processor.process_document(
            req.source, 
            req.chunk_size, 
            req.chunk_overlap
        )
        return result
    except Exception as e:
        raise HTTPException(500, str(e))

@router.post("/ingest/url")
def ingest_url(req: IngestURLReq):
    """Ingest document from URL"""
    try:
        result = document_processor.process_url(
            req.url, 
            req.chunk_size, 
            req.chunk_overlap
        )
        return result
    except Exception as e:
        raise HTTPException(500, str(e))

@router.post("/chat")
def chat_endpoint(req: ChatRequest):
    """Chat endpoint with intent detection and routing"""
    return chat_service.process_chat(req)
