from fastapi import APIRouter, HTTPException
from app.models.schemas import (
    AddDocsRequest, QueryRequest, StoreChunkRequest,
    IngestURLReq, DocumentUploadRequest, ChatRequest,
    ImageAnalysisRequest, ImageAnalysisResponse,
    AIConsultationRequest, AIConsultationResponse
)
from app.services.embedding_service import embed_via_gemini, check_gemini_embedding_health
from app.services.document_service import document_processor
from app.services.chat_service import chat_service
from app.services.gemini_service import gemini_service
from app.services.contractor_service import contractor_embedding_service
from app.core.database import store_chunks, search_similar_vectors, get_document_count
from app.config.settings import settings
import json
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Any
from pydantic import BaseModel

router = APIRouter()

# In-memory rate limiter for AI consultation (5 messages per day per user)
class RateLimiter:
    def __init__(self, max_messages_per_day: int = 5):
        self.max_messages = max_messages_per_day
        # user_id -> list of timestamps
        self.user_messages: Dict[str, List[datetime]] = defaultdict(list)

    def check_and_increment(self, user_id: str) -> tuple[bool, int]:
        """
        Check if user can send message and increment counter

        Returns:
            (can_send, remaining_messages)
        """
        now = datetime.now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        # Lọc bỏ các message cũ (không phải hôm nay)
        self.user_messages[user_id] = [
            ts for ts in self.user_messages[user_id]
            if ts >= today_start
        ]

        current_count = len(self.user_messages[user_id])

        if current_count >= self.max_messages:
            return False, 0

        # Thêm message mới
        self.user_messages[user_id].append(now)
        remaining = self.max_messages - (current_count + 1)

        return True, remaining

# Global rate limiter instance
ai_rate_limiter = RateLimiter(max_messages_per_day=5)

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

@router.post("/analyze-incident", response_model=ImageAnalysisResponse)
async def analyze_incident(request: ImageAnalysisRequest):
    """
    Phân tích ảnh sự cố xây dựng bằng AI

    - Nhận ảnh base64 + báo cáo sơ bộ
    - Trả về báo cáo chi tiết + đề xuất giải pháp
    """
    if not request.images:
        raise HTTPException(status_code=400, detail="Vui lòng cung cấp ít nhất 1 ảnh sự cố.")

    if not gemini_service.is_configured():
        raise HTTPException(
            status_code=500,
            detail="Hệ thống AI chưa được cấu hình. Vui lòng liên hệ quản trị viên."
        )

    # Gọi service phân tích
    result = await gemini_service.analyze_incident_images(
        images_b64=request.images,
        incident_report=request.incident_report,
        context=request.context
    )

    return ImageAnalysisResponse(
        incident_report=result.get("incident_report", ""),
        recommendations=result.get("recommendations", "")
    )

@router.post("/ai-consultation", response_model=AIConsultationResponse)
async def ai_consultation(request: AIConsultationRequest):
    """
    Tư vấn AI chuyên về kỹ thuật xây dựng, pháp luật xây dựng, và an toàn lao động

    Rate limit: 5 tin nhắn mỗi ngày cho mỗi user
    """
    if not request.user_id or not request.message.strip():
        raise HTTPException(
            status_code=400,
            detail="Vui lòng cung cấp user_id và nội dung câu hỏi."
        )

    if not gemini_service.is_configured():
        raise HTTPException(
            status_code=500,
            detail="Hệ thống AI chưa được cấu hình. Vui lòng liên hệ quản trị viên."
        )

    # Kiểm tra rate limit
    can_send, remaining = ai_rate_limiter.check_and_increment(request.user_id)

    if not can_send:
        raise HTTPException(
            status_code=429,
            detail="Bạn đã hết lượt tư vấn hôm nay. Vui lòng quay lại vào ngày mai. (Giới hạn: 5 tin nhắn/ngày)"
        )

    # Gọi AI để tư vấn
    response_text = gemini_service.construction_consultation(request.message)

    if not response_text:
        raise HTTPException(
            status_code=500,
            detail="Hệ thống AI gặp lỗi khi xử lý câu hỏi. Vui lòng thử lại."
        )

    return AIConsultationResponse(
        response=response_text,
        remaining_messages=remaining
    )

# Pydantic models for contractor embedding
class ContractorData(BaseModel):
    contractor_id: str
    contractor_name: str
    contractor_slug: str
    description: str
    specialties: List[str] = []
    budget_range: str
    location: str
    rating: float
    years_of_experience: int = 0
    team_size: int = 0
    is_verified: bool = False

class EmbedContractorsRequest(BaseModel):
    contractors: List[Dict[str, Any]]
    chunk_size: int = 500
    chunk_overlap: int = 50

class EmbedContractorsResponse(BaseModel):
    message: str
    contractors_processed: int
    chunks_created: int
    chunks_stored: int
    chunk_ids: List[int]

@router.post("/embed/contractors", response_model=EmbedContractorsResponse)
async def embed_contractors(request: EmbedContractorsRequest):
    """
    Embed contractors from backend into RAG database

    Accepts a list of contractor objects and creates searchable embeddings.
    This allows AI to recommend contractors based on user queries.
    """
    if not request.contractors:
        raise HTTPException(
            status_code=400,
            detail="Vui lòng cung cấp danh sách contractors."
        )

    try:
        result = contractor_embedding_service.embed_contractors(
            contractors=request.contractors,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap
        )

        return EmbedContractorsResponse(
            message=result["message"],
            contractors_processed=result["contractors_processed"],
            chunks_created=result["chunks_created"],
            chunks_stored=result["chunks_stored"],
            chunk_ids=result["chunk_ids"]
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Có lỗi xảy ra khi embedding contractors: {str(e)}"
        )
