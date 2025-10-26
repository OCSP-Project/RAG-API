from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum

class DocItem(BaseModel):
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class AddDocsRequest(BaseModel):
    docs: List[DocItem]

class QueryRequest(BaseModel):
    query: str
    k: int = 5

class StoreChunkRequest(BaseModel):
    content: str
    embedding: List[float]
    metadata: Dict[str, Any] = Field(default_factory=dict)

class IngestURLReq(BaseModel):
    url: str
    chunk_size: int = 1200
    chunk_overlap: int = 150

class DocumentUploadRequest(BaseModel):
    source: str  # URL hoặc file path
    chunk_size: int = 500
    chunk_overlap: int = 50

class ContractorAction(BaseModel):
    contractor_id: str  # UUID từ backend DB
    contractor_name: str
    contractor_slug: str
    description: str
    budget_range: str
    rating: float
    location: str
    profile_url: str
    contact_url: str

class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = None
    top_k: int = 5

class ChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)

class EnhancedChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    contractors: List[ContractorAction] = Field(default_factory=list)
    has_recommendations: bool = False

class ChatIntent(str, Enum):
    GREETING = "greeting"
    GENERAL = "general_chat"
    CONTRACTOR_FULL = "contractor_request"
    CONTRACTOR_PARTIAL = "incomplete_request"
