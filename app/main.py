import os, json
from typing import List, Dict, Optional, Any
import requests
import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from datetime import datetime
import google.generativeai as genai
from fastapi.responses import StreamingResponse
from docling.document_converter import DocumentConverter
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time, io, requests as _rq
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import logging
import re
from typing import Optional, Tuple, List
from enum import Enum
logger = logging.getLogger("rag-api")
logger.setLevel(logging.INFO)


APP_TITLE = "OCSP RAG API"
APP_VERSION = "1.0.0"

DB_URL = os.getenv("URL") or os.getenv("DATABASE_URL") or ""
COLAB_EMBEDDING_URL = (os.getenv("COLAB_EMBEDDING_URL") or "").rstrip("/")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))  # ✅ Fixed: 768 cho model tiếng Việt

# Cấu hình Gemini
if GEMINI_API_KEY and GEMINI_API_KEY != "your-gemini-key-here":
    genai.configure(api_key=GEMINI_API_KEY)
    try:
        GEMINI_MODEL = genai.GenerativeModel("gemini-1.5-flash")
    except Exception:
        GEMINI_MODEL = None
else:
    GEMINI_MODEL = None

app = FastAPI(title=APP_TITLE, version=APP_VERSION)

# CORS: nếu dùng credentials, KHÔNG được dùng "*"
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "")  # ví dụ: https://your-frontend.example.com
allow_origins = ["*"] if not FRONTEND_ORIGIN else [FRONTEND_ORIGIN]
allow_credentials = bool(os.getenv("CORS_ALLOW_CREDENTIALS", "true").lower() == "true")
if allow_credentials and allow_origins == ["*"]:
    # Tự động chuyển sang không credentials nếu bạn vẫn muốn "*"
    allow_credentials = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================== Pydantic Models =====================
class DocItem(BaseModel):
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class AddDocsRequest(BaseModel):
    docs: List[DocItem]



# ===================== Config & Constants =====================
class ChatIntent(str, Enum):
    GREETING = "greeting"
    GENERAL = "general_chat"
    CONTRACTOR_FULL = "contractor_request"
    CONTRACTOR_PARTIAL = "incomplete_request"

# Compile regex một lần duy nhất (performance boost)
BUDGET_PATTERNS = [
    re.compile(r'(\d+)\s*tỷ'),
    re.compile(r'dưới\s*(\d+)\s*tỷ'),
    re.compile(r'khoảng\s*(\d+)\s*tỷ'),
    re.compile(r'(\d+)\s*-\s*(\d+)\s*tỷ')
]

GREETINGS = {'hello', 'hi', 'chào', 'xin chào', 'hey'}
CONTRACTOR_KW = {'nhà thầu', 'thi công', 'xây dựng', 'công trình', 'tư vấn', 'báo giá', 'ngân sách'}
CONTRACTOR_PHRASES = [
    r"nhà thầu", r"thi công", r"xây dựng", r"công trình",
    r"tư vấn", r"báo giá", r"ngân sách", r"gợi ý"
]
PROJECT_TYPES = {
    'nhà phố':   [r"nhà phố", r"townhouse"],
    'biệt thự':  [r"biệt thự", r"villa"],
    'chung cư':  [r"chung cư", r"căn hộ", r"apartment"],
    'văn phòng': [r"văn phòng", r"office"],
    'nhà xưởng': [r"nhà xưởng", r"factory", r"xưởng"],
    'khách sạn': [r"khách sạn", r"hotel"],
    'nhà hàng':  [r"nhà hàng", r"restaurant"],
}

class QueryRequest(BaseModel):
    query: str
    k: int = 5

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

class EnhancedChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    contractors: List[ContractorAction] = Field(default_factory=list)
    has_recommendations: bool = False

class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = None
    top_k: int = 5  # ✅ Fixed: Added missing field

class ChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)

class StoreChunkRequest(BaseModel):
    content: str
    embedding: List[float]
    metadata: Dict[str, Any] = Field(default_factory=dict)

class IngestURLReq(BaseModel):
    url: str
    chunk_size: int = 1200
    chunk_overlap: int = 150

# ===================== DB Helpers =====================
def get_db_connection():
    if not DB_URL:
        raise HTTPException(500, "DB URL chưa cấu hình (env URL hoặc DATABASE_URL).")
    try:
        return psycopg2.connect(DB_URL)
    except Exception as e:
        raise HTTPException(500, f"Không kết nối được DB: {e}")

def _vec_to_pg(v: List[float]) -> str:
    return "[" + ",".join(f"{float(x):.6f}" for x in v) + "]"

# ===================== Embedding Service =====================
def embed_via_colab(texts: List[str]) -> List[List[float]]:
    if not COLAB_EMBEDDING_URL:
        raise HTTPException(500, "COLAB_EMBEDDING_URL chưa cấu hình")
    
    url = f"{COLAB_EMBEDDING_URL}/embed"
    headers = {"Content-Type": "application/json"}
    
    # ✅ Fixed: Simplified payload handling
    payload = {"texts": texts}
    
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()  # Raise exception for HTTP errors
        
        data = r.json()
        vecs = data.get("embeddings") or data.get("vectors") or data.get("data")
        
        # Handle response format variations
        if isinstance(vecs, list) and vecs and isinstance(vecs[0], dict) and "embedding" in vecs[0]:
            vecs = [item["embedding"] for item in vecs]
            
        if not isinstance(vecs, list):
            raise ValueError(f"Invalid response format: {data}")
            
        if len(vecs) != len(texts):
            if len(texts) > 1 and len(vecs) == 1:
                vecs = vecs * len(texts)
            else:
                raise ValueError(f"Vector count ({len(vecs)}) != text count ({len(texts)})")
                
        # Validate dimensions
        for i, v in enumerate(vecs):
            if not isinstance(v, list) or len(v) != EMBED_DIM:
                raise ValueError(f"Wrong vector dimension at {i}: {len(v)} != {EMBED_DIM}")
                
        return vecs
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(500, f"Embedding service connection error: {e}")
    except Exception as e:
        raise HTTPException(500, f"Embedding service error: {e}")

def check_colab_health() -> str:
    if not COLAB_EMBEDDING_URL:
        return "not_configured"
    try:
        r = requests.get(f"{COLAB_EMBEDDING_URL}/health", timeout=10)
        return "connected" if r.status_code == 200 else "error"
    except Exception:
        return "disconnected"


# ===================== Intent Detection Helper =====================
def detect_intent_and_extract_info(message: str) -> Tuple[str, Optional[dict]]:
    """
    Phát hiện ý định và trích xuất thông tin từ tin nhắn
    
    Returns:
        (intent, extracted_info)
        - intent: 'greeting', 'contractor_request', 'general_chat'
        - extracted_info: {'budget': str, 'project_type': str, 'location': str}
    """
    message_lower = message.lower()
    
    # 1. Phát hiện lời chào
    greetings = ['hello', 'hi', 'chào', 'xin chào', 'hey', 'chào bạn']
    if any(g in message_lower for g in greetings) and len(message_lower.split()) <= 3:
        return ('greeting', None)
    
    # 2. Phát hiện yêu cầu tư vấn nhà thầu
    contractor_keywords = [
        'nhà thầu', 'thi công', 'xây dựng', 'công trình', 
        'tư vấn', 'báo giá', 'ngân sách', 'gợi ý'
    ]
    
    has_contractor_intent = any(kw in message_lower for kw in contractor_keywords)
    
    if not has_contractor_intent:
        return ('general_chat', None)
    
    # 3. Trích xuất thông tin (ngân sách, loại công trình, địa điểm)
    extracted = {}
    
    # Trích xuất ngân sách
    budget_patterns = [
        r'(\d+)\s*tỷ',  # "3 tỷ", "2tỷ"
        r'dưới\s*(\d+)\s*tỷ',  # "dưới 2 tỷ"
        r'khoảng\s*(\d+)\s*tỷ',  # "khoảng 3 tỷ"
        r'(\d+)\s*-\s*(\d+)\s*tỷ',  # "2-3 tỷ"
    ]
    
    for pattern in budget_patterns:
        match = re.search(pattern, message_lower)
        if match:
            extracted['budget'] = match.group(0)
            break
    
    # Trích xuất loại công trình
    project_types = {
        'nhà phố': ['nhà phố', 'townhouse'],
        'biệt thự': ['biệt thự', 'villa'],
        'chung cư': ['chung cư', 'apartment', 'căn hộ'],
        'văn phòng': ['văn phòng', 'office'],
        'nhà xưởng': ['nhà xưởng', 'factory', 'xưởng'],
        'khách sạn': ['khách sạn', 'hotel'],
        'nhà hàng': ['nhà hàng', 'restaurant'],
    }
    
    for ptype, keywords in project_types.items():
        if any(kw in message_lower for kw in keywords):
            extracted['project_type'] = ptype
            break
    
    # Trích xuất địa điểm
    locations = ['đà nẵng', 'da nang', 'hà nội', 'hanoi', 'sài gòn', 'saigon', 'hồ chí minh']
    for loc in locations:
        if loc in message_lower:
            extracted['location'] = loc
            break
    
    # 4. Kiểm tra đủ thông tin
    has_enough_info = 'budget' in extracted and 'project_type' in extracted
    
    if has_enough_info:
        return ('contractor_request', extracted)
    else:
        return ('incomplete_request', extracted)

# ===================== Intent Detection (Optimized) =====================
def detect_intent(msg: str) -> Tuple[ChatIntent, dict]:
    raw = msg or ""
    msg_lower = raw.lower().strip()
    msg_nod = _strip_accents(msg_lower)

    # 1) Greeting
    if len(msg_lower.split()) <= 4 and any(g in msg_lower for g in GREETINGS):
        return (ChatIntent.GREETING, {})

    # 2) Intent nhà thầu
    has_contractor_intent = any(
        _contains_phrase(msg_lower, p) or _contains_phrase(msg_nod, _strip_accents(p))
        for p in CONTRACTOR_PHRASES
    )
    if not has_contractor_intent:
        return (ChatIntent.GENERAL, {})

    # 3) Trích xuất info
    info = {}

    # Budget
    for pat in BUDGET_PATTERNS:
        m = pat.search(msg_lower)
        if m:
            info['budget'] = m.group(0)
            break

    # Project type
    for ptype, kws in PROJECT_TYPES.items():
        if any(_contains_phrase(msg_lower, k) or _contains_phrase(msg_nod, _strip_accents(k))
               for k in kws):
            info['project_type'] = ptype
            break

    # Location (phủ các biến thể có/không dấu)
    for loc in [r"đà nẵng", r"da nang", r"hà nội", r"hanoi",
                r"hồ chí minh", r"ho chi minh", r"sài gòn", r"saigon"]:
        if _contains_phrase(msg_lower, loc) or _contains_phrase(msg_nod, _strip_accents(loc)):
            info['location'] = loc
            break

    # 4) Quyết định intent
    if 'budget' in info and 'project_type' in info:
        return (ChatIntent.CONTRACTOR_FULL, info)
    elif 'budget' in info or 'project_type' in info:
        return (ChatIntent.CONTRACTOR_PARTIAL, info)
    else:
        return (ChatIntent.GENERAL, {})

# ===================== Response Generators =====================
def generate_greeting() -> str:
    """Simple greeting response"""
    return "Xin chào! Tôi có thể tư vấn nhà thầu xây dựng cho bạn. Bạn cần loại công trình nào và ngân sách bao nhiêu?"

def generate_missing_info(info: dict) -> str:
    """Ask for missing information"""
    missing = []
    if 'budget' not in info:
        missing.append("ngân sách")
    if 'project_type' not in info:
        missing.append("loại công trình")
    
    return f"Để gợi ý chính xác, cho tôi biết thêm {' và '.join(missing)} nhé!\n\nVí dụ: 'Tìm nhà thầu xây nhà phố 3 tỷ'"

def generate_no_contractors(info: dict) -> str:
    """No contractors found"""
    return f"""Chưa tìm thấy nhà thầu phù hợp với:
• Loại: {info.get('project_type', '?')}
• Ngân sách: {info.get('budget', '?')}

Thử điều chỉnh điều kiện hoặc liên hệ hotline nhé!"""


# ===================== Improved Contractor Extraction =====================

    """Extract contractor information với limit có thể điều chỉnh"""
    contractors = []
    seen_ids = set()  # Tránh trùng lặp
    
    for chunk in chunks:
        content = chunk.get('content', '')
        
        # Parse table format
        if '|' in content and any(uuid in content for uuid in ['0fa72a73', '8c7628fd', 'fd268472']):
            parts = [p.strip() for p in content.split('|') if p.strip()]
            
            if len(parts) >= 9:
                try:
                    contractor_id = parts[1]
                    
                    # Skip nếu đã có
                    if contractor_id in seen_ids:
                        continue
                    seen_ids.add(contractor_id)
                    
                    name = parts[2]
                    code = parts[3]
                    description = parts[4]
                    specialty = parts[5]
                    budget = parts[6]
                    location = parts[7]
                    rating_str = parts[8]
                    
                    rating = float(rating_str) if rating_str.replace('.', '').replace(',', '').isdigit() else 4.0
                    
                    FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
                    
                    contractor = ContractorAction(
                        contractor_id=contractor_id,
                        contractor_name=name,
                        contractor_slug=code,
                        description=description,
                        budget_range=budget,
                        rating=rating,
                        location=location,
                        profile_url=f"{FRONTEND_URL}/contractors/{contractor_id}",
                        contact_url=f"{FRONTEND_URL}/contractors/{contractor_id}?action=contact"
                    )
                    contractors.append(contractor)
                    
                    # Dừng khi đủ số lượng
                    if len(contractors) >= limit:
                        break
                    
                except (IndexError, ValueError) as e:
                    logger.error(f"Error parsing contractor: {e}")
                    continue
    
    return contractors



def extract_contractor_info(chunks: List[dict]) -> List[ContractorAction]:
    """Extract contractor information với UUID và URLs"""
    contractors = []
    
    for chunk in chunks:
        content = chunk.get('content', '')
        
        # Parse table format: | STT | UUID | Tên | Code | Description | Lĩnh vực | Ngân sách | Khu vực | Đánh giá |
        if '|' in content and any(uuid in content for uuid in ['0fa72a73', '8c7628fd', 'fd268472']):
            parts = [p.strip() for p in content.split('|') if p.strip()]
            
            if len(parts) >= 9:
                try:
                    stt = parts[0]
                    contractor_id = parts[1]  # UUID from backend
                    name = parts[2]
                    code = parts[3]
                    description = parts[4]
                    specialty = parts[5]
                    budget = parts[6]
                    location = parts[7]
                    rating_str = parts[8]
                    
                    # Parse rating
                    rating = float(rating_str) if rating_str.replace('.', '').replace(',', '').isdigit() else 4.0
                    
                    # Frontend URLs
                    FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
                    
                    contractor = ContractorAction(
                        contractor_id=contractor_id,
                        contractor_name=name,
                        contractor_slug=code,  # Use code as slug
                        description=description,
                        budget_range=budget,
                        rating=rating,
                        specialties=[specialty],
                        location=location,
                        profile_url=f"{FRONTEND_URL}/contractors/{contractor_id}",
                        contact_url=f"{FRONTEND_URL}/contractors/{contractor_id}?action=contact"
                    )
                    contractors.append(contractor)
                    
                except (IndexError, ValueError) as e:
                    logger.error(f"Error parsing contractor: {e}")
                    continue
    
    return contractors[:3] 

def simple_keyword_search(query: str, k: int = 5) -> List[dict]:
    """Simple keyword search fallback"""
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    try:
        # Basic keyword matching
        keywords = []
        query_lower = query.lower()
        
        # Budget keywords
        if "3 tỷ" in query_lower or "3000000000" in query_lower:
            keywords.extend(["1500000000", "3000000000", "4000000000", "6000000000"])
        if "2 tỷ" in query_lower:
            keywords.extend(["1000000000", "1500000000", "2000000000"])
        if "dưới 2 tỷ" in query_lower:
            keywords.extend(["1000000000", "1200000000", "1500000000"])
            
        # Company names
        companies = ["DNG", "SBS", "Group 4N", "CDC", "Tín An"]
        
        # Location
        if "đà nẵng" in query_lower or "da nang" in query_lower:
            keywords.append("Da Nang")
            
        # Build search conditions
        conditions = []
        params = []
        
        for keyword in keywords:
            conditions.append("content ILIKE %s")
            params.append(f"%{keyword}%")
            
        for company in companies:
            conditions.append("content ILIKE %s") 
            params.append(f"%{company}%")
            
        # If no specific keywords, search all
        if not conditions:
            conditions = ["content IS NOT NULL"]
            
        where_clause = " OR ".join(conditions)
        sql = f"""
            SELECT id, content, metadata, 0.8 as score
            FROM document_chunks 
            WHERE {where_clause}
            ORDER BY id
            LIMIT %s
        """
        params.append(k)
        
        cur.execute(sql, params)
        rows = cur.fetchall()
        
        return [
            {
                "id": r["id"],
                "content": r["content"],
                "metadata": r["metadata"],
                "score": float(r["score"])
            }
            for r in rows
        ]
        
    except Exception as e:
        logger.error(f"Keyword search error: {e}")
        return []
    finally:
        cur.close()
        conn.close()

# ===================== Startup =====================
@app.on_event("startup")
def on_startup():
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        try:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        except Exception as e:
            # Không dừng app nếu thiếu quyền; log warning
            print(f"[startup] WARNING: cannot create extension vector: {e}")
        cur.execute(f"""
        CREATE TABLE IF NOT EXISTS document_chunks (
            id BIGSERIAL PRIMARY KEY,
            content TEXT NOT NULL,
            embedding VECTOR({EMBED_DIM}) NOT NULL,
            metadata JSONB DEFAULT '{{}}'::jsonb
        );
        """)
        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_chunks_embedding_ivfflat
        ON document_chunks
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_metadata ON document_chunks USING GIN (metadata);")
        conn.commit()
    finally:
        conn.close()

# ===================== Health & Root =====================
@app.get("/")
def root():
    return {"message": f"{APP_TITLE} is running!", "version": APP_VERSION, "docs": "/docs"}

@app.get("/health")
def health_check():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM document_chunks;")
        doc_count = cur.fetchone()[0]
        cur.close()
        conn.close()
        db_status = "connected"
    except Exception as e:
        doc_count = 0
        db_status = f"error: {e}"
    return {
        "status": "healthy" if db_status == "connected" else "degraded",
        "database": db_status,
        "document_count": doc_count,
        "colab_embedding_service": check_colab_health(),
        "gemini_configured": GEMINI_MODEL is not None,
        "embed_dim": EMBED_DIM,
        "timestamp": datetime.now().isoformat()
    }

# ===================== Core Endpoints =====================
@app.post("/add_docs")
def add_docs(req: AddDocsRequest):
    if not req.docs:
        raise HTTPException(400, "docs rỗng")
    texts = [d.content for d in req.docs]
    vecs = embed_via_colab(texts)
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        ids = []
        for d, v in zip(req.docs, vecs):
            v_str = _vec_to_pg(v)
            cur.execute("""
                INSERT INTO document_chunks (content, embedding, metadata)
                VALUES (%s, %s::vector, %s)
                RETURNING id
            """, (d.content, v_str, json.dumps(d.metadata)))
            ids.append(cur.fetchone()[0])
        conn.commit()
        return {"added": len(ids), "ids": ids}
    except Exception as e:
        conn.rollback()
        raise HTTPException(500, str(e))
    finally:
        cur.close()
        conn.close()

@app.post("/store-chunk")
def store_chunk(request: StoreChunkRequest):
    if len(request.embedding) != EMBED_DIM:
        raise HTTPException(400, f"Embedding phải có {EMBED_DIM} chiều.")
    v_str = _vec_to_pg(request.embedding)
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO document_chunks (content, embedding, metadata)
            VALUES (%s, %s::vector, %s)
            RETURNING id
        """, (request.content, v_str, json.dumps(request.metadata)))
        cid = cur.fetchone()[0]
        conn.commit()
        return {"message": "Chunk stored successfully", "id": cid}
    except Exception as e:
        conn.rollback()
        raise HTTPException(500, str(e))
    finally:
        cur.close()
        conn.close()

@app.post("/search-similar")
def search_similar(req: QueryRequest):
    q_vec = embed_via_colab([req.query])[0]
    v_str = _vec_to_pg(q_vec)
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cur.execute("""
            SELECT id, content, metadata, 1 - (embedding <=> %s::vector) AS score
            FROM document_chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (v_str, v_str, req.k))
        rows = cur.fetchall()
        return {"query": req.query, "results": [
            {"id": r["id"], "content": r["content"], "metadata": r["metadata"], "score": float(r["score"])}
            for r in rows
        ]}
    finally:
        cur.close()
        conn.close()

@app.post("/rag")
def rag(req: QueryRequest):
    try:
        res = search_similar(req)
        hits = res["results"]
        ctx = "\n\n---\n\n".join(f"[{i+1}] {h['content']}" for i, h in enumerate(hits)) or "N/A"
        if not GEMINI_MODEL:
            raise HTTPException(500, "Gemini API chưa cấu hình")
        prompt = f"""Bạn là trợ lý AI chuyên về xây dựng tại Việt Nam.

Câu hỏi: {req.query}

Ngữ cảnh:
{ctx}

Yêu cầu:
- Trả lời dựa trên ngữ cảnh. Nếu thiếu, nói rõ là chưa đủ dữ liệu.
- Dùng tiếng Việt, ngắn gọn.
- Trích dẫn [1], [2]... ứng với thứ tự ngữ cảnh khi có thể.
""".strip()
        try:
            resp = GEMINI_MODEL.generate_content(prompt)
            answer = getattr(resp, "text", None) or (resp.candidates[0].content.parts[0].text if getattr(resp, "candidates", None) else "")
            if not answer:
                answer = "Xin lỗi, tôi chưa thể tạo câu trả lời từ ngữ cảnh hiện có."
        except Exception as e:
            raise HTTPException(500, f"Lỗi model: {e}")
        return {"answer": answer, "retrieved": hits}
    except Exception as e:
        raise HTTPException(500, f"RAG error: {e}")

# ===== Document Upload & Processing =====
class DocumentUploadRequest(BaseModel):
    source: str  # URL hoặc file path
    chunk_size: int = 500
    chunk_overlap: int = 50

@app.post("/upload_document")
async def upload_document(req: DocumentUploadRequest):
    """Upload và parse document từ URL bằng IBM Docling"""
    try:
        
        try:
            from docling.document_converter import DocumentConverter
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            import time
        except ImportError:
            raise HTTPException(500, "Docling chưa được cài đặt. Cần cài docling và langchain-text-splitters")

        # Convert document
        converter = DocumentConverter()
        start_time = time.time()
        result = converter.convert(req.source)
        end_time = time.time()

        # Export to markdown
        markdown_content = result.document.export_to_markdown()

        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=req.chunk_size,
            chunk_overlap=req.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_text(markdown_content)

        # Create docs for embedding
        docs = []
        for i, chunk in enumerate(chunks):
            docs.append({
                "content": chunk.strip(),
                "metadata": {
                    "type": "document",
                    "source": req.source,
                    "chunk_index": i,
                    "processing_time": end_time - start_time,
                    "timestamp": datetime.now().isoformat()
                }
            })

        # Embed và store chunks
        if docs:
            texts = [d["content"] for d in docs]
            vecs = embed_via_colab(texts)

            conn = get_db_connection()
            cur = conn.cursor()
            try:
                ids = []
                for d, v in zip(docs, vecs):
                    v_str = _vec_to_pg(v)
                    cur.execute(
                        """
                        INSERT INTO document_chunks (content, embedding, metadata)
                        VALUES (%s, %s::vector, %s)
                        RETURNING id
                        """,
                        (d["content"], v_str, json.dumps(d["metadata"]))
                    )
                    ids.append(cur.fetchone()[0])
                conn.commit()

                return {
                    "message": "Document processed successfully",
                    "source": req.source,
                    "processing_time": end_time - start_time,
                    "chunks_created": len(chunks),
                    "chunks_stored": len(ids),
                    "chunk_ids": ids
                }
            except Exception as e:
                conn.rollback()
                raise HTTPException(500, f"Database error: {str(e)}")
            finally:
                cur.close()
                conn.close()
        else:
            return {"message": "No content to process", "source": req.source}

    except Exception as e:
        raise HTTPException(500, f"Document processing error: {str(e)}")

@app.post("/ingest/url")
def ingest_url(req: IngestURLReq):
    # Fix: Pass URL directly to Docling, don't download first
    conv = DocumentConverter()
    try:
        result = conv.convert(req.url)  # Pass URL directly
        md = result.document.export_to_markdown()
    except Exception as e:
        raise HTTPException(500, f"Document conversion failed: {e}")

    # Rest of code stays same
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=req.chunk_size, chunk_overlap=req.chunk_overlap,
        separators=["\n\n","\n"," ",""]
    )
    chunks = splitter.split_text(md)

    vecs = embed_via_colab(chunks)
    conn = get_db_connection(); cur = conn.cursor()
    try:
        for c,v in zip(chunks, vecs):
            cur.execute("""
              INSERT INTO document_chunks(content, embedding, metadata)
              VALUES (%s, %s::vector, %s)
            """, (c, _vec_to_pg(v), json.dumps({"source": req.url})))
        conn.commit()
        return {"ok": True, "chunks": len(chunks)}
    finally:
        cur.close(); conn.close()

# ===================== Chat (RAG wrapper) =====================
@app.post("/chat", response_model=EnhancedChatResponse)
def chat_endpoint(req: ChatRequest):
    """Optimized chat with intent routing"""
    try:
        # Step 1: Fast intent detection
        intent, info = detect_intent(req.message)
        
        # Step 2: Route based on intent (early returns)
        if intent == ChatIntent.GREETING:
            return EnhancedChatResponse(
                response=generate_greeting(),
                sources=[],
                contractors=[],
                has_recommendations=False
            )
        
        if intent == ChatIntent.CONTRACTOR_PARTIAL:
            return EnhancedChatResponse(
                response=generate_missing_info(info),
                sources=[],
                contractors=[],
                has_recommendations=False
            )
        
        # Step 3: Search only when needed
        if intent == ChatIntent.CONTRACTOR_FULL:
            return handle_contractor_request(req, info)
        
        # Step 4: General chat fallback
        return handle_general_chat(req)
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return EnhancedChatResponse(
            response="Có lỗi xảy ra, thử lại sau nhé!",
            contractors=[],
            has_recommendations=False
        )

@app.post("/debug/search-detailed")
def debug_search_detailed(req: QueryRequest):
    """Debug search with detailed scoring"""
    q_vec = embed_via_colab([req.query])[0]
    v_str = _vec_to_pg(q_vec)
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cur.execute("""
            SELECT id, content, metadata, 
                   1 - (embedding <=> %s::vector) AS score,
                   embedding <=> %s::vector AS distance
            FROM document_chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (v_str, v_str, v_str, req.k))
        rows = cur.fetchall()
        return {
            "query": req.query, 
            "results": [
                {
                    "id": r["id"], 
                    "content": r["content"][:200] + "..." if len(r["content"]) > 200 else r["content"],
                    "metadata": r["metadata"], 
                    "score": float(r["score"]),
                    "distance": float(r["distance"])
                }
                for r in rows
            ]
        }
    finally:
        cur.close()
        conn.close()    


# ===================== Handlers (Separated for clarity) =====================
def handle_contractor_request(req: ChatRequest, info: dict) -> EnhancedChatResponse:
    """Handle contractor search with full info"""
    # Search with higher limit
    hits = simple_keyword_search(req.message, req.top_k * 2)
    contractors = extract_contractor_info_improved(hits, limit=min(req.top_k, 5))
    
    if not contractors:
        return EnhancedChatResponse(
            response=generate_no_contractors(info),
            sources=[],
            contractors=[],
            has_recommendations=False
        )
    
    # Generate AI response
    ctx = "\n".join([f"[{i+1}] {h['content'][:200]}" for i, h in enumerate(hits[:3])])
    prompt = f"""Giới thiệu ngắn gọn (2 câu) {len(contractors)} nhà thầu phù hợp với:
• Loại: {info.get('project_type')}
• Ngân sách: {info.get('budget')}

Kết thúc: "Xem chi tiết bên dưới!"

Danh sách: {', '.join(c.contractor_name for c in contractors)}
Chi tiết: {ctx[:500]}"""
    
    # AI generation (with timeout protection)
    answer = f"Gợi ý {len(contractors)} nhà thầu phù hợp. Xem chi tiết bên dưới!"
    if GEMINI_MODEL:
        try:
            out = GEMINI_MODEL.generate_content(prompt)
            answer = out.text or answer
        except Exception as e:
            logger.warning(f"Gemini fallback: {e}")
    
    return EnhancedChatResponse(
        response=answer,
        sources=[{"id": int(h["id"]), "score": float(h.get("score", 0))} for h in hits[:5]],
        contractors=contractors,
        has_recommendations=True
    )

def handle_general_chat(req: ChatRequest) -> EnhancedChatResponse:
    """Handle general questions with RAG"""
    hits = simple_keyword_search(req.message, 3)
    
    if not hits:
        return EnhancedChatResponse(
            response="Tôi chưa tìm thấy thông tin liên quan. Bạn có thể hỏi về nhà thầu xây dựng không?",
            sources=[],
            contractors=[],
            has_recommendations=False
        )
    
    ctx = "\n".join([h['content'][:150] for h in hits])
    prompt = f"Trả lời ngắn:\n\nCâu hỏi: {req.message}\n\nThông tin: {ctx}"
    
    answer = "Tôi cần thêm thông tin để trả lời."
    if GEMINI_MODEL:
        try:
            out = GEMINI_MODEL.generate_content(prompt)
            answer = out.text or answer
        except:
            pass
    
    return EnhancedChatResponse(
        response=answer,
        sources=[{"id": int(h["id"]), "score": 0} for h in hits],
        contractors=[],
        has_recommendations=False
    )

# ===================== Optimized Contractor Extraction =====================
def extract_contractor_info_improved(chunks: List[dict], limit: int = 5) -> List[ContractorAction]:
    """Fast contractor extraction with early exit"""
    contractors = []
    seen = set()
    frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")
    
    for chunk in chunks:
        if len(contractors) >= limit:
            break
            
        content = chunk.get('content', '')
        if '|' not in content:
            continue
        
        parts = [p.strip() for p in content.split('|') if p.strip()]
        if len(parts) < 9:
            continue
        
        contractor_id = parts[1]
        if contractor_id in seen:
            continue
        seen.add(contractor_id)
        
        try:
            contractors.append(ContractorAction(
                contractor_id=contractor_id,
                contractor_name=parts[2],
                contractor_slug=parts[3],
                description=parts[4],
                budget_range=parts[6],
                rating=float(parts[8]) if parts[8].replace('.','').isdigit() else 4.0,
                location=parts[7],
                profile_url=f"{frontend_url}/contractors/{contractor_id}",
                contact_url=f"{frontend_url}/contractors/{contractor_id}?action=contact"
            ))
        except (IndexError, ValueError):
            continue
    
    return contractors

# ===================== SSE Streaming Chat =====================
def _build_rag_prompt(question: str, hits: list) -> str:
    ctx_blocks = []
    for i, h in enumerate(hits, 1):
        md = h.get("metadata") or {}
        src = md.get("source") or md.get("url") or md.get("path")
        snippet = h.get("content") or ""
        if len(snippet) > 1200:
            snippet = snippet[:1200] + "..."
        ctx_blocks.append(f"[{i}] (source: {src or 'N/A'})\n{snippet}")
    context = "\n\n".join(ctx_blocks) if ctx_blocks else "—"
    return f"""Bạn là trợ lý kỹ thuật xây dựng, trả lời ngắn gọn, bám sát ngữ cảnh.
Nếu thiếu thông tin thì nói 'mình không chắc từ tài liệu'.
Cuối câu trả lời nên liệt kê nguồn theo dạng [1], [2] tương ứng đoạn đã dùng.

[CNTX]
{context}

[CÂU HỎI]
{question}
"""

@app.get("/chat/stream")
def chat_stream(q: str, k: int = 5):
    """Server-Sent Events: stream câu trả lời LLM theo thời gian thực"""
    if not GEMINI_MODEL:
        raise HTTPException(500, "GEMINI_API_KEY chưa cấu hình")

    # 1) Embed + retrieve
    q_vec = embed_via_colab([q])[0]
    v_str = _vec_to_pg(q_vec)
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cur.execute("""
            SELECT id, content, metadata, 1 - (embedding <=> %s::vector) AS score
            FROM document_chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (v_str, v_str, k))
        hits = cur.fetchall()
    finally:
        cur.close()
        conn.close()

    prompt = _build_rag_prompt(q, hits)

    def event_gen():
        # Gửi nguồn trước
        try:
            import json as _json
            yield "event: sources\n"
            yield "data: " + _json.dumps([
                {"id": int(h["id"]), "score": float(h["score"]), "source": (h.get("metadata") or {}).get("source")}
                for h in hits
            ]) + "\n\n"
        except Exception:
            # Không chặn stream nếu lỗi serialize nguồn
            pass

        # 2) Stream từ Gemini
        try:
            for chunk in GEMINI_MODEL.generate_content(prompt, stream=True):
                txt = getattr(chunk, "text", "") or ""
                if txt:
                    # Mỗi message là một 'data: ...\n\n' theo SSE
                    yield "data: " + txt + "\n\n"
        except Exception as e:
            yield f"event: error\ndata: {str(e)}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")