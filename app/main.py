import os, json
from typing import List, Dict, Optional, Any
import requests
import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import google.generativeai as genai

APP_TITLE = "OCSP RAG API"; APP_VERSION = "1.0.0"

DB_URL = os.getenv("URL") or os.getenv("DATABASE_URL") or ""
COLAB_EMBEDDING_URL = (os.getenv("COLAB_EMBEDDING_URL") or "").rstrip("/")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
EMBED_DIM = 768

if GEMINI_API_KEY and GEMINI_API_KEY != "your-gemini-key-here":
    genai.configure(api_key=GEMINI_API_KEY)
    try:
        GEMINI_MODEL = genai.GenerativeModel("gemini-1.5-flash")
    except Exception:
        GEMINI_MODEL = None
else:
    GEMINI_MODEL = None

app = FastAPI(title=APP_TITLE, version=APP_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

class DocItem(BaseModel):
    content: str
    metadata: Dict[str, Any] = {}

class AddDocsRequest(BaseModel):
    docs: List[DocItem]

class QueryRequest(BaseModel):
    query: str
    k: int = 5

class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    sources: List[Dict] = []

class StoreChunkRequest(BaseModel):
    content: str
    embedding: List[float]
    metadata: Dict[str, Any] = {}

def get_db_connection():
    if not DB_URL:
        raise HTTPException(500, "DB URL chưa cấu hình (env URL hoặc DATABASE_URL).")
    try:
        return psycopg2.connect(DB_URL)
    except Exception as e:
        raise HTTPException(500, f"Không kết nối được DB: {e}")

def _vec_to_pg(v: List[float]) -> str:
    return "[" + ",".join(f"{float(x):.6f}" for x in v) + "]"

def embed_via_colab(texts: List[str]) -> List[List[float]]:
    if not COLAB_EMBEDDING_URL:
        raise HTTPException(500, "COLAB_EMBEDDING_URL chưa cấu hình")
    url = f"{COLAB_EMBEDDING_URL}/embed"
    headers = {"Content-Type": "application/json"}
    tries = [
        {"json": {"texts": texts}},
        {"json": {"text": texts[0] if len(texts)==1 else "\n".join(texts)}},
        {"json": {"inputs": texts}},
    ]
    last_err = None
    for payload in tries:
        try:
            r = requests.post(url, headers=headers, timeout=60, **payload)
            if r.status_code == 200:
                data = r.json()
                vecs = data.get("vectors") or data.get("embeddings") or data.get("data")
                if isinstance(vecs, list) and vecs and isinstance(vecs[0], dict) and "embedding" in vecs[0]:
                    vecs = [item["embedding"] for item in vecs]
                if not isinstance(vecs, list):
                    raise ValueError(f"Phản hồi không hợp lệ: {data}")
                if len(vecs) != len(texts):
                    if len(texts) > 1 and len(vecs) == 1:
                        vecs = vecs * len(texts)
                    else:
                        raise ValueError(f"Độ dài vectors ({len(vecs)}) != texts ({len(texts)})")
                for i, v in enumerate(vecs):
                    if not isinstance(v, list) or len(v) != EMBED_DIM:
                        raise ValueError(f"Chiều vector sai tại {i}")
                return vecs
            else:
                last_err = f"{r.status_code} {r.text[:300]}"
        except Exception as e:
            last_err = str(e)
    raise HTTPException(500, f"Embedding service lỗi/không tương thích payload. Chi tiết: {last_err}")

def check_colab_health() -> str:
    if not COLAB_EMBEDDING_URL:
        return "not_configured"
    try:
        r = requests.get(f"{COLAB_EMBEDDING_URL}/health", timeout=10)
        return "connected" if r.status_code == 200 else "error"
    except Exception:
        return "disconnected"

@app.on_event("startup")
def on_startup():
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
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

@app.get("/")
def root():
    return {"message": f"{APP_TITLE} is running!", "version": APP_VERSION, "docs": "/docs"}

@app.get("/health")
def health_check():
    try:
        conn = get_db_connection(); cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM document_chunks;")
        doc_count = cur.fetchone()[0]
        cur.close(); conn.close()
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
        "timestamp": datetime.now().isoformat()
    }

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
        cur.close(); conn.close()

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
        cur.close(); conn.close()

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
        cur.close(); conn.close()

@app.post("/rag")
def rag(req: QueryRequest):
    res = search_similar(req)
    hits = res["results"]
    ctx = "\n\n---\n\n".join(f"[{i+1}] {h['content']}" for i, h in enumerate(hits)) or "N/A"
    if not GEMINI_MODEL:
        raise HTTPException(500, "Gemini API chưa cấu hình")
    prompt = f"""
Bạn là trợ lý AI chuyên về xây dựng tại Việt Nam.

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


# ===== Document Upload & Processing =====
class DocumentUploadRequest(BaseModel):
    source: str  # URL hoặc file path
    chunk_size: int = 500
    chunk_overlap: int = 50

@app.post("/upload_document")
async def upload_document(req: DocumentUploadRequest):
    """Upload và parse document từ URL bằng IBM Docling"""
    try:
        # Import docling (cần cài trong container)
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
                cur.close(); conn.close()
        else:
            return {
                "message": "No content to process",
                "source": req.source
            }
            
    except Exception as e:
        raise HTTPException(500, f"Document processing error: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    try:
        rag_response = rag(QueryRequest(query=request.message, k=3))
        return ChatResponse(
            response=rag_response["answer"],
            sources=[
                {
                    "type": hit.get("metadata", {}).get("type", "unknown"),
                    "content": (hit["content"][:200] + "...") if hit.get("content") else "",
                    "score": hit.get("score")
                }
                for hit in rag_response.get("retrieved", [])
            ]
        )
    except Exception as e:
        return ChatResponse(
            response=f"Xin lỗi, tôi gặp lỗi khi xử lý câu hỏi: {str(e)}",
            sources=[]
        )
