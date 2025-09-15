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

class QueryRequest(BaseModel):
    query: str
    k: int = 5

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
                cur.close()
                conn.close()
        else:
            return {"message": "No content to process", "source": req.source}

    except Exception as e:
        raise HTTPException(500, f"Document processing error: {str(e)}")

@app.post("/ingest/url")
def ingest_url(req: IngestURLReq):
    # 1) Tải file
    resp = _rq.get(req.url, timeout=60)
    resp.raise_for_status()
    # 2) Docling convert
    conv = DocumentConverter()
    result = conv.convert(io.BytesIO(resp.content))
    md = result.document.export_to_markdown()

    # 3) Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=req.chunk_size, chunk_overlap=req.chunk_overlap,
        separators=["\n\n","\n"," ",""]
    )
    chunks = splitter.split_text(md)

    # 4) Embed + lưu
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
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # ✅ Fixed: Use proper search_similar function
    search_req = QueryRequest(query=req.message, k=req.top_k)
    search_result = search_similar(search_req)
    hits = search_result["results"]

    # Xây prompt ngữ cảnh
    ctx = "\n\n".join(
        f"[{i+1}] {h['content']}" for i,h in enumerate(hits)
    ) or "—"
    prompt = (
        "Bạn là trợ lý tư vấn nhà thầu xây dựng, bạn tư vấn thông tin các nhà thầu có trong hệ thống, chỉ dùng thông tin trong [CNTX]. "
        "Nếu không chắc thì nói 'chưa đủ dữ liệu'. "
        "Luôn ghi nguồn [1],[2]...\n\n"
        f"[CNTX]\n{ctx}\n\n[CÂU HỎI]\n{req.message}"
    )

    if not GEMINI_MODEL:
        raise HTTPException(500, "Gemini API chưa cấu hình")
    
    try:
        out = GEMINI_MODEL.generate_content(prompt)
        answer = out.text
    except Exception as e:
        raise HTTPException(500, f"Lỗi model: {e}")
        
    return ChatResponse(
        response=answer,
        sources=[
            {"id": int(h["id"]), "score": float(h["score"]), "source": (h.get("metadata") or {}).get("source")}
            for h in hits
        ],
    )

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