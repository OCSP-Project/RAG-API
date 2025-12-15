import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Any
import logging
from app.config.settings import settings

logger = logging.getLogger("rag-api")

def get_db_connection():
    """Get database connection"""
    if not settings.DATABASE_URL:
        raise Exception("DB URL chưa cấu hình (env URL hoặc DATABASE_URL).")
    try:
        return psycopg2.connect(settings.DATABASE_URL)
    except Exception as e:
        raise Exception(f"Không kết nối được DB: {e}")

def _vec_to_pg(v: List[float]) -> str:
    """Convert vector to PostgreSQL format"""
    return "[" + ",".join(f"{float(x):.6f}" for x in v) + "]"

def init_database():
    """Initialize database with required tables and extensions"""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        try:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        except Exception as e:
            logger.warning(f"Cannot create extension vector: {e}")
        
        cur.execute(f"""
        CREATE TABLE IF NOT EXISTS document_chunks (
            id BIGSERIAL PRIMARY KEY,
            content TEXT NOT NULL,
            embedding VECTOR({settings.EMBED_DIM}) NOT NULL,
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
        logger.info("Database initialized successfully")
    finally:
        conn.close()

def store_chunks(chunks_data: List[Dict[str, Any]]) -> List[int]:
    """Store multiple chunks in database"""
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        ids = []
        for chunk_data in chunks_data:
            v_str = _vec_to_pg(chunk_data['embedding'])
            cur.execute("""
                INSERT INTO document_chunks (content, embedding, metadata)
                VALUES (%s, %s::vector, %s)
                RETURNING id
            """, (chunk_data['content'], v_str, chunk_data['metadata']))
            ids.append(cur.fetchone()[0])
        conn.commit()
        return ids
    except Exception as e:
        conn.rollback()
        raise Exception(f"Database error: {str(e)}")
    finally:
        cur.close()
        conn.close()

def search_similar_vectors(query_vector: List[float], k: int = 5) -> List[Dict[str, Any]]:
    """Search for similar vectors in database"""
    v_str = _vec_to_pg(query_vector)
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cur.execute("""
            SELECT id, content, metadata, 1 - (embedding <=> %s::vector) AS score
            FROM document_chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (v_str, v_str, k))
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
    finally:
        cur.close()
        conn.close()

def get_document_count() -> int:
    """Get total document count"""
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT COUNT(*) FROM document_chunks;")
        return cur.fetchone()[0]
    except Exception:
        return 0
    finally:
        cur.close()
        conn.close()

def delete_all_chunks() -> int:
    """Delete all chunks from database"""
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("DELETE FROM document_chunks;")
        deleted_count = cur.rowcount
        conn.commit()
        logger.info(f"Deleted {deleted_count} chunks from database")
        return deleted_count
    except Exception as e:
        conn.rollback()
        raise Exception(f"Failed to delete chunks: {str(e)}")
    finally:
        cur.close()
        conn.close()

def delete_chunks_by_type(chunk_type: str) -> int:
    """Delete chunks by metadata type (e.g., 'contractor', 'document')"""
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            DELETE FROM document_chunks
            WHERE metadata->>'type' = %s;
        """, (chunk_type,))
        deleted_count = cur.rowcount
        conn.commit()
        logger.info(f"Deleted {deleted_count} chunks with type '{chunk_type}'")
        return deleted_count
    except Exception as e:
        conn.rollback()
        raise Exception(f"Failed to delete chunks by type: {str(e)}")
    finally:
        cur.close()
        conn.close()
