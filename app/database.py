import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

def get_db_connection():
    database_url = os.getenv("DATABASE_URL")
    conn = psycopg2.pool.SimpleConnectionPool(1, 10, database_url)
    return conn

def init_db():
    """Initialize database with required tables"""
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Enable pgvector extension
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
    # Create table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS document_chunks (
            id SERIAL PRIMARY KEY,
            content TEXT NOT NULL,
            embedding vector(768),
            metadata JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    # Create indexes
    cur.execute("""
        CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx 
        ON document_chunks USING ivfflat (embedding vector_cosine_ops) 
        WITH (lists = 100);
    """)
    
    cur.execute("""
        CREATE INDEX IF NOT EXISTS document_chunks_metadata_idx 
        ON document_chunks USING GIN (metadata);
    """)
    
    conn.commit()
    cur.close()
    conn.close()
    print("Database initialized successfully!")
