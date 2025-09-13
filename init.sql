CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS document_chunks (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(768),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx 
ON document_chunks USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);

CREATE INDEX IF NOT EXISTS document_chunks_metadata_idx 
ON document_chunks USING GIN (metadata);
