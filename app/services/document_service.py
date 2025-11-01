import logging
from typing import List, Dict, Any
from docling.document_converter import DocumentConverter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.services.embedding_service import embed_via_gemini
from app.core.database import store_chunks
from datetime import datetime
import json

logger = logging.getLogger("rag-api")

class DocumentProcessor:
    """Service for processing documents using Docling"""
    
    def __init__(self):
        self.converter = DocumentConverter()
    
    def process_url(self, url: str, chunk_size: int = 500, chunk_overlap: int = 50) -> Dict[str, Any]:
        """Process document from URL"""
        try:
            # Convert document
            result = self.converter.convert(url)
            
            # Export to markdown
            markdown_content = result.document.export_to_markdown()
            
            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            chunks = text_splitter.split_text(markdown_content)
            
            # Prepare chunks for embedding
            chunks_data = []
            for i, chunk in enumerate(chunks):
                chunks_data.append({
                    "content": chunk.strip(),
                    "metadata": {
                        "type": "document",
                        "source": url,
                        "chunk_index": i,
                        "timestamp": datetime.now().isoformat()
                    }
                })
            
            # Get embeddings
            texts = [chunk["content"] for chunk in chunks_data]
            embeddings = embed_via_gemini(texts)
            
            # Add embeddings to chunks
            for i, embedding in enumerate(embeddings):
                chunks_data[i]["embedding"] = embedding
                chunks_data[i]["metadata"] = json.dumps(chunks_data[i]["metadata"])
            
            # Store in database
            chunk_ids = store_chunks(chunks_data)
            
            return {
                "message": "Document processed successfully",
                "source": url,
                "chunks_created": len(chunks),
                "chunks_stored": len(chunk_ids),
                "chunk_ids": chunk_ids
            }
            
        except Exception as e:
            logger.error(f"Document processing error: {e}")
            raise Exception(f"Document processing failed: {e}")
    
    def process_document(self, source: str, chunk_size: int = 500, chunk_overlap: int = 50) -> Dict[str, Any]:
        """Process document from source (URL or file path)"""
        try:
            # Convert document
            result = self.converter.convert(source)
            
            # Export to markdown
            markdown_content = result.document.export_to_markdown()
            
            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            chunks = text_splitter.split_text(markdown_content)
            
            # Prepare chunks for embedding
            chunks_data = []
            for i, chunk in enumerate(chunks):
                chunks_data.append({
                    "content": chunk.strip(),
                    "metadata": {
                        "type": "document",
                        "source": source,
                        "chunk_index": i,
                        "timestamp": datetime.now().isoformat()
                    }
                })
            
            # Get embeddings
            texts = [chunk["content"] for chunk in chunks_data]
            embeddings = embed_via_gemini(texts)
            
            # Add embeddings to chunks
            for i, embedding in enumerate(embeddings):
                chunks_data[i]["embedding"] = embedding
                chunks_data[i]["metadata"] = json.dumps(chunks_data[i]["metadata"])
            
            # Store in database
            chunk_ids = store_chunks(chunks_data)
            
            return {
                "message": "Document processed successfully",
                "source": source,
                "chunks_created": len(chunks),
                "chunks_stored": len(chunk_ids),
                "chunk_ids": chunk_ids
            }
            
        except Exception as e:
            logger.error(f"Document processing error: {e}")
            raise Exception(f"Document processing failed: {e}")

# Global instance
document_processor = DocumentProcessor()
