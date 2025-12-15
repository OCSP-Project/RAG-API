import logging
from typing import List, Dict, Any
from datetime import datetime
import json
from app.services.embedding_service import embed_via_gemini
from app.core.database import store_chunks

logger = logging.getLogger("rag-api")

class ContractorEmbeddingService:
    """Service for embedding contractor data"""

    def embed_contractors(
        self,
        contractors: List[Dict[str, Any]],
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ) -> Dict[str, Any]:
        """
        Embed contractor data into the RAG database

        Args:
            contractors: List of contractor dictionaries with fields:
                - contractor_id: str
                - contractor_name: str
                - contractor_slug: str
                - description: str
                - specialties: List[str]
                - budget_range: str
                - location: str
                - rating: float
                - years_of_experience: int
                - team_size: int
                - is_verified: bool
            chunk_size: Size of text chunks for embedding
            chunk_overlap: Overlap between chunks

        Returns:
            Dictionary with processing results
        """
        try:
            chunks_data = []

            for contractor in contractors:
                # Format contractor data as structured text for embedding
                content = self._format_contractor_text(contractor)

                # Create chunk with metadata
                chunks_data.append({
                    "content": content,
                    "metadata": {
                        "type": "contractor",
                        "contractor_id": contractor.get("contractor_id"),
                        "contractor_name": contractor.get("contractor_name"),
                        "contractor_slug": contractor.get("contractor_slug"),
                        "location": contractor.get("location"),
                        "rating": contractor.get("rating"),
                        "budget_range": contractor.get("budget_range"),
                        "is_verified": contractor.get("is_verified", False),
                        "timestamp": datetime.now().isoformat()
                    }
                })

            # Get embeddings
            logger.info(f"Embedding {len(chunks_data)} contractor entries...")
            texts = [chunk["content"] for chunk in chunks_data]
            embeddings = embed_via_gemini(texts)

            # Add embeddings and serialize metadata
            for i, embedding in enumerate(embeddings):
                chunks_data[i]["embedding"] = embedding
                chunks_data[i]["metadata"] = json.dumps(chunks_data[i]["metadata"])

            # Store in database
            logger.info(f"Storing {len(chunks_data)} chunks in database...")
            chunk_ids = store_chunks(chunks_data)

            return {
                "message": "Contractors embedded successfully",
                "contractors_processed": len(contractors),
                "chunks_created": len(chunks_data),
                "chunks_stored": len(chunk_ids),
                "chunk_ids": chunk_ids
            }

        except Exception as e:
            logger.error(f"Contractor embedding error: {e}")
            raise Exception(f"Failed to embed contractors: {e}")

    def _format_contractor_text(self, contractor: Dict[str, Any]) -> str:
        """
        Format contractor data as structured text for embedding

        Format: ID | Name | Slug | Description | Specialties | Budget | Location | Rating
        """
        # Extract fields with defaults
        contractor_id = contractor.get("contractor_id", "")
        name = contractor.get("contractor_name", "")
        slug = contractor.get("contractor_slug", "")
        description = contractor.get("description", "")
        specialties = ", ".join(contractor.get("specialties", []))
        budget_range = contractor.get("budget_range", "Liên hệ")
        location = contractor.get("location", "")
        rating = contractor.get("rating", 0)
        years_exp = contractor.get("years_of_experience", 0)
        team_size = contractor.get("team_size", 0)
        is_verified = "Đã xác thực" if contractor.get("is_verified", False) else ""

        # Build structured content
        content_parts = [
            f"ID: {contractor_id}",
            f"| {contractor_id}",
            f"| {name}",
            f"| {slug}",
            f"| {description}",
            f"| {specialties}",
            f"| {budget_range}",
            f"| {location}",
            f"| {rating}",
            f"| Kinh nghiệm: {years_exp} năm",
            f"| Quy mô: {team_size} người",
            f"| {is_verified}"
        ]

        # Add natural language description for better search
        natural_desc = f"""
Nhà thầu {name} chuyên về {description or 'xây dựng'}.
Địa điểm: {location}.
Ngân sách: {budget_range}.
Đánh giá: {rating}/5.
Kinh nghiệm: {years_exp} năm.
{is_verified}
"""

        content = " ".join(content_parts) + "\n" + natural_desc
        return content.strip()

# Global instance
contractor_embedding_service = ContractorEmbeddingService()
