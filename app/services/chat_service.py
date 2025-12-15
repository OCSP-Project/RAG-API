import logging
from typing import List, Dict, Any
from app.core.database import search_similar_vectors
from app.services.embedding_service import embed_via_gemini
from app.services.intent_service import (
    detect_intent, extract_contractor_info, generate_greeting, 
    generate_missing_info, generate_no_contractors
)
from app.services.gemini_service import gemini_service
from app.models.schemas import ChatRequest, EnhancedChatResponse, ChatIntent

logger = logging.getLogger("rag-api")

class ChatService:
    """Service for handling chat interactions"""
    
    def __init__(self):
        self.gemini = gemini_service
    
    def process_chat(self, request: ChatRequest) -> EnhancedChatResponse:
        """Process chat request with intent detection and routing"""
        try:
            # Detect intent
            intent, info = detect_intent(request.message)
            
            # Route based on intent
            if intent == ChatIntent.GREETING:
                from app.models.schemas import SourceItem
                return EnhancedChatResponse(
                    response=generate_greeting(),
                    sources=[],
                    contractors=[],
                    has_recommendations=False
                )
            
            if intent == ChatIntent.CONTRACTOR_PARTIAL:
                # Still search and show contractors, but ask for more info
                return self._handle_contractor_request(request, info)
            
            if intent == ChatIntent.CONTRACTOR_FULL:
                return self._handle_contractor_request(request, info)
            
            # General chat fallback
            return self._handle_general_chat(request)
            
        except Exception as e:
            logger.error(f"Chat processing error: {e}")
            return EnhancedChatResponse(
                response="Có lỗi xảy ra, thử lại sau nhé!",
                sources=[],
                contractors=[],
                has_recommendations=False
            )
    
    def _handle_contractor_request(self, request: ChatRequest, info: dict) -> EnhancedChatResponse:
        """Handle contractor search with full info"""
        # Search for relevant chunks (search more to ensure we get enough unique contractors)
        query_vector = embed_via_gemini([request.message])[0]
        search_limit = request.top_k * 5
        logger.info(f"Searching for {search_limit} similar chunks for query: '{request.message}'")
        hits = search_similar_vectors(query_vector, search_limit)  # Search 5x more chunks
        logger.info(f"Vector search returned {len(hits)} chunks")

        # Extract contractor information
        extract_limit = min(request.top_k, 5)
        logger.info(f"Attempting to extract up to {extract_limit} contractors from {len(hits)} chunks")
        contractors = extract_contractor_info(hits, limit=extract_limit)
        
        if not contractors:
            from app.models.schemas import SourceItem
            return EnhancedChatResponse(
                response=generate_no_contractors(info),
                sources=[],
                contractors=[],
                has_recommendations=False
            )
        
        # Generate AI response with contractor details
        contractor_list = "\n".join([
            f"• **{c.contractor_name}** - {c.location} - {c.rating}⭐ - {c.budget_range}"
            for c in contractors
        ])

        # Create contextual prompt based on available info
        criteria = []
        if info.get('project_type'):
            criteria.append(f"Loại công trình: {info['project_type']}")
        if info.get('budget'):
            criteria.append(f"Ngân sách: {info['budget']}")
        if info.get('location'):
            criteria.append(f"Địa điểm: {info['location']}")

        criteria_text = "\n".join(f"• {c}" for c in criteria) if criteria else "Hiển thị tất cả nhà thầu có sẵn"

        prompt = f"""Viết câu giới thiệu ngắn gọn, thân thiện (1-2 câu) cho {len(contractors)} nhà thầu sau:

{contractor_list}

Tiêu chí tìm kiếm:
{criteria_text}

Kết thúc bằng: "Xem chi tiết và liên hệ bên dưới!"
"""

        # Generate response
        answer = f"Tìm thấy {len(contractors)} nhà thầu phù hợp. Xem chi tiết bên dưới!"
        if self.gemini.is_configured():
            ai_response = self.gemini.generate_response(prompt)
            if ai_response:
                answer = ai_response
        
        from app.models.schemas import SourceItem
        return EnhancedChatResponse(
            response=answer,
            sources=[SourceItem(id=int(h["id"]), score=float(h.get("score", 0))) for h in hits[:5]],
            contractors=contractors,
            has_recommendations=True
        )
    
    def _handle_general_chat(self, request: ChatRequest) -> EnhancedChatResponse:
        """Handle general questions with RAG"""
        # Search for relevant chunks
        query_vector = embed_via_gemini([request.message])[0]
        hits = search_similar_vectors(query_vector, 3)
        
        if not hits:
            return EnhancedChatResponse(
                response="Tôi chưa tìm thấy thông tin liên quan. Bạn có thể hỏi về nhà thầu xây dựng không?",
                sources=[],
                contractors=[],
                has_recommendations=False
            )
        
        # Generate response
        ctx = "\n".join([h['content'][:150] for h in hits])
        prompt = f"Trả lời ngắn:\n\nCâu hỏi: {request.message}\n\nThông tin: {ctx}"
        
        answer = "Tôi cần thêm thông tin để trả lời."
        if self.gemini.is_configured():
            ai_response = self.gemini.generate_response(prompt)
            if ai_response:
                answer = ai_response
        
        from app.models.schemas import SourceItem
        return EnhancedChatResponse(
            response=answer,
            sources=[SourceItem(id=int(h["id"]), score=float(h.get("score", 0))) for h in hits],
            contractors=[],
            has_recommendations=False
        )
    
    def search_similar(self, query: str, k: int = 5) -> Dict[str, Any]:
        """Search for similar documents"""
        try:
            query_vector = embed_via_gemini([query])[0]
            hits = search_similar_vectors(query_vector, k)
            
            return {
                "query": query,
                "results": hits
            }
        except Exception as e:
            logger.error(f"Search error: {e}")
            return {"query": query, "results": []}

# Global instance
chat_service = ChatService()
