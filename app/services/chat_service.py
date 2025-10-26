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
            
            if intent == ChatIntent.CONTRACTOR_FULL:
                return self._handle_contractor_request(request, info)
            
            # General chat fallback
            return self._handle_general_chat(request)
            
        except Exception as e:
            logger.error(f"Chat processing error: {e}")
            return EnhancedChatResponse(
                response="Có lỗi xảy ra, thử lại sau nhé!",
                contractors=[],
                has_recommendations=False
            )
    
    def _handle_contractor_request(self, request: ChatRequest, info: dict) -> EnhancedChatResponse:
        """Handle contractor search with full info"""
        # Search for relevant chunks
        query_vector = embed_via_gemini([request.message])[0]
        hits = search_similar_vectors(query_vector, request.top_k * 2)
        
        # Extract contractor information
        contractors = extract_contractor_info(hits, limit=min(request.top_k, 5))
        
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
        
        # Generate response
        answer = f"Gợi ý {len(contractors)} nhà thầu phù hợp. Xem chi tiết bên dưới!"
        if self.gemini.is_configured():
            ai_response = self.gemini.generate_response(prompt)
            if ai_response:
                answer = ai_response
        
        return EnhancedChatResponse(
            response=answer,
            sources=[{"id": int(h["id"]), "score": float(h.get("score", 0))} for h in hits[:5]],
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
        
        return EnhancedChatResponse(
            response=answer,
            sources=[{"id": int(h["id"]), "score": 0} for h in hits],
            contractors=[],
            has_recommendations=False
        )
    
    def search_similar(self, query: str, k: int = 5) -> Dict[str, Any]:
        """Search for similar documents"""
        try:
            query_vector = embed_via_api([query])[0]
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
