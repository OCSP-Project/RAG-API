import requests
import logging
import json
import base64
from io import BytesIO
from typing import Optional, List, Dict
from app.config.settings import settings

logger = logging.getLogger("rag-api")

class GeminiService:
    """Service for interacting with Gemini API"""
    
    def __init__(self):
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize Gemini model"""
        if settings.GEMINI_API_KEY and settings.GEMINI_API_KEY != "your-gemini-key-here":
            try:
                import google.generativeai as genai
                genai.configure(api_key=settings.GEMINI_API_KEY)
                self.model = genai.GenerativeModel("gemini-1.5-flash")
                logger.info("Gemini model initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini model: {e}")
                self.model = None
        else:
            logger.warning("GEMINI_API_KEY not configured")
    
    def generate_response(self, prompt: str) -> Optional[str]:
        """Generate response using Gemini API"""
        if not self.model:
            return None
        
        try:
            response = self.model.generate_content(prompt)
            answer = getattr(response, "text", None) or (
                response.candidates[0].content.parts[0].text 
                if getattr(response, "candidates", None) else ""
            )
            return answer
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            return None
    
    def generate_streaming_response(self, prompt: str):
        """Generate streaming response using Gemini API"""
        if not self.model:
            return None
        
        try:
            return self.model.generate_content(prompt, stream=True)
        except Exception as e:
            logger.error(f"Gemini streaming error: {e}")
            return None
    
    def is_configured(self) -> bool:
        """Check if Gemini is properly configured"""
        return self.model is not None

    async def analyze_incident_images(
        self,
        images_b64: List[str],
        incident_report: str = "",
        context: str = ""
    ) -> Dict[str, str]:
        """
        Phân tích ảnh sự cố xây dựng và trả về báo cáo + đề xuất

        Args:
            images_b64: Danh sách ảnh dạng Base64
            incident_report: Báo cáo sự cố từ người dùng
            context: Ngữ cảnh bổ sung

        Returns:
            Dict với keys: incident_report, recommendations
        """
        if not self.model:
            return {
                "incident_report": "Hệ thống AI chưa được cấu hình.",
                "recommendations": "Vui lòng liên hệ quản trị viên để cấu hình Gemini API."
            }

        try:
            # Import PIL tại đây để tránh lỗi nếu chưa cài
            from PIL import Image

            # Chuẩn bị prompt chuyên gia
            prompt = f"""
Vai trò: Bạn là Chuyên gia Giám sát Xây dựng với 20 năm kinh nghiệm, chuyên về phát hiện và xử lý sự cố thi công.

Thông tin hiện có:
- Báo cáo sơ bộ: {incident_report if incident_report else "Chưa có"}
- Ngữ cảnh: {context if context else "Chưa có"}

Nhiệm vụ: Phân tích kỹ lưỡng các hình ảnh hiện trường và tạo báo cáo chuyên nghiệp.

YÊU CẦU ĐẦU RA (JSON Format):
{{
    "incident_report": "Phân tích chi tiết hiện trạng từ ảnh. Mô tả rõ: (1) Hạng mục đang thi công, (2) Các sự cố phát hiện (vết nứt, thấm, lún, sai kỹ thuật, mất an toàn...), (3) Mức độ nghiêm trọng, (4) Vị trí cụ thể.",
    "recommendations": "Đề xuất giải pháp xử lý cụ thể, chi tiết: (1) Biện pháp khắc phục ngay (nếu có sự cố), (2) Quy trình thi công đúng cần áp dụng, (3) Vật liệu/công cụ cần thiết, (4) Tiêu chuẩn kỹ thuật liên quan, (5) Biện pháp phòng ngừa cho giai đoạn tiếp theo."
}}

LƯU Ý:
- Sử dụng thuật ngữ xây dựng chính xác (Tiếng Việt)
- Văn phong khách quan, chuyên nghiệp
- Nếu không phát hiện sự cố, ghi nhận công trình đạt tiêu chuẩn và đưa ra khuyến nghị duy trì chất lượng
- QUAN TRỌNG: Chỉ trả về JSON thuần túy, không thêm markdown, code block hay text nào khác
"""

            # Chuẩn bị input content
            input_content = [prompt]

            # Xử lý từng ảnh Base64
            for b64_str in images_b64:
                try:
                    # Xử lý cắt header nếu có (data:image/jpeg;base64,...)
                    if "base64," in b64_str:
                        b64_str = b64_str.split("base64,")[1]

                    # Decode base64 to image
                    image_data = base64.b64decode(b64_str)
                    image = Image.open(BytesIO(image_data))

                    # Convert to RGB nếu cần (RGBA -> RGB)
                    if image.mode != 'RGB':
                        image = image.convert('RGB')

                    input_content.append(image)

                except Exception as img_err:
                    logger.error(f"Lỗi xử lý ảnh: {img_err}")
                    continue

            # Gọi Gemini API với multimodal input
            logger.info(f"Sending {len(input_content)-1} images to Gemini for analysis...")
            response = self.model.generate_content(input_content)

            # Parse JSON response
            response_text = response.text.strip()

            # Xử lý trường hợp Gemini trả về markdown code block
            if response_text.startswith("```json"):
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif response_text.startswith("```"):
                response_text = response_text.split("```")[1].split("```")[0].strip()

            result = json.loads(response_text)

            logger.info("Incident analysis completed successfully")
            return {
                "incident_report": result.get("incident_report", "Không có thông tin"),
                "recommendations": result.get("recommendations", "Không có đề xuất")
            }

        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}. Raw response: {response_text if 'response_text' in locals() else 'N/A'}")
            return {
                "incident_report": "Lỗi xử lý phản hồi từ AI. Vui lòng thử lại.",
                "recommendations": f"Chi tiết kỹ thuật: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Gemini incident analysis error: {str(e)}")
            return {
                "incident_report": "Hệ thống AI gặp lỗi khi phân tích ảnh.",
                "recommendations": f"Chi tiết lỗi: {str(e)}. Vui lòng kiểm tra lại ảnh hoặc thử lại sau."
            }

# Global instance
gemini_service = GeminiService()
