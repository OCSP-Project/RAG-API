RAG-API/
├── app/ # Thư mục ứng dụng chính
│ ├── **init**.py # Module khởi tạo
│ ├── main.py # Entry point chính của ứng dụng FastAPI
│ ├── main_new.py # Phiên bản mới của main.py (backup)
│ ├── main_old.py # Phiên bản cũ của main.py (backup)
│ ├── main.py.backup # File backup
│ ├── database.py # Module database (legacy)
│ │
│ ├── api/ # API Endpoints
│ │ ├── **init**.py # Router chính cho /api/v1
│ │ └── endpoints.py # Định nghĩa các endpoint API
│ │
│ ├── config/ # Cấu hình
│ │ ├── **init**.py
│ │ └── settings.py # Load và quản lý biến môi trường
│ │
│ ├── core/ # Core functionality
│ │ ├── **init**.py
│ │ └── database.py # Database connection, queries với pgvector
│ │
│ ├── models/ # Data Models
│ │ ├── **init**.py
│ │ └── schemas.py # Pydantic schemas cho request/response
│ │
│ ├── services/ # Business Logic Services
│ │ ├── **init**.py
│ │ ├── chat_service.py # Xử lý chat với intent detection
│ │ ├── document_service.py # Xử lý document (Docling)
│ │ ├── embedding_service.py # Tạo embeddings qua Gemini
│ │ ├── gemini_service.py # Tích hợp Google Gemini API
│ │ └── intent_service.py # Phát hiện intent và extract thông tin
│ │
│ └── utils/ # Utilities
│ └── **init**.py
│
├── docker-compose.yml # Docker Compose config
├── Dockerfile # Docker image build config
├── init.sql # Database initialization script
├── requirements.txt # Python dependencies
└── deploy.sh # Deployment script

```

```

Bước 1: Định nghĩa Data Model (Schemas)
Tạo cấu trúc dữ liệu input/output cho API phân tích ảnh.

Mở file: app/models/schemas.py (hoặc tạo mới nếu chưa có)

Python

from pydantic import BaseModel
from typing import List, Optional

# Request từ Frontend gửi lên

class ImageAnalysisRequest(BaseModel):
images: List[str] # Danh sách ảnh dạng Base64 string
context: Optional[str] = "" # Ngữ cảnh bổ sung (ví dụ: "Đang đổ sàn tầng 2")

# Response trả về cho Frontend

class ImageAnalysisResponse(BaseModel):
incident_report: str # Nội dung báo cáo sự cố
recommendations: str # Nội dung đề xuất
Bước 2: Viết Logic AI trong Service
Chúng ta sẽ mở rộng GeminiService để thêm khả năng nhìn ảnh (Multimodal).

Mở file: app/services/gemini_service.py

Python

import google.generativeai as genai
import os
import json
import base64
from io import BytesIO
from PIL import Image
from app.config.settings import settings

class GeminiService:
def **init**(self): # Config API Key từ biến môi trường
genai.configure(api_key=settings.GEMINI_API_KEY) # Đảm bảo settings.py đã load biến này

        # Cấu hình Model: Dùng Flash cho nhanh và rẻ
        self.vision_model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config={
                "temperature": 0.4, # Thấp để AI nghiêm túc, ít bịa
                "response_mime_type": "application/json" # Bắt buộc trả về JSON
            }
        )

    def is_configured(self):
        return settings.GEMINI_API_KEY is not None

    async def analyze_images(self, images_b64: list[str], context: str = "") -> dict:
        """
        Phân tích danh sách ảnh xây dựng và trả về báo cáo + đề xuất
        """
        try:
            # 1. Chuẩn bị Prompt chuyên gia
            prompt = f"""
            Vai trò: Bạn là một Chuyên gia Giám sát Xây dựng (Construction Supervisor) với 20 năm kinh nghiệm.

            Ngữ cảnh bổ sung: {context}

            Nhiệm vụ: Xem xét kỹ lưỡng các hình ảnh hiện trường được cung cấp và tạo báo cáo JSON.

            Yêu cầu đầu ra (JSON Format):
            {{
                "incident_report": "Mô tả chi tiết hiện trạng. Nhận diện các hạng mục đang thi công. Nếu thấy vết nứt, thấm, sai kỹ thuật, hoặc vi phạm an toàn lao động (thiếu mũ, giàn giáo yếu...), hãy chỉ rõ.",
                "recommendations": "Đưa ra các biện pháp khắc phục cụ thể cho sự cố (nếu có) hoặc các lưu ý kỹ thuật để đảm bảo chất lượng cho công đoạn tiếp theo."
            }}

            Lưu ý: Sử dụng thuật ngữ chuyên ngành xây dựng Tiếng Việt. Văn phong khách quan, chuyên nghiệp.
            """

            # 2. Xử lý ảnh (Base64 -> PIL Image)
            input_content = [prompt]

            for b64_str in images_b64:
                # Xử lý cắt header nếu có (data:image/jpeg;base64,...)
                if "base64," in b64_str:
                    b64_str = b64_str.split("base64,")[1]

                image_data = base64.b64decode(b64_str)
                image = Image.open(BytesIO(image_data))
                input_content.append(image)

            # 3. Gọi Gemini
            # Lưu ý: generate_content là hàm đồng bộ (sync), nếu muốn async chuẩn cần chạy trong executor
            # hoặc dùng thư viện async của google (hiện tại bản python client chủ yếu là sync wrapper)
            response = self.vision_model.generate_content(input_content)

            # 4. Parse kết quả
            return json.loads(response.text)

        except Exception as e:
            print(f"Lỗi Gemini Vision: {str(e)}")
            # Trả về fallback nếu lỗi để không crash app
            return {
                "incident_report": "Hệ thống AI đang bận hoặc không thể xử lý ảnh này.",
                "recommendations": f"Chi tiết lỗi: {str(e)}"
            }

# Khởi tạo Singleton instance để dùng chung

gemini_service = GeminiService()
Bước 3: Tạo API Endpoint
Tạo endpoint để Frontend gọi vào.

Mở file: app/api/endpoints.py (hoặc tạo file mới app/api/construction.py rồi include vào router chính)

Python

from fastapi import APIRouter, HTTPException
from app.models.schemas import ImageAnalysisRequest, ImageAnalysisResponse
from app.services.gemini_service import gemini_service

router = APIRouter()

@router.post("/analyze-site", response_model=ImageAnalysisResponse)
async def analyze_site_images(request: ImageAnalysisRequest):
"""
API nhận ảnh hiện trường -> Trả về gợi ý nhật ký công trình
"""
if not request.images:
raise HTTPException(status_code=400, detail="Vui lòng cung cấp ít nhất 1 ảnh.")

    if not gemini_service.is_configured():
        raise HTTPException(status_code=500, detail="Chưa cấu hình Gemini API Key.")

    # Gọi service xử lý
    result = await gemini_service.analyze_images(
        images_b64=request.images,
        context=request.context
    )

    return ImageAnalysisResponse(
        incident_report=result.get("incident_report", ""),
        recommendations=result.get("recommendations", "")
    )

Bước 4: Đăng ký Router (Nếu chưa có)
Trong file app/api/**init**.py hoặc nơi bạn gom router:

Python

from fastapi import APIRouter
from app.api import endpoints

api_router = APIRouter()

# Include router vừa tạo

api_router.include_router(endpoints.router, prefix="/v1/construction", tags=["construction"])
Bước 5: Kết nối Frontend (Nhắc lại)
Bây giờ Frontend Next.js của bạn chỉ cần gọi vào endpoint mới này:

URL: http://localhost:8000/api/v1/construction/analyze-site (hoặc path tương ứng bạn config)

Method: POST

Body:

JSON

{
"images": ["base64string_anh_1", "base64string_anh_2"],
"context": "Đổ bê tông sàn tầng 3, trời nắng"
}
⚠️ Một lưu ý nhỏ về Connection Pool (Fix code của bạn)
Code bạn đưa ở đầu:

Python

def get_db_connection():
database_url = os.getenv("DATABASE_URL") # NGUY HIỂM: Dòng này tạo một POOL MỚI mỗi lần gọi hàm!
conn = psycopg2.pool.SimpleConnectionPool(1, 10, database_url)
return conn
Cách này vẫn chưa đúng bản chất của Pool. Nếu hàm get_db_connection được gọi 100 lần, bạn sẽ tạo ra 100 cái Pool, mỗi Pool mở 1-10 connection -> Sập DB ngay lập tức.

Cách sửa đúng (Singleton Pattern): Bạn cần khởi tạo Pool một lần duy nhất ở bên ngoài hàm hoặc trong startup event.

Python

# app/core/database.py

import psycopg2.pool
import os

# Biến toàn cục lưu pool

pg_pool = None

def init_db_pool():
global pg_pool
database_url = os.getenv("DATABASE_URL") # Chỉ tạo nếu chưa có
if pg_pool is None:
pg_pool = psycopg2.pool.ThreadedConnectionPool(1, 20, database_url)
print("Database Pool Created!")

def get_db_connection():
global pg_pool
if pg_pool is None:
init_db_pool() # Lấy 1 connection từ pool có sẵn
return pg_pool.getconn()

def release_db_connection(conn):
global pg_pool
if pg_pool and conn: # Trả connection về pool
pg_pool.putconn(conn)
