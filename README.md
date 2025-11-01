# RAG API - Hệ thống Chatbot Tư vấn Nhà thầu Xây dựng

## 📋 Tổng quan

RAG API là hệ thống backend sử dụng RAG (Retrieval-Augmented Generation) để tư vấn và gợi ý nhà thầu xây dựng dựa trên ngân sách và loại công trình. Hệ thống được xây dựng với FastAPI, PostgreSQL với pgvector extension, và Google Gemini AI.

## 🏗️ Cấu trúc Dự án

```
RAG-API/
├── app/                          # Thư mục ứng dụng chính
│   ├── __init__.py              # Module khởi tạo
│   ├── main.py                  # Entry point chính của ứng dụng FastAPI
│   ├── main_new.py              # Phiên bản mới của main.py (backup)
│   ├── main_old.py              # Phiên bản cũ của main.py (backup)
│   ├── main.py.backup           # File backup
│   ├── database.py              # Module database (legacy)
│   │
│   ├── api/                     # API Endpoints
│   │   ├── __init__.py          # Router chính cho /api/v1
│   │   └── endpoints.py         # Định nghĩa các endpoint API
│   │
│   ├── config/                  # Cấu hình
│   │   ├── __init__.py
│   │   └── settings.py          # Load và quản lý biến môi trường
│   │
│   ├── core/                    # Core functionality
│   │   ├── __init__.py
│   │   └── database.py          # Database connection, queries với pgvector
│   │
│   ├── models/                  # Data Models
│   │   ├── __init__.py
│   │   └── schemas.py           # Pydantic schemas cho request/response
│   │
│   ├── services/                # Business Logic Services
│   │   ├── __init__.py
│   │   ├── chat_service.py      # Xử lý chat với intent detection
│   │   ├── document_service.py  # Xử lý document (Docling)
│   │   ├── embedding_service.py # Tạo embeddings qua Gemini
│   │   ├── gemini_service.py   # Tích hợp Google Gemini API
│   │   └── intent_service.py   # Phát hiện intent và extract thông tin
│   │
│   └── utils/                   # Utilities
│       └── __init__.py
│
├── docker-compose.yml           # Docker Compose config
├── Dockerfile                   # Docker image build config
├── init.sql                     # Database initialization script
├── requirements.txt             # Python dependencies
└── deploy.sh                   # Deployment script
```

---

**Biến cấu hình**:

- `APP_TITLE`: Tên ứng dụng (default: "OCSP RAG API")
- `APP_VERSION`: Phiên bản (default: "1.0.0")
- `DATABASE_URL`: PostgreSQL connection string
- `GEMINI_API_KEY`: API key cho Google Gemini
- `EMBED_DIM`: Chiều embedding (default: 768)
- `FRONTEND_URL`: URL frontend
- `FRONTEND_ORIGIN`: Origin cho CORS
- `CORS_ALLOW_CREDENTIALS`: CORS credentials flag

**CORS configuration**:

- Nếu `FRONTEND_ORIGIN` trống → cho phép tất cả origins (`*`)
- Nếu có `FRONTEND_ORIGIN` → chỉ cho phép origin đó

---

### 3. `app/core/database.py`
