# RAG API - Optimized Version (Gemini Integration)

## Cấu trúc thư mục mới (Clean Architecture)

```
app/
├── config/           # Cấu hình ứng dụng
│   ├── __init__.py
│   └── settings.py
├── core/             # Core business logic
│   ├── __init__.py
│   └── database.py
├── models/           # Pydantic models
│   ├── __init__.py
│   └── schemas.py
├── services/         # Business services
│   ├── __init__.py
│   ├── chat_service.py
│   ├── document_service.py
│   ├── embedding_service.py  # Gemini embedding
│   ├── gemini_service.py
│   └── intent_service.py
├── api/              # API endpoints
│   ├── __init__.py
│   └── endpoints.py
├── utils/             # Utilities
│   └── __init__.py
└── main.py           # FastAPI app
```

## Những thay đổi chính

### ✅ Đã tối ưu:

1. **Sử dụng Gemini API cho cả embedding và response**: Không cần external embedding service
2. **Tách code thành modules**: Clean architecture với separation of concerns
3. **Giảm dependencies**: Loại bỏ các package không cần thiết
4. **Tối ưu Docker**: Chỉ cần PostgreSQL và API service
5. **Cải thiện error handling**: Centralized error handling

### 🔧 Các tính năng được giữ lại:

- ✅ Document processing với Docling
- ✅ Chat với intent detection
- ✅ Contractor recommendations
- ✅ Vector search
- ✅ Streaming responses
- ✅ Gemini embedding và response generation

### 🗑️ Đã loại bỏ:

- ❌ External embedding service
- ❌ Complex contractor extraction logic
- ❌ Unused utility functions
- ❌ Redundant API endpoints
- ❌ Embedding service container

## Environment Variables

```bash
# Database
DATABASE_URL=postgresql://postgres:root@localhost:5432/rag_db

# Gemini API (cho cả embedding và response)
GEMINI_API_KEY=your-gemini-api-key

# Configuration
EMBED_DIM=768
FRONTEND_URL=http://localhost:3000
FRONTEND_ORIGIN=http://localhost:3000
```

## Chạy ứng dụng

### Development

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker-compose up -d
```

## API Endpoints

### Core Endpoints

- `POST /api/v1/add_docs` - Thêm documents
- `POST /api/v1/store-chunk` - Lưu chunk với embedding
- `POST /api/v1/search-similar` - Tìm kiếm tương tự
- `POST /api/v1/chat` - Chat endpoint
- `POST /api/v1/upload_document` - Upload document
- `POST /api/v1/ingest/url` - Ingest từ URL

### Utility Endpoints

- `GET /health` - Health check
- `GET /` - Root endpoint
- `GET /chat/stream` - Streaming chat

## Tính năng mới có thể thêm

1. **Authentication & Authorization**
2. **Rate limiting**
3. **Caching layer**
4. **Monitoring & Logging**
5. **Batch processing**
6. **Multi-language support**
