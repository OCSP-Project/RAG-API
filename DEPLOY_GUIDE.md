# RAG API Deployment Guide for EC2

## 📋 Những gì đã thêm mới

### 1. Contractor Embedding Service
- **File:** `app/services/contractor_service.py`
- **Chức năng:** Embed contractors từ backend vào RAG database
- **Dependencies:** Sử dụng các thư viện đã có trong requirements.txt

### 2. API Endpoint mới
- **Endpoint:** `POST /api/v1/embed/contractors`
- **File:** `app/api/endpoints.py` (đã cập nhật)
- **Chức năng:** Nhận danh sách contractors và embedding vào vector database

## 🚀 Cách Deploy lên EC2

### Bước 1: Kiểm tra code trước khi push

```bash
# Vào thư mục RAG API
cd "D:\Ky 9\do_an_tot_nghiep\full\rag-api\RAG-API"

# Kiểm tra git status
git status

# Xem những file đã thay đổi
git diff
```

### Bước 2: Commit và Push code

```bash
# Add files mới và đã sửa
git add app/services/contractor_service.py
git add app/api/endpoints.py
git add DEPLOY_GUIDE.md

# Commit với message rõ ràng
git commit -m "feat: Add contractor embedding service for RAG

- Add ContractorEmbeddingService to format and embed contractors
- Add POST /api/v1/embed/contractors endpoint
- Support bulk embedding from admin panel
- Format contractor data as structured text for better search"

# Push lên repository
git push origin main
```

### Bước 3: Deploy trên EC2

```bash
# SSH vào EC2 instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Vào thư mục project
cd /path/to/rag-api

# Pull code mới
git pull origin main

# Restart service (nếu dùng systemd)
sudo systemctl restart rag-api

# Hoặc nếu dùng Docker
docker-compose down
docker-compose up -d --build

# Hoặc nếu dùng PM2/uvicorn trực tiếp
pm2 restart rag-api
```

### Bước 4: Verify deployment

```bash
# Kiểm tra health endpoint
curl http://your-ec2-ip:8000/health

# Kiểm tra API docs
curl http://your-ec2-ip:8000/docs

# Test embed endpoint
curl -X POST http://your-ec2-ip:8000/api/v1/embed/contractors \
  -H "Content-Type: application/json" \
  -d '{
    "contractors": [
      {
        "contractor_id": "test-id",
        "contractor_name": "Test Contractor",
        "contractor_slug": "test-contractor",
        "description": "Test description",
        "specialties": [],
        "budget_range": "1-5 tỷ",
        "location": "Hà Nội",
        "rating": 4.5,
        "years_of_experience": 5,
        "team_size": 10,
        "is_verified": true
      }
    ],
    "chunk_size": 500,
    "chunk_overlap": 50
  }'
```

## 📝 Environment Variables cần có trên EC2

Đảm bảo file `.env` trên EC2 có các biến sau:

```env
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/dbname

# Gemini API
GEMINI_API_KEY=your-gemini-api-key-here
EMBED_DIM=768

# Frontend (cho contractor URLs)
FRONTEND_URL=http://your-frontend-domain.com
```

## 🔍 Troubleshooting

### Lỗi: "Module not found: contractor_service"
```bash
# Kiểm tra file có tồn tại
ls -la app/services/contractor_service.py

# Nếu thiếu, pull lại code
git pull origin main
```

### Lỗi: "Database connection failed"
```bash
# Kiểm tra PostgreSQL đang chạy
sudo systemctl status postgresql

# Test connection
psql -U user -d dbname -c "SELECT 1"
```

### Lỗi: "GEMINI_API_KEY not configured"
```bash
# Kiểm tra .env file
cat .env | grep GEMINI_API_KEY

# Nếu thiếu, thêm vào
echo "GEMINI_API_KEY=your-key-here" >> .env
```

### Lỗi: Import error
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Hoặc nếu dùng Docker
docker-compose build --no-cache
```

## ✅ Checklist trước khi deploy

- [ ] Code đã được test local (nếu có môi trường)
- [ ] Không có syntax errors trong Python files
- [ ] Git commit message rõ ràng
- [ ] Dependencies trong requirements.txt đầy đủ
- [ ] Environment variables đã được cấu hình trên EC2
- [ ] Database connection string đúng
- [ ] GEMINI_API_KEY hợp lệ
- [ ] Đã backup database trước khi deploy (nếu cần)

## 🎯 Kiểm tra sau khi deploy

1. **Health check:** `GET /health` → Status "healthy"
2. **API docs:** `GET /docs` → Thấy endpoint `/api/v1/embed/contractors`
3. **Test embedding:** Call endpoint với sample data
4. **Check logs:** Không có errors trong application logs
5. **Frontend test:** Admin panel có thể gọi endpoint thành công

## 📚 Files đã thay đổi

1. ✅ `app/services/contractor_service.py` - File mới
2. ✅ `app/api/endpoints.py` - Đã cập nhật (thêm import và endpoint)
3. ✅ `DEPLOY_GUIDE.md` - File hướng dẫn này

## 🔗 Related Documentation

- FastAPI: https://fastapi.tiangolo.com/
- Gemini API: https://ai.google.dev/docs
- PostgreSQL: https://www.postgresql.org/docs/
