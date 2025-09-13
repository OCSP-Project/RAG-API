FROM python:3.11-slim

# Tối ưu runtime
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Cài CA certificates để gọi HTTPS (ngrok/Gemini) an toàn
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cài deps
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

EXPOSE 8000

# Chạy uvicorn (không --reload trong prod)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
