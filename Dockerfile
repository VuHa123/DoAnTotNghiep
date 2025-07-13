# Multi-stage Dockerfile cho Mental Health Chatbot
# Stage 1: Base image với dependencies chung
FROM python:3.11-slim as base

# Thiết lập environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Cài đặt system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Thiết lập working directory
WORKDIR /app

# Copy requirements và cài đặt Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Tạo thư mục cho logs
RUN mkdir -p /app/logs

# Stage 2: Backend API
FROM base as backend

# Expose port cho backend
EXPOSE 8000

# Health check cho backend
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run backend application
CMD ["uvicorn", "api_gateway.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Stage 3: Frontend Gradio
FROM base as frontend

# Expose port cho frontend
EXPOSE 7860

# Run frontend application
CMD ["python", "frontend/app.py"]

# Stage 4: Development (default)
FROM base as development

# Expose ports cho development
EXPOSE 8000 7860

# Run development server
CMD ["python", "run_servers.py"] 