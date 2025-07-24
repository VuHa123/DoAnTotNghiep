#!/bin/bash

# Cấu hình
APP_MODULE="main:app"
PORT=8000
CLOUDFLARED_PATH="cloudflared"  # Đường dẫn tới file cloudflared

# Bước 1: Khởi động uvicorn trong nền
echo "Khởi động server trên cổng $PORT..."
uvicorn $APP_MODULE --host 0.0.0.0 --port $PORT &

# Bước 2: Đợi uvicorn chạy (có thể điều chỉnh delay nếu cần)
sleep 3

# Bước 3: Mở Cloudflare Tunnel
echo "Đang public thông qua Cloudflare Tunnel..."
$CLOUDFLARED_PATH tunnel --url http://localhost:$PORT --no-autoupdate
