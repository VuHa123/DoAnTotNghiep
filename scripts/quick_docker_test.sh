#!/bin/bash

echo "🔧 Quick Docker Test - Mental Health Chatbot"
echo "=============================================="

# Kiểm tra Docker
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker không chạy. Vui lòng khởi động Docker Desktop"
    exit 1
fi

echo "✅ Docker đang chạy"

# Kiểm tra docker-compose
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose không được cài đặt"
    exit 1
fi

echo "✅ docker-compose đã sẵn sàng"

# Kiểm tra scripts
if [ ! -f "scripts/docker_manage.sh" ]; then
    echo "❌ Script docker_manage.sh không tồn tại"
    exit 1
fi

echo "✅ Scripts đã sẵn sàng"

# Test build
echo ""
echo "🔨 Testing build process..."
./scripts/docker_manage.sh build

if [ $? -eq 0 ]; then
    echo "✅ Build test thành công"
else
    echo "❌ Build test thất bại"
    exit 1
fi

# Test backup
echo ""
echo "💾 Testing backup process..."
./scripts/docker_manage.sh backup

if [ $? -eq 0 ]; then
    echo "✅ Backup test thành công"
else
    echo "❌ Backup test thất bại"
fi

# Show status
echo ""
echo "📊 Current status:"
./scripts/docker_manage.sh status

echo ""
echo "🎉 Quick test completed!"
echo "Bây giờ bạn có thể sử dụng:"
echo "  ./scripts/docker_manage.sh start  # Khởi động services"
echo "  ./scripts/docker_manage.sh stop   # Dừng services"
echo "  ./scripts/docker_manage.sh status # Xem trạng thái" 