#!/bin/bash

echo "🚀 Building and running inference test in Docker..."

# Kiểm tra xem có GPU không
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected"
    echo "GPU Info:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
else
    echo "⚠️  No NVIDIA GPU detected. Inference will run on CPU (slower)"
fi

# Build Docker image
echo "🔨 Building Docker image..."
docker build -f Dockerfile.inference -t chatbot-inference .

if [ $? -eq 0 ]; then
    echo "✅ Docker image built successfully"
else
    echo "❌ Failed to build Docker image"
    exit 1
fi

# Run inference test
echo "🧪 Running inference test..."
docker run --rm \
    --gpus all \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/logs:/app/logs \
    chatbot-inference

echo "✅ Inference test completed!" 