#!/bin/bash

echo "🚀 Setup and Run Inference Test"
echo "================================"

# Kiểm tra xem có HF_TOKEN không
if [ -z "$HF_TOKEN" ]; then
    echo "❌ HF_TOKEN not found in environment variables"
    echo "Please set your HuggingFace token:"
    echo "export HF_TOKEN=your_token_here"
    echo ""
    echo "Or create a .env file with:"
    echo "HF_TOKEN=your_token_here"
    exit 1
fi

echo "✅ HF_TOKEN found"

# Kiểm tra xem có base model local không
if [ ! -d "models/weights/base_model" ]; then
    echo "📥 Base model not found locally. Downloading..."
    echo "This may take a while..."
    
    # Chạy script download base model
    python download_base_model.py
    
    if [ $? -ne 0 ]; then
        echo "❌ Failed to download base model"
        exit 1
    fi
else
    echo "✅ Base model found locally"
fi

# Build Docker image
echo ""
echo "🔨 Building Docker image..."
docker build -f Dockerfile.inference -t chatbot-inference .

if [ $? -eq 0 ]; then
    echo "✅ Docker image built successfully"
else
    echo "❌ Failed to build Docker image"
    exit 1
fi

# Run inference test
echo ""
echo "🧪 Running inference test..."
docker run --rm \
    --gpus all \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/logs:/app/logs \
    -e HF_TOKEN=$HF_TOKEN \
    chatbot-inference

echo ""
echo "✅ Inference test completed!" 