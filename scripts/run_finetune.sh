#!/bin/bash

# Script chạy fine-tuning cho Mental Health Chatbot
# Sử dụng QLoRA với model open source

echo "=== Mental Health Chatbot Fine-tuning ==="
echo "Starting fine-tuning process..."

# Kiểm tra GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
else
    echo "No GPU detected, will use CPU (not recommended for fine-tuning)"
fi

# Tạo thư mục output nếu chưa có
mkdir -p models/chatbot_finetuned

# Chạy fine-tuning với các tham số tối ưu
python scripts/finetune_chatbot.py \
    --model_name "microsoft/DialoGPT-medium" \
    --data_path "Dataset/llama_instruction_data.jsonl" \
    --output_dir "models/weights/chatbot_finetuned" \
    --batch_size 4 \
    --gradient_accumulation_steps 8 \
    --epochs 3 \
    --learning_rate 2e-4 \
    --warmup_steps 100 \
    --max_length 512 \
    --save_steps 500 \
    --logging_steps 50 \
    --run_name "mental-health-chatbot-finetune" \
    --use_wandb

echo "Fine-tuning completed!"
echo "Model saved to: models/chatbot_finetuned/final_model" 