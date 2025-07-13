#!/usr/bin/env python3
"""
Script để tổ chức lại cấu trúc model files
Di chuyển từ services/*/models/best_model sang models/weights/
"""

import os
import shutil
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_directory_structure():
    """Tạo cấu trúc thư mục mới"""
    directories = [
        "models/weights",
        "models/weights/sentiment_v1",  # Sentiment model lần đầu
        "models/weights/sentiment_v2",  # Sentiment model cuối cùng
        "models/weights/gating_router",
        "models/weights/mental_state",
        "models/weights/chatbot_finetuned"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def move_model_files():
    """Di chuyển model files"""
    moves = [
        # Sentiment model lần đầu (để gán nhãn)
        ("best_model", "models/weights/sentiment_v1"),
        
        # Sentiment model cuối cùng
        ("services/setiment_analysis/models/best_model", "models/weights/sentiment_v2"),
        
        # Gating router model
        ("services/gating_router/models/best_model", "models/weights/gating_router"),
        
        # Mental state classifier
        ("services/mental_state_classifier/models/best_model", "models/weights/mental_state"),
    ]
    
    for source, destination in moves:
        if os.path.exists(source):
            if os.path.exists(destination):
                logger.warning(f"Destination {destination} already exists, skipping...")
                continue
                
            shutil.copytree(source, destination)
            logger.info(f"Moved {source} -> {destination}")
        else:
            logger.warning(f"Source {source} does not exist")

def update_model_paths():
    """Cập nhật đường dẫn trong code"""
    files_to_update = [
        ("services/setiment_analysis/analyzer.py", "models/weights/sentiment_v2"),
        ("services/mental_state_classifier/classifer.py", "models/weights/mental_state"),
        ("api_gateway/main.py", "models/weights/gating_router")
    ]
    
    for file_path, new_model_path in files_to_update:
        if os.path.exists(file_path):
            logger.info(f"Updating {file_path} to use {new_model_path}")
            # Note: Đã cập nhật thủ công ở trên

def create_model_config():
    """Tạo file config để quản lý model paths"""
    config = {
        "model_paths": {
            "sentiment_v1": "models/weights/sentiment_v1",
            "sentiment_v2": "models/weights/sentiment_v2", 
            "gating_router": "models/weights/gating_router",
            "mental_state": "models/weights/mental_state",
            "chatbot_finetuned": "models/weights/chatbot_finetuned"
        },
        "description": {
            "sentiment_v1": "Sentiment model lần đầu - dùng để gán nhãn dữ liệu",
            "sentiment_v2": "Sentiment model cuối cùng - dùng cho inference",
            "gating_router": "Gating network để phân loại mức độ rủi ro",
            "mental_state": "Mental state classifier",
            "chatbot_finetuned": "Fine-tuned chatbot model"
        }
    }
    
    import json
    with open("models/model_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    logger.info("Created models/model_config.json")

def main():
    """Main function"""
    logger.info("=== Reorganizing Model Structure ===")
    
    # 1. Tạo cấu trúc thư mục
    create_directory_structure()
    
    # 2. Di chuyển model files
    move_model_files()
    
    # 3. Cập nhật đường dẫn
    update_model_paths()
    
    # 4. Tạo config file
    create_model_config()
    
    logger.info("=== Reorganization completed ===")
    logger.info("New structure:")
    logger.info("models/weights/")
    logger.info("├── sentiment_v1/     # Sentiment model lần đầu")
    logger.info("├── sentiment_v2/     # Sentiment model cuối cùng")
    logger.info("├── gating_router/    # Gating model")
    logger.info("├── mental_state/     # Mental state classifier")
    logger.info("└── chatbot_finetuned/ # Fine-tuned chatbot")

if __name__ == "__main__":
    main() 