#!/usr/bin/env python3
"""
Script để tải model Llama-3.2-1B-Instruct từ Hugging Face
"""

import os
import sys
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load environment variables
load_dotenv("token.env")

def download_llama_model():
    """Tải model Llama-3.2-1B-Instruct"""
    try:
        print("🔄 Downloading Llama-3.2-1B-Instruct model...")
        
        # Model path
        model_name = "vinai/phobert-base"
        output_dir = "/home/aero/DoAnTotNghiep/models/weights/phobert-base/models--vinai--phobert-base"
        
        # Tạo thư mục output
        os.makedirs(output_dir, exist_ok=True)
        
        # Lấy token
        token = os.getenv("HF_TOKEN")
        if not token:
            print("❌ HF_TOKEN not found in environment variables")
            return False
        
        print(f"✅ Using token: {token[:10]}...")
        
        # Tải tokenizer
        print("📥 Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=token,
            trust_remote_code=True
        )
        
        # Tải model
        print("📥 Downloading model (this may take a while)...")
        model = AutoModelForCausalLM.from_pretrained(
                model_name,
                # device_map="auto",
                torch_dtype=torch.float16,
                use_safetensors=True  # <-- thêm dòng này
            )

        
        # Lưu model và tokenizer
        print(f"💾 Saving model to {output_dir}...")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        print("✅ Model downloaded successfully!")
        print(f"📁 Model saved to: {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return False

def check_model_access():
    """Kiểm tra quyền truy cập model"""
    try:
        from huggingface_hub import HfApi
        
        api = HfApi()
        token = os.getenv("HF_TOKEN")
        
        if not token:
            print("❌ HF_TOKEN not found")
            return False
        
        # Kiểm tra token
        user = api.whoami(token=token)
        print(f"✅ Logged in as: {user}")
        
        # Kiểm tra quyền truy cập model
        model_id = "vinai/phobert-base"
        try:
            model_info = api.model_info(model_id, token=token)
            print(f"✅ Access to {model_id} granted")
            return True
        except Exception as e:
            print(f"❌ No access to {model_id}: {e}")
            print("💡 Please visit: https://huggingface.co/vinai/phobert-base")
            print("   And click 'Agree and Access' to get permission")
            return False
            
    except Exception as e:
        print(f"❌ Error checking access: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Llama Model Downloader")
    print("=" * 50)
    
    # Kiểm tra quyền truy cập
    print("1. Checking model access...")
    if not check_model_access():
        print("❌ Cannot access model. Please check your token and permissions.")
        sys.exit(1)
    
    # Tải model
    print("\n2. Downloading model...")
    if download_llama_model():
        print("🎉 Model downloaded successfully!")
    else:
        print("❌ Failed to download model")
        sys.exit(1) 