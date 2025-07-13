#!/usr/bin/env python3
"""
📥 Download Base Model Script

Tải base model meta-llama/Llama-3.2-1B-Instruct về local
"""

import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

def download_base_model():
    """Download base model to local directory"""
    
    # Tạo thư mục cho base model
    base_model_dir = "models/weights/base_model"
    os.makedirs(base_model_dir, exist_ok=True)
    
    print("🔐 Checking HuggingFace authentication...")
    
    # Kiểm tra token
    token = os.getenv("HF_TOKEN")
    if not token:
        print("❌ HF_TOKEN not found in environment variables")
        print("Please set your HuggingFace token:")
        print("export HF_TOKEN=your_token_here")
        print("Or create a .env file with: HF_TOKEN=your_token_here")
        return False
    
    try:
        # Đăng nhập HuggingFace
        login(token=token)
        print("✅ HuggingFace authentication successful")
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return False
    
    print(f"\n📥 Downloading base model to {base_model_dir}...")
    
    try:
        # Tải tokenizer
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.2-1B-Instruct",
            trust_remote_code=True,
            cache_dir=base_model_dir
        )
        tokenizer.save_pretrained(base_model_dir)
        print("✅ Tokenizer downloaded")
        
        # Tải model
        print("Downloading model (this may take a while)...")
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-1B-Instruct",
            trust_remote_code=True,
            cache_dir=base_model_dir,
            torch_dtype="auto"
        )
        model.save_pretrained(base_model_dir)
        print("✅ Model downloaded")
        
        print(f"\n🎉 Base model successfully downloaded to {base_model_dir}")
        print("You can now run the inference test!")
        
        return True
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

def main():
    """Main function"""
    print("🚀 Download Base Model Script")
    print("="*50)
    
    if download_base_model():
        print("\n✅ Setup completed successfully!")
        print("Now you can run: python test_chatbot_inference_local.py")
    else:
        print("\n❌ Setup failed. Please check the errors above.")

if __name__ == "__main__":
    main() 