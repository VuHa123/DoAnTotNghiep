#!/usr/bin/env python3
"""
📥 Download Alternative Model Script

Tải model tương tự Llama-3.2-1B-Instruct từ các nguồn miễn phí
"""

import os
from transformers import AutoTokenizer, AutoModelForCausalLM

def download_alternative_model():
    """Download alternative model that's similar to Llama-3.2-1B-Instruct"""
    
    # Tạo thư mục cho base model
    base_model_dir = "models/weights/base_model"
    os.makedirs(base_model_dir, exist_ok=True)
    
    print("🔍 Looking for alternative models...")
    
    # Danh sách các model tương tự có thể dùng
    alternative_models = [
        "microsoft/DialoGPT-medium",  # Model chat nhỏ, không cần token
        "gpt2",  # Model GPT-2 cơ bản
        "distilgpt2",  # Phiên bản nhỏ hơn của GPT-2
    ]
    
    for model_name in alternative_models:
        print(f"\n📥 Trying to download {model_name}...")
        
        try:
            # Tải tokenizer
            print(f"Downloading tokenizer for {model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir=base_model_dir
            )
            
            # Thêm pad_token nếu cần
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            tokenizer.save_pretrained(base_model_dir)
            print(f"✅ Tokenizer for {model_name} downloaded")
            
            # Tải model
            print(f"Downloading model {model_name} (this may take a while)...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir=base_model_dir,
                torch_dtype="auto"
            )
            model.save_pretrained(base_model_dir)
            print(f"✅ Model {model_name} downloaded successfully!")
            
            print(f"\n🎉 Alternative model {model_name} downloaded to {base_model_dir}")
            print("Note: This is a different model from the original Llama-3.2-1B-Instruct")
            print("The adapter might not work perfectly, but you can test the structure.")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to download {model_name}: {e}")
            continue
    
    print("\n❌ All alternative models failed to download")
    return False

def create_model_info():
    """Tạo file thông tin về model đã tải"""
    info_file = "models/weights/base_model/model_info.txt"
    with open(info_file, 'w') as f:
        f.write("Model Information:\n")
        f.write("==================\n")
        f.write("This is an alternative model downloaded for testing purposes.\n")
        f.write("Original checkpoint was trained on meta-llama/Llama-3.2-1B-Instruct\n")
        f.write("This alternative model may not work perfectly with the LoRA adapter.\n")
        f.write("It's mainly for testing the inference pipeline structure.\n")

def main():
    """Main function"""
    print("🚀 Download Alternative Model Script")
    print("="*50)
    
    if download_alternative_model():
        create_model_info()
        print("\n✅ Setup completed successfully!")
        print("Now you can test the inference pipeline structure.")
        print("Note: For full functionality, you'll need the original Llama-3.2-1B-Instruct model.")
    else:
        print("\n❌ Setup failed. Please check your internet connection.")

if __name__ == "__main__":
    main() 