#!/usr/bin/env python3
"""
Script setup để cài đặt ExLlama và dependencies cho GPTQ inference
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(command: str, cwd: str = None) -> bool:
    """
    Chạy command và trả về True nếu thành công
    
    Args:
        command: Command để chạy
        cwd: Working directory
        
    Returns:
        True nếu thành công, False nếu thất bại
    """
    try:
        print(f"🔄 Running: {command}")
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"✅ Success: {command}")
            return True
        else:
            print(f"❌ Failed: {command}")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def check_git_installed() -> bool:
    """Kiểm tra git đã được cài đặt chưa"""
    return shutil.which("git") is not None

def check_cuda_available() -> bool:
    """Kiểm tra CUDA có available không"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

def install_exllama():
    """Cài đặt ExLlama"""
    print("📦 Installing ExLlama...")
    
    # Kiểm tra git
    if not check_git_installed():
        print("❌ Git chưa được cài đặt. Vui lòng cài đặt git trước.")
        return False
    
    # Clone ExLlama repository
    exllama_path = Path(__file__).parent.parent / "exllama"
    
    if exllama_path.exists():
        print(f"⚠️  ExLlama đã tồn tại tại: {exllama_path}")
        response = input("Bạn có muốn cài đặt lại không? (y/N): ")
        if response.lower() != 'y':
            print("✅ Sử dụng ExLlama hiện có")
            return True
        else:
            shutil.rmtree(exllama_path)
    
    # Clone repository
    if not run_command(f"git clone https://github.com/turboderp/exllama.git {exllama_path}"):
        return False
    
    print("✅ ExLlama đã được cài đặt thành công!")
    return True

def install_dependencies():
    """Cài đặt các dependencies cần thiết"""
    print("📦 Installing dependencies...")
    
    # Danh sách packages cần cài đặt
    packages = [
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "auto-gptq>=0.5.0",
        "accelerate>=0.24.0",
        "peft>=0.7.0",
        "bitsandbytes>=0.41.0",
        "sentencepiece>=0.1.99",
        "protobuf>=3.20.0",
        "ninja>=1.10.0"
    ]
    
    for package in packages:
        if not run_command(f"pip install {package}"):
            print(f"❌ Không thể cài đặt {package}")
            return False
    
    print("✅ Dependencies đã được cài đặt thành công!")
    return True

def check_installation():
    """Kiểm tra installation"""
    print("🔍 Checking installation...")
    
    # Kiểm tra CUDA
    if check_cuda_available():
        print("✅ CUDA available")
    else:
        print("⚠️  CUDA không available - có thể ảnh hưởng đến performance")
    
    # Kiểm tra ExLlama
    exllama_path = Path(__file__).parent.parent / "exllama"
    if exllama_path.exists():
        print("✅ ExLlama repository found")
    else:
        print("❌ ExLlama repository not found")
        return False
    
    # Kiểm tra imports
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    
    try:
        import transformers
        print(f"✅ Transformers version: {transformers.__version__}")
    except ImportError:
        print("❌ Transformers not installed")
        return False
    
    try:
        import auto_gptq
        print("✅ Auto-GPTQ installed")
    except ImportError:
        print("❌ Auto-GPTQ not installed")
        return False
    
    # Test ExLlama import
    sys.path.append(str(exllama_path))
    try:
        from exllama.model import ExLlama, ExLlamaConfig
        print("✅ ExLlama imports working")
    except ImportError as e:
        print(f"❌ ExLlama imports failed: {e}")
        return False
    
    print("✅ Installation check completed successfully!")
    return True

def create_example_scripts():
    """Tạo các script example"""
    print("📝 Creating example scripts...")
    
    # Tạo script convert example
    convert_example = '''#!/usr/bin/env python3
# Example: Convert LoRA adapter to GPTQ
python scripts/convert_to_gptq.py \\
    --base_model "meta-llama/Llama-3.2-1B-Instruct" \\
    --lora_path "models/weights/chatbot_finetuned_nf4" \\
    --output_path "models/weights/chatbot_gptq" \\
    --bits 4 \\
    --group_size 128
'''
    
    # Tạo script inference example
    inference_example = '''#!/usr/bin/env python3
# Example: Run inference with ExLlama
python scripts/exllama_inference.py \\
    --model_path "models/weights/chatbot_gptq" \\
    --max_seq_len 2048 \\
    --temperature 0.7 \\
    --test

# Interactive mode
python scripts/exllama_inference.py \\
    --model_path "models/weights/chatbot_gptq" \\
    --interactive
'''
    
    # Lưu examples
    with open("scripts/convert_example.sh", "w") as f:
        f.write(convert_example)
    
    with open("scripts/inference_example.sh", "w") as f:
        f.write(inference_example)
    
    # Make executable
    os.chmod("scripts/convert_example.sh", 0o755)
    os.chmod("scripts/inference_example.sh", 0o755)
    
    print("✅ Example scripts created!")

def main():
    """Main function"""
    print("🚀 Setting up ExLlama GPTQ inference environment...")
    print("=" * 60)
    
    # 1. Cài đặt dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        return False
    
    # 2. Cài đặt ExLlama
    if not install_exllama():
        print("❌ Failed to install ExLlama")
        return False
    
    # 3. Kiểm tra installation
    if not check_installation():
        print("❌ Installation check failed")
        return False
    
    # 4. Tạo example scripts
    create_example_scripts()
    
    print("\n" + "=" * 60)
    print("✅ Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Convert your fine-tuned model to GPTQ format:")
    print("   bash scripts/convert_example.sh")
    print("\n2. Run inference with ExLlama:")
    print("   bash scripts/inference_example.sh")
    print("\n3. For interactive chat:")
    print("   python scripts/exllama_inference.py --model_path models/weights/chatbot_gptq --interactive")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 