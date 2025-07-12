#!/usr/bin/env python3
"""
Script để cập nhật requirements.txt với dependencies cho ExLlama GPTQ
"""

import os
from pathlib import Path

def update_requirements():
    """Cập nhật requirements.txt với dependencies cho ExLlama GPTQ"""
    
    # Đường dẫn đến requirements.txt
    req_file = Path(__file__).parent.parent / "requirements.txt"
    
    # Dependencies cần thêm cho ExLlama GPTQ
    exllama_deps = [
        "# ExLlama GPTQ dependencies",
        "auto-gptq>=0.5.0",
        "sentencepiece>=0.1.99", 
        "protobuf>=3.20.0",
        "ninja>=1.10.0",
        "exllama>=0.0.1",  # Placeholder - ExLlama được install từ git
    ]
    
    # Đọc requirements.txt hiện tại
    if req_file.exists():
        with open(req_file, 'r', encoding='utf-8') as f:
            current_content = f.read()
        
        # Kiểm tra xem đã có ExLlama dependencies chưa
        if "auto-gptq" not in current_content:
            print("📝 Adding ExLlama GPTQ dependencies to requirements.txt...")
            
            # Thêm dependencies mới
            updated_content = current_content.rstrip() + "\n\n" + "\n".join(exllama_deps)
            
            # Backup file cũ
            backup_file = req_file.with_suffix('.txt.backup')
            with open(backup_file, 'w', encoding='utf-8') as f:
                f.write(current_content)
            print(f"✅ Backup created: {backup_file}")
            
            # Ghi file mới
            with open(req_file, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            
            print("✅ Requirements.txt updated successfully!")
        else:
            print("✅ ExLlama dependencies already present in requirements.txt")
    else:
        print("❌ requirements.txt not found")
        return False
    
    return True

def create_exllama_requirements():
    """Tạo requirements file riêng cho ExLlama"""
    
    exllama_req_file = Path(__file__).parent.parent / "requirements_exllama.txt"
    
    exllama_deps = [
        "# ExLlama GPTQ Requirements",
        "# Install with: pip install -r requirements_exllama.txt",
        "",
        "# Core dependencies",
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "accelerate>=0.24.0",
        "peft>=0.7.0",
        "bitsandbytes>=0.41.0",
        "",
        "# GPTQ dependencies", 
        "auto-gptq>=0.5.0",
        "sentencepiece>=0.1.99",
        "protobuf>=3.20.0",
        "ninja>=1.10.0",
        "",
        "# Additional utilities",
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "requests>=2.31.0",
        "python-dotenv>=1.0.0",
        "",
        "# Note: ExLlama is installed from git repository",
        "# git clone https://github.com/turboderp/exllama.git",
    ]
    
    with open(exllama_req_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(exllama_deps))
    
    print(f"✅ Created {exllama_req_file}")
    return True

def main():
    """Main function"""
    print("📦 Updating requirements for ExLlama GPTQ...")
    
    # Cập nhật requirements.txt chính
    if not update_requirements():
        return False
    
    # Tạo requirements file riêng cho ExLlama
    create_exllama_requirements()
    
    print("\n✅ Requirements update completed!")
    print("\n📋 Next steps:")
    print("1. Install ExLlama dependencies:")
    print("   pip install -r requirements_exllama.txt")
    print("\n2. Setup ExLlama:")
    print("   python scripts/setup_exllama.py")
    print("\n3. Convert your model:")
    print("   python scripts/convert_to_gptq.py --help")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 