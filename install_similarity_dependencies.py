#!/usr/bin/env python3
"""
Script cài đặt dependencies cho tính năng Knowledge Similarity Filtering
"""

import subprocess
import sys
import os

def install_package(package):
    """Cài đặt package sử dụng pip"""
    try:
        print(f"📦 Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False

def check_package(package):
    """Kiểm tra package đã được cài đặt chưa"""
    try:
        __import__(package)
        return True
    except ImportError:
        return False

def main():
    print("🚀 Installing Knowledge Similarity Filtering Dependencies")
    print("=" * 60)
    
    # Danh sách packages cần thiết
    required_packages = [
        "scikit-learn==1.5.2",
        "qdrant-client==1.9.0", 
        "sentence-transformers==3.0.0"
    ]
    
    # Kiểm tra và cài đặt từng package
    installed_count = 0
    for package in required_packages:
        package_name = package.split("==")[0]
        
        if check_package(package_name):
            print(f"✅ {package_name} already installed")
            installed_count += 1
        else:
            if install_package(package):
                installed_count += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Installation Summary:")
    print(f"   Total packages: {len(required_packages)}")
    print(f"   Successfully installed: {installed_count}")
    print(f"   Failed: {len(required_packages) - installed_count}")
    
    if installed_count == len(required_packages):
        print("\n🎉 All dependencies installed successfully!")
        print("\n📋 Next steps:")
        print("1. Start Qdrant server: docker-compose up qdrant")
        print("2. Start API server: python api_gateway/api.py")
        print("3. Test the feature: python test_knowledge_similarity.py")
    else:
        print("\n⚠️ Some packages failed to install. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 