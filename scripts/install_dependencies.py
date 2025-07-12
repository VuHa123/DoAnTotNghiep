#!/usr/bin/env python3
"""
Script to install dependencies for Mental Health Chatbot
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install requirements from requirements.txt"""
    print("📦 Installing Python dependencies...")
    
    project_root = Path(__file__).parent.parent
    requirements_file = project_root / "requirements.txt"
    
    if not requirements_file.exists():
        print("❌ requirements.txt not found!")
        return False
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ], cwd=project_root, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Dependencies installed successfully!")
            return True
        else:
            print("❌ Failed to install dependencies:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def install_nltk_data():
    """Install NLTK data"""
    print("\n📚 Installing NLTK data...")
    
    try:
        import nltk
        
        # Download required NLTK data
        nltk_data = [
            "punkt",
            "stopwords", 
            "wordnet",
            "averaged_perceptron_tagger"
        ]
        
        for data in nltk_data:
            print(f"Downloading {data}...")
            nltk.download(data, quiet=True)
        
        print("✅ NLTK data installed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error installing NLTK data: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    project_root = Path(__file__).parent.parent
    directories = [
        "logs",
        "models/weights",
        "data",
        "htmlcov"  # for test coverage reports
    ]
    
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}")
    
    return True

def check_gpu():
    """Check GPU availability"""
    print("\n🖥️  Checking GPU...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU detected: {gpu_name} ({gpu_count} device(s))")
            return True
        else:
            print("⚠️  No GPU detected. Will use CPU (slower for ML tasks)")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed, cannot check GPU")
        return False

def main():
    """Main function"""
    print("🚀 Mental Health Chatbot Setup")
    print("=" * 50)
    
    # Install Python dependencies
    deps_ok = install_requirements()
    
    # Install NLTK data
    nltk_ok = install_nltk_data()
    
    # Create directories
    dirs_ok = create_directories()
    
    # Check GPU
    gpu_ok = check_gpu()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Setup Summary:")
    print(f"Dependencies: {'✅ INSTALLED' if deps_ok else '❌ FAILED'}")
    print(f"NLTK Data: {'✅ INSTALLED' if nltk_ok else '❌ FAILED'}")
    print(f"Directories: {'✅ CREATED' if dirs_ok else '❌ FAILED'}")
    print(f"GPU: {'✅ AVAILABLE' if gpu_ok else '⚠️  CPU ONLY'}")
    
    if deps_ok and nltk_ok and dirs_ok:
        print("\n🎉 Setup completed successfully!")
        print("\n📝 Next steps:")
        print("1. Copy env.example to .env and configure your API keys")
        print("2. Run: python scripts/run_tests.py")
        print("3. Run: python api_gateway/main.py")
        return 0
    else:
        print("\n⚠️  Setup completed with issues!")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 