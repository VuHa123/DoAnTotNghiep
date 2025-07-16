#!/usr/bin/env python3
"""
Script setup môi trường development cho Mental Health Chatbot
Thay thế cho Docker deployment
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_python_version():
    """Kiểm tra phiên bản Python"""
    if sys.version_info < (3, 8):
        logger.error("❌ Python 3.8+ is required")
        return False
    logger.info(f"✅ Python version: {sys.version}")
    return True

def create_virtual_environment():
    """Tạo virtual environment"""
    venv_path = Path(".venv")
    
    if venv_path.exists():
        logger.info("✅ Virtual environment already exists")
        return True
    
    try:
        logger.info("🔧 Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", ".venv"], check=True)
        logger.info("✅ Virtual environment created")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to create virtual environment: {e}")
        return False

def install_dependencies():
    """Cài đặt dependencies"""
    try:
        logger.info("📦 Installing dependencies...")
        
        # Kích hoạt virtual environment
        if os.name == "nt":  # Windows
            pip_path = ".venv/Scripts/pip"
        else:  # Linux/Mac
            pip_path = ".venv/bin/pip"
        
        # Cài đặt requirements
        subprocess.run([pip_path, "install", "-r", "requirements.txt"], check=True)
        logger.info("✅ Dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install dependencies: {e}")
        return False

def setup_directories():
    """Tạo các thư mục cần thiết"""
    directories = [
        "logs",
        "models/weights",
        "Dataset/Data",
        "Database"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Created directory: {directory}")

def check_model_files():
    """Kiểm tra các file model cần thiết"""
    required_models = [
        "models/weights/base_model/meta-llama/Llama-3.2-1B-Instruct/config.json",
        "models/weights/chatbot_finetuned/checkpoint-1098",
        "models/weights/gating_router/model.safetensors",
        "models/weights/mental_state/model.safetensors",
        "models/weights/sentiment/model.safetensors"
    ]
    
    missing_models = []
    for model_path in required_models:
        if not Path(model_path).exists():
            missing_models.append(model_path)
    
    if missing_models:
        logger.warning(f"⚠️ Missing model files: {missing_models}")
        logger.info("💡 You may need to download models or run training scripts")
    else:
        logger.info("✅ All model files are present")

def create_dev_scripts():
    """Tạo các script development"""
    scripts = {
        "start_dev.py": """#!/usr/bin/env python3
import subprocess
import sys
import os

# Kích hoạt virtual environment
if os.name == "nt":  # Windows
    python_path = ".venv/Scripts/python"
else:  # Linux/Mac
    python_path = ".venv/bin/python"

# Chạy development server
subprocess.run([python_path, "run_dev.py"])
""",
        "install_dev.py": """#!/usr/bin/env python3
import subprocess
import sys
import os

# Kích hoạt virtual environment
if os.name == "nt":  # Windows
    pip_path = ".venv/Scripts/pip"
else:  # Linux/Mac
    pip_path = ".venv/bin/pip"

# Cài đặt dependencies
subprocess.run([pip_path, "install", "-r", "requirements.txt"])
""",
        "test_dev.py": """#!/usr/bin/env python3
import requests
import time
import sys

def test_servers():
    print("🧪 Testing development servers...")
    
    # Test API Gateway
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ API Gateway is running")
        else:
            print("❌ API Gateway health check failed")
            return False
    except Exception as e:
        print(f"❌ API Gateway not accessible: {e}")
        return False
    
    # Test Model Server
    try:
        response = requests.get("http://localhost:8001/health", timeout=5)
        if response.status_code == 200:
            print("✅ Model Server is running")
        else:
            print("❌ Model Server health check failed")
            return False
    except Exception as e:
        print(f"❌ Model Server not accessible: {e}")
        return False
    
    print("🎉 All servers are working correctly!")
    return True

if __name__ == "__main__":
    success = test_servers()
    sys.exit(0 if success else 1)
"""
    }
    
    for filename, content in scripts.items():
        with open(filename, "w") as f:
            f.write(content)
        
        # Làm cho script executable trên Linux/Mac
        if os.name != "nt":
            os.chmod(filename, 0o755)
        
        logger.info(f"✅ Created script: {filename}")

def main():
    """Main setup function"""
    logger.info("🏗️ Setting up Mental Health Chatbot Development Environment...")
    
    # Kiểm tra Python version
    if not check_python_version():
        return False
    
    # Tạo virtual environment
    if not create_virtual_environment():
        return False
    
    # Cài đặt dependencies
    if not install_dependencies():
        return False
    
    # Tạo thư mục
    setup_directories()
    
    # Kiểm tra model files
    check_model_files()
    
    # Tạo development scripts
    create_dev_scripts()
    
    logger.info("""
🎉 Development environment setup completed!

📋 Next steps:
1. Activate virtual environment:
   - Windows: .venv\\Scripts\\activate
   - Linux/Mac: source .venv/bin/activate

2. Start development servers:
   python run_dev.py

3. Test the application:
   python test_dev.py

4. Access the application:
   - API Gateway: http://localhost:8000
   - API Docs: http://localhost:8000/docs
   - Model Server: http://localhost:8001

💡 Quick commands:
- Start: python run_dev.py
- Test: python test_dev.py
- Install deps: python install_dev.py
    """)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 