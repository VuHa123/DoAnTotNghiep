#!/usr/bin/env python3
"""
Script kiểm tra trạng thái tổng quan của development environment
"""

import os
import sys
import requests
import logging
from pathlib import Path
import psutil

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_python_environment():
    """Kiểm tra môi trường Python"""
    logger.info("🐍 Checking Python environment...")
    
    # Kiểm tra Python version
    version = sys.version_info
    logger.info(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version < (3, 8):
        logger.error("❌ Python 3.8+ is required")
        return False
    
    # Kiểm tra virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        logger.info("✅ Virtual environment is active")
    else:
        logger.warning("⚠️ Virtual environment not detected")
    
    return True

def check_dependencies():
    """Kiểm tra dependencies"""
    logger.info("📦 Checking dependencies...")
    
    required_packages = [
        'torch', 'transformers', 'fastapi', 'uvicorn',
        'requests', 'psutil', 'numpy', 'pandas'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package}")
        except ImportError:
            logger.error(f"❌ {package} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"❌ Missing packages: {missing_packages}")
        return False
    
    return True

def check_files_and_directories():
    """Kiểm tra files và directories"""
    logger.info("📁 Checking files and directories...")
    
    required_files = [
        "model_server.py",
        "api_gateway/main.py",
        "services/chatbot/bot_service.py",
        "services/gating_router/router.py",
        "requirements.txt",
        "run_dev.py",
        "setup_dev.py"
    ]
    
    required_dirs = [
        "logs",
        "models/weights",
        "services",
        "api_gateway"
    ]
    
    # Kiểm tra files
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            logger.info(f"✅ {file}")
        else:
            logger.error(f"❌ {file} - Missing")
            missing_files.append(file)
    
    # Kiểm tra directories
    missing_dirs = []
    for directory in required_dirs:
        if os.path.exists(directory):
            logger.info(f"✅ {directory}/")
        else:
            logger.error(f"❌ {directory}/ - Missing")
            missing_dirs.append(directory)
    
    if missing_files or missing_dirs:
        logger.error(f"❌ Missing: {missing_files + missing_dirs}")
        return False
    
    return True

def check_model_files():
    """Kiểm tra model files"""
    logger.info("🤖 Checking model files...")
    
    model_paths = [
        "models/weights/base_model/meta-llama/Llama-3.2-1B-Instruct/",
        "models/weights/chatbot_finetuned/checkpoint-1098/",
        "models/weights/gating_router/",
        "models/weights/mental_state/",
        "models/weights/sentiment/"
    ]
    
    missing_models = []
    for model_path in model_paths:
        if os.path.exists(model_path):
            # Kiểm tra xem có file config hoặc model không
            if any(os.listdir(model_path)):
                logger.info(f"✅ {model_path}")
            else:
                logger.warning(f"⚠️ {model_path} - Empty")
                missing_models.append(model_path)
        else:
            logger.error(f"❌ {model_path} - Missing")
            missing_models.append(model_path)
    
    if missing_models:
        logger.warning(f"⚠️ Missing model files: {missing_models}")
        logger.info("💡 You may need to download models or run training scripts")
    
    return len(missing_models) == 0

def check_ports():
    """Kiểm tra ports"""
    logger.info("🔌 Checking ports...")
    
    ports_to_check = [8000, 8001]
    port_status = {}
    
    for port in ports_to_check:
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=5)
            if response.status_code == 200:
                logger.info(f"✅ Port {port} - Service running")
                port_status[port] = "running"
            else:
                logger.warning(f"⚠️ Port {port} - Service responding but unhealthy")
                port_status[port] = "unhealthy"
        except requests.exceptions.ConnectionError:
            logger.info(f"ℹ️ Port {port} - No service")
            port_status[port] = "free"
        except Exception as e:
            logger.error(f"❌ Port {port} - Error: {e}")
            port_status[port] = "error"
    
    return port_status

def check_processes():
    """Kiểm tra processes"""
    logger.info("🔄 Checking processes...")
    
    process_patterns = [
        "model_server.py",
        "api_gateway/main.py",
        "uvicorn",
        "run_dev.py"
    ]
    
    found_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
            for pattern in process_patterns:
                if pattern in cmdline:
                    found_processes.append({
                        'name': proc.name(),
                        'pid': proc.pid,
                        'pattern': pattern
                    })
                    break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    if found_processes:
        logger.info("✅ Found development processes:")
        for proc in found_processes:
            logger.info(f"  - {proc['name']} (PID: {proc['pid']}) - {proc['pattern']}")
    else:
        logger.info("ℹ️ No development processes found")
    
    return found_processes

def check_logs():
    """Kiểm tra log files"""
    logger.info("📝 Checking log files...")
    
    log_files = [
        "logs/model_server.log",
        "logs/api_gateway.log"
    ]
    
    for log_file in log_files:
        if os.path.exists(log_file):
            size = os.path.getsize(log_file)
            logger.info(f"✅ {log_file} ({size} bytes)")
        else:
            logger.info(f"ℹ️ {log_file} - Not created yet")

def generate_summary(python_ok, deps_ok, files_ok, models_ok, port_status, processes):
    """Tạo summary"""
    logger.info("\n" + "="*50)
    logger.info("📊 DEVELOPMENT ENVIRONMENT SUMMARY")
    logger.info("="*50)
    
    # Environment status
    env_status = "✅ READY" if python_ok and deps_ok and files_ok else "❌ NOT READY"
    logger.info(f"Environment: {env_status}")
    
    # Services status
    running_services = sum(1 for status in port_status.values() if status == "running")
    total_services = len(port_status)
    services_status = f"✅ {running_services}/{total_services} services running"
    logger.info(f"Services: {services_status}")
    
    # Models status
    models_status = "✅ READY" if models_ok else "⚠️ PARTIAL"
    logger.info(f"Models: {models_status}")
    
    # Processes
    process_count = len(processes)
    logger.info(f"Processes: {process_count} development processes found")
    
    # Recommendations
    logger.info("\n💡 Recommendations:")
    
    if not python_ok or not deps_ok or not files_ok:
        logger.info("  - Run: python setup_dev.py")
    
    if running_services == 0:
        logger.info("  - Start services: python run_dev.py")
    elif running_services < total_services:
        logger.info("  - Some services not running, check logs")
    
    if not models_ok:
        logger.info("  - Download or train models")
    
    logger.info("="*50)

def main():
    """Main function"""
    logger.info("🔍 Checking Mental Health Chatbot Development Environment...")
    
    # Kiểm tra từng component
    python_ok = check_python_environment()
    deps_ok = check_dependencies()
    files_ok = check_files_and_directories()
    models_ok = check_model_files()
    
    # Kiểm tra runtime status
    port_status = check_ports()
    processes = check_processes()
    check_logs()
    
    # Tạo summary
    generate_summary(python_ok, deps_ok, files_ok, models_ok, port_status, processes)
    
    # Return status
    overall_ok = python_ok and deps_ok and files_ok
    return overall_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 