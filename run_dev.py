#!/usr/bin/env python3
"""
Script để chạy ứng dụng Mental Health Chatbot trong môi trường development
Thay thế cho Docker deployment
"""

import os
import sys
import time
import signal
import logging
import subprocess
import threading
from pathlib import Path
from typing import Dict, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DevelopmentServerManager:
    def __init__(self):
        self.processes: Dict[str, subprocess.Popen] = {}
        self.running = False
        
    def check_dependencies(self) -> bool:
        """Kiểm tra các dependencies cần thiết"""
        logger.info("🔍 Checking dependencies...")
        
        required_files = [
            "model_server.py",
            "api_gateway/main.py",
            "services/chatbot/bot_service.py",
            "services/gating_router/router.py",
            "services/mental_state_classifier/classifer.py",
            "services/setiment_analysis/analyzer.py"
        ]
        
        missing_files = []
        for file in required_files:
            if not os.path.exists(file):
                missing_files.append(file)
        
        if missing_files:
            logger.error(f"❌ Missing required files: {missing_files}")
            return False
        
        # Kiểm tra Python environment
        try:
            import torch
            import transformers
            import fastapi
            import uvicorn
            logger.info("✅ All Python dependencies are available")
        except ImportError as e:
            logger.error(f"❌ Missing Python dependency: {e}")
            return False
        
        return True
    
    def setup_environment(self):
        """Thiết lập môi trường development"""
        logger.info("🔧 Setting up development environment...")
        
        # Tạo thư mục logs nếu chưa có
        os.makedirs("logs", exist_ok=True)
        
        # Thiết lập environment variables
        os.environ.setdefault("PYTHONPATH", str(Path(__file__).parent))
        os.environ.setdefault("DEV_MODE", "true")
        
        logger.info("✅ Environment setup completed")
    
    def start_model_server(self) -> bool:
        """Khởi động Model Server"""
        try:
            logger.info("🚀 Starting Model Server on port 8001...")
            
            # Tạo log file cho model server
            log_file = open("logs/model_server.log", "w")
            
            process = subprocess.Popen([
                sys.executable, "model_server.py"
            ], stdout=log_file, stderr=log_file)
            
            self.processes["model_server"] = process
            logger.info(f"✅ Model Server started with PID: {process.pid}")
            
            # Đợi server khởi động
            time.sleep(8)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start Model Server: {e}")
            return False
    
    def start_api_gateway(self) -> bool:
        """Khởi động API Gateway"""
        try:
            logger.info("🚀 Starting API Gateway on port 8000...")
            
            # Tạo log file cho API gateway
            log_file = open("logs/api_gateway.log", "w")
            
            process = subprocess.Popen([
                "uvicorn", "api_gateway.main:app", 
                "--host", "0.0.0.0", 
                "--port", "8000",
                "--reload"
            ], stdout=log_file, stderr=log_file)
            
            self.processes["api_gateway"] = process
            logger.info(f"✅ API Gateway started with PID: {process.pid}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start API Gateway: {e}")
            return False
    
    def check_server_health(self, url: str, name: str) -> bool:
        """Kiểm tra health của server"""
        try:
            import requests
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                logger.info(f"✅ {name} is healthy")
                return True
            else:
                logger.warning(f"⚠️ {name} returned status {response.status_code}")
                return False
        except Exception as e:
            logger.warning(f"⚠️ {name} health check failed: {e}")
            return False
    
    def wait_for_servers(self) -> bool:
        """Đợi servers khởi động"""
        logger.info("⏳ Waiting for servers to start...")
        
        # Đợi Model Server
        for i in range(60):  # 60 giây timeout
            if self.check_server_health("http://localhost:8001/health", "Model Server"):
                break
            time.sleep(1)
        else:
            logger.error("❌ Model Server failed to start within timeout")
            return False
        
        # Đợi API Gateway
        for i in range(30):  # 30 giây timeout
            if self.check_server_health("http://localhost:8000/health", "API Gateway"):
                break
            time.sleep(1)
        else:
            logger.error("❌ API Gateway failed to start within timeout")
            return False
        
        logger.info("🎉 All servers are running!")
        return True
    
    def monitor_servers(self):
        """Monitor trạng thái của các servers"""
        while self.running:
            try:
                for name, process in self.processes.items():
                    if process.poll() is not None:
                        logger.error(f"❌ {name} has stopped unexpectedly")
                        self.running = False
                        break
                time.sleep(5)
            except KeyboardInterrupt:
                break
    
    def stop_all(self):
        """Dừng tất cả servers"""
        logger.info("🛑 Stopping all servers...")
        self.running = False
        
        for name, process in self.processes.items():
            try:
                logger.info(f"Stopping {name} (PID: {process.pid})...")
                process.terminate()
                
                # Đợi process dừng
                try:
                    process.wait(timeout=10)
                    logger.info(f"✅ {name} stopped gracefully")
                except subprocess.TimeoutExpired:
                    logger.warning(f"⚠️ {name} didn't stop gracefully, killing...")
                    process.kill()
                    process.wait()
                    logger.info(f"✅ {name} killed")
                    
            except Exception as e:
                logger.error(f"❌ Error stopping {name}: {e}")
    
    def run(self) -> bool:
        """Chạy ứng dụng trong development mode"""
        try:
            # Kiểm tra dependencies
            if not self.check_dependencies():
                return False
            
            # Thiết lập môi trường
            self.setup_environment()
            
            # Khởi động Model Server trước
            if not self.start_model_server():
                return False
            
            # Khởi động API Gateway
            if not self.start_api_gateway():
                return False
            
            # Đợi servers khởi động
            if not self.wait_for_servers():
                return False
            
            self.running = True
            
            # Hiển thị thông tin
            logger.info("""
🎉 Development servers are running!

📋 URLs:
- API Gateway: http://localhost:8000
- Model Server: http://localhost:8001
- API Docs: http://localhost:8000/docs
- Model Server Docs: http://localhost:8001/docs

📁 Logs:
- Model Server: logs/model_server.log
- API Gateway: logs/api_gateway.log

💡 Test endpoints:
- Health check: curl http://localhost:8000/health
- Chat: curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"user_input": "Hello", "history": [], "session_id": "test"}'

Press Ctrl+C to stop all servers
            """)
            
            # Bắt đầu monitor trong thread riêng
            monitor_thread = threading.Thread(target=self.monitor_servers)
            monitor_thread.daemon = True
            monitor_thread.start()
            
            # Đợi cho đến khi user dừng
            try:
                while self.running:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Received interrupt signal")
                
        except Exception as e:
            logger.error(f"❌ Error running development servers: {e}")
            return False
        finally:
            self.stop_all()
        
        return True

def signal_handler(signum, frame):
    """Signal handler để dừng servers gracefully"""
    logger.info("Received signal to stop servers")
    sys.exit(0)

def main():
    """Main function"""
    logger.info("🏗️ Starting Mental Health Chatbot in Development Mode...")
    
    # Đăng ký signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Khởi động servers
    manager = DevelopmentServerManager()
    success = manager.run()
    
    if success:
        logger.info("✅ Development servers stopped successfully")
    else:
        logger.error("❌ Development servers failed to start or run properly")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 