#!/usr/bin/env python3
"""
Script để chạy từng service riêng biệt trong development mode
Cho phép chạy Model Server và API Gateway độc lập
"""

import os
import sys
import time
import signal
import logging
import subprocess
import argparse
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IndividualServiceRunner:
    def __init__(self):
        self.process = None
        self.running = False
        
    def check_dependencies(self, service: str) -> bool:
        """Kiểm tra dependencies cho service cụ thể"""
        logger.info(f"🔍 Checking dependencies for {service}...")
        
        if service == "model_server":
            required_files = ["model_server.py"]
        elif service == "api_gateway":
            required_files = [
                "api_gateway/main.py",
                "services/chatbot/bot_service.py",
                "services/gating_router/router.py"
            ]
        else:
            logger.error(f"❌ Unknown service: {service}")
            return False
        
        missing_files = []
        for file in required_files:
            if not os.path.exists(file):
                missing_files.append(file)
        
        if missing_files:
            logger.error(f"❌ Missing required files: {missing_files}")
            return False
        
        logger.info(f"✅ Dependencies for {service} are available")
        return True
    
    def start_model_server(self) -> bool:
        """Khởi động Model Server"""
        try:
            logger.info("🚀 Starting Model Server on port 8001...")
            
            # Tạo log file
            log_file = open("logs/model_server.log", "w")
            
            process = subprocess.Popen([
                sys.executable, "model_server.py"
            ], stdout=log_file, stderr=log_file)
            
            self.process = process
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
            
            # Tạo log file
            log_file = open("logs/api_gateway.log", "w")
            
            process = subprocess.Popen([
                "uvicorn", "api_gateway.main:app", 
                "--host", "0.0.0.0", 
                "--port", "8000",
                "--reload"
            ], stdout=log_file, stderr=log_file)
            
            self.process = process
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
    
    def wait_for_server(self, service: str) -> bool:
        """Đợi server khởi động"""
        logger.info(f"⏳ Waiting for {service} to start...")
        
        if service == "model_server":
            url = "http://localhost:8001/health"
            name = "Model Server"
            timeout = 60
        elif service == "api_gateway":
            url = "http://localhost:8000/health"
            name = "API Gateway"
            timeout = 30
        else:
            logger.error(f"❌ Unknown service: {service}")
            return False
        
        for i in range(timeout):
            if self.check_server_health(url, name):
                logger.info(f"🎉 {service} is running!")
                return True
            time.sleep(1)
        
        logger.error(f"❌ {service} failed to start within timeout")
        return False
    
    def stop_service(self):
        """Dừng service"""
        if self.process:
            try:
                logger.info(f"🛑 Stopping service (PID: {self.process.pid})...")
                self.process.terminate()
                
                try:
                    self.process.wait(timeout=10)
                    logger.info("✅ Service stopped gracefully")
                except subprocess.TimeoutExpired:
                    logger.warning("⚠️ Service didn't stop gracefully, killing...")
                    self.process.kill()
                    self.process.wait()
                    logger.info("✅ Service killed")
                    
            except Exception as e:
                logger.error(f"❌ Error stopping service: {e}")
    
    def run(self, service: str) -> bool:
        """Chạy service cụ thể"""
        try:
            # Kiểm tra dependencies
            if not self.check_dependencies(service):
                return False
            
            # Tạo thư mục logs
            os.makedirs("logs", exist_ok=True)
            
            # Khởi động service
            if service == "model_server":
                if not self.start_model_server():
                    return False
            elif service == "api_gateway":
                if not self.start_api_gateway():
                    return False
            else:
                logger.error(f"❌ Unknown service: {service}")
                return False
            
            # Đợi server khởi động
            if not self.wait_for_server(service):
                return False
            
            self.running = True
            
            # Hiển thị thông tin
            if service == "model_server":
                logger.info("""
🎉 Model Server is running!

📋 URLs:
- Model Server: http://localhost:8001
- Model Server Docs: http://localhost:8001/docs

📁 Logs:
- Model Server: logs/model_server.log

💡 Test endpoints:
- Health check: curl http://localhost:8001/health
- Generate: curl -X POST http://localhost:8001/generate -H "Content-Type: application/json" -d '{"prompt": "Hello"}'

Press Ctrl+C to stop the server
                """)
            else:
                logger.info("""
🎉 API Gateway is running!

📋 URLs:
- API Gateway: http://localhost:8000
- API Docs: http://localhost:8000/docs

📁 Logs:
- API Gateway: logs/api_gateway.log

💡 Test endpoints:
- Health check: curl http://localhost:8000/health
- Chat: curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"user_input": "Hello", "history": [], "session_id": "test"}'

Press Ctrl+C to stop the server
                """)
            
            # Đợi cho đến khi user dừng
            try:
                while self.running:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Received interrupt signal")
                
        except Exception as e:
            logger.error(f"❌ Error running {service}: {e}")
            return False
        finally:
            self.stop_service()
        
        return True

def signal_handler(signum, frame):
    """Signal handler để dừng service gracefully"""
    logger.info("Received signal to stop service")
    sys.exit(0)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Run individual services in development mode")
    parser.add_argument("service", choices=["model_server", "api_gateway"], 
                       help="Service to run")
    
    args = parser.parse_args()
    
    logger.info(f"🏗️ Starting {args.service} in Development Mode...")
    
    # Đăng ký signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Khởi động service
    runner = IndividualServiceRunner()
    success = runner.run(args.service)
    
    if success:
        logger.info(f"✅ {args.service} stopped successfully")
    else:
        logger.error(f"❌ {args.service} failed to start or run properly")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 