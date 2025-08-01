#!/usr/bin/env python3
"""
Script để chạy cả API Gateway và Model Server
"""

import subprocess
import time
import sys
import os
import signal
import logging
from pathlib import Path

# Setup logging
# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ServerManager:
    def __init__(self):
        self.processes = {}
        
    def start_model_server(self):
        """Khởi động Model Server"""
        try:
            logger.info("🚀 Starting Model Server on port 8001...")
            
            # Kiểm tra xem model server có tồn tại không
            if not os.path.exists("model_server.py"):
                logger.error("❌ model_server.py not found!")
                return False
                
            process = subprocess.Popen([
                sys.executable, "model_server.py"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.processes["model_server"] = process
            logger.info(f"✅ Model Server started with PID: {process.pid}")
            
            # Đợi một chút để server khởi động
            time.sleep(5)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start Model Server: {e}")
            return False
    
    def start_api_gateway(self):
        """Khởi động API Gateway"""
        try:
            logger.info("🚀 Starting API Gateway on port 8000...")
            
            process = subprocess.Popen([
                "uvicorn", "api_gateway.main:app", 
                "--host", "0.0.0.0", 
                "--port", "8000",
                "--reload"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
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
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                logger.info(f"✅ {name} is healthy")
                return True
            else:
                logger.warning(f"⚠️ {name} returned status {response.status_code}")
                return False
        except Exception as e:
            logger.warning(f"⚠️ {name} health check failed: {e}")
            return False
    
    def wait_for_servers(self):
        """Đợi servers khởi động"""
        logger.info("⏳ Waiting for servers to start...")
        
        # Đợi Model Server
        for i in range(30):  # 30 giây timeout
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
    
    def stop_all(self):
        """Dừng tất cả servers"""
        logger.info("🛑 Stopping all servers...")
        
        for name, process in self.processes.items():
            try:
                logger.info(f"Stopping {name} (PID: {process.pid})...")
                process.terminate()
                process.wait(timeout=10)
                logger.info(f"✅ {name} stopped")
            except subprocess.TimeoutExpired:
                logger.warning(f"⚠️ {name} didn't stop gracefully, killing...")
                process.kill()
            except Exception as e:
                logger.error(f"❌ Error stopping {name}: {e}")
    
    def run(self):
        """Chạy cả hai servers"""
        try:
            # Khởi động Model Server trước
            if not self.start_model_server():
                return False
            
            # Khởi động API Gateway
            if not self.start_api_gateway():
                return False
            
            # Đợi servers khởi động
            if not self.wait_for_servers():
                return False
            
            logger.info("""
🎉 Servers are running!

📋 URLs:
- API Gateway: http://localhost:8000
- Model Server: http://localhost:8001
- API Docs: http://localhost:8000/docs
- Model Server Docs: http://localhost:8001/docs

💡 Test endpoints:
- Health check: curl http://localhost:8000/health
- Chat: curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"user_input": "Hello", "history": [], "session_id": "test"}'

Press Ctrl+C to stop all servers
            """)
            
            # Đợi cho đến khi user dừng
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Received interrupt signal")
                
        except Exception as e:
            logger.error(f"❌ Error running servers: {e}")
            return False
        finally:
            self.stop_all()

def main():
    """Main function"""
    logger.info("🏗️ Starting Mental Health Chatbot Servers...")
    
    # Kiểm tra dependencies
    required_files = [
        "model_server.py",
        "api_gateway/main.py",
        "services/chatbot/llama_service.py",
        "services/chatbot/gemini_service.py"
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            logger.error(f"❌ Required file not found: {file}")
            return False
    
    # Khởi động servers
    manager = ServerManager()
    return manager.run()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 