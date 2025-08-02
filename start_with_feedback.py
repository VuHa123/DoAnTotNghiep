#!/usr/bin/env python3
"""
Script khởi động hệ thống chatbot với tính năng feedback
"""

import subprocess
import sys
import os
import time
import signal
import threading

def check_mongodb():
    """Kiểm tra MongoDB có đang chạy không"""
    try:
        import pymongo
        client = pymongo.MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
        client.server_info()
        client.close()
        return True
    except Exception as e:
        print(f"❌ MongoDB không khả dụng: {e}")
        return False

def install_dependencies():
    """Cài đặt dependencies nếu cần"""
    try:
        import pymongo
        print("✅ pymongo đã được cài đặt")
        return True
    except ImportError:
        print("📦 Cài đặt pymongo...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "pymongo==4.6.1"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ pymongo đã được cài đặt thành công!")
            return True
        else:
            print(f"❌ Lỗi khi cài đặt pymongo: {result.stderr}")
            return False

def test_feedback_system():
    """Test hệ thống feedback"""
    try:
        print("🧪 Testing feedback system...")
        result = subprocess.run([
            sys.executable, "test_mongodb_connection.py"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Feedback system test thành công!")
            return True
        else:
            print(f"❌ Feedback system test thất bại: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Lỗi khi test feedback system: {e}")
        return False

def start_servers():
    """Khởi động các server"""
    try:
        print("🚀 Khởi động servers...")
        
        # Khởi động model server (nếu cần)
        model_server_process = None
        try:
            model_server_process = subprocess.Popen([
                sys.executable, "llmserver/main.py"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("✅ Model server đã khởi động")
        except Exception as e:
            print(f"⚠️ Không thể khởi động model server: {e}")
        
        # Đợi một chút để model server khởi động
        time.sleep(3)
        
        # Khởi động UI server
        ui_server_process = subprocess.Popen([
            sys.executable, "UI/main.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        print("✅ UI server đã khởi động")
        print("\n🎉 Hệ thống đã sẵn sàng!")
        print("📱 Truy cập: http://localhost:8000")
        print("📊 API docs: http://localhost:8000/docs")
        print("\n💡 Tính năng feedback đã được kích hoạt:")
        print("   - Sau mỗi câu trả lời sẽ có nút 👍/👎")
        print("   - Feedback được lưu vào MongoDB")
        print("   - Xem thống kê: python view_feedback_stats.py")
        
        # Giữ script chạy
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Đang dừng servers...")
            
            if model_server_process:
                model_server_process.terminate()
                print("✅ Model server đã dừng")
            
            ui_server_process.terminate()
            print("✅ UI server đã dừng")
            
    except Exception as e:
        print(f"❌ Lỗi khi khởi động servers: {e}")
        return False

def main():
    """Main function"""
    print("🤖 CHATBOT TÂM LÝ VỚI TÍNH NĂNG FEEDBACK")
    print("=" * 50)
    
    # Kiểm tra dependencies
    if not install_dependencies():
        print("❌ Không thể cài đặt dependencies")
        sys.exit(1)
    
    # Kiểm tra MongoDB
    if not check_mongodb():
        print("❌ MongoDB không khả dụng")
        print("💡 Hãy khởi động MongoDB trước:")
        print("   sudo systemctl start mongod")
        sys.exit(1)
    
    # Test feedback system
    if not test_feedback_system():
        print("❌ Feedback system test thất bại")
        sys.exit(1)
    
    # Khởi động servers
    start_servers()

if __name__ == "__main__":
    main() 