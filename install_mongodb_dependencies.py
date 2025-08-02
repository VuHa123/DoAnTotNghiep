#!/usr/bin/env python3
"""
Script cài đặt MongoDB dependencies
"""

import subprocess
import sys
import os

def install_mongodb_dependencies():
    """Cài đặt pymongo và các dependencies cần thiết"""
    try:
        print("📦 Đang cài đặt MongoDB dependencies...")
        
        # Cài đặt pymongo
        print("🔧 Cài đặt pymongo...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "pymongo==4.6.1"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ pymongo đã được cài đặt thành công!")
        else:
            print(f"❌ Lỗi khi cài đặt pymongo: {result.stderr}")
            return False
        
        # Kiểm tra cài đặt
        print("\n🔍 Kiểm tra cài đặt...")
        try:
            import pymongo
            print(f"✅ pymongo version: {pymongo.version}")
        except ImportError as e:
            print(f"❌ Không thể import pymongo: {e}")
            return False
        
        print("\n🎉 Tất cả dependencies đã được cài đặt thành công!")
        print("\n📋 Hướng dẫn tiếp theo:")
        print("1. Đảm bảo MongoDB server đang chạy")
        print("2. Chạy: python test_mongodb_connection.py")
        print("3. Chạy: python view_feedback_stats.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Lỗi khi cài đặt dependencies: {e}")
        return False

if __name__ == "__main__":
    success = install_mongodb_dependencies()
    if not success:
        sys.exit(1) 