#!/usr/bin/env python3
"""
Script khởi tạo database với bảng feedback
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from Database.core import create_db, engine, Base
from sqlalchemy import text

def init_feedback_database():
    """Khởi tạo database với bảng feedback"""
    try:
        print("🔄 Đang khởi tạo database...")
        
        # Tạo tất cả bảng
        create_db()
        
        # Kiểm tra xem bảng feedback đã được tạo chưa
        with engine.connect() as conn:
            result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='feedback'"))
            if result.fetchone():
                print("✅ Bảng feedback đã được tạo thành công!")
            else:
                print("❌ Bảng feedback chưa được tạo!")
                return False
        
        print("✅ Database đã được khởi tạo thành công!")
        print("📊 Các bảng đã tạo:")
        
        # Liệt kê tất cả bảng
        with engine.connect() as conn:
            result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
            tables = result.fetchall()
            for table in tables:
                print(f"   - {table[0]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Lỗi khi khởi tạo database: {e}")
        return False

if __name__ == "__main__":
    success = init_feedback_database()
    if success:
        print("\n🎉 Khởi tạo database hoàn tất!")
    else:
        print("\n💥 Khởi tạo database thất bại!")
        sys.exit(1) 