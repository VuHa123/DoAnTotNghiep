#!/usr/bin/env python3
"""
Script khởi tạo database cho chatbot
"""

from Database.core import create_db

if __name__ == "__main__":
    print("Đang tạo database và các tables...")
    create_db()
    print("✅ Database đã được tạo thành công!")
    print("Các tables đã tạo:")
    print("- conversations")
    print("- emergency_logs") 
    print("- user_sessions")
    print("- mental_state_history") 