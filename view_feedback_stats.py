#!/usr/bin/env python3
"""
Script xem thống kê feedback
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from services.mongodb_feedback_service import mongodb_feedback_service
import json
from datetime import datetime

def view_feedback_stats():
    """Xem thống kê feedback"""
    try:
        print("📊 THỐNG KÊ FEEDBACK")
        print("=" * 50)
        
        # Lấy thống kê tổng quan
        stats = mongodb_feedback_service.get_feedback_stats()
        
        print(f"📈 Tổng số feedback: {stats['total']}")
        print(f"👍 Like: {stats['likes']}")
        print(f"👎 Dislike: {stats['dislikes']}")
        print(f"📊 Tỷ lệ hài lòng: {stats['satisfaction_rate']:.1f}%")
        
        if stats['total'] > 0:
            print("\n" + "=" * 50)
            print("📋 CHI TIẾT FEEDBACK DISLIKE")
            print("=" * 50)
            
            # Lấy danh sách feedback dislike
            dislikes = mongodb_feedback_service.get_dislike_feedback(limit=10)
            
            if dislikes:
                for i, dislike in enumerate(dislikes, 1):
                    print(f"\n{i}. Session: {dislike['session_id']}")
                    print(f"   📅 Thời gian: {dislike['timestamp']}")
                    print(f"   💬 User input: {dislike['user_input'][:100]}...")
                    print(f"   🤖 Bot response: {dislike['bot_response'][:100]}...")
                    if dislike['user_feedback_text']:
                        print(f"   📝 Lý do: {dislike['user_feedback_text']}")
                    if dislike['risk_level']:
                        print(f"   ⚠️ Risk level: {dislike['risk_level']}")
                    if dislike['emotion_label']:
                        print(f"   😊 Emotion: {dislike['emotion_label']}")
                    print("-" * 30)
            else:
                print("🎉 Không có feedback dislike nào!")
        
        return True
        
    except Exception as e:
        print(f"❌ Lỗi khi xem thống kê: {e}")
        return False

def export_feedback_data():
    """Xuất dữ liệu feedback ra file JSONL"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"feedback_export_{timestamp}.jsonl"
        
        print(f"📤 Đang xuất dữ liệu feedback ra file {filename}...")
        
        success = mongodb_feedback_service.export_feedback_to_jsonl(filename)
        
        if success:
            print(f"✅ Đã xuất dữ liệu thành công ra file: {filename}")
            return True
        else:
            print("❌ Lỗi khi xuất dữ liệu")
            return False
            
    except Exception as e:
        print(f"❌ Lỗi khi xuất dữ liệu: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "export":
        success = export_feedback_data()
    else:
        success = view_feedback_stats()
    
    if not success:
        sys.exit(1) 