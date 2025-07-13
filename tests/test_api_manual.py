#!/usr/bin/env python3
"""
Script test API cho Mental Health Chatbot
Kiểm tra các endpoint chính: /health, /chat, /emergency
"""

import requests
import json
import time
import uuid

# Cấu hình
API_BASE_URL = "http://localhost:8000"
SESSION_ID = str(uuid.uuid4())

def test_health():
    """Test endpoint /health"""
    print("🔍 Testing /health endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            print("✅ /health: OK")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ /health: Failed with status {response.status_code}")
    except Exception as e:
        print(f"❌ /health: Error - {e}")

def test_chat_normal():
    """Test endpoint /chat với message bình thường"""
    print("\n🔍 Testing /chat with normal message...")
    payload = {
        "user_input": "Xin chào, hôm nay tôi cảm thấy khá ổn.",
        "history": [],
        "session_id": SESSION_ID
    }
    try:
        response = requests.post(f"{API_BASE_URL}/chat", json=payload)
        if response.status_code == 200:
            result = response.json()
            print("✅ /chat (normal): OK")
            print(f"   Risk level: {result.get('risk_level', 'N/A')}")
            print(f"   Confidence: {result.get('confidence', 'N/A')}")
            print(f"   Response: {result.get('bot_response', 'N/A')[:100]}...")
        else:
            print(f"❌ /chat (normal): Failed with status {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ /chat (normal): Error - {e}")

def test_chat_risky():
    """Test endpoint /chat với message có vấn đề"""
    print("\n🔍 Testing /chat with risky message...")
    payload = {
        "user_input": "Tôi cảm thấy rất lo lắng và stress về công việc.",
        "history": [],
        "session_id": SESSION_ID
    }
    try:
        response = requests.post(f"{API_BASE_URL}/chat", json=payload)
        if response.status_code == 200:
            result = response.json()
            print("✅ /chat (risky): OK")
            print(f"   Risk level: {result.get('risk_level', 'N/A')}")
            print(f"   Confidence: {result.get('confidence', 'N/A')}")
            print(f"   Response: {result.get('bot_response', 'N/A')[:100]}...")
        else:
            print(f"❌ /chat (risky): Failed with status {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ /chat (risky): Error - {e}")

def test_chat_emergency():
    """Test endpoint /chat với message khẩn cấp"""
    print("\n🔍 Testing /chat with emergency message...")
    payload = {
        "user_input": "Tôi muốn tự tử, tôi không thể chịu đựng được nữa.",
        "history": [],
        "session_id": SESSION_ID
    }
    try:
        response = requests.post(f"{API_BASE_URL}/chat", json=payload)
        if response.status_code == 200:
            result = response.json()
            print("✅ /chat (emergency): OK")
            print(f"   Risk level: {result.get('risk_level', 'N/A')}")
            print(f"   Confidence: {result.get('confidence', 'N/A')}")
            print(f"   Response: {result.get('bot_response', 'N/A')[:100]}...")
        else:
            print(f"❌ /chat (emergency): Failed with status {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ /chat (emergency): Error - {e}")

def test_multi_turn():
    """Test chat đa lượt"""
    print("\n🔍 Testing multi-turn conversation...")
    history = []
    messages = [
        "Xin chào, tôi cảm thấy hơi lo lắng.",
        "Tôi lo lắng về công việc và tương lai.",
        "Cảm ơn bạn đã lắng nghe."
    ]
    
    for i, message in enumerate(messages):
        print(f"   Turn {i+1}: {message}")
        payload = {
            "user_input": message,
            "history": history,
            "session_id": SESSION_ID
        }
        try:
            response = requests.post(f"{API_BASE_URL}/chat", json=payload)
            if response.status_code == 200:
                result = response.json()
                bot_reply = result.get('bot_response', '')
                history.append((message, bot_reply))
                print(f"   ✅ Response: {bot_reply[:50]}...")
            else:
                print(f"   ❌ Failed: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

def test_emergency_endpoint():
    """Test endpoint /emergency"""
    print("\n🔍 Testing /emergency endpoint...")
    try:
        response = requests.post(f"{API_BASE_URL}/emergency")
        if response.status_code == 200:
            print("✅ /emergency: OK")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ /emergency: Failed with status {response.status_code}")
    except Exception as e:
        print(f"❌ /emergency: Error - {e}")

def main():
    """Chạy tất cả test"""
    print("🚀 Starting API tests for Mental Health Chatbot")
    print("=" * 50)
    
    # Test các endpoint
    test_health()
    test_chat_normal()
    test_chat_risky()
    test_chat_emergency()
    test_multi_turn()
    test_emergency_endpoint()
    
    print("\n" + "=" * 50)
    print("🏁 API tests completed!")

if __name__ == "__main__":
    main() 