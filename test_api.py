#!/usr/bin/env python3
"""
Simple test script for production API
"""
import requests
import json

def test_api_gateway():
    """Test API Gateway with dict prompt"""
    url = "http://localhost:8000/chat"
    
    # Test prompt object
    test_data = {
        "user_input": "Tôi cảm thấy rất lo lắng về tương lai",
        "history": [],
        "session_id": "test_session"
    }
    
    try:
        response = requests.post(url, json=test_data)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_model_server():
    """Test Model Server with dict prompt"""
    url = "http://localhost:8001/generate"
    
    # Test prompt object
    test_data = {
        "prompt": {
            "input": "Tôi cảm thấy rất lo lắng về tương lai",
            "context": {
                "mental_state": "lo_lang",
                "sentiment_intensity": "cao",
                "risk_level": "normal"
            }
        },
        "max_length": 200,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(url, json=test_data)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    print("Testing API Gateway...")
    test_api_gateway()
    
    print("\nTesting Model Server...")
    test_model_server() 