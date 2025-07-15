#!/usr/bin/env python3
"""
Script test nhanh cho development environment
"""

import requests
import json
import time
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_api_gateway():
    """Test API Gateway"""
    logger.info("🧪 Testing API Gateway...")
    
    try:
        # Health check
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            logger.info("✅ API Gateway health check passed")
        else:
            logger.error(f"❌ API Gateway health check failed: {response.status_code}")
            return False
        
        # Chat endpoint test
        chat_data = {
            "user_input": "Hello, how are you?",
            "history": [],
            "session_id": "test_session"
        }
        
        response = requests.post(
            "http://localhost:8000/chat",
            json=chat_data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info("✅ Chat endpoint test passed")
            logger.info(f"   Response: {result.get('bot_response', 'No response')[:50]}...")
            return True
        else:
            logger.error(f"❌ Chat endpoint test failed: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        logger.error("❌ API Gateway not accessible")
        return False
    except Exception as e:
        logger.error(f"❌ API Gateway test error: {e}")
        return False

def test_model_server():
    """Test Model Server"""
    logger.info("🧪 Testing Model Server...")
    
    try:
        # Health check
        response = requests.get("http://localhost:8001/health", timeout=5)
        if response.status_code == 200:
            logger.info("✅ Model Server health check passed")
        else:
            logger.error(f"❌ Model Server health check failed: {response.status_code}")
            return False
        
        # Generate endpoint test
        generate_data = {
            "prompt": "Hello, how are you?",
            "max_length": 100,
            "temperature": 0.7
        }
        
        response = requests.post(
            "http://localhost:8001/generate",
            json=generate_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info("✅ Generate endpoint test passed")
            logger.info(f"   Response: {result.get('response', 'No response')[:50]}...")
            logger.info(f"   Inference time: {result.get('inference_time', 0):.2f}s")
            return True
        else:
            logger.error(f"❌ Generate endpoint test failed: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        logger.error("❌ Model Server not accessible")
        return False
    except Exception as e:
        logger.error(f"❌ Model Server test error: {e}")
        return False

def test_integration():
    """Test integration between services"""
    logger.info("🧪 Testing service integration...")
    
    try:
        # Test chat with different types of messages
        test_cases = [
            {
                "name": "Normal greeting",
                "input": "Hello, how are you today?",
                "expected_risk": "normal"
            },
            {
                "name": "Mental health concern",
                "input": "I've been feeling really sad lately",
                "expected_risk": "risky"
            }
        ]
        
        for test_case in test_cases:
            logger.info(f"   Testing: {test_case['name']}")
            
            chat_data = {
                "user_input": test_case["input"],
                "history": [],
                "session_id": f"test_{int(time.time())}"
            }
            
            response = requests.post(
                "http://localhost:8000/chat",
                json=chat_data,
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                risk_level = result.get('risk_level', 'unknown')
                logger.info(f"     ✅ Response received, risk level: {risk_level}")
            else:
                logger.error(f"     ❌ Failed: {response.status_code}")
                return False
        
        logger.info("✅ Integration test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration test error: {e}")
        return False

def main():
    """Main test function"""
    logger.info("🚀 Starting Quick Test for Mental Health Chatbot...")
    
    # Test API Gateway
    api_ok = test_api_gateway()
    
    # Test Model Server
    model_ok = test_model_server()
    
    # Test Integration
    integration_ok = test_integration()
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("📊 QUICK TEST RESULTS")
    logger.info("="*50)
    
    logger.info(f"API Gateway: {'✅ PASS' if api_ok else '❌ FAIL'}")
    logger.info(f"Model Server: {'✅ PASS' if model_ok else '❌ FAIL'}")
    logger.info(f"Integration: {'✅ PASS' if integration_ok else '❌ FAIL'}")
    
    overall_success = api_ok and model_ok and integration_ok
    
    if overall_success:
        logger.info("\n🎉 All tests passed! Development environment is working correctly.")
    else:
        logger.info("\n❌ Some tests failed. Check the logs above for details.")
        logger.info("💡 Try running: python status_dev.py for detailed diagnostics")
    
    logger.info("="*50)
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 