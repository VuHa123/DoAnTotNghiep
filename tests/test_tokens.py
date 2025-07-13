#!/usr/bin/env python3
"""
Test script để kiểm tra token loading
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from utils.token_loader import token_loader
from utils.api_manager import api_manager

def test_token_loading():
    """Test loading tất cả tokens"""
    print("=== Testing Token Loading ===")
    
    # Test token loader
    print("\n1. Token Loader Test:")
    validation = token_loader.validate_tokens()
    print(f"Validation: {validation}")
    
    print(f"\nHF Token: {'✓' if token_loader.get_hf_token() else '✗'}")
    print(f"GitHub Token: {'✓' if token_loader.get_github_token() else '✗'}")
    print(f"Gemini Keys: {len(token_loader.get_gemini_keys())} keys")
    
    # Test API manager
    print("\n2. API Manager Test:")
    print(f"Loaded {len(api_manager.api_keys)} Gemini API keys")
    
    if api_manager.api_keys:
        print("First few keys:")
        for i, key in enumerate(api_manager.api_keys[:3]):
            print(f"  Key {i+1}: {key[:10]}...{key[-10:]}")
    
    # Test getting best key
    best_key = api_manager.get_best_api_key()
    if best_key:
        print(f"\nBest API key: {best_key[:10]}...{best_key[-10:]}")
    else:
        print("\nNo valid API key found")
    
    # Test stats
    stats = api_manager.get_key_stats()
    print(f"\nAPI Stats: {stats['total_keys']} total keys")

def test_gemini_integration():
    """Test Gemini service integration"""
    print("\n=== Testing Gemini Integration ===")
    
    try:
        from services.chatbot.gemini_service import gemini_service
        
        # Test simple prompt
        test_prompt = "Hello, how are you?"
        result = gemini_service.get_response(test_prompt)
        
        print(f"Gemini test result: {result['success']}")
        if result['success']:
            print(f"Response: {result['response'][:100]}...")
        else:
            print(f"Error: {result['response']}")
            
    except Exception as e:
        print(f"Error testing Gemini: {e}")

if __name__ == "__main__":
    test_token_loading()
    test_gemini_integration() 