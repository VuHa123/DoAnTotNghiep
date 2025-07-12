#!/usr/bin/env python3
"""
Script to run all tests for Mental Health Chatbot
"""

import subprocess
import sys
import os
from pathlib import Path

def run_tests():
    """Run all tests"""
    print("🧪 Running Mental Health Chatbot Tests...")
    
    # Get project root
    project_root = Path(__file__).parent.parent
    
    # Run pytest
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "tests/", 
            "-v", 
            "--cov=services",
            "--cov=api_gateway",
            "--cov-report=html:htmlcov",
            "--cov-report=term-missing"
        ], cwd=project_root, capture_output=True, text=True)
        
        print("✅ Tests completed!")
        print(f"Exit code: {result.returncode}")
        
        if result.stdout:
            print("\n📋 Test Output:")
            print(result.stdout)
        
        if result.stderr:
            print("\n⚠️  Test Errors:")
            print(result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

def run_linting():
    """Run code linting"""
    print("\n🔍 Running code linting...")
    
    project_root = Path(__file__).parent.parent
    
    try:
        # Run flake8
        result = subprocess.run([
            sys.executable, "-m", "flake8", 
            "services/", "api_gateway/", "tests/",
            "--max-line-length=100",
            "--ignore=E501,W503"
        ], cwd=project_root, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Linting passed!")
        else:
            print("❌ Linting failed:")
            print(result.stdout)
            print(result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Error running linting: {e}")
        return False

def main():
    """Main function"""
    print("🚀 Mental Health Chatbot Test Suite")
    print("=" * 50)
    
    # Run tests
    tests_passed = run_tests()
    
    # Run linting
    linting_passed = run_linting()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    print(f"Tests: {'✅ PASSED' if tests_passed else '❌ FAILED'}")
    print(f"Linting: {'✅ PASSED' if linting_passed else '❌ FAILED'}")
    
    if tests_passed and linting_passed:
        print("\n🎉 All checks passed!")
        return 0
    else:
        print("\n⚠️  Some checks failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 