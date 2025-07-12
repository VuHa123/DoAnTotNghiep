#!/usr/bin/env python3
"""
Script để chạy toàn bộ workflow từ fine-tune đến ExLlama inference
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path

def run_command(command: str, cwd: str = None, check: bool = True) -> bool:
    """
    Chạy command và trả về True nếu thành công
    
    Args:
        command: Command để chạy
        cwd: Working directory
        check: Có check return code không
        
    Returns:
        True nếu thành công, False nếu thất bại
    """
    try:
        print(f"🔄 Running: {command}")
        start_time = time.time()
        
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ Success ({duration:.2f}s): {command}")
            if result.stdout.strip():
                print(f"Output: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ Failed ({duration:.2f}s): {command}")
            if result.stderr.strip():
                print(f"Error: {result.stderr.strip()}")
            if check:
                return False
            else:
                return True
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def check_prerequisites():
    """Kiểm tra prerequisites"""
    print("🔍 Checking prerequisites...")
    
    # Kiểm tra CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA not available - performance may be affected")
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    
    # Kiểm tra ExLlama
    exllama_path = Path(__file__).parent.parent / "exllama"
    if exllama_path.exists():
        print("✅ ExLlama repository found")
    else:
        print("❌ ExLlama not found - will install during setup")
    
    # Kiểm tra model paths
    base_model_path = Path(__file__).parent.parent / "models" / "weights" / "chatbot_finetuned_nf4"
    if base_model_path.exists():
        print("✅ Fine-tuned model found")
    else:
        print("❌ Fine-tuned model not found")
        return False
    
    return True

def setup_environment():
    """Setup environment"""
    print("📦 Setting up environment...")
    
    # Cài đặt ExLlama và dependencies
    setup_script = Path(__file__).parent / "setup_exllama.py"
    if not run_command(f"python {setup_script}"):
        print("❌ Failed to setup environment")
        return False
    
    return True

def convert_model_to_gptq(base_model: str, lora_path: str, output_path: str):
    """Convert model sang GPTQ format"""
    print("🔄 Converting model to GPTQ format...")
    
    convert_script = Path(__file__).parent / "convert_to_gptq.py"
    command = f"python {convert_script} --base_model {base_model} --lora_path {lora_path} --output_path {output_path} --bits 4 --group_size 128"
    
    if not run_command(command):
        print("❌ Failed to convert model")
        return False
    
    return True

def test_inference(model_path: str):
    """Test inference"""
    print("🧪 Testing inference...")
    
    inference_script = Path(__file__).parent / "exllama_inference.py"
    command = f"python {inference_script} --model_path {model_path} --test"
    
    if not run_command(command):
        print("❌ Failed to test inference")
        return False
    
    return True

def run_interactive_chat(model_path: str):
    """Chạy interactive chat"""
    print("💬 Starting interactive chat...")
    print("Press Ctrl+C to exit")
    
    inference_script = Path(__file__).parent / "exllama_inference.py"
    command = f"python {inference_script} --model_path {model_path} --interactive"
    
    # Chạy interactive mode (không capture output)
    try:
        subprocess.run(command, shell=True)
    except KeyboardInterrupt:
        print("\n👋 Interactive chat ended")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Run ExLlama GPTQ workflow")
    parser.add_argument("--base_model", type=str, 
                       default="meta-llama/Llama-3.2-1B-Instruct",
                       help="Base model name")
    parser.add_argument("--lora_path", type=str,
                       default="models/weights/chatbot_finetuned_nf4",
                       help="Path to LoRA adapter")
    parser.add_argument("--output_path", type=str,
                       default="models/weights/chatbot_gptq",
                       help="Output path for GPTQ model")
    parser.add_argument("--setup", action="store_true",
                       help="Setup environment (install ExLlama)")
    parser.add_argument("--convert", action="store_true",
                       help="Convert model to GPTQ")
    parser.add_argument("--test", action="store_true",
                       help="Test inference")
    parser.add_argument("--interactive", action="store_true",
                       help="Run interactive chat")
    parser.add_argument("--all", action="store_true",
                       help="Run all steps")
    
    args = parser.parse_args()
    
    print("🚀 ExLlama GPTQ Workflow")
    print("=" * 50)
    
    # Kiểm tra prerequisites
    if not check_prerequisites():
        print("❌ Prerequisites check failed")
        return False
    
    # Setup environment nếu cần
    if args.setup or args.all:
        if not setup_environment():
            print("❌ Environment setup failed")
            return False
    
    # Convert model nếu cần
    if args.convert or args.all:
        if not convert_model_to_gptq(args.base_model, args.lora_path, args.output_path):
            print("❌ Model conversion failed")
            return False
    
    # Test inference nếu cần
    if args.test or args.all:
        if not test_inference(args.output_path):
            print("❌ Inference test failed")
            return False
    
    # Interactive chat nếu cần
    if args.interactive:
        if not run_interactive_chat(args.output_path):
            print("❌ Interactive chat failed")
            return False
    
    print("\n" + "=" * 50)
    print("✅ Workflow completed successfully!")
    
    if args.all:
        print("\n📋 Summary:")
        print(f"✅ Environment setup: Completed")
        print(f"✅ Model conversion: {args.output_path}")
        print(f"✅ Inference test: Passed")
        print(f"✅ Interactive chat: Available")
        
        print("\n🎯 Next steps:")
        print(f"1. Test your model: python scripts/exllama_inference.py --model_path {args.output_path} --test")
        print(f"2. Interactive chat: python scripts/exllama_inference.py --model_path {args.output_path} --interactive")
        print(f"3. Integrate into API: from models.exllama_chatbot import create_exllama_chatbot")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 