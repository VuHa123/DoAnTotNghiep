#!/usr/bin/env python3
"""
Script test nhanh các giải pháp memory mà không cần train full
"""

import os
import gc
import torch
import sys
from pathlib import Path

# Setup memory optimization
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def clear_memory():
    """Clear memory và cache"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    print("🧹 Memory cleared!")

def get_memory_info():
    """Lấy thông tin memory hiện tại"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free = total - reserved
        return {
            'allocated': allocated,
            'reserved': reserved,
            'total': total,
            'free': free
        }
    return None

def test_memory_usage(solution_name):
    """Test memory usage với solution cụ thể"""
    print(f"\n{'='*50}")
    print(f"🧪 Testing solution: {solution_name}")
    print(f"{'='*50}")
    
    # Clear memory trước test
    clear_memory()
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        
        # Cấu hình theo solution
        if solution_name == "ultra_light":
            settings = {
                'MAX_LENGTH': 64,
                'LORA_R': 2,
                'LORA_ALPHA': 8,
            }
        elif solution_name == "light":
            settings = {
                'MAX_LENGTH': 128,
                'LORA_R': 4,
                'LORA_ALPHA': 16,
            }
        elif solution_name == "original":
            settings = {
                'MAX_LENGTH': 256,
                'LORA_R': 8,
                'LORA_ALPHA': 32,
            }
        else:
            settings = {}
        
        print(f"📊 Settings: {settings}")
        
        # Memory trước khi load model
        mem_before = get_memory_info()
        if mem_before:
            print(f"💾 Memory before model load: {mem_before['allocated']:.2f}GB")
        
        # Load model
        MODEL_NAME = 'meta-llama/Llama-3.2-1B-Instruct'
        
        # Quantization config
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_group_size=128,
        )
        
        # Load model
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map='auto',
            quantization_config=quant_config,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        base_model = prepare_model_for_kbit_training(base_model)
        
        # Memory sau khi load base model
        mem_after_base = get_memory_info()
        if mem_after_base:
            print(f"💾 Memory after base model: {mem_after_base['allocated']:.2f}GB")
        
        # LoRA config
        peft_config = LoraConfig(
            r=settings.get('LORA_R', 8),
            lora_alpha=settings.get('LORA_ALPHA', 32),
            target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj'],
            lora_dropout=0.1,
            bias='none',
            task_type='CAUSAL_LM'
        )
        model = get_peft_model(base_model, peft_config)
        
        # Memory sau khi load LoRA
        mem_after_lora = get_memory_info()
        if mem_after_lora:
            print(f"💾 Memory after LoRA: {mem_after_lora['allocated']:.2f}GB")
        
        # Test với một batch nhỏ
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Tạo dummy data
        dummy_text = "### Instruction:\nTest instruction\n\n### Input:\nTest input\n\n### Response:\nTest response"
        inputs = tokenizer(
            dummy_text, 
            truncation=True, 
            padding='max_length', 
            max_length=settings.get('MAX_LENGTH', 256), 
            return_tensors='pt'
        )
        
        # Test forward pass
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Memory sau forward pass
        mem_after_forward = get_memory_info()
        if mem_after_forward:
            print(f"💾 Memory after forward pass: {mem_after_forward['allocated']:.2f}GB")
        
        # Tính toán memory usage
        if mem_before and mem_after_forward:
            total_used = mem_after_forward['allocated'] - mem_before['allocated']
            print(f"📈 Total memory used: {total_used:.2f}GB")
            print(f"📈 Peak memory: {mem_after_forward['allocated']:.2f}GB")
            print(f"📈 Free memory: {mem_after_forward['free']:.2f}GB")
        
        # Đánh giá
        if mem_after_forward and mem_after_forward['free'] > 1.0:
            print("✅ Solution PASSED - Có đủ memory để train")
            return True
        else:
            print("❌ Solution FAILED - Không đủ memory")
            return False
            
    except Exception as e:
        print(f"❌ Error testing solution {solution_name}: {e}")
        return False
    finally:
        clear_memory()

def main():
    """Test tất cả các solutions"""
    print("🔧 Testing Memory Solutions for OutOfMemoryError")
    print("="*60)
    
    # Test các solutions
    solutions = [
        ("ultra_light", "Ultra Light Settings"),
        ("light", "Light Settings"),
        ("original", "Original Settings")
    ]
    
    results = {}
    
    for solution_name, description in solutions:
        print(f"\n🎯 Testing: {description}")
        success = test_memory_usage(solution_name)
        results[solution_name] = success
        
        if success:
            print(f"✅ {solution_name} PASSED")
        else:
            print(f"❌ {solution_name} FAILED")
    
    # Tổng kết
    print(f"\n{'='*60}")
    print("📊 TEST RESULTS SUMMARY")
    print(f"{'='*60}")
    
    for solution_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{solution_name:15} : {status}")
    
    # Khuyến nghị
    print(f"\n💡 RECOMMENDATIONS:")
    if results.get("ultra_light", False):
        print("✅ Use ULTRA_LIGHT solution - Best for 6GB GPU")
    elif results.get("light", False):
        print("✅ Use LIGHT solution - Good for 8GB GPU")
    elif results.get("original", False):
        print("✅ Use ORIGINAL settings - Sufficient memory")
    else:
        print("❌ All solutions failed - Consider:")
        print("   - Using smaller model")
        print("   - Reducing dataset size")
        print("   - Using CPU training")
        print("   - Upgrading GPU")

if __name__ == "__main__":
    main() 