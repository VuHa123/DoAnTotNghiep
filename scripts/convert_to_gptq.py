#!/usr/bin/env python3
"""
Script để convert model fine-tuned từ LoRA adapter sang GPTQ format cho ExLlama
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import argparse

def convert_lora_to_gptq(
    base_model_name: str,
    lora_adapter_path: str,
    output_path: str,
    bits: int = 4,
    group_size: int = 128,
    desc_act: bool = True
):
    """
    Convert LoRA adapter sang GPTQ format
    
    Args:
        base_model_name: Tên base model (ví dụ: meta-llama/Llama-3.2-1B-Instruct)
        lora_adapter_path: Đường dẫn đến LoRA adapter đã fine-tune
        output_path: Đường dẫn output cho GPTQ model
        bits: Số bit quantization (4 hoặc 8)
        group_size: Group size cho quantization
        desc_act: Có sử dụng desc_act hay không
    """
    
    print(f"🔄 Loading base model: {base_model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"🔄 Loading LoRA adapter from: {lora_adapter_path}")
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    
    # Merge LoRA weights vào base model
    print("🔄 Merging LoRA weights...")
    model = model.merge_and_unload()
    
    # Tạo quantization config
    quantize_config = BaseQuantizeConfig(
        bits=bits,
        group_size=group_size,
        desc_act=desc_act
    )
    
    print(f"🔄 Converting to GPTQ format (bits={bits}, group_size={group_size})...")
    
    # Convert sang GPTQ
    gptq_model = AutoGPTQForCausalLM.from_pretrained(
        model,
        quantize_config=quantize_config,
        device_map="auto"
    )
    
    # Save model và tokenizer
    os.makedirs(output_path, exist_ok=True)
    gptq_model.save_quantized(output_path)
    tokenizer.save_pretrained(output_path)
    
    print(f"✅ Model đã được convert và lưu tại: {output_path}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Convert LoRA adapter sang GPTQ format")
    parser.add_argument("--base_model", type=str, required=True,
                       help="Tên base model (ví dụ: meta-llama/Llama-3.2-1B-Instruct)")
    parser.add_argument("--lora_path", type=str, required=True,
                       help="Đường dẫn đến LoRA adapter")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Đường dẫn output cho GPTQ model")
    parser.add_argument("--bits", type=int, default=4, choices=[4, 8],
                       help="Số bit quantization (4 hoặc 8)")
    parser.add_argument("--group_size", type=int, default=128,
                       help="Group size cho quantization")
    parser.add_argument("--desc_act", action="store_true", default=True,
                       help="Sử dụng desc_act cho quantization")
    
    args = parser.parse_args()
    
    convert_lora_to_gptq(
        base_model_name=args.base_model,
        lora_adapter_path=args.lora_path,
        output_path=args.output_path,
        bits=args.bits,
        group_size=args.group_size,
        desc_act=args.desc_act
    )

if __name__ == "__main__":
    main() 