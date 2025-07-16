#!/usr/bin/env python3
"""
Script để merge checkpoint fine-tune thành final model
"""

import os
import sys
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def merge_checkpoint_to_final_model(
    base_model_path: str,
    checkpoint_path: str,
    output_path: str,
    checkpoint_name: str = "checkpoint-1098"
):
    """
    Merge checkpoint thành final model
    
    Args:
        base_model_path: Đường dẫn đến base model
        checkpoint_path: Đường dẫn đến thư mục checkpoint
        output_path: Đường dẫn output cho final model
        checkpoint_name: Tên checkpoint cụ thể (checkpoint-549 hoặc checkpoint-1098)
    """
    
    try:
        logger.info(f"🔄 Starting merge process...")
        logger.info(f"Base model: {base_model_path}")
        logger.info(f"Checkpoint: {checkpoint_path}/{checkpoint_name}")
        logger.info(f"Output: {output_path}")
        
        # Tạo thư mục output
        os.makedirs(output_path, exist_ok=True)
        
        # Load base model
        logger.info("Loading base model...")
        
        # Kiểm tra GPU availability
        if torch.cuda.is_available():
            # GPU quantization config
            from transformers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                device_map="auto",
                quantization_config=quant_config,
                trust_remote_code=True
            )
        else:
            # CPU - không sử dụng quantization
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load LoRA adapter từ checkpoint
        checkpoint_full_path = os.path.join(checkpoint_path, checkpoint_name)
        logger.info(f"Loading LoRA adapter from: {checkpoint_full_path}")
        
        if not os.path.exists(checkpoint_full_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_full_path}")
        
        # Load LoRA adapter
        model = PeftModel.from_pretrained(base_model, checkpoint_full_path)
        
        # Merge LoRA weights vào base model
        logger.info("Merging LoRA weights...")
        merged_model = model.merge_and_unload()
        
        # Save merged model
        logger.info(f"Saving merged model to: {output_path}")
        merged_model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        
        # Copy các file cần thiết khác
        import shutil
        files_to_copy = [
            "adapter_config.json",
            "chat_template.jinja",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "vocab.json"
        ]
        
        for file_name in files_to_copy:
            src_path = os.path.join(checkpoint_full_path, file_name)
            dst_path = os.path.join(output_path, file_name)
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
                logger.info(f"Copied {file_name}")
        
        logger.info("✅ Merge completed successfully!")
        logger.info(f"Final model saved to: {output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error during merge: {e}")
        return False

def list_available_checkpoints(checkpoint_path: str):
    """Liệt kê các checkpoint có sẵn"""
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint path not found: {checkpoint_path}")
        return []
    
    checkpoints = []
    for item in os.listdir(checkpoint_path):
        item_path = os.path.join(checkpoint_path, item)
        if os.path.isdir(item_path) and item.startswith("checkpoint-"):
            checkpoints.append(item)
    
    return sorted(checkpoints)

def main():
    parser = argparse.ArgumentParser(description="Merge checkpoint to final model")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.2-1B-Instruct",
                       help="Base model path")
    parser.add_argument("--checkpoint_path", type=str, default="models/weights/chatbot_finetuned",
                       help="Path to checkpoint directory")
    parser.add_argument("--output_path", type=str, default="models/weights/chatbot_finetuned/checkpoint-1098",
                       help="Output path for final model")
    parser.add_argument("--checkpoint_name", type=str, default="checkpoint-10",
                       help="Specific checkpoint name (checkpoint-549 or checkpoint-1098)")
    parser.add_argument("--list_checkpoints", action="store_true",
                       help="List available checkpoints")
    
    args = parser.parse_args()
    
    try:
        if args.list_checkpoints:
            checkpoints = list_available_checkpoints(args.checkpoint_path)
            if checkpoints:
                logger.info("Available checkpoints:")
                for cp in checkpoints:
                    logger.info(f"  - {cp}")
            else:
                logger.warning("No checkpoints found")
            return
        
        # Merge checkpoint
        success = merge_checkpoint_to_final_model(
            base_model_path=args.base_model,
            checkpoint_path=args.checkpoint_path,
            output_path=args.output_path,
            checkpoint_name=args.checkpoint_name
        )
        
        if success:
            logger.info("🎉 Merge process completed successfully!")
            logger.info(f"You can now use the model at: {args.output_path}")
        else:
            logger.error("❌ Merge process failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 