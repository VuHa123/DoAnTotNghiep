#!/usr/bin/env python3
"""
Fine-tuning script cho Mental Health Chatbot
Sử dụng QLoRA với model open source
"""

import os
import sys
import logging
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_device():
    """Setup device và kiểm tra GPU"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    return device

def load_data(data_path):
    """Load và kiểm tra dataset"""
    try:
        dataset = load_dataset("json", data_files=data_path, split="train")
        logger.info(f"Dataset loaded successfully. Number of samples: {len(dataset)}")
        
        # Kiểm tra cấu trúc dữ liệu
        sample = dataset[0]
        logger.info(f"Sample data structure: {sample.keys()}")
        
        return dataset
        
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

def load_tokenizer(model_name):
    """Load tokenizer với error handling"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        
        # Đảm bảo có pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info(f"Tokenizer loaded successfully. Vocab size: {tokenizer.vocab_size}")
        
        return tokenizer
        
    except Exception as e:
        logger.error(f"Error loading tokenizer: {e}")
        raise

def tokenize_dataset(dataset, tokenizer, max_length=512):
    """Tokenize dataset với format mental health chatbot"""
    def tokenize_function(example):
        try:
            # Format cho mental health conversation
            if 'instruction' in example and 'input' in example and 'output' in example:
                prompt = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
            else:
                # Fallback format
                prompt = str(example)
            
            # Tokenize với padding và truncation
            result = tokenizer(
                prompt, 
                truncation=True, 
                padding="max_length", 
                max_length=max_length,
                return_tensors="pt"
            )
            
            return {
                "input_ids": result["input_ids"].flatten(),
                "attention_mask": result["attention_mask"].flatten(),
                "labels": result["input_ids"].flatten()
            }
            
        except Exception as e:
            logger.error(f"Error tokenizing example: {e}")
            return None

    # Tokenize dataset
    tokenized_dataset = dataset.map(
        tokenize_function, 
        remove_columns=dataset.column_names, 
        batched=False
    )
    
    logger.info(f"Tokenization completed. Dataset size: {len(tokenized_dataset)}")
    return tokenized_dataset

def load_model(model_name, device):
    """Load model với QLoRA"""
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_4bit=True,
            trust_remote_code=True
        )
        
        # Prepare for k-bit training
        model = prepare_model_for_kbit_training(model)
        
        # QLoRA configuration tối ưu cho mental health chatbot
        peft_config = LoraConfig(
            r=16,                    # Rank
            lora_alpha=32,           # Alpha parameter
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.1,        # Dropout
            bias="none",
            task_type="CAUSAL_LM",
            inference_mode=False
        )
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
        logger.info("Model loaded and QLoRA applied successfully")
        
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def setup_trainer(model, tokenized_dataset, tokenizer, output_dir, args):
    """Setup trainer với configuration tối ưu"""
    training_args = TrainingArguments(
        output_dir=output_dir,
        
        # Batch size configuration
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        
        # Training configuration
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        
        # Optimization
        optim="adamw_torch",
        weight_decay=0.01,
        
        # Saving and logging
        save_strategy="steps",
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        eval_strategy="no",
        
        # Mixed precision
        bf16=True,
        
        # Memory optimization
        gradient_checkpointing=True,
        save_total_limit=2,
        
        # Monitoring
        report_to="wandb" if args.use_wandb else "none",
        run_name=args.run_name,
        
        # Other
        dataloader_pin_memory=False,
        remove_unused_columns=False
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False
    )
    
    # Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        max_seq_length=args.max_length,
    )
    
    logger.info("Trainer configured successfully")
    return trainer

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Mental Health Chatbot")
    
    # Model và data arguments
    parser.add_argument("--model_name", type=str, default="microsoft/DialoGPT-medium",
                       help="Model name to fine-tune")
    parser.add_argument("--data_path", type=str, default="../Dataset/llama_instruction_data.jsonl",
                       help="Path to training data")
    parser.add_argument("--output_dir", type=str, default="../models/weights/chatbot_finetuned",
                       help="Output directory for model")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=4, help="Per device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, 
                       help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    
    # Logging arguments
    parser.add_argument("--save_steps", type=int, default=500, help="Save steps")
    parser.add_argument("--logging_steps", type=int, default=50, help="Logging steps")
    parser.add_argument("--use_wandb", action="store_true", help="Use wandb for logging")
    parser.add_argument("--run_name", type=str, default="mental-health-chatbot-finetune",
                       help="Run name for wandb")
    
    args = parser.parse_args()
    
    try:
        # Setup
        device = setup_device()
        
        # Load data
        dataset = load_data(args.data_path)
        
        # Load tokenizer
        tokenizer = load_tokenizer(args.model_name)
        
        # Tokenize dataset
        tokenized_dataset = tokenize_dataset(dataset, tokenizer, args.max_length)
        
        # Load model
        model = load_model(args.model_name, device)
        
        # Setup trainer
        trainer = setup_trainer(model, tokenized_dataset, tokenizer, args.output_dir, args)
        
        # Training
        logger.info("Starting training...")
        trainer.train()
        
        # Save model
        final_model_path = f"{args.output_dir}/final_model"
        trainer.save_model(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        
        logger.info(f"Training completed successfully. Model saved to {final_model_path}")
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 