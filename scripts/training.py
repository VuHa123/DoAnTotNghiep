#!/usr/bin/env python3
"""
Comprehensive Training Script for Mental Health Chatbot
Combines functionality from fixed_training_script.py and fix_and_train.py
"""

import os
import sys
import logging
import torch
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_environment():
    """Kiểm tra môi trường và dependencies"""
    logger.info("🔍 Checking environment...")
    
    # Kiểm tra GPU
    if torch.cuda.is_available():
        logger.info(f"✅ GPU available: {torch.cuda.get_device_name()}")
        logger.info(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        logger.warning("⚠️ No GPU available, training will be slow")
    
    # Kiểm tra dependencies
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, DataCollatorForLanguageModeling, EarlyStoppingCallback
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from trl import SFTTrainer
        from datasets import load_dataset
        logger.info("✅ All required packages available")
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        return False
    
    return True

def train_chatbot_model():
    """Train chatbot model with LoRA fine-tuning"""
    logger.info("🚀 Starting chatbot model training...")
    
    # Cấu hình
    MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
    DATA_PATH = "Dataset/llama_instruction_data.jsonl"
    OUTPUT_DIR = "models/weights/chatbot_finetuned"
    BATCH_SIZE = 1
    GRAD_ACC = 32
    EPOCHS = 3
    LEARNING_RATE = 2e-4
    MAX_LENGTH = 384
    WARMUP_STEPS = 100
    PATIENCE = 3
    VALIDATION_SPLIT = 0.1
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        # Load dataset
        raw_dataset = load_dataset("json", data_files=DATA_PATH, split="train")
        logger.info(f"Loaded {len(raw_dataset)} samples")
        
        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Tokenize function
        def tokenize(example):
            prompt = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
            result = tokenizer(prompt, truncation=True, padding="max_length", max_length=MAX_LENGTH, return_tensors="pt")
            return {
                "input_ids": result["input_ids"].squeeze(0),
                "attention_mask": result["attention_mask"].squeeze(0),
                "labels": result["input_ids"].squeeze(0)
            }
        
        # Split dataset
        tokenized = raw_dataset.train_test_split(test_size=VALIDATION_SPLIT)
        train_dataset = tokenized["train"].map(tokenize, batched=False, remove_columns=raw_dataset.column_names)
        eval_dataset = tokenized["test"].map(tokenize, batched=False, remove_columns=raw_dataset.column_names)
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_4bit=True,
            trust_remote_code=True
        )
        base_model = prepare_model_for_kbit_training(base_model)
        
        # LoRA config
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        model = get_peft_model(base_model, peft_config)
        model.print_trainable_parameters()
        
        # Training args
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACC,
            num_train_epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            lr_scheduler_type="reduce_lr_on_plateau",
            warmup_steps=WARMUP_STEPS,
            optim="adamw_torch",
            weight_decay=0.01,
            save_strategy="epoch",
            logging_steps=500,
            eval_strategy="epoch",
            bf16=False,
            gradient_checkpointing=True,
            save_total_limit=2,
            report_to="none",
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False
        )
        
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        
        # Trainer
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)]
        )
        
        # Training
        logger.info("🚀 Start training...")
        trainer.train()
        
        # Save model
        final_model_path = os.path.join(OUTPUT_DIR, "final_model")
        trainer.save_model(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        logger.info(f"✅ Model saved to: {final_model_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        return False

def train_qlora_model():
    """Train model with QLoRA (Quantized LoRA)"""
    logger.info("🚀 Starting QLoRA training...")
    
    # Cấu hình cho QLoRA
    MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
    DATA_PATH = "Dataset/llama_instruction_data.jsonl"
    OUTPUT_DIR = "models/weights/qlora_finetuned"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        from transformers import BitsAndBytesConfig
        
        # Quantization config
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_group_size=128,
        )
        
        # Load model with quantization
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            quantization_config=quant_config,
            trust_remote_code=True
        )
        base_model = prepare_model_for_kbit_training(base_model)
        
        # LoRA config for QLoRA
        peft_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.3,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        model = get_peft_model(base_model, peft_config)
        model.print_trainable_parameters()
        
        logger.info("✅ QLoRA model setup completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ QLoRA training failed: {e}")
        return False

def test_model():
    """Test trained model"""
    logger.info("🧪 Testing trained model...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Load trained model
        model_path = "models/weights/chatbot_finetuned/final_model"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        
        # Test inference
        test_prompt = "### Instruction:\nHãy tư vấn cho tôi khi tôi cảm thấy lo lắng.\n\n### Input:\nTôi rất lo lắng về công việc.\n\n### Response:\n"
        
        inputs = tokenizer(test_prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=200, temperature=0.7)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        logger.info(f"✅ Model test successful")
        logger.info(f"Test response: {response}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model test failed: {e}")
        return False

def main():
    """Main training function"""
    logger.info("🎯 Starting comprehensive training pipeline")
    
    # Check environment
    if not check_environment():
        logger.error("❌ Environment check failed")
        return False
    
    # Train chatbot model
    if not train_chatbot_model():
        logger.error("❌ Chatbot training failed")
        return False
    
    # Train QLoRA model (optional)
    if not train_qlora_model():
        logger.warning("⚠️ QLoRA training failed, continuing...")
    
    # Test model
    if not test_model():
        logger.error("❌ Model testing failed")
        return False
    
    logger.info("🎉 Training pipeline completed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 