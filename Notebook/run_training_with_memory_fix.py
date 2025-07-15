#!/usr/bin/env python3
"""
Script để chạy training với các giải pháp khắc phục OutOfMemoryError
"""

import os
import sys
import gc
import torch
from pathlib import Path

# Thêm thư mục gốc vào path
sys.path.append(str(Path(__file__).parent.parent))

def setup_memory_optimization():
    """Thiết lập các environment variables để tối ưu memory"""
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'
    print("⚙️ Memory optimization settings applied!")

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

def run_training_with_solution(solution_name, **kwargs):
    """Chạy training với solution cụ thể"""
    print(f"\n{'='*50}")
    print(f"🚀 Running training with solution: {solution_name}")
    print(f"{'='*50}")
    
    # Clear memory trước khi bắt đầu
    clear_memory()
    
    # Import và chạy training
    try:
        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
            TrainingArguments,
            DataCollatorForLanguageModeling,
            BitsAndBytesConfig,
            EarlyStoppingCallback
        )
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from trl import SFTTrainer
        from datasets import load_dataset
        
        # Cấu hình cơ bản
        MODEL_NAME = 'meta-llama/Llama-3.2-1B-Instruct'
        DATA_PATH = '../Dataset/llama_instruction_data.jsonl'
        OUTPUT_DIR = f'../models/weights/chatbot_finetuned_{solution_name}'
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Áp dụng solution-specific settings
        if solution_name == "ultra_light":
            # Solution 1: Ultra light settings
            settings = {
                'GRAD_ACC': 8,
                'MAX_LENGTH': 64,
                'VALIDATION_SPLIT': 0.02,
                'EPOCHS': 2,
                'WARMUP_STEPS': 20,
                'LORA_R': 2,
                'LORA_ALPHA': 8,
                'EVAL_ACCUMULATION_STEPS': 32,
                'PER_DEVICE_EVAL_BATCH_SIZE': 1,
            }
        elif solution_name == "light":
            # Solution 2: Light settings
            settings = {
                'GRAD_ACC': 16,
                'MAX_LENGTH': 128,
                'VALIDATION_SPLIT': 0.05,
                'EPOCHS': 3,
                'WARMUP_STEPS': 50,
                'LORA_R': 4,
                'LORA_ALPHA': 16,
                'EVAL_ACCUMULATION_STEPS': 16,
                'PER_DEVICE_EVAL_BATCH_SIZE': 1,
            }
        elif solution_name == "no_eval":
            # Solution 3: No evaluation during training
            settings = {
                'GRAD_ACC': 32,
                'MAX_LENGTH': 256,
                'VALIDATION_SPLIT': 0.1,
                'EPOCHS': 3,
                'WARMUP_STEPS': 100,
                'LORA_R': 8,
                'LORA_ALPHA': 32,
                'EVAL_ACCUMULATION_STEPS': 1,
                'PER_DEVICE_EVAL_BATCH_SIZE': 1,
                'EVAL_STRATEGY': 'no',  # Không eval trong training
            }
        else:
            # Default settings
            settings = kwargs
        
        print(f"📊 Using settings: {settings}")
        
        # Load dataset
        raw_dataset = load_dataset('json', data_files=DATA_PATH, split='train')
        print(f'📈 Loaded samples: {len(raw_dataset)}')
        
        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        def tokenize(example, max_length=settings.get('MAX_LENGTH', 256)):
            prompt = f'### Instruction:\n{example["instruction"]}\n\n### Input:\n{example["input"]}\n\n### Response:\n{example["output"]}'
            result = tokenizer(prompt, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')
            return {
                'input_ids': result['input_ids'].squeeze(0),
                'attention_mask': result['attention_mask'].squeeze(0),
                'labels': result['input_ids'].squeeze(0)
            }
        
        # Split dataset
        tokenized = raw_dataset.train_test_split(test_size=settings.get('VALIDATION_SPLIT', 0.1))
        train_dataset = tokenized['train'].map(
            lambda x: tokenize(x, max_length=settings.get('MAX_LENGTH', 256)), 
            batched=False, 
            remove_columns=raw_dataset.column_names
        )
        eval_dataset = tokenized['test'].map(
            lambda x: tokenize(x, max_length=settings.get('MAX_LENGTH', 256)), 
            batched=False, 
            remove_columns=raw_dataset.column_names
        )
        
        # Clear memory
        del raw_dataset, tokenized
        clear_memory()
        
        # Model quantization
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
        model.print_trainable_parameters()
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=settings.get('PER_DEVICE_EVAL_BATCH_SIZE', 1),
            gradient_accumulation_steps=settings.get('GRAD_ACC', 32),
            num_train_epochs=settings.get('EPOCHS', 3),
            learning_rate=2e-4,
            lr_scheduler_type='reduce_lr_on_plateau',
            warmup_steps=settings.get('WARMUP_STEPS', 100),
            optim='adamw_torch',
            weight_decay=0.01,
            save_strategy='epoch',
            logging_steps=500,
            eval_strategy=settings.get('EVAL_STRATEGY', 'epoch'),
            bf16=False,
            gradient_checkpointing=True,
            save_total_limit=1,
            report_to='none',
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            load_best_model_at_end=True,
            metric_for_best_model='eval_loss',
            greater_is_better=False,
            dataloader_drop_last=True,
            prediction_loss_only=True,
            eval_accumulation_steps=settings.get('EVAL_ACCUMULATION_STEPS', 4),
            max_grad_norm=1.0,
            logging_first_step=True,
        )
        
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        
        def compute_metrics(eval_pred):
            loss = eval_pred.loss
            return {'eval_loss': loss}
        
        # Trainer
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset if settings.get('EVAL_STRATEGY', 'epoch') != 'no' else None,
            args=training_args,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Memory info trước training
        mem_info = get_memory_info()
        if mem_info:
            print(f"💾 GPU Memory before training: {mem_info['allocated']:.2f}GB allocated, {mem_info['free']:.2f}GB free")
        
        # Training
        print("🚀 Starting training...")
        trainer.train()
        
        # Save model
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"✅ Model saved to: {OUTPUT_DIR}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed with solution {solution_name}: {e}")
        return False

def main():
    """Main function để chạy các solutions khác nhau"""
    setup_memory_optimization()
    
    print("🔧 Available solutions for OutOfMemoryError:")
    print("1. ultra_light - Cài đặt cực nhẹ")
    print("2. light - Cài đặt nhẹ")
    print("3. no_eval - Không eval trong training")
    
    # Chạy từng solution
    solutions = [
        ("ultra_light", "Ultra Light Settings"),
        ("light", "Light Settings"), 
        ("no_eval", "No Evaluation During Training")
    ]
    
    for solution_name, description in solutions:
        print(f"\n{'='*60}")
        print(f"🎯 Trying solution: {description}")
        print(f"{'='*60}")
        
        success = run_training_with_solution(solution_name)
        
        if success:
            print(f"✅ Solution '{solution_name}' completed successfully!")
            break
        else:
            print(f"❌ Solution '{solution_name}' failed, trying next...")
            clear_memory()
    
    print("\n🎉 Training process completed!")

if __name__ == "__main__":
    main() 