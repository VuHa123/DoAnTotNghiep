import torch
print('CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('VRAM:', torch.cuda.get_device_properties(0).total_memory / 1024**3, 'GB')

from datasets import load_dataset
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
import os
import logging
import gc

# Thiết lập environment variables để tối ưu memory
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 2. Cấu hình
MODEL_NAME = 'meta-llama/Llama-3.2-1B-Instruct'
DATA_PATH = '../Dataset/llama_instruction_data.jsonl'
OUTPUT_DIR = '../models/weights/chatbot_finetuned_nf4_optimized'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Giảm các tham số để tiết kiệm memory
GRAD_ACC = 16  # Giảm từ 32 xuống 16
EPOCHS = 3
LEARNING_RATE = 2e-4
MAX_LENGTH = 128  # Giảm từ 256 xuống 128
WARMUP_STEPS = 50  # Giảm từ 100 xuống 50
PATIENCE = 3
VALIDATION_SPLIT = 0.05  # Giảm validation set

# 3. Load dataset
raw_dataset = load_dataset('json', data_files=DATA_PATH, split='train')
print('Loaded samples:', len(raw_dataset))

# 4. Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize(example, max_length=MAX_LENGTH):
    prompt = f'### Instruction:\n{example["instruction"]}\n\n### Input:\n{example["input"]}\n\n### Response:\n{example["output"]}'
    result = tokenizer(prompt, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')
    return {
        'input_ids': result['input_ids'].squeeze(0),
        'attention_mask': result['attention_mask'].squeeze(0),
        'labels': result['input_ids'].squeeze(0)
    }

tokenized = raw_dataset.train_test_split(test_size=VALIDATION_SPLIT)
train_dataset = tokenized['train'].map(lambda x: tokenize(x, max_length=MAX_LENGTH), batched=False, remove_columns=raw_dataset.column_names)
eval_dataset = tokenized['test'].map(lambda x: tokenize(x, max_length=MAX_LENGTH), batched=False, remove_columns=raw_dataset.column_names)

# Clear memory
del raw_dataset, tokenized
gc.collect()
torch.cuda.empty_cache()

# 5. Quantization config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_group_size=128,
)

# 6. Load model 4-bit
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map='auto',
    quantization_config=quant_config,
    trust_remote_code=True,
    torch_dtype=torch.float16,  # Thêm dtype để tiết kiệm memory
)
base_model = prepare_model_for_kbit_training(base_model)

# 7. LoRA config - giảm rank để tiết kiệm memory
peft_config = LoraConfig(
    r=4,  # Giảm từ 8 xuống 4
    lora_alpha=16,  # Giảm từ 32 xuống 16
    target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj'],
    lora_dropout=0.1,  # Giảm dropout
    bias='none',
    task_type='CAUSAL_LM'
)
model = get_peft_model(base_model, peft_config)
model.print_trainable_parameters()

# 8. Training args với tối ưu memory
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,  # Batch size cho evaluation
    gradient_accumulation_steps=GRAD_ACC,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type='reduce_lr_on_plateau',
    warmup_steps=WARMUP_STEPS,
    optim='adamw_torch',
    weight_decay=0.01,
    save_strategy='epoch',
    logging_steps=500,
    eval_strategy='epoch',
    bf16=False,
    gradient_checkpointing=True,
    save_total_limit=1,
    report_to='none',
    dataloader_pin_memory=False,
    dataloader_num_workers=0,  # Giảm workers
    remove_unused_columns=False,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    greater_is_better=False,
    # Memory optimization settings
    dataloader_drop_last=True,
    prediction_loss_only=True,  # Chỉ tính loss, không lưu predictions
    eval_accumulation_steps=8,  # Accumulate eval steps
    # Thêm các settings để tránh OOM
    max_grad_norm=1.0,
    logging_first_step=True,
    logging_dir=os.path.join(OUTPUT_DIR, 'logs'),
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

def compute_metrics(eval_pred):
    loss = eval_pred.loss
    return {'eval_loss': loss}

# 9. Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)]
)

# 10. Clear memory trước khi train
gc.collect()
torch.cuda.empty_cache()

# 11. Training với memory monitoring
print('🚀 Start training...')
print(f'GPU Memory before training: {torch.cuda.memory_allocated() / 1024**3:.2f} GB')

try:
    trainer.train()
    print('✅ Training completed successfully!')
except Exception as e:
    print(f'❌ Training failed: {e}')
    # Clear memory và thử lại với settings khác
    gc.collect()
    torch.cuda.empty_cache()
    print('🔄 Trying with reduced settings...')
    
    # Thử với settings còn nhẹ hơn
    training_args.per_device_eval_batch_size = 1
    training_args.eval_accumulation_steps = 16
    training_args.gradient_accumulation_steps = 8
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)]
    )
    
    trainer.train()

# 12. Save model
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print('✅ Model saved to:', OUTPUT_DIR) 