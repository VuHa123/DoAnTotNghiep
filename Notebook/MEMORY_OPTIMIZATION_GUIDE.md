# Hướng dẫn khắc phục OutOfMemoryError trong Training

## 🔍 Vấn đề
Lỗi `OutOfMemoryError: CUDA out of memory` xảy ra khi GPU hết bộ nhớ trong quá trình evaluation sau mỗi epoch. Đây là vấn đề phổ biến khi train model lớn trên GPU có bộ nhớ hạn chế (như RTX 3060 6GB).

## 🛠️ Các giải pháp

### 1. **Giải pháp Ultra Light** (Khuyến nghị đầu tiên)
```python
# Cài đặt cực nhẹ để tiết kiệm memory tối đa
settings = {
    'GRAD_ACC': 8,                    # Giảm từ 32 xuống 8
    'MAX_LENGTH': 64,                 # Giảm từ 256 xuống 64
    'VALIDATION_SPLIT': 0.02,         # Giảm validation set
    'EPOCHS': 2,                      # Giảm epochs
    'WARMUP_STEPS': 20,               # Giảm warmup
    'LORA_R': 2,                      # Giảm LoRA rank
    'LORA_ALPHA': 8,                  # Giảm LoRA alpha
    'EVAL_ACCUMULATION_STEPS': 32,    # Tăng eval accumulation
}
```

### 2. **Giải pháp Light**
```python
# Cài đặt nhẹ hơn original nhưng vẫn đảm bảo chất lượng
settings = {
    'GRAD_ACC': 16,                   # Giảm từ 32 xuống 16
    'MAX_LENGTH': 128,                # Giảm từ 256 xuống 128
    'VALIDATION_SPLIT': 0.05,         # Giảm validation set
    'EPOCHS': 3,                      # Giữ nguyên epochs
    'WARMUP_STEPS': 50,               # Giảm warmup
    'LORA_R': 4,                      # Giảm LoRA rank
    'LORA_ALPHA': 16,                 # Giảm LoRA alpha
    'EVAL_ACCUMULATION_STEPS': 16,    # Tăng eval accumulation
}
```

### 3. **Giải pháp No Evaluation**
```python
# Không eval trong training, chỉ train và save
settings = {
    'EVAL_STRATEGY': 'no',            # Không eval trong training
    'GRAD_ACC': 32,                   # Giữ nguyên
    'MAX_LENGTH': 256,                # Giữ nguyên
    'VALIDATION_SPLIT': 0.1,          # Giữ nguyên
    'EPOCHS': 3,                      # Giữ nguyên
    'WARMUP_STEPS': 100,              # Giữ nguyên
    'LORA_R': 8,                      # Giữ nguyên
    'LORA_ALPHA': 32,                 # Giữ nguyên
}
```

## 🔧 Các cài đặt tối ưu memory

### Environment Variables
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_NO_CUDA_MEMORY_CACHING=1
```

### Training Arguments
```python
training_args = TrainingArguments(
    # Memory optimization
    per_device_eval_batch_size=1,     # Batch size nhỏ cho eval
    eval_accumulation_steps=8,        # Accumulate eval steps
    prediction_loss_only=True,         # Chỉ tính loss, không lưu predictions
    dataloader_num_workers=0,         # Giảm workers
    dataloader_drop_last=True,        # Drop last batch
    dataloader_pin_memory=False,      # Không pin memory
    
    # Gradient optimization
    gradient_checkpointing=True,       # Tiết kiệm memory
    max_grad_norm=1.0,                # Clip gradients
    
    # Other optimizations
    bf16=False,                       # Không dùng bf16
    remove_unused_columns=False,      # Giữ columns
)
```

## 📊 Monitoring Memory

### Sử dụng Memory Monitor
```python
from memory_monitor import MemoryMonitor, clear_memory

# Tạo monitor
monitor = MemoryMonitor(interval=2.0)
monitor.start_monitoring()

try:
    # Training code here
    trainer.train()
finally:
    monitor.stop_monitoring()
    clear_memory()
```

### Manual Memory Check
```python
def get_memory_info():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free = total - reserved
        print(f"GPU: {allocated:.2f}GB allocated, {free:.2f}GB free / {total:.2f}GB total")
```

## 🚀 Cách sử dụng

### 1. Chạy script tự động
```bash
cd Notebook
python run_training_with_memory_fix.py
```

### 2. Chạy từng solution thủ công
```python
# Trong notebook
from run_training_with_memory_fix import run_training_with_solution

# Thử solution ultra_light
success = run_training_with_solution("ultra_light")
if not success:
    # Thử solution light
    success = run_training_with_solution("light")
```

### 3. Sử dụng notebook đã tối ưu
```python
# Chạy file chatbot_finetune_optimized.py
exec(open('chatbot_finetune_optimized.py').read())
```

## 📈 So sánh các giải pháp

| Giải pháp | Memory Usage | Training Speed | Model Quality | Khuyến nghị |
|-----------|--------------|----------------|---------------|-------------|
| Ultra Light | ~2-3GB | Nhanh | Cơ bản | ✅ Cho GPU 6GB |
| Light | ~3-4GB | Trung bình | Tốt | ✅ Cho GPU 8GB |
| No Eval | ~4-5GB | Nhanh | Tốt | ⚠️ Không có eval |

## 🔍 Debugging Tips

### 1. Kiểm tra memory trước training
```python
print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
```

### 2. Clear memory thường xuyên
```python
import gc
gc.collect()
torch.cuda.empty_cache()
```

### 3. Monitor trong quá trình training
```python
# Thêm vào training loop
if step % 100 == 0:
    print(f"Step {step}: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
```

## ⚠️ Lưu ý quan trọng

1. **Luôn clear memory** trước khi bắt đầu training mới
2. **Monitor memory** trong quá trình training
3. **Thử từng solution** theo thứ tự: ultra_light → light → no_eval
4. **Backup model** sau mỗi epoch thành công
5. **Sử dụng gradient checkpointing** để tiết kiệm memory

## 🎯 Kết quả mong đợi

- **Ultra Light**: Training thành công với memory ~2-3GB
- **Light**: Training thành công với memory ~3-4GB  
- **No Eval**: Training thành công với memory ~4-5GB

## 📞 Hỗ trợ

Nếu vẫn gặp lỗi, hãy:
1. Kiểm tra GPU memory usage
2. Thử giảm thêm các parameters
3. Sử dụng model nhỏ hơn
4. Chia nhỏ dataset 