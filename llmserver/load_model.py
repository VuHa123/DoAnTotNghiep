# load_model.py
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

def load_model_tokenizer(model_path: str):
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cuda"
    # type=torch.float16 if torch.cuda.is_available() else torch.float32
    type = torch.float16
    
    # Lấy HF_TOKEN từ environment variable
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable is required for loading models from HuggingFace")
    
    print(f"Loading model: {model_path}")
    print(f"Using HF token: {hf_token[:10]}...")
    
    config = PeftConfig.from_pretrained(model_path, token=hf_token)
    base_model_path = config.base_model_name_or_path
    print(f"Base model: {base_model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, 
        torch_dtype=type,
        token=hf_token
    ).to(device)
    
    # Áp dụng adapter PEFT vào model
    model = PeftModel.from_pretrained(model, model_path, token=hf_token)
    model = model.to(device)
    
    # Sử dụng pad token khác với eos token để tránh cảnh báo
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        # Thêm pad token vào vocab nếu chưa có
        if tokenizer.pad_token not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))
    
    print(f"Model loaded successfully on device: {device}")
    return model, tokenizer, device
