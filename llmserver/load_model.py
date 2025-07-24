# load_model.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
def load_model_tokenizer(model_path: str):
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cuda"
    # type=torch.float16 if torch.cuda.is_available() else torch.float32
    type = torch.float16
    config = PeftConfig.from_pretrained(model_path)
    base_model_path = config.base_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=type).to(device)
    # Áp dụng adapter PEFT vào model
    model = PeftModel.from_pretrained(model, model_path)
    model = model.to(device)
    
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, device
