# load_model.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model_tokenizer(model_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_auth_token=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32).to(device)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, device
