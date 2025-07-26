#!/usr/bin/env python3
"""
Model Inference Server cho Fine-tuned LLaMA
Server riêng biệt để serve model đã fine-tune
"""

import os
import sys
import logging
import torch
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional, List
import uvicorn
import re
import unicodedata

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

# Import model components
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from services.gating_router.prompt_builder import build_prompt_from_object

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LLaMA Model Inference Server", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
tokenizer = None
device = None

class InferenceRequest(BaseModel):
    prompt: dict
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True

class InferenceResponse(BaseModel):
    response: str
    model_name: str
    inference_time: float
    tokens_generated: int

def load_model():
    """Load fine-tuned model"""
    global model, tokenizer, device
    
    try:
        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Model paths
        base_model_path = "models/weights/base_model/meta-llama/Llama-3.2-1B-Instruct"
        checkpoint_path = "models/weights/chatbot_finetuned/checkpoint-1098"
        
        # Check if checkpoint exists
        if os.path.exists(checkpoint_path):
            logger.info(f"Loading checkpoint from: {checkpoint_path}")
            model_path = base_model_path
            adapter_path = checkpoint_path
        else:
            logger.warning(f"Checkpoint not found at {checkpoint_path}, using base model")
            model_path = base_model_path
            adapter_path = None
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(adapter_path if adapter_path else model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        model = model.to(device)
        
        # If using base model, try to load LoRA adapter
        if adapter_path and os.path.exists(adapter_path):
            try:
                logger.info("Loading LoRA adapter...")
                model = PeftModel.from_pretrained(model, adapter_path)
                logger.info("LoRA adapter loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load LoRA adapter: {e}")
        
        model.eval()
        logger.info("Model loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def generate_response(prompt: dict, max_length: int = 512, temperature: float = 0.7, 
                     top_p: float = 0.9, do_sample: bool = True) -> str:
    """Generate response using the model"""
    
    try:
        if tokenizer is None or model is None:
            raise RuntimeError("Model and tokenizer must be loaded before inference. Call load_model() first.")
        
        # Chỉ hỗ trợ prompt dict cho production API
        if not isinstance(prompt, dict):
            raise ValueError("Prompt phải là dict object cho production API")
        
        formatted_prompt = build_prompt_from_object(prompt, include_template=True)
        # Tokenize
        inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, 
                          max_length=max_length, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the response part
        if "### Response:" in response:
            response = response.split("### Response:")[-1].strip()
        # Làm sạch token đặc biệt
        response = clean_response(response)
        return response
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return "Xin lỗi, tôi đang gặp sự cố kỹ thuật. Vui lòng thử lại sau."

def clean_response(text: str) -> str:
    """
    Làm sạch phản hồi từ mô hình sinh, loại bỏ token đặc biệt và ký tự không mong muốn.
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Loại bỏ MỌI token dạng <|...|> (bao gồm cả multiline) - ưu tiên xử lý trước
    text = re.sub(r"<\|[^|]*?\|>", "", text, flags=re.DOTALL | re.MULTILINE)
    
    # 2. Loại bỏ token đặc biệt với các pattern cụ thể
    special_tokens = [
        r"<\|closuresnippet\|>",   # closure snippet token
        r"<\|fim\|>",              # fim token (thêm vào)
        r"<\|fim_system\|>",       # fim system token
        r"<\|fim_user\|>",         # fim user token  
        r"<\|fim_assistant\|>",    # fim assistant token
        r"<\|fim_[^|]*?\|>",       # các fim tokens khác
        r"<\|end[^|]*?\|>",        # end tokens
        r"<\|start[^|]*?\|>",      # start tokens
        r"<\|eot_id\|>",           # end of turn token
        r"<\|begin_of_text\|>",    # begin text token
        r"<\|end_of_text\|>",      # end text token
        r"</?s>",                  # sentence tokens
        r"<unk>",                  # unknown tokens
        r"<pad>",                  # padding tokens
        r"<mask>",                 # mask tokens
    ]
    
    for pattern in special_tokens:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)
    
    # 3. Loại bỏ MỌI token dạng <...> (bất kể nội dung, bao gồm self-closing tags)
    text = re.sub(r"<[^<>]*?/?>\s*", "", text, flags=re.DOTALL | re.MULTILINE)
    
    # 4. Loại các từ đặc biệt bị sót lại (mở rộng danh sách)
    blacklist_words = [
        "closuresnippet", "startoftext", "endoftext", "endofprompt", 
        "startofresponse", "assistant", "user", "system", "human",
        "cách thức trả lời", "cho phép", "Yes/No", "fim_system",
        "fim_prefix", "fim_middle", "fim_suffix", "eot_id", "start_header_id",
        "end_header_id", "begin", "end", "instruction", "response"
    ]
    
    # Tạo pattern với word boundaries và case insensitive
    pattern = r"\b(" + "|".join(re.escape(word) for word in blacklist_words) + r")\b"
    text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    
    # 5. Loại bỏ các ký tự Unicode vô hình và điều khiển
    invisible_chars = [
        r"[\u200b\u200c\u200d\u200e\u200f]",  # Zero-width chars
        r"[\ufeff\ufffe\uffff]",              # BOM chars
        r"[\u00a0\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]",  # Various spaces
        r"[\u0000-\u001f\u007f-\u009f]",      # Control characters
    ]
    
    for char_pattern in invisible_chars:
        text = re.sub(char_pattern, "", text)
    
    # 6. Xử lý đặc biệt cho token ở cuối text (thường gặp nhất)
    # Loại bỏ token cuối câu/đoạn trước khi xử lý dòng
    text = re.sub(r"\s*<\|[^|]*?\|>\s*$", "", text, flags=re.MULTILINE | re.DOTALL)
    text = re.sub(r"\s*<[^<>]*?>\s*$", "", text, flags=re.MULTILINE | re.DOTALL)
    
    # Xử lý token ở giữa text
    text = re.sub(r"\s*<\|[^|]*?\|>\s*", " ", text, flags=re.DOTALL)
    text = re.sub(r"\s*<[^<>]*?>\s*", " ", text, flags=re.DOTALL)
    
    # 7. Loại bỏ các ký tự lạ còn sót lại (non-printable characters)
    text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C' or char in '\n\t\r ')
    
    # 8. Chuẩn hóa khoảng trắng
    text = re.sub(r'\s+', ' ', text)  # Nhiều space thành 1 space
    text = re.sub(r'\n\s*\n', '\n', text)  # Nhiều newline thành 1 newline
    
    # 9. Xóa dòng trống dư thừa và trim
    lines = []
    for line in text.splitlines():
        line = line.strip()
        # Chỉ giữ lại dòng có nội dung có nghĩa
        if line and not re.match(r'^[\s\W]*$', line):
            lines.append(line)
    
    # 10. Aggressive final cleanup - loại bỏ TOÀN BỘ token còn sót lại
    result = "\n".join(lines).strip()
    
    # Multiple passes để đảm bảo loại bỏ hết token
    for _ in range(3):  # Lặp nhiều lần để bắt token lồng nhau
        # Loại bỏ token với pipe
        result = re.sub(r'<\|[^|]*?\|>', '', result, flags=re.DOTALL)
        # Loại bỏ token HTML-style  
        result = re.sub(r'<[^<>]*?>', '', result, flags=re.DOTALL)
        # Loại bỏ token ở đầu/cuối với whitespace
        result = re.sub(r'^\s*<[^>]*>\s*', '', result, flags=re.MULTILINE)
        result = re.sub(r'\s*<[^>]*>\s*$', '', result, flags=re.MULTILINE)
    
    # Xử lý đặc biệt cho fim token ở cuối
    result = re.sub(r'\s*<\|fim\|>\s*$', '', result, flags=re.MULTILINE | re.DOTALL)
    result = re.sub(r'\s*<\|fim_user\|>\s*$', '', result, flags=re.MULTILINE | re.DOTALL)
    result = re.sub(r'\s*<\|fim_system\|>\s*$', '', result, flags=re.MULTILINE | re.DOTALL)
    result = re.sub(r'\s*<\|fim_assistant\|>\s*$', '', result, flags=re.MULTILINE | re.DOTALL)
    
    return result


# Hàm bổ sung để kiểm tra chất lượng text sau khi clean
def validate_cleaned_text(text: str) -> bool:
    """
    Kiểm tra xem text đã được clean có còn ký tự lạ không
    """
    if not text:
        return False
    
    # Kiểm tra các pattern không mong muốn
    unwanted_patterns = [
        r'<[^>]*>',           # HTML/XML tags
        r'<\|[^|]*\|>',       # Special tokens
        r'\b(assistant|user|system)\b',  # Role tokens
        r'[\u200b-\u200f]',   # Zero-width chars
    ]
    
    for pattern in unwanted_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return False
    
    return True


# Test function
def test_clean_response():
    """
    Test cases để kiểm tra hàm clean_response
    """
    test_cases = [
        "Xin chào <|fim_system|>",
        "Đây là câu trả lời <s> với token </s>",
        "Text với user assistant system",
        "Nội dung bình thường không có token lạ",
        "<|start|>Content<|end|>",
        "Text\u200bwith\u200einvisible\u200fchars",
        "Hy vọng những kỹ thuật này sẽ giúp bạn! <|fim_user|>",  # Case thực tế
        "Response content <|fim_assistant|> more content",
        "Final answer <|eot_id|>",
    ]
    
    for i, test in enumerate(test_cases):
        cleaned = clean_response(test)
        is_valid = validate_cleaned_text(cleaned)
        print(f"Test {i+1}: {'✓' if is_valid else '✗'}")
        print(f"Input: {repr(test)}")
        print(f"Output: {repr(cleaned)}")
        print(f"Valid: {is_valid}\n")

# Uncomment để test
# test_clean_response()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        return {
            "status": "healthy",
            "model_loaded": model is not None,
            "device": str(device) if device else None,
            "model_name": "llama-3.2-1b-finetuned" if model else None
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Service unavailable")

@app.post("/generate", response_model=InferenceResponse)
async def generate_endpoint(request: InferenceRequest):
    """Generate response endpoint"""
    try:
        import time
        start_time = time.time()
        
        logger.info(f"Received generation request: {request.prompt[:50]}...")
        
        # Generate response
        response = generate_response(
            prompt=request.prompt,
            max_length=request.max_length,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=request.do_sample
        )
        
        inference_time = time.time() - start_time
        
        if tokenizer is None:
            raise RuntimeError("Tokenizer must be loaded before counting tokens. Call load_model() first.")
        tokens_generated = len(tokenizer.encode(response))
        
        logger.info(f"Generated response in {inference_time:.2f}s, {tokens_generated} tokens")
        
        return InferenceResponse(
            response=response,
            model_name="llama-3.2-1b-finetuned",
            inference_time=inference_time,
            tokens_generated=tokens_generated
        )
        
    except Exception as e:
        logger.error(f"Error in generate endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.get("/model-info")
async def model_info():
    """Get model information"""
    try:
        return {
            "model_name": "llama-3.2-1b-finetuned",
            "base_model": "meta-llama/Llama-3.2-1B-Instruct",
            "fine_tuned": True,
            "device": str(device) if device else None,
            "parameters": sum(p.numel() for p in model.parameters()) if model else 0,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad) if model else 0
        }
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail="Failed to get model info")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info") 