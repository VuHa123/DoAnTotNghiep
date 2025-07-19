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
    prompt: str
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
        tokenizer = AutoTokenizer.from_pretrained(model_path)
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

def generate_response(prompt: str, max_length: int = 512, temperature: float = 0.7, 
                     top_p: float = 0.9, do_sample: bool = True) -> str:
    """Generate response using the model"""
    
    try:
        if tokenizer is None or model is None:
            raise RuntimeError("Model and tokenizer must be loaded before inference. Call load_model() first.")
        # Nếu prompt là dict (object), build lại prompt string
        if isinstance(prompt, dict):
            prompt_str = build_prompt_from_object(prompt)
        else:
            prompt_str = prompt
        # Format prompt for mental health chatbot
        formatted_prompt = f"""### Instruction:\nBạn là một chatbot hỗ trợ tâm lý chuyên nghiệp. Hãy trả lời người dùng một cách thân thiện, đồng cảm và hữu ích.\n\n### Input:\n{prompt_str}\n\n### Response:\n"""
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
        response = re.sub(r"<\|.*?\|>", "", response)
        return response
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return "Xin lỗi, tôi đang gặp sự cố kỹ thuật. Vui lòng thử lại sau."

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("Starting Model Inference Server...")
    load_model()

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