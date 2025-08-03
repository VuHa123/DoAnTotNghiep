# main.py
import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from load_model import load_model_tokenizer
from generate import generate_stream

# Load environment variables
load_dotenv("token.env")

MODEL_PATH = "NV9523/MentalGPT"  # Hoặc đường dẫn local nếu đã tải model

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép mọi domain gọi (bạn có thể thay bằng domain cụ thể)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model with authentication
try:
    model, tokenizer, device = load_model_tokenizer(MODEL_PATH)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model, tokenizer, device = None, None, None

class PromptRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 1024

@app.post("/model/generate/")
async def generate_text(req: PromptRequest):
    if model is None or tokenizer is None:
        return {"error": "Model not loaded properly"}
    return StreamingResponse(generate_stream(model, tokenizer, device, req.prompt, req.max_new_tokens), media_type="text/plain")
