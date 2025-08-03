# main.py
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from load_model import load_model_tokenizer
from generate import generate_stream

MODEL_PATH = "NV9523/MentalGPT"  # Hoặc đường dẫn local nếu đã tải model

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép mọi domain gọi (bạn có thể thay bằng domain cụ thể)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
model, tokenizer, device = load_model_tokenizer(MODEL_PATH)

class PromptRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 1024

@app.post("/model/generate/")
async def generate_text(req: PromptRequest):
    return StreamingResponse(generate_stream(model, tokenizer, device, req.prompt, req.max_new_tokens), media_type="text/plain")