# main.py
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from load_model import load_model_tokenizer
from generate import generate_stream

MODEL_PATH = "NV9523/MentalGPT"  # Hoặc đường dẫn local nếu đã tải model

app = FastAPI()
model, tokenizer, device = load_model_tokenizer(MODEL_PATH)

class PromptRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 1024

@app.post("model/generate/")
async def generate_text(req: PromptRequest):
    generator = generate_stream(model, tokenizer, device, req.prompt, req.max_new_tokens)
    return StreamingResponse((word for word in generator), media_type="text/plain")
