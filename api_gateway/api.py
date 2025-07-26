import sys
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import logging
import time

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from services.gating_router.router import MessageRouter
from services.mental_state_classifier.classifer import detect_mental_state
from services.setiment_analysis.analyzer import detect_sentiment_label
from services.chatbot.response_generator import call_gemini_llm
from services.emergency_handler.handler import EmergencyHandler
from services.context_tracking.tracker import update_context
from api_gateway.chatbot_api import router as chatbot_router
from services.common_schemas import SentimentOutput, MentalStateOutput
from services.gating_router.prompt_builder import build_prompt_from_object
from services.semantic_search import SemanticIndexer

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Mental Health Chatbot API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include chatbot router
app.include_router(chatbot_router, prefix="/api/v1", tags=["chatbot"])

# Middleware logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = int((time.time() - start)*1000)
    logging.info(f"{request.method} {request.url} {duration}ms")
    return response

# Khởi tạo services
# chatbot_service = ChatbotService()
# emergency_handler = EmergencyHandler()
router = MessageRouter(model_path="models/weights/gating_router")

# Schema definitions
class ChatRequest(BaseModel):
    user_input: str
    history: list[str]
    session_id: str = None

class ChatResponse(BaseModel):
    bot_response: str
    emotion_label: str = "Normal"
    risk_level: str = "normal"
    confidence: float = 0.0
    suggestion: str = ""
    knowledge: list = []  # Thêm trường knowledge để trả về các đoạn semantic search

class EmergencyRequest(BaseModel):
    user_id: str
    location: Optional[str] = None
    contact: Optional[str] = None

class SemanticSearchRequest(BaseModel):
    query: str
    top_k: int = 5

class SemanticSearchResponse(BaseModel):
    results: list

indexer = SemanticIndexer()

# Health check endpoint: Kiểm tra trạng thái API Gateway.

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        return {
            "status": "healthy",
            "message": "Mental Health Chatbot API is running",
            "services": {
                "chatbot": "available",
                "context_tracker": "available", 
                "emergency_handler": "available",
                "gating_router": "available"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Service unavailable")

# Main chat endpoint:  Nhận message từ frontend, xử lý toàn bộ luồng (gọi Gating Router, các service, Model Server...).
@app.post("/chat", response_model=ChatResponse)
async def handle_chat(req: ChatRequest):
    """
    Chat handler: Luôn semantic search, luôn build prompt với context đầy đủ (risk, mental_state, sentiment, knowledge, history), luôn gọi LLM.
    Nếu risk_level là 'emergency', trả về cả phản hồi LLM và cảnh báo khẩn cấp.
    """
    try:
        logger.info(f"Received chat request: {req.user_input[:50]}...")
        # 1. Phân tích risk, mental, sentiment
        risk_level, confidence = router.route(req.user_input)
        mental_state_obj = detect_mental_state(req.user_input)
        sentiment_obj = detect_sentiment_label(req.user_input)
        # 2. Semantic search Qdrant
        knowledge_chunks = indexer.query(req.user_input, top_k=5)
        print(f"[DEBUG] Raw knowledge_chunks: {knowledge_chunks}")
        knowledge_texts = [chunk.get("chunk_text", "") for chunk in knowledge_chunks]
        # Log knowledge ra terminal
        print("[RAG/Knowledge] Các đoạn semantic search tìm được:")
        for idx, chunk in enumerate(knowledge_texts, 1):
            print(f"  [{idx}] {chunk}")
        # 3. Build prompt object cho LLM
        prompt_obj = {
            "input": req.user_input,
            "context": {
                "history": req.history[-5:] if req.history else [],
                "risk_level": risk_level,
                "mental_state": getattr(mental_state_obj, 'mental_state', ''),
                "sentiment_intensity": getattr(sentiment_obj, 'sentiment', ''),
                "knowledge": knowledge_texts
            }
        }
        # 4. Gọi LLM
        import json
        prompt_str = json.dumps(prompt_obj, ensure_ascii=False)
        reply = call_gemini_llm(prompt_str)
        update_context(
            req.history,
            req.user_input,
            sentiment=getattr(sentiment_obj, 'sentiment', ''),
            mental_state=getattr(mental_state_obj, 'mental_state', ''),
            session_id=req.session_id
        )
        # 5. Nếu khẩn cấp, gọi emergency handler và trả về cảnh báo kèm phản hồi LLM
        if risk_level == "emergency":
            emergency_result = EmergencyHandler().check_emergency(req.session_id or "anonymous", req.user_input)
            return ChatResponse(
                bot_response=f"[CẢNH BÁO KHẨN CẤP]: {emergency_result['message']}\n\n[Phản hồi trợ lý]: {reply}",
                risk_level=risk_level,
                confidence=confidence,
                emotion_label=getattr(sentiment_obj, 'sentiment', ''),
                knowledge=knowledge_texts  # Trả về knowledge
            )
        else:
            return ChatResponse(
                bot_response=reply,
                risk_level=risk_level,
                confidence=confidence,
                emotion_label=getattr(sentiment_obj, 'sentiment', ''),
                knowledge=knowledge_texts  # Trả về knowledge
            )
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Emergency endpoint:Xử lý khẩn cấp.
@app.post("/emergency")
async def emergency_endpoint(request: EmergencyRequest):
    """Emergency handling endpoint"""
    try:
        logger.info(f"Emergency request from user: {request.user_id}")
        result: EmergencyOutput = EmergencyHandler().handle_emergency(
            user_id=request.user_id,
            location=request.location,
            contact=request.contact
        )
        return {
            "status": result.status,
            "message": result.message,
            "user_id": request.user_id,
            "action": result.action
        }
    except Exception as e:
        logger.error(f"Error in emergency endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Emergency handling failed: {str(e)}")

# Semantic search endpoint
@app.post("/semantic_search", response_model=SemanticSearchResponse)
async def semantic_search(req: SemanticSearchRequest):
    try:
        results = indexer.query(req.query, top_k=req.top_k)
        return SemanticSearchResponse(results=results)
    except Exception as e:
        logger.error(f"Error in semantic_search endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Semantic search failed: {str(e)}")

# Context management endpoints
@app.get("/context/{user_id}")#lấy/xóa context hội thoại.
async def get_context(user_id: str):
    """Get conversation context for a user"""
    try:
        # context = context_tracker.get_context(user_id) # This line was removed as per the edit hint
        return {
            "user_id": user_id,
            "context": "Context tracking is currently disabled." # Placeholder as context_tracker is removed
        }
    except Exception as e:
        logger.error(f"Error getting context: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get context: {str(e)}")

@app.delete("/context/{user_id}")
async def clear_context(user_id: str):
    """Clear conversation context for a user"""
    try:
        # context_tracker.clear_context(user_id) # This line was removed as per the edit hint
        return {"message": f"Context clearing is currently disabled for user {user_id}"} # Placeholder as context_tracker is removed
    except Exception as e:
        logger.error(f"Error clearing context: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear context: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
