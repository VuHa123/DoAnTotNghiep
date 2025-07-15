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
from services.chatbot.bot_service import generate_reply
from services.emergency_handler.handler import EmergencyHandler
from services.context_tracking.tracker import update_context
from api_gateway.chatbot_api import router as chatbot_router
from services.common_schemas import ChatServiceInput, ChatServiceOutput, SentimentOutput, MentalStateOutput, EmergencyOutput

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
emergency_handler = EmergencyHandler()
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

class EmergencyRequest(BaseModel):
    user_id: str
    location: Optional[str] = None
    contact: Optional[str] = None

# Health check endpoint
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

# Main chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def handle_chat(req: ChatRequest):
    """
    Enhanced chat handler with comprehensive risk assessment
    LƯU Ý: Mọi request chat đều PHẢI routing qua Gating Router trước khi xử lý tiếp!
    Flow:
      1. Nhận request từ frontend
      2. Gọi Gating Router để xác định risk_level (bình thường, có vấn đề, khẩn cấp)
      3. Tùy risk_level, gọi các service phù hợp (LLaMA, Sentiment, Mental, Emergency...)
    """
    try:
        logger.info(f"Received chat request: {req.user_input[:50]}...")
        # BƯỚC QUAN TRỌNG: Routing qua Gating Router để xác định risk_level
        risk_level, confidence = router.route(req.user_input)
        # Chuẩn hóa input cho các service
        chat_input = ChatServiceInput(
            user_message=req.user_input,
            sentiment=None,
            mental_state=None,
            risk_level=risk_level
        )
        if risk_level == "normal":
            # Low risk: use simple prompt
            reply = generate_reply(req.user_input, req.history, sentiment="", mental_state="")
            update_context(req.history, req.user_input, sentiment="", mental_state="", session_id=req.session_id)
            return ChatResponse(
                bot_response=reply,
                risk_level=risk_level,
                confidence=confidence
            )
        elif risk_level == "risky":
            # Medium risk: deeper analysis
            mental_state_obj = detect_mental_state(req.user_input)
            sentiment_obj = detect_sentiment_label(req.user_input)
            update_context(req.history, req.user_input, sentiment_obj.sentiment, mental_state_obj.mental_state, session_id=req.session_id)
            reply = generate_reply(req.user_input, req.history, sentiment_obj.sentiment, mental_state_obj.mental_state)
            return ChatResponse(
                bot_response=reply,
                risk_level=risk_level,
                confidence=confidence,
                emotion_label=sentiment_obj.sentiment
            )
        else:  # emergency
            # High risk: emergency handling
            update_context(req.history, req.user_input, sentiment="emergency", mental_state="emergency", session_id=req.session_id)
            emergency_result = emergency_handler.check_emergency(req.session_id or "anonymous", req.user_input)
            return ChatResponse(
                bot_response=emergency_result.message,
                risk_level=risk_level,
                confidence=confidence
            )
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Emergency endpoint
@app.post("/emergency")
async def emergency_endpoint(request: EmergencyRequest):
    """Emergency handling endpoint"""
    try:
        logger.info(f"Emergency request from user: {request.user_id}")
        result: EmergencyOutput = emergency_handler.handle_emergency(
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

# Context management endpoints
@app.get("/context/{user_id}")
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
