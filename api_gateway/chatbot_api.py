#!/usr/bin/env python3
"""
Main Chatbot API - Tích hợp tất cả modules
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel

# Import các services
from services.gating_router.quick_check import QuickCheckModel
from services.gating_router.prompt_builder import build_prompt_from_object
from services.chatbot.response_generator import call_gemini_llm
from services.setiment_analysis.analyzer import detect_sentiment_label
from services.mental_state_classifier.classifer import detect_mental_state
from services.common_schemas import SentimentOutput, MentalStateOutput

logger = logging.getLogger(__name__)

# Khởi tạo router
router = APIRouter()

# Khởi tạo gating router
gating_model = QuickCheckModel("models/weights/gating_router")

class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    history: Optional[list] = None  # Thêm history nếu có

class ChatResponse(BaseModel):
    response: str
    sentiment: str
    mental_state: str
    risk_level: str
    warning: Optional[str] = None

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main chatbot endpoint - tối ưu: chỉ phân tích sentiment/mental_state khi risk_level khác 'normal'
    """
    try:
        user_message = request.message
        # 1. Gating router - đánh giá rủi ro
        risk_proba = gating_model.predict_proba(user_message)
        risk_level = max(risk_proba, key=lambda k: risk_proba[k])
        logger.info(f"Risk level: {risk_level}, Proba: {risk_proba}")
        # 2. Xử lý theo risk_level
        if risk_level == "normal":
            prompt = user_message
            response_text = call_gemini_llm(prompt)
            sentiment = ""
            mental_state = ""
        else:
            sentiment_obj: SentimentOutput = detect_sentiment_label(user_message)
            logger.info(f"Sentiment: {sentiment_obj}")
            mental_state_obj: MentalStateOutput = detect_mental_state(user_message)
            logger.info(f"Mental state: {mental_state_obj}")
            prompt_obj = {
                "input": user_message,
                "context": {
                    "history": request.history[-5:] if request.history else [],
                    "risk_level": risk_level,
                    "mental_state": getattr(mental_state_obj, 'mental_state', ''),
                    "sentiment_intensity": getattr(sentiment_obj, 'sentiment', ''),
                    "knowledge": []
                }
            }
            prompt = build_prompt_from_object(prompt_obj)
            response_text = call_gemini_llm(prompt)
            sentiment = sentiment_obj.sentiment
            mental_state = mental_state_obj.mental_state
        # 3. Xử lý warning nếu cần
        warning = None
        if risk_level == "emergency":
            warning = "⚠️ KHẨN CẤP: Nếu bạn cần hỗ trợ khẩn cấp, hãy liên hệ hotline 1900xxxx ngay lập tức!"
        elif risk_level == "risky":
            warning = "⚠️ RỦI RO: Bạn có thể cân nhắc liên hệ chuyên gia tâm lý để được hỗ trợ tốt hơn."
        elif sentiment and mental_state and (sentiment in ["3", "negative"] and mental_state != "normal"):
            warning = "💡 Gợi ý: Hãy thử các hoạt động thư giãn như thiền, tập thể dục, hoặc nói chuyện với người thân."
        return ChatResponse(
            response=response_text,
            sentiment=sentiment,
            mental_state=mental_state,
            risk_level=risk_level,
            warning=warning
        )
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/api-stats")
async def get_api_stats():
    """Lấy thống kê API usage"""
    try:
        gemini_stats = gemini_service.get_api_stats()
        llama_health = llama_service.check_server_health()
        
        return {
            "success": True,
            "gemini_stats": gemini_stats,
            "llama_health": llama_health
        }
    except Exception as e:
        logger.error(f"Error getting API stats: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        llama_health = llama_service.check_server_health()
        return {
            "status": "healthy",
            "services": {
                "sentiment_analysis": "active",
                "mental_state_classifier": "active", 
                "gating_router": "active",
                    "gemini_api": "active",
                    "llama_model_server": llama_health["status"]
                },
                "llama_server": llama_health
            }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "degraded",
            "error": str(e)
        } 