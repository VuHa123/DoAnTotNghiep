#!/usr/bin/env python3
"""
Main Chatbot API - Tích hợp tất cả modules
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel

# Import các services
from services.setiment_analysis.analyzer import detect_sentiment_label
from services.mental_state_classifier.classifer import detect_mental_state
from services.gating_router.quick_check import QuickCheckModel
from services.chatbot.gemini_service import gemini_service
from services.chatbot.llama_service import llama_service
from services.chatbot.inference_service import ChatbotInference
from services.common_schemas import ChatServiceInput, ChatServiceOutput, SentimentOutput, MentalStateOutput

logger = logging.getLogger(__name__)

# Khởi tạo router
router = APIRouter()

# Khởi tạo gating router
gating_model = QuickCheckModel("models/weights/gating_router")

class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    prefer_model: str = "auto"  # "llama", "gemini", "auto"

class ChatResponse(BaseModel):
    response: str
    sentiment: str
    mental_state: str
    risk_level: str
    source: str
    warning: Optional[str] = None
    model_used: Optional[str] = None

class DirectGenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 200
    temperature: float = 0.7
    top_p: float = 0.9
    checkpoint: str = "checkpoint-1000"

class DirectGenerateResponse(BaseModel):
    response: str
    success: bool
    error: str = ""

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main chatbot endpoint - tích hợp tất cả modules
    """
    try:
        user_message = request.message
        # 1. Phân tích sentiment
        sentiment_obj: SentimentOutput = detect_sentiment_label(user_message)
        logger.info(f"Sentiment: {sentiment_obj}")
        # 2. Phân loại mental state
        mental_state_obj: MentalStateOutput = detect_mental_state(user_message)
        logger.info(f"Mental state: {mental_state_obj}")
        # 3. Gating router - đánh giá rủi ro
        risk_proba = gating_model.predict_proba(user_message)
        risk_level = max(risk_proba, key=lambda k: risk_proba[k])
        logger.info(f"Risk level: {risk_level}, Proba: {risk_proba}")
        # 4. Chọn model và gọi API
        if risk_level == "normal":
            # Gọi Model Server luôn, không phân tích sentiment/mental_state
            chat_input = ChatServiceInput(
                user_message=user_message,
                sentiment=None,
                mental_state=None,
                risk_level=risk_level
            )
        else:
            # Phân tích sentiment/mental_state trước
            sentiment_obj = detect_sentiment_label(user_message)
            mental_state_obj = detect_mental_state(user_message)
            chat_input = ChatServiceInput(
                user_message=user_message,
                sentiment=sentiment_obj.sentiment,
                mental_state=mental_state_obj.mental_state,
                risk_level=risk_level
            )
        response_obj: ChatServiceOutput = await get_response_with_fallback(
            chat_input=chat_input,
            prefer_model=request.prefer_model
        )
        # 5. Xử lý response và warning
        warning = None
        if risk_level == "emergency":
            warning = "⚠️ KHẨN CẤP: Nếu bạn cần hỗ trợ khẩn cấp, hãy liên hệ hotline 1900xxxx ngay lập tức!"
        elif risk_level == "risky":
            warning = "⚠️ RỦI RO: Bạn có thể cân nhắc liên hệ chuyên gia tâm lý để được hỗ trợ tốt hơn."
        elif sentiment_obj.sentiment in ["3", "negative"] and mental_state_obj.mental_state != "normal":
            warning = "💡 Gợi ý: Hãy thử các hoạt động thư giãn như thiền, tập thể dục, hoặc nói chuyện với người thân."
        return ChatResponse(
            response=response_obj.response,
            sentiment=sentiment_obj.sentiment,
            mental_state=mental_state_obj.mental_state,
            risk_level=risk_level,
            source=response_obj.source,
            warning=warning,
            model_used=getattr(response_obj, "model_used", None)
        )
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

async def get_response_with_fallback(chat_input: ChatServiceInput, prefer_model: str = "auto") -> ChatServiceOutput:
    """Lấy response với fallback logic, truyền object schema"""
    llama_health = llama_service.check_server_health()
    llama_available = llama_health["status"] == "healthy" and llama_health.get("model_loaded", False)
    logger.info(f"LLaMA server health: {llama_health}")
    logger.info(f"Prefer model: {prefer_model}")
    if prefer_model == "llama" and llama_available:
        llama_result_main = llama_service.get_response(
            chat_input.user_message,
            chat_input.sentiment or '',
            chat_input.mental_state or '',
            chat_input.risk_level or ''
        )
        llama_result_main = ChatServiceOutput(**llama_result_main)
        if llama_result_main.success:
            return llama_result_main
        else:
            logger.warning("LLaMA failed, falling back to Gemini")
    elif prefer_model == "gemini":
        gemini_result_main = gemini_service.get_response(
            chat_input.user_message,
            chat_input.sentiment or '',
            chat_input.mental_state or '',
            chat_input.risk_level or ''
        )
        gemini_result_main = ChatServiceOutput(**gemini_result_main)
        if gemini_result_main.success:
            return gemini_result_main
        else:
            logger.warning("Gemini failed")
            return ChatServiceOutput(success=False, response="Xin lỗi, tôi đang gặp sự cố kỹ thuật.", source="fallback")
    else:  # prefer_model == "auto"
        if llama_available:
            llama_result_auto = llama_service.get_response(
                chat_input.user_message,
                chat_input.sentiment or '',
                chat_input.mental_state or '',
                chat_input.risk_level or ''
            )
            llama_result_auto = ChatServiceOutput(**llama_result_auto)
            if llama_result_auto.success:
                return llama_result_auto
            else:
                logger.warning("LLaMA failed, falling back to Gemini")
        gemini_result_auto = gemini_service.get_response(
            chat_input.user_message,
            chat_input.sentiment or '',
            chat_input.mental_state or '',
            chat_input.risk_level or ''
        )
        gemini_result_auto = ChatServiceOutput(**gemini_result_auto)
        if gemini_result_auto.success:
            return gemini_result_auto
        else:
            logger.error("Both LLaMA and Gemini failed")
            return ChatServiceOutput(success=False, response="Xin lỗi, tôi đang gặp sự cố kỹ thuật.", source="fallback")
    return ChatServiceOutput(success=False, response="Xin lỗi, tôi đang gặp sự cố kỹ thuật.", source="fallback")

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