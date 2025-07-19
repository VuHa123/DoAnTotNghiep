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
gating_model = QuickCheckModel("models/weights/gating_router/best_model")

# Khởi tạo instance inference toàn cục (chỉ load 1 lần)
chatbot_inference = ChatbotInference()
model_loaded = chatbot_inference.load_model()

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
    checkpoint: str = "checkpoint-1098"

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
        llama_result: ChatServiceOutput = llama_service.get_response(chat_input)
        if llama_result.success:
            return llama_result
        else:
            logger.warning("LLaMA failed, falling back to Gemini")
    elif prefer_model == "gemini":
        gemini_result: ChatServiceOutput = gemini_service.get_response(chat_input)
        if gemini_result.success:
            return gemini_result
        else:
            logger.warning("Gemini failed")
            return ChatServiceOutput(success=False, response="Xin lỗi, tôi đang gặp sự cố kỹ thuật.", source="fallback")
    else:  # prefer_model == "auto"
        if llama_available:
            llama_result: ChatServiceOutput = llama_service.get_response(chat_input)
            if llama_result.success:
                return llama_result
            else:
                logger.warning("LLaMA failed, falling back to Gemini")
        gemini_result: ChatServiceOutput = gemini_service.get_response(chat_input)
        if gemini_result.success:
            return gemini_result
        else:
            logger.error("Both LLaMA and Gemini failed")
            return ChatServiceOutput(success=False, response="Xin lỗi, tôi đang gặp sự cố kỹ thuật.", source="fallback")
    return ChatServiceOutput(success=False, response="Xin lỗi, tôi đang gặp sự cố kỹ thuật.", source="fallback")

@router.post("/generate-direct", response_model=DirectGenerateResponse)
async def generate_direct_endpoint(
    req: DirectGenerateRequest = Body(...)
):
    """
    Sinh response trực tiếp từ mô hình fine-tuned local (không qua server trung gian)
    """
    try:
        # Nếu checkpoint khác, reload model
        if req.checkpoint != chatbot_inference.checkpoint_name:
            chatbot_inference.checkpoint_name = req.checkpoint
            if not chatbot_inference.load_model():
                return DirectGenerateResponse(response="", success=False, error="Không load được checkpoint mới")
        if chatbot_inference.model is None:
            return DirectGenerateResponse(response="", success=False, error="Model chưa được load")
        response = chatbot_inference.generate_response(
            req.prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p
        )
        return DirectGenerateResponse(response=response, success=True)
    except Exception as e:
        logger.error(f"Error in /generate-direct: {e}")
        return DirectGenerateResponse(response="", success=False, error=str(e))

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