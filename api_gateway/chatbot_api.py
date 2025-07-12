#!/usr/bin/env python3
"""
Main Chatbot API - Tích hợp tất cả modules
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# Import các services
from services.setiment_analysis.analyzer import detect_sentiment_label
from services.mental_state_classifier.classifer import detect_mental_state
from services.gating_router.quick_check import QuickCheckModel
from services.chatbot.gemini_service import gemini_service
from services.chatbot.llama_service import llama_service

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

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main chatbot endpoint - tích hợp tất cả modules
    """
    try:
        user_message = request.message
        
        # 1. Phân tích sentiment
        sentiment = detect_sentiment_label(user_message)
        logger.info(f"Sentiment: {sentiment}")
        
        # 2. Phân loại mental state
        mental_state = detect_mental_state(user_message)
        logger.info(f"Mental state: {mental_state}")
        
        # 3. Gating router - đánh giá rủi ro
        risk_proba = gating_model.predict_proba(user_message)
        risk_level = max(risk_proba, key=lambda k: risk_proba[k])
        logger.info(f"Risk level: {risk_level}, Proba: {risk_proba}")
        
        # 4. Chọn model và gọi API
        response, source, model_used = await get_response_with_fallback(
            user_message=user_message,
            sentiment=sentiment,
            mental_state=mental_state,
            risk_level=risk_level,
            prefer_model=request.prefer_model
        )
        
        # 5. Xử lý response và warning
        warning = None
        
        # Thêm warning nếu cần
        if risk_level == "emergency":
            warning = "⚠️ KHẨN CẤP: Nếu bạn cần hỗ trợ khẩn cấp, hãy liên hệ hotline 1900xxxx ngay lập tức!"
        elif risk_level == "risky":
            warning = "⚠️ RỦI RO: Bạn có thể cân nhắc liên hệ chuyên gia tâm lý để được hỗ trợ tốt hơn."
        elif sentiment in ["3", "negative"] and mental_state != "normal":
            warning = "💡 Gợi ý: Hãy thử các hoạt động thư giãn như thiền, tập thể dục, hoặc nói chuyện với người thân."
        
        return ChatResponse(
            response=response,
            sentiment=sentiment,
            mental_state=mental_state,
            risk_level=risk_level,
            source=source,
            warning=warning,
            model_used=model_used
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

async def get_response_with_fallback(user_message: str, sentiment: str, 
                                   mental_state: str, risk_level: str, 
                                   prefer_model: str = "auto") -> tuple[str, str, str]:
    """Lấy response với fallback logic"""
    
    # Kiểm tra health của LLaMA Model Server
    llama_health = llama_service.check_server_health()
    llama_available = llama_health["status"] == "healthy" and llama_health.get("model_loaded", False)
    
    logger.info(f"LLaMA server health: {llama_health}")
    logger.info(f"Prefer model: {prefer_model}")
    
    # Logic chọn model
    if prefer_model == "llama" and llama_available:
        # Thử LLaMA trước
        llama_result = llama_service.get_response(
            user_message=user_message,
            sentiment=sentiment,
            mental_state=mental_state,
            risk_level=risk_level
        )
        
        if llama_result["success"]:
            return llama_result["response"], llama_result["source"], "llama"
        else:
            logger.warning("LLaMA failed, falling back to Gemini")
            
    elif prefer_model == "gemini":
        # Chỉ dùng Gemini
        gemini_result = gemini_service.get_response(
            user_message=user_message,
            sentiment=sentiment,
            mental_state=mental_state,
            risk_level=risk_level
        )
        
        if gemini_result["success"]:
            return gemini_result["response"], gemini_result["source"], "gemini"
        else:
            logger.warning("Gemini failed")
            return "Xin lỗi, tôi đang gặp sự cố kỹ thuật.", "fallback", "none"
    
    else:  # prefer_model == "auto"
        # Auto logic: thử LLaMA trước, fallback về Gemini
        if llama_available:
            llama_result = llama_service.get_response(
                user_message=user_message,
                sentiment=sentiment,
                mental_state=mental_state,
                risk_level=risk_level
            )
            
            if llama_result["success"]:
                return llama_result["response"], llama_result["source"], "llama"
            else:
                logger.warning("LLaMA failed, falling back to Gemini")
        
        # Fallback to Gemini
        gemini_result = gemini_service.get_response(
            user_message=user_message,
            sentiment=sentiment,
            mental_state=mental_state,
            risk_level=risk_level
        )
        
        if gemini_result["success"]:
            return gemini_result["response"], gemini_result["source"], "gemini"
        else:
            logger.error("Both LLaMA and Gemini failed")
            return "Xin lỗi, tôi đang gặp sự cố kỹ thuật.", "fallback", "none"
    
    # Final fallback
    return "Xin lỗi, tôi đang gặp sự cố kỹ thuật.", "fallback", "none"

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