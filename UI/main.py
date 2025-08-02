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

# from services.gating_router.router import MessageRouter
# from services.mental_state_classifier.classifer import detect_mental_state
# from services.setiment_analysis.analyzer import detect_sentiment_label
import requests
import re
# from services.emergency_handler.handler import EmergencyHandler
# from services.context_tracking.tracker import update_context
# from api_gateway.chatbot_api import router as chatbot_router
# from services.common_schemas import ChatServiceInput, ChatServiceOutput, SentimentOutput, MentalStateOutput, EmergencyOutput
# from services.gating_router.prompt_builder import build_prompt_from_object

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

# # Include chatbot router
# app.include_router(chatbot_router, prefix="/api/v1", tags=["chatbot"])

# # Middleware logging
# @app.middleware("http")
# async def log_requests(request: Request, call_next):
#     start = time.time()
#     response = await call_next(request)
#     duration = int((time.time() - start)*1000)
#     logging.info(f"{request.method} {request.url} {duration}ms")
#     return response

# # Khởi tạo services
# # chatbot_service = ChatbotService()
# emergency_handler = EmergencyHandler()
# router = MessageRouter(model_path="models/weights/gating_router")

# Schema definitions
class ChatRequest(BaseModel):
    user_input: str
    history: Optional[List[str]] = []
    session_id: Optional[str] = None

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

# Health check endpoint: Kiểm tra trạng thái API Gateway.

# @app.get("/health")
# async def health_check():
#     """Health check endpoint"""
#     try:
#         return {
#             "status": "healthy",
#             "message": "Mental Health Chatbot API is running",
#             "services": {
#                 "chatbot": "available",
#                 "context_tracker": "available", 
#                 "emergency_handler": "available",
#                 "gating_router": "available"
#             }
#         }
#     except Exception as e:
#         logger.error(f"Health check failed: {e}")
#         raise HTTPException(status_code=500, detail="Service unavailable")

# Main chat endpoint:  Nhận message từ frontend, xử lý toàn bộ luồng (gọi Gating Router, các service, Model Server...).
def extract_main_response(text: str) -> str:
    """
    Trích xuất phần nội dung chính từ response của LLM, bắt đầu từ "Chào bạn..."
    Loại bỏ các phần giải thích thêm và chỉ giữ lại phần nội dung chính.
    """
    if not isinstance(text, str):
        return ""
    
    # Pattern 1: Tìm phần trong dấu ngoặc kép sau "Chào bạn"
    chao_ban_quote_pattern = re.compile(r'Chào bạn[^"]*"([^"]*)"', re.IGNORECASE | re.DOTALL)
    match = chao_ban_quote_pattern.search(text)
    
    if match:
        return match.group(1).strip()
    
    # Pattern 2: Tìm phần từ "Chào bạn" đến hết (không có dấu ngoặc kép)
    chao_ban_pattern = re.compile(r'(Chào bạn.*?)(?=\n\n|\nTôi hy vọng|\nNếu bạn|$)', re.IGNORECASE | re.DOTALL)
    match = chao_ban_pattern.search(text)
    
    if match:
        return match.group(1).strip()
    
    # Pattern 3: Tìm phần trong dấu ngoặc kép đầu tiên
    quote_pattern = re.compile(r'"([^"]*)"', re.DOTALL)
    quote_match = quote_pattern.search(text)
    
    if quote_match:
        return quote_match.group(1).strip()
    
    # Pattern 4: Nếu không có pattern nào khớp, trả về text đã clean
    # Loại bỏ các phần giải thích thêm ở cuối
    lines = text.split('\n')
    main_lines = []
    for line in lines:
        line = line.strip()
        if line and not any(keyword in line.lower() for keyword in [
            'tôi hy vọng', 'nếu bạn cần', 'hãy cho tôi biết', 'cảm ơn bạn'
        ]):
            main_lines.append(line)
        elif line and any(keyword in line.lower() for keyword in [
            'tôi hy vọng', 'nếu bạn cần', 'hãy cho tôi biết', 'cảm ơn bạn'
        ]):
            break
    
    result = '\n'.join(main_lines).strip()
    return result if result else text

@app.post("/chat", response_model=ChatResponse)
async def handle_chat(req: ChatRequest):
    user_input = req.user_input

    print("User input:", user_input)
    
    try:
        # Gọi Model Server
        model_server_url = "http://localhost:8001/model/generate/"
        payload = {
            "prompt": f"Bạn là một chatbot hỗ trợ tâm lý thân thiện và cảm thông. Hãy trả lời người dùng một cách nhẹ nhàng, hỗ trợ và chuyên nghiệp. Bắt đầu câu trả lời bằng 'Chào bạn' và chỉ đưa ra nội dung chính, không cần giải thích thêm hay kết luận.\n\nUser: {user_input}\n\nAssistant:",
            "max_new_tokens": 1024
        }
        
        response = requests.post(model_server_url, json=payload, timeout=30)
        
        if response.status_code == 200:
            # Lấy response text từ streaming response
            raw_response = response.text
            
            # Trích xuất phần nội dung chính từ "Chào bạn..."
            final_response = extract_main_response(raw_response)
            
            return ChatResponse(
                bot_response=final_response,
            )
        else:
            logger.error(f"Model Server error: {response.status_code}")
            return ChatResponse(
                bot_response="Xin lỗi, tôi gặp lỗi khi xử lý yêu cầu của bạn. Vui lòng thử lại.",
            )
            
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return ChatResponse(
            bot_response="Xin lỗi, tôi gặp lỗi khi xử lý yêu cầu của bạn. Vui lòng thử lại.",
        )
#     """
#     Enhanced chat handler with comprehensive risk assessment
#     LƯU Ý: Mọi request chat đều PHẢI routing qua Gating Router trước khi xử lý tiếp!
#     Flow:
#       1. Nhận request từ frontend
#       2. Gọi Gating Router để xác định risk_level (bình thường, có vấn đề, khẩn cấp)
#       3. Tùy risk_level, gọi các service phù hợp (LLaMA, Sentiment, Mental, Emergency...)
#     """
#     try:
#         logger.info(f"Received chat request: {req.user_input[:50]}...")
#         # BƯỚC QUAN TRỌNG: Routing qua Gating Router để xác định risk_level
#         risk_level, confidence = router.route(req.user_input)
#         # Chuẩn hóa input cho các service
#         chat_input = ChatServiceInput(
#             user_message=req.user_input,
#             sentiment=None,
#             mental_state=None,
#             risk_level=risk_level
#         )
#         # Build prompt object for generate_reply
#         prompt_obj = {
#             "instruction": "Bạn là một chatbot hỗ trợ tâm lý. Hãy phản hồi nhẹ nhàng và cảm thông.",
#             "input": req.user_input,
#             "context": {
#                 "history": req.history[-5:] if req.history else [],
#                 "risk_level": risk_level
#             }
#         }
#         if risk_level == "normal":
#             # Low risk: use simple prompt
#             reply = generate_reply(req.user_input, req.history, sentiment="", mental_state="")
#             update_context(req.history, req.user_input, sentiment="", mental_state="", session_id=req.session_id)
#             return ChatResponse(
#                 bot_response=reply,
#                 risk_level=risk_level,
#                 confidence=confidence
#             )
#         elif risk_level == "risky":
#             # Medium risk: deeper analysis
#             mental_state_obj = detect_mental_state(req.user_input)
#             sentiment_obj = detect_sentiment_label(req.user_input)
#             update_context(req.history, req.user_input, sentiment_obj.sentiment, mental_state_obj.mental_state, session_id=req.session_id)
#             reply = generate_reply(req.user_input, req.history, sentiment_obj.sentiment, mental_state_obj.mental_state)
#             return ChatResponse(
#                 bot_response=reply,
#                 risk_level=risk_level,
#                 confidence=confidence,
#                 emotion_label=sentiment_obj.sentiment
#             )
#         else:  # emergency
#             # High risk: emergency handling
#             update_context(req.history, req.user_input, sentiment="emergency", mental_state="emergency", session_id=req.session_id)
#             emergency_result = emergency_handler.check_emergency(req.session_id or "anonymous", req.user_input)
#             return ChatResponse(
#                 bot_response=emergency_result.message,
#                 risk_level=risk_level,
#                 confidence=confidence
#             )
#     except Exception as e:
#         logger.error(f"Error in chat endpoint: {e}")
#         raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# # Emergency endpoint:Xử lý khẩn cấp.
# @app.post("/emergency")
# async def emergency_endpoint(request: EmergencyRequest):
#     """Emergency handling endpoint"""
#     try:
#         logger.info(f"Emergency request from user: {request.user_id}")
#         result: EmergencyOutput = emergency_handler.handle_emergency(
#             user_id=request.user_id,
#             location=request.location,
#             contact=request.contact
#         )
#         return {
#             "status": result.status,
#             "message": result.message,
#             "user_id": request.user_id,
#             "action": result.action
#         }
#     except Exception as e:
#         logger.error(f"Error in emergency endpoint: {e}")
#         raise HTTPException(status_code=500, detail=f"Emergency handling failed: {str(e)}")

# # Context management endpoints
# @app.get("/context/{user_id}")#lấy/xóa context hội thoại.
# async def get_context(user_id: str):
#     """Get conversation context for a user"""
#     try:
#         # context = context_tracker.get_context(user_id) # This line was removed as per the edit hint
#         return {
#             "user_id": user_id,
#             "context": "Context tracking is currently disabled." # Placeholder as context_tracker is removed
#         }
#     except Exception as e:
#         logger.error(f"Error getting context: {e}")
#         raise HTTPException(status_code=500, detail=f"Failed to get context: {str(e)}")

# @app.delete("/context/{user_id}")
# async def clear_context(user_id: str):
#     """Clear conversation context for a user"""
#     try:
#         # context_tracker.clear_context(user_id) # This line was removed as per the edit hint
#         return {"message": f"Context clearing is currently disabled for user {user_id}"} # Placeholder as context_tracker is removed
#     except Exception as e:
#         logger.error(f"Error clearing context: {e}")
#         raise HTTPException(status_code=500, detail=f"Failed to clear context: {str(e)}")

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
