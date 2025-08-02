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
import uuid

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'services'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'llmserver'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'services', 'chatbot'))

import requests
import re

# Import các hàm từ response_generator để tránh code duplication
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import importlib.util

# Import response_generator module
spec = importlib.util.spec_from_file_location(
    "response_generator", 
    os.path.join(os.path.dirname(__file__), '..', 'services', 'chatbot', 'response_generator.py')
)
response_generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(response_generator)

extract_main_response = response_generator.extract_main_response
final_cleanup = response_generator.final_cleanup

# Import feedback từ core
from Database.core import Feedback, mongodb_manager

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
    session_id: Optional[str] = None

class FeedbackRequest(BaseModel):
    session_id: str
    user_input: str
    bot_response: str
    feedback_type: str  # 'like' or 'dislike'
    user_feedback_text: Optional[str] = None
    risk_level: Optional[str] = None
    emotion_label: Optional[str] = None

class EmergencyRequest(BaseModel):
    user_id: str
    location: Optional[str] = None
    contact: Optional[str] = None


@app.post("/chat", response_model=ChatResponse)
async def handle_chat(req: ChatRequest):
    user_input = req.user_input
    
    # Tạo session_id nếu chưa có
    session_id = req.session_id or f"session_{uuid.uuid4().hex[:8]}"

    print("User input:", user_input)
    print("Session ID:", session_id)
    
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
                        
            # Làm sạch response trước khi trích xuất
            cleaned_response = final_cleanup(raw_response)
            
            # Trích xuất phần nội dung chính từ "Chào bạn..."
            final_response = extract_main_response(cleaned_response)
            
            return ChatResponse(
                bot_response=final_response,
                session_id=session_id
            )
        else:
            logger.error(f"Model Server error: {response.status_code}")
            return ChatResponse(
                bot_response="Xin lỗi, tôi gặp lỗi khi xử lý yêu cầu của bạn. Vui lòng thử lại.",
                session_id=session_id
            )
            
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return ChatResponse(
            bot_response="Xin lỗi, tôi gặp lỗi khi xử lý yêu cầu của bạn. Vui lòng thử lại.",
            session_id=session_id
        )

@app.post("/feedback")
async def handle_feedback(req: FeedbackRequest):
    """
    Xử lý feedback từ người dùng
    """
    try:
        logger.info(f"Received feedback: {req.feedback_type} for session {req.session_id}")
        
        # Tạo feedback object và lưu
        feedback = Feedback(
            session_id=req.session_id,
            user_input=req.user_input,
            bot_response=req.bot_response,
            feedback_type=req.feedback_type,
            user_feedback_text=req.user_feedback_text,
            risk_level=req.risk_level,
            emotion_label=req.emotion_label
        )
        feedback_id = feedback.save()
        
        if feedback_id:
            result = {
                "status": "success",
                "message": "Feedback đã được lưu thành công",
                "feedback_id": feedback_id
            }
        else:
            result = {
                "status": "error",
                "message": "Lỗi khi lưu feedback"
            }
        
        if result["status"] == "success":
            return {
                "status": "success",
                "message": "Cảm ơn bạn đã đưa ra phản hồi!",
                "feedback_id": result["feedback_id"]
            }
        else:
            raise HTTPException(status_code=500, detail=result["message"])
            
    except Exception as e:
        logger.error(f"Error in feedback endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý feedback: {str(e)}")

@app.get("/feedback/stats")
async def get_feedback_stats(session_id: Optional[str] = None):
    """
    Lấy thống kê feedback
    """
    try:
        # Lấy thống kê feedback
        query = {}
        if session_id:
            query["session_id"] = session_id
        
        total = mongodb_manager.user_feedback.count_documents(query)
        likes = mongodb_manager.user_feedback.count_documents({**query, "feedback_type": "like"})
        dislikes = mongodb_manager.user_feedback.count_documents({**query, "feedback_type": "dislike"})
        
        satisfaction_rate = (likes / total * 100) if total > 0 else 0
        
        stats = {
            "total": total,
            "likes": likes,
            "dislikes": dislikes,
            "satisfaction_rate": satisfaction_rate
        }
        return stats
    except Exception as e:
        logger.error(f"Error getting feedback stats: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi lấy thống kê: {str(e)}")

@app.get("/feedback/dislikes")
async def get_dislike_feedback(limit: int = 50):
    """
    Lấy danh sách feedback dislike để phân tích
    """
    try:
        # Lấy danh sách feedback dislike
        dislikes_docs = list(mongodb_manager.user_feedback.find(
            {"feedback_type": "dislike"}
        ).sort("timestamp", -1).limit(limit))
        
        dislikes = []
        for doc in dislikes_docs:
            dislikes.append({
                "id": str(doc["_id"]),
                "session_id": doc["session_id"],
                "user_input": doc["user_input"],
                "bot_response": doc["bot_response"],
                "user_feedback_text": doc.get("user_feedback_text"),
                "timestamp": doc["timestamp"].isoformat(),
                "risk_level": doc.get("risk_level"),
                "emotion_label": doc.get("emotion_label")
            })
        return {
            "dislikes": dislikes,
            "count": len(dislikes)
        }
    except Exception as e:
        logger.error(f"Error getting dislike feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi lấy feedback: {str(e)}")

@app.post("/feedback/export")
async def export_feedback(filepath: str = "feedback_data.jsonl"):
    """
    Xuất feedback ra file JSONL để fine-tuning
    """
    try:
        # Xuất feedback ra file JSONL
        all_feedback = list(mongodb_manager.user_feedback.find().sort("timestamp", -1))
        
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            for doc in all_feedback:
                record = {
                    "id": str(doc["_id"]),
                    "session_id": doc["session_id"],
                    "user_input": doc["user_input"],
                    "bot_response": doc["bot_response"],
                    "feedback_type": doc["feedback_type"],
                    "user_feedback_text": doc.get("user_feedback_text"),
                    "timestamp": doc["timestamp"].isoformat(),
                    "risk_level": doc.get("risk_level"),
                    "emotion_label": doc.get("emotion_label")
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        success = True
        if success:
            return {
                "status": "success",
                "message": f"Đã xuất feedback ra file {filepath}",
                "filepath": filepath
            }
        else:
            raise HTTPException(status_code=500, detail="Lỗi khi xuất feedback")
    except Exception as e:
        logger.error(f"Error exporting feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi xuất feedback: {str(e)}")

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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
