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
import json

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from services.gating_router.router import MessageRouter
from services.mental_state_classifier.classifer import detect_mental_state
from services.setiment_analysis.analyzer import detect_sentiment_label
from services.chatbot.response_generator import call_gemini_llm
from services.emergency_handler.handler import EmergencyHandler


from services.gating_router.prompt_builder import build_prompt_from_object
from services.semantic_search import SemanticIndexer

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
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
try:
    # Không cần truyền model_path vì sẽ load từ HuggingFace
    router = MessageRouter(model_path="")  # Empty path, will use HuggingFace only
    print("✅ Gating router initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize gating router: {e}")
    router = None

# Schema definitions
class ChatRequest(BaseModel):
    user_input: str
    history: list[str]
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    bot_response: str
    emotion_label: str = "Normal"
    risk_level: str = "normal"
    confidence: float = 0.0
    suggestion: str = ""
    knowledge: list = []  # Thêm trường knowledge để trả về các đoạn semantic search
    knowledge_similarity_scores: list = []  # Thêm trường để trả về độ tương đồng của knowledge

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
    Chat handler: Kết hợp logic tối ưu:
    - normal: chỉ semantic search, build prompt, LLM
    - risky: semantic search + sentiment + mental state, build prompt, LLM
    - emergency: gọi cảnh báo, semantic search, build prompt, LLM
    """
    try:
        user_message = req.user_input
        # Debug: Log received history
        logger.info(f"[chat] 📨 Received history: {req.history}")
        logger.info(f"[chat] 📨 Current user input: {user_message}")
        logger.info(f"[chat] 📨 History length: {len(req.history)}")
        logger.info(f"[chat] 📨 History type: {type(req.history)}")
        
        # Validate history - should not contain current user input
        if req.history and user_message in req.history:
            logger.warning(f"[chat] ⚠️ History contains current user input! Filtering out...")
            # Remove current user input from history if it exists
            req.history = [msg for msg in req.history if msg != user_message]
            logger.info(f"[chat] 📨 Filtered history: {req.history}")
        
        # Log history for debugging
        logger.info(f"[chat] 📨 History contains {len(req.history)} user messages")
        if req.history:
            logger.info(f"[chat] 📨 History messages: {[msg[:50] + '...' if len(msg) > 50 else msg for msg in req.history]}")
        
        # 1. Gating router - đánh giá rủi ro
        logger.info(f"[chat] 🚦 Gating router - đánh giá rủi ro cho: '{user_message[:50]}...'")
        if router is None:
            # Fallback khi router không load được
            logger.warning("⚠️ Gating router not available, using fallback risk assessment")
            risk_level, confidence = "normal", 0.5
        else:
            risk_level, confidence = router.route(user_message)
            logger.info(f"[chat] 🚦 Gating router result: risk_level='{risk_level}', confidence={confidence:.3f}")
        sentiment_obj = None
        mental_state_obj = None
        warning = None
        # 2. Semantic search Qdrant với lọc theo độ tương đồng
        knowledge_chunks = indexer.query(user_message, top_k=5)  # Lấy nhiều hơn để lọc
        # Lọc knowledge theo độ tương đồng > 80%
        filtered_chunks = indexer.filter_knowledge_by_similarity(
            user_input=user_message, 
            knowledge_chunks=knowledge_chunks, 
            similarity_threshold=0.8
        )
        knowledge_texts = [chunk.get("chunk_text", "") for chunk in filtered_chunks]
        knowledge_similarity_scores = [chunk.get("similarity_score", 0.0) for chunk in filtered_chunks]
        logger.info(f"[chat] 📚 Tìm được {len(knowledge_chunks)} đoạn knowledge, lọc còn {len(knowledge_texts)} đoạn có độ tương đồng > 80%")
        # 3. Build prompt object cho LLM
        if risk_level == "normal":
            prompt_obj = {
                "input": user_message,
                "context": {
                    "history": req.history[-1:] if req.history else [],
                    "knowledge": knowledge_texts[:1]
                }
            }
        elif risk_level == "risky":
            mental_state_obj = detect_mental_state(user_message)
            sentiment_obj = detect_sentiment_label(user_message)
            prompt_obj = {
                "input": user_message,
                "context": {
                    "history": req.history[-2:] if req.history else [],
                    "mental_state": getattr(mental_state_obj, 'mental_state', ''),
                    "sentiment_intensity": getattr(sentiment_obj, 'sentiment', ''),
                    "knowledge": knowledge_texts
                }
            }
        elif risk_level == "emergency":
            mental_state_obj = detect_mental_state(user_message)
            sentiment_obj = detect_sentiment_label(user_message)
            emergency_result = EmergencyHandler().handle_emergency(
                user_id=req.session_id or "anonymous",
                location=None,
                contact=None
            )
            warning = emergency_result.get("message", "") if emergency_result else "⚠️ Cần hỗ trợ khẩn cấp"
            prompt_obj = {
                "input": user_message,
                "context": {
                    "history": req.history[-2:] if req.history else [],
                    "mental_state": getattr(mental_state_obj, 'mental_state', ''),
                    "sentiment_intensity": getattr(sentiment_obj, 'sentiment', ''),
                    "knowledge": knowledge_texts,
                    "warning": warning
                }
            }
        else:
            prompt_obj = {
                "input": user_message,
                "context": {
                    "history": req.history[-1:] if req.history else [],
                    "knowledge": knowledge_texts[:1]
                }
            }
        # 4. Build prompt từ context
        logger.info(f"[chat] 🔧 Building prompt from object: {json.dumps(prompt_obj, ensure_ascii=False)[:200]}...")
        try:
            prompt = build_prompt_from_object(prompt_obj)
            logger.info(f"[chat] ✅ Prompt built successfully ({len(prompt)} chars)")
        except Exception as e:
            logger.warning(f"Prompt builder failed, using fallback: {e}")
            prompt = json.dumps(prompt_obj, ensure_ascii=False)
            logger.info(f"[chat] ⚠️ Using fallback prompt: {prompt[:200]}...")
        # 5. Gọi model server custom
        reply = call_gemini_llm(prompt)
        # 6. Xử lý warning nếu cần (ngoài emergency)
        if not warning:
            if risk_level == "risky":
                warning = "⚠️ RỦI RO: Bạn có thể cân nhắc liên hệ chuyên gia tâm lý để được hỗ trợ tốt hơn."
            elif sentiment_obj and (getattr(sentiment_obj, 'sentiment', None) in ["3", "negative"]) and (mental_state_obj and getattr(mental_state_obj, 'mental_state', None) != "normal"):
                warning = "💡 Gợi ý: Hãy thử các hoạt động thư giãn như thiền, tập thể dục, hoặc nói chuyện với người thân."
        return ChatResponse(
            bot_response=reply,
            risk_level=risk_level,
            confidence=confidence,
            emotion_label=getattr(sentiment_obj, 'sentiment', '') if sentiment_obj else '',
            mental_state=getattr(mental_state_obj, 'mental_state', '') if mental_state_obj else '',
            suggestion=warning or '',
            knowledge=knowledge_texts,
            knowledge_similarity_scores=knowledge_similarity_scores
        )
    except Exception as e:
        import traceback
        logger.error(f"Error in chat endpoint: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")



# Emergency endpoint:Xử lý khẩn cấp.
@app.post("/emergency")
async def emergency_endpoint(request: EmergencyRequest):
    """Emergency handling endpoint"""
    try:
        logger.info(f"Emergency request from user: {request.user_id}")
        result = EmergencyHandler().handle_emergency(
            user_id=request.user_id,
            location=request.location,
            contact=request.contact
        )
        if result:
            return {
                "status": result.status,
                "message": result.message,
                "user_id": request.user_id,
                "action": result.action
            }
        else:
            return {
                "status": "error",
                "message": "Emergency handler returned None",
                "user_id": request.user_id,
                "action": "none"
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
