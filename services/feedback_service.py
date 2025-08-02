import uuid
import json
from datetime import datetime
from typing import Optional, Dict, Any
from Database.core import SessionLocal, Feedback
import logging

logger = logging.getLogger(__name__)

class FeedbackService:
    def __init__(self):
        self.db = SessionLocal()
    
    def save_feedback(self, 
                     session_id: str,
                     user_input: str,
                     bot_response: str,
                     feedback_type: str,
                     user_feedback_text: Optional[str] = None,
                     risk_level: Optional[str] = None,
                     emotion_label: Optional[str] = None) -> Dict[str, Any]:
        """
        Lưu feedback từ người dùng vào database
        """
        try:
            # Tạo record feedback mới
            feedback_record = Feedback(
                session_id=session_id,
                user_input=user_input,
                bot_response=bot_response,
                feedback_type=feedback_type,
                user_feedback_text=user_feedback_text,
                risk_level=risk_level,
                emotion_label=emotion_label,
                timestamp=datetime.utcnow()
            )
            
            self.db.add(feedback_record)
            self.db.commit()
            
            logger.info(f"Feedback saved: {feedback_type} for session {session_id}")
            
            return {
                "status": "success",
                "message": "Feedback đã được lưu thành công",
                "feedback_id": feedback_record.id
            }
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error saving feedback: {e}")
            return {
                "status": "error",
                "message": f"Lỗi khi lưu feedback: {str(e)}"
            }
    
    def get_feedback_stats(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Lấy thống kê feedback
        """
        try:
            query = self.db.query(Feedback)
            
            if session_id:
                query = query.filter(Feedback.session_id == session_id)
            
            total_feedback = query.count()
            likes = query.filter(Feedback.feedback_type == 'like').count()
            dislikes = query.filter(Feedback.feedback_type == 'dislike').count()
            
            return {
                "total": total_feedback,
                "likes": likes,
                "dislikes": dislikes,
                "satisfaction_rate": (likes / total_feedback * 100) if total_feedback > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting feedback stats: {e}")
            return {
                "total": 0,
                "likes": 0,
                "dislikes": 0,
                "satisfaction_rate": 0
            }
    
    def get_dislike_feedback(self, limit: int = 50) -> list:
        """
        Lấy danh sách feedback dislike để phân tích và cải thiện
        """
        try:
            dislikes = self.db.query(Feedback).filter(
                Feedback.feedback_type == 'dislike'
            ).order_by(Feedback.timestamp.desc()).limit(limit).all()
            
            return [
                {
                    "id": f.id,
                    "session_id": f.session_id,
                    "user_input": f.user_input,
                    "bot_response": f.bot_response,
                    "user_feedback_text": f.user_feedback_text,
                    "timestamp": f.timestamp.isoformat(),
                    "risk_level": f.risk_level,
                    "emotion_label": f.emotion_label
                }
                for f in dislikes
            ]
            
        except Exception as e:
            logger.error(f"Error getting dislike feedback: {e}")
            return []
    
    def export_feedback_to_jsonl(self, filepath: str) -> bool:
        """
        Xuất tất cả feedback ra file JSONL để fine-tuning
        """
        try:
            all_feedback = self.db.query(Feedback).order_by(Feedback.timestamp.desc()).all()
            
            with open(filepath, 'w', encoding='utf-8') as f:
                for feedback in all_feedback:
                    record = {
                        "id": feedback.id,
                        "session_id": feedback.session_id,
                        "user_input": feedback.user_input,
                        "bot_response": feedback.bot_response,
                        "feedback_type": feedback.feedback_type,
                        "user_feedback_text": feedback.user_feedback_text,
                        "timestamp": feedback.timestamp.isoformat(),
                        "risk_level": feedback.risk_level,
                        "emotion_label": feedback.emotion_label
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            logger.info(f"Exported {len(all_feedback)} feedback records to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting feedback: {e}")
            return False
    
    def __del__(self):
        if hasattr(self, 'db'):
            self.db.close()

# Singleton instance
feedback_service = FeedbackService() 