import uuid
import json
from datetime import datetime
from typing import Optional, Dict, Any, List
import logging
from pymongo import MongoClient
from pymongo.errors import PyMongoError

logger = logging.getLogger(__name__)

class MongoDBFeedbackService:
    def __init__(self, connection_string: str = "mongodb://localhost:27017/"):
        """
        Khởi tạo MongoDB connection
        """
        try:
            self.client = MongoClient(connection_string)
            self.db = self.client.qdrant_mongoDB  # Sử dụng database qdrant_mongoDB như yêu cầu
            self.feedback_collection = self.db.feedback  # Collection feedback
            
            # Tạo index cho các trường thường query
            self.feedback_collection.create_index([("session_id", 1)])
            self.feedback_collection.create_index([("feedback_type", 1)])
            self.feedback_collection.create_index([("timestamp", -1)])
            
            logger.info("MongoDB feedback service initialized successfully")
            
        except PyMongoError as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    def save_feedback(self, 
                     session_id: str,
                     user_input: str,
                     bot_response: str,
                     feedback_type: str,
                     user_feedback_text: Optional[str] = None,
                     risk_level: Optional[str] = None,
                     emotion_label: Optional[str] = None) -> Dict[str, Any]:
        """
        Lưu feedback từ người dùng vào MongoDB
        """
        try:
            # Tạo document feedback
            feedback_doc = {
                "session_id": session_id,
                "user_input": user_input,
                "bot_response": bot_response,
                "feedback_type": feedback_type,  # 'like' hoặc 'dislike'
                "user_feedback_text": user_feedback_text,
                "risk_level": risk_level,
                "emotion_label": emotion_label,
                "timestamp": datetime.utcnow(),
                "created_at": datetime.utcnow()
            }
            
            # Insert vào collection
            result = self.feedback_collection.insert_one(feedback_doc)
            
            logger.info(f"Feedback saved to MongoDB: {feedback_type} for session {session_id}")
            
            return {
                "status": "success",
                "message": "Feedback đã được lưu thành công vào MongoDB",
                "feedback_id": str(result.inserted_id)
            }
            
        except PyMongoError as e:
            logger.error(f"Error saving feedback to MongoDB: {e}")
            return {
                "status": "error",
                "message": f"Lỗi khi lưu feedback vào MongoDB: {str(e)}"
            }
    
    def get_feedback_stats(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Lấy thống kê feedback từ MongoDB
        """
        try:
            # Build query
            query = {}
            if session_id:
                query["session_id"] = session_id
            
            # Đếm tổng số feedback
            total_feedback = self.feedback_collection.count_documents(query)
            
            # Đếm likes
            like_query = query.copy()
            like_query["feedback_type"] = "like"
            likes = self.feedback_collection.count_documents(like_query)
            
            # Đếm dislikes
            dislike_query = query.copy()
            dislike_query["feedback_type"] = "dislike"
            dislikes = self.feedback_collection.count_documents(dislike_query)
            
            # Tính tỷ lệ hài lòng
            satisfaction_rate = (likes / total_feedback * 100) if total_feedback > 0 else 0
            
            return {
                "total": total_feedback,
                "likes": likes,
                "dislikes": dislikes,
                "satisfaction_rate": satisfaction_rate
            }
            
        except PyMongoError as e:
            logger.error(f"Error getting feedback stats from MongoDB: {e}")
            return {
                "total": 0,
                "likes": 0,
                "dislikes": 0,
                "satisfaction_rate": 0
            }
    
    def get_dislike_feedback(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Lấy danh sách feedback dislike để phân tích và cải thiện
        """
        try:
            # Query feedback dislike, sắp xếp theo timestamp giảm dần
            dislikes = list(self.feedback_collection.find(
                {"feedback_type": "dislike"}
            ).sort("timestamp", -1).limit(limit))
            
            # Convert ObjectId to string và format timestamp
            result = []
            for doc in dislikes:
                result.append({
                    "id": str(doc["_id"]),
                    "session_id": doc["session_id"],
                    "user_input": doc["user_input"],
                    "bot_response": doc["bot_response"],
                    "user_feedback_text": doc.get("user_feedback_text"),
                    "timestamp": doc["timestamp"].isoformat(),
                    "risk_level": doc.get("risk_level"),
                    "emotion_label": doc.get("emotion_label")
                })
            
            return result
            
        except PyMongoError as e:
            logger.error(f"Error getting dislike feedback from MongoDB: {e}")
            return []
    
    def get_feedback_by_session(self, session_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Lấy tất cả feedback của một session
        """
        try:
            feedbacks = list(self.feedback_collection.find(
                {"session_id": session_id}
            ).sort("timestamp", -1).limit(limit))
            
            result = []
            for doc in feedbacks:
                result.append({
                    "id": str(doc["_id"]),
                    "session_id": doc["session_id"],
                    "user_input": doc["user_input"],
                    "bot_response": doc["bot_response"],
                    "feedback_type": doc["feedback_type"],
                    "user_feedback_text": doc.get("user_feedback_text"),
                    "timestamp": doc["timestamp"].isoformat(),
                    "risk_level": doc.get("risk_level"),
                    "emotion_label": doc.get("emotion_label")
                })
            
            return result
            
        except PyMongoError as e:
            logger.error(f"Error getting feedback by session from MongoDB: {e}")
            return []
    
    def export_feedback_to_jsonl(self, filepath: str) -> bool:
        """
        Xuất tất cả feedback ra file JSONL để fine-tuning
        """
        try:
            # Lấy tất cả feedback, sắp xếp theo timestamp
            all_feedback = list(self.feedback_collection.find().sort("timestamp", -1))
            
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
            
            logger.info(f"Exported {len(all_feedback)} feedback records to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting feedback: {e}")
            return False
    
    def get_feedback_analytics(self) -> Dict[str, Any]:
        """
        Lấy analytics chi tiết về feedback
        """
        try:
            # Pipeline aggregation để tính toán
            pipeline = [
                {
                    "$group": {
                        "_id": "$feedback_type",
                        "count": {"$sum": 1},
                        "avg_risk_level": {
                            "$avg": {
                                "$cond": [
                                    {"$eq": ["$risk_level", "normal"]}, 0,
                                    {"$cond": [{"$eq": ["$risk_level", "risky"]}, 1, 2]}
                                ]
                            }
                        }
                    }
                }
            ]
            
            results = list(self.feedback_collection.aggregate(pipeline))
            
            analytics = {
                "feedback_distribution": {},
                "total_feedback": 0,
                "avg_risk_by_feedback_type": {}
            }
            
            for result in results:
                feedback_type = result["_id"]
                count = result["count"]
                analytics["feedback_distribution"][feedback_type] = count
                analytics["total_feedback"] += count
                analytics["avg_risk_by_feedback_type"][feedback_type] = result["avg_risk_level"]
            
            return analytics
            
        except PyMongoError as e:
            logger.error(f"Error getting feedback analytics: {e}")
            return {}
    
    def close_connection(self):
        """
        Đóng kết nối MongoDB
        """
        if hasattr(self, 'client'):
            self.client.close()
            logger.info("MongoDB connection closed")

# Singleton instance
mongodb_feedback_service = MongoDBFeedbackService() 