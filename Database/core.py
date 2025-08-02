"""
MongoDB Core Models - Tách biệt Feedback Collections và RAG Collections trong cùng Database
"""

from datetime import datetime
from typing import Optional, Dict, Any
from pymongo import MongoClient
from pymongo.errors import PyMongoError
import logging

logger = logging.getLogger(__name__)

class MongoDBManager:
    def __init__(self, connection_string: str = "mongodb://localhost:27017/"):
        """Khởi tạo MongoDB connection"""
        try:
            self.client = MongoClient(connection_string)
            
            # Sử dụng cùng database qdrant_mongoDB nhưng tách biệt collections
            self.db = self.client.qdrant_mongoDB
            
            # RAG Collections (cho knowledge base và chatbot operations)
            self.conversations = self.db.conversations
            self.emergency_logs = self.db.emergency_logs
            self.user_sessions = self.db.user_sessions
            self.mental_state_history = self.db.mental_state_history
            
            # Feedback Collections (cho user feedback và analytics)
            self.user_feedback = self.db.user_feedback
            self.feedback_analytics = self.db.feedback_analytics
            
            # Tạo indexes
            self._create_indexes()
            
            logger.info("MongoDB Manager initialized successfully")
            logger.info("Database: qdrant_mongoDB")
            logger.info("RAG Collections: conversations, emergency_logs, user_sessions, mental_state_history")
            logger.info("Feedback Collections: user_feedback, feedback_analytics")
            
        except PyMongoError as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    def _create_indexes(self):
        """Tạo indexes cho các collections"""
        try:
            # RAG Collections indexes
            self.conversations.create_index([("created_at", -1)])
            self.emergency_logs.create_index([("user_id", 1)])
            self.emergency_logs.create_index([("timestamp", -1)])
            self.user_sessions.create_index([("started_at", -1)])
            self.mental_state_history.create_index([("detected_at", -1)])
            
            # Feedback Collections indexes
            self.user_feedback.create_index([("session_id", 1)])
            self.user_feedback.create_index([("feedback_type", 1)])
            self.user_feedback.create_index([("timestamp", -1)])
            self.user_feedback.create_index([("risk_level", 1)])
            self.user_feedback.create_index([("emotion_label", 1)])
            
            # Analytics indexes
            self.feedback_analytics.create_index([("date", -1)])
            self.feedback_analytics.create_index([("session_id", 1)])
            
        except PyMongoError as e:
            logger.error(f"Error creating indexes: {e}")

# Singleton instance
mongodb_manager = MongoDBManager()

# Model classes để tương thích với code hiện tại
class Conversation:
    def __init__(self, user_input: str, bot_response: str, created_at: Optional[datetime] = None):
        self.user_input = user_input
        self.bot_response = bot_response
        self.created_at = created_at or datetime.utcnow()
    
    def save(self):
        """Lưu conversation vào RAG collection"""
        try:
            doc = {
                "user_input": self.user_input,
                "bot_response": self.bot_response,
                "created_at": self.created_at
            }
            result = mongodb_manager.conversations.insert_one(doc)
            return str(result.inserted_id)
        except PyMongoError as e:
            logger.error(f"Error saving conversation: {e}")
            return None

class EmergencyLog:
    def __init__(self, user_id: str, message: str, action: str, status: str, timestamp: Optional[datetime] = None):
        self.user_id = user_id
        self.message = message
        self.action = action
        self.status = status
        self.timestamp = timestamp or datetime.utcnow()
    
    def save(self):
        """Lưu emergency log vào RAG collection"""
        try:
            doc = {
                "user_id": self.user_id,
                "message": self.message,
                "action": self.action,
                "status": self.status,
                "timestamp": self.timestamp
            }
            result = mongodb_manager.emergency_logs.insert_one(doc)
            return str(result.inserted_id)
        except PyMongoError as e:
            logger.error(f"Error saving emergency log: {e}")
            return None

class UserSession:
    def __init__(self, started_at: Optional[datetime] = None):
        self.started_at = started_at or datetime.utcnow()
    
    def save(self):
        """Lưu user session vào RAG collection"""
        try:
            doc = {
                "started_at": self.started_at
            }
            result = mongodb_manager.user_sessions.insert_one(doc)
            return str(result.inserted_id)
        except PyMongoError as e:
            logger.error(f"Error saving user session: {e}")
            return None

class MentalStateHistory:
    def __init__(self, mental_state: str, detected_at: Optional[datetime] = None):
        self.mental_state = mental_state
        self.detected_at = detected_at or datetime.utcnow()
    
    def save(self):
        """Lưu mental state history vào RAG collection"""
        try:
            doc = {
                "mental_state": self.mental_state,
                "detected_at": self.detected_at
            }
            result = mongodb_manager.mental_state_history.insert_one(doc)
            return str(result.inserted_id)
        except PyMongoError as e:
            logger.error(f"Error saving mental state history: {e}")
            return None

class Feedback:
    def __init__(self, session_id: str, user_input: str, bot_response: str, 
                 feedback_type: str, user_feedback_text: Optional[str] = None,
                 risk_level: Optional[str] = None, emotion_label: Optional[str] = None,
                 timestamp: Optional[datetime] = None):
        self.session_id = session_id
        self.user_input = user_input
        self.bot_response = bot_response
        self.feedback_type = feedback_type
        self.user_feedback_text = user_feedback_text
        self.risk_level = risk_level
        self.emotion_label = emotion_label
        self.timestamp = timestamp or datetime.utcnow()
    
    def save(self):
        """Lưu feedback vào Feedback collection (riêng biệt)"""
        try:
            doc = {
                "session_id": self.session_id,
                "user_input": self.user_input,
                "bot_response": self.bot_response,
                "feedback_type": self.feedback_type,
                "user_feedback_text": self.user_feedback_text,
                "risk_level": self.risk_level,
                "emotion_label": self.emotion_label,
                "timestamp": self.timestamp,
                "created_at": self.timestamp
            }
            result = mongodb_manager.user_feedback.insert_one(doc)
            return str(result.inserted_id)
        except PyMongoError as e:
            logger.error(f"Error saving feedback: {e}")
            return None

# Functions để tương thích với code hiện tại
def create_db():
    """Tạo database và collections (MongoDB tự động tạo)"""
    try:
        # Test connection
        mongodb_manager.client.server_info()
        logger.info("MongoDB database ready")
        logger.info("Database: qdrant_mongoDB")
        logger.info("Collections: conversations, emergency_logs, user_sessions, mental_state_history, user_feedback, feedback_analytics")
        return True
    except PyMongoError as e:
        logger.error(f"Error creating database: {e}")
        return False

# SessionLocal tương thích (không cần thiết cho MongoDB nhưng giữ để tương thích)
class SessionLocal:
    def __init__(self):
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def add(self, obj):
        """Add object to MongoDB"""
        if hasattr(obj, 'save'):
            return obj.save()
        return None
    
    def commit(self):
        """Commit changes (MongoDB tự động commit)"""
        pass
    
    def close(self):
        """Close session"""
        pass
