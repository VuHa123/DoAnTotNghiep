# utils/common.py - Common utilities and shared components

from pydantic import BaseModel
from typing import List
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chatbot")

class ChatRequest(BaseModel):
    user_input: str
    history: List[str]

def anonymize(text: str) -> str:
    """Anonymize sensitive data in text"""
    return text.replace("@", "[email]")

class EmergencyException(Exception):
    """Custom exception for emergency situations"""
    def __init__(self, message: str):
        super().__init__(message)

def encrypt(data: str) -> str:
    """Simple encryption for sensitive data"""
    return data[::-1]

def is_safe_text(text: str) -> bool:
    """Check if text contains safe content"""
    unsafe_keywords = ["suicide", "kill", "die", "death"]
    return not any(keyword in text.lower() for keyword in unsafe_keywords)

def sanitize_input(text: str) -> str:
    """Sanitize user input"""
    import re
    # Remove potentially dangerous characters
    sanitized = re.sub(r'[<>"\']', '', text)
    return sanitized.strip()

def validate_session_id(session_id: str) -> bool:
    """Validate session ID format"""
    import re
    # UUID format validation
    uuid_pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE)
    return bool(uuid_pattern.match(session_id)) 