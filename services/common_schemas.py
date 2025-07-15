from pydantic import BaseModel
from typing import Optional

class ChatServiceInput(BaseModel):
    user_message: str
    sentiment: Optional[str] = None
    mental_state: Optional[str] = None
    risk_level: Optional[str] = None

class ChatServiceOutput(BaseModel):
    success: bool
    response: str
    source: str
    warning: Optional[str] = None

class RiskAssessmentOutput(BaseModel):
    risk_level: str
    confidence: float

class SentimentOutput(BaseModel):
    sentiment: str
    confidence: float

class MentalStateOutput(BaseModel):
    mental_state: str
    confidence: float

class EmergencyOutput(BaseModel):
    status: str
    message: str
    action: Optional[str] = None 