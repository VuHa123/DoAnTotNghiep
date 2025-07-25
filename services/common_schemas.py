from pydantic import BaseModel
from typing import Optional

# Chỉ giữ lại các schema còn dùng thực tế cho API mới
# (Ví dụ: SentimentOutput, MentalStateOutput nếu còn dùng ở chatbot_api.py)

class SentimentOutput(BaseModel):
    sentiment: str
    score: Optional[float] = None

class MentalStateOutput(BaseModel):
    mental_state: str
    confidence: Optional[float] = None 