from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
from services.setiment_analysis.analyzer import detect_sentiment_label

Base = declarative_base()
engine = create_engine("sqlite:///chatbot.db")
SessionLocal = sessionmaker(bind=engine)

class Conversation(Base):
    __tablename__ = 'conversations'
    id = Column(Integer, primary_key=True)
    user_input = Column(Text)
    bot_response = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

class EmergencyLog(Base):
    __tablename__ = 'emergency_logs'
    id = Column(Integer, primary_key=True)
    user_id = Column(String)
    message = Column(Text)
    action = Column(String)  # hotline, staff_alert, etc.
    status = Column(String)  # success, failed, etc.
    timestamp = Column(DateTime, default=datetime.utcnow)

class UserSession(Base):
    __tablename__ = 'user_sessions'
    id = Column(Integer, primary_key=True)
    started_at = Column(DateTime, default=datetime.utcnow)

class MentalStateHistory(Base):
    __tablename__ = 'mental_state_history'
    id = Column(Integer, primary_key=True)
    mental_state = Column(String)
    detected_at = Column(DateTime, default=datetime.utcnow)

class Feedback(Base):
    __tablename__ = 'feedback'
    id = Column(Integer, primary_key=True)
    session_id = Column(String, nullable=False)
    user_input = Column(Text, nullable=False)
    bot_response = Column(Text, nullable=False)
    feedback_type = Column(String, nullable=False)  # 'like' or 'dislike'
    user_feedback_text = Column(Text)  # Optional reason from user
    timestamp = Column(DateTime, default=datetime.utcnow)
    risk_level = Column(String)  # Store risk level for analysis
    emotion_label = Column(String)  # Store emotion for analysis

def create_db():
    Base.metadata.create_all(bind=engine)
