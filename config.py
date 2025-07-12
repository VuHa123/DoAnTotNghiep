import os
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "demo-key")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///chatbot.db")
HOTLINE = "0984.104.115"
HOURS = {"start": "07:30", "end": "22:00"}
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
