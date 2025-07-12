import os
from dotenv import load_dotenv
load_dotenv()

class Settings:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    DATABASE_URL = os.getenv("DATABASE_URL")
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"

settings = Settings()