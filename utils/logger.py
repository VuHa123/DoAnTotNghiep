import os
import json
from datetime import datetime

LOG_DIR = "logs/chat_logs"
os.makedirs(LOG_DIR, exist_ok=True)

def save_conversation_log(history):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(LOG_DIR, f"chat_{timestamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    return path