import os
import json
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs', 'chat_logs')
os.makedirs(LOG_DIR, exist_ok=True)

def save_conversation_log(history):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = os.path.join(LOG_DIR, f"conversation_{timestamp}.json")

    with open(log_filename, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=4)

    return log_filename