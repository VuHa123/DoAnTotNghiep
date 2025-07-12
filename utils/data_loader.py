import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "Dataset", "llama_instruction_data.jsonl")

def load_chat_format_data(path="/Dataset/llama_instruction_data.jsonl"):
    """
    Nạp dữ liệu hội thoại từ file JSON.

    Args:
        path (str): Đường dẫn đến file JSON chứa dữ liệu hội thoại. Mặc định là "../Dataset/llama_instruction_data.jsonl".

    Returns:
        list: Dữ liệu hội thoại dưới dạng danh sách các cặp câu hỏi và trả lời.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data