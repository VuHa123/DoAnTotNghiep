import json

def load_chat_format_data(path="data/conversations.json"):
    """
    Nạp dữ liệu hội thoại từ file JSON.

    Args:
        path (str): Đường dẫn đến file JSON chứa dữ liệu hội thoại. Mặc định là "data/conversations.json".

    Returns:
        list: Dữ liệu hội thoại dưới dạng danh sách các cặp câu hỏi và trả lời.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data