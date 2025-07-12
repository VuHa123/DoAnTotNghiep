import time

# Context tracker đơn giản dùng dictionary lưu trạng thái
default_context_store = {}

# Cho phép truyền session_id để tracking theo user/session thực

def update_context(history: list, user_input: str, sentiment: str, mental_state: str, session_id: str = None):
    global default_context_store
    if not session_id:
        session_id = str(int(time.time()))  # dùng timestamp làm id tạm
    default_context_store[session_id] = {
        "history": history,
        "latest_input": user_input,
        "sentiment": sentiment,
        "mental_state": mental_state,
        "updated_at": time.time()
    }