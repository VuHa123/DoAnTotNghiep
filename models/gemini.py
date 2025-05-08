def gemini_chat(user_input, history):
    # Giả lập phản hồi từ mô hình Gemini
    reply = f"[Gemini phản hồi] Mình nghe bạn nói: '{user_input}'. Mình luôn ở đây nếu bạn cần chia sẻ."
    history.append((user_input, reply))
    return reply, history
