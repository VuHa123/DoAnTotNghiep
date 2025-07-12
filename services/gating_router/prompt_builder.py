def build_prompt(user_message: str, mental_state: str, sentiment_intensity: str) -> str:
    instruction = (
        "Bạn là một chuyên gia tâm lý. Hãy trả lời người dùng với giọng điệu nhẹ nhàng, đồng cảm. "
        "Dựa vào trạng thái tâm lý và mức độ cảm xúc của họ."
    )
    input_text = (
        f"Tin nhắn: {user_message}\n"
        f"Trạng thái tâm lý: {mental_state}\n"
        f"Mức độ cảm xúc: {sentiment_intensity}\n"
        "Phản hồi:"
    )
    return f"{instruction}\n{input_text}"
