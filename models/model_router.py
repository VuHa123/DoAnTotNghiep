from .chatbot_model import chatbot_response, end_chat_and_save
from .gemini import gemini_chat

def get_response(user_input, history, model_name="llama"):
    if model_name == "llama":
        return chatbot_response(user_input, history)
    elif model_name == "gemini":
        return gemini_chat(user_input, history)
    else:
        raise ValueError("Model không hợp lệ")