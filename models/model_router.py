from models.llama_model import llama_chat
from models.gemini_api import gemini_chat

def get_response(user_input, history, model_name="llama"):
    if model_name == "gemini":
        return gemini_chat(user_input, history)
    else:
        return llama_chat(user_input, history)