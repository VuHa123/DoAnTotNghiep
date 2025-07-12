from .chatbot_model import chatbot_response, load_chatbot_pipeline

pipeline = load_chatbot_pipeline()

def llama_chat(user_input, history):
    return chatbot_response(pipeline, user_input, history)