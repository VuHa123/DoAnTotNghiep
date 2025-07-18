from fastapi import FastAPI, Request
from pydantic import BaseModel

# Import hoặc khởi tạo model chatbot của bạn
from chatbot_inference import ChatbotInference

app = FastAPI()
chatbot = ChatbotInference(checkpoint_name="checkpoint-1098")
chatbot.load_model()

class ChatRequest(BaseModel):
    user_input: str
    chat_history: list = []

@app.post("/chatbot/generate")
def generate_response(request: ChatRequest):
    # Gọi model sinh phản hồi
    response = chatbot.generate_response(request.user_input)
    return {"response": response}
