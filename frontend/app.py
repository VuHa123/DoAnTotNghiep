# frontend/app.py - Enhanced Mental Health Chatbot Interface

import gradio as gr
import requests
import uuid
from dotenv import load_dotenv
load_dotenv("token.env")
import os
print("HF_TOKEN:", os.environ.get("HF_TOKEN"))

# API configuration
API_URL = "http://localhost:8000/chat"
SESSION_ID = str(uuid.uuid4())

# Enhanced suggestions based on emotion labels
suggestions = {
    "Depression": "👉 Hãy thử viết nhật ký, nghỉ ngơi, hoặc tâm sự với người đáng tin cậy.",
    "Anxiety": "👉 Bạn có thể thử hít thở sâu, thiền định hoặc nghỉ giải lao ngắn.",
    "Normal": "👍 Tiếp tục duy trì trạng thái tích cực nhé!"
}

def chat_with_bot(user_input, chat_history=[], model_select="llama"):
    """Chat function chỉ gọi API Gateway, không fallback nội bộ"""
    payload = {
        "user_input": user_input,
        "history": chat_history,
        "session_id": SESSION_ID
    }
    try:
        res = requests.post(API_URL, json=payload)
        res.raise_for_status()
        result = res.json()
        bot_reply = result.get("bot_response", "")
        emotion_label = result.get("emotion_label", "Normal")
        risk_level = result.get("risk_level", "normal")
        suggestion = suggestions.get(emotion_label, "Hãy tiếp tục chia sẻ cảm xúc của bạn.")
    except Exception as e:
        # Không fallback, chỉ báo lỗi cho người dùng
        bot_reply = "[Lỗi] Hệ thống đang bảo trì hoặc quá tải. Vui lòng thử lại sau."
        emotion_label = ""
        suggestion = ""
        risk_level = ""
    chat_history.append((user_input, bot_reply))
    return "", chat_history, emotion_label, suggestion, risk_level

def save_conversation():
    """Tạm thời vô hiệu hóa lưu hội thoại ở frontend, chỉ thực hiện ở backend qua API Gateway nếu cần"""
    return "Chức năng lưu hội thoại hiện chỉ hỗ trợ qua hệ thống backend."

def clear_conversation():
    """Clear conversation history"""
    return "", [], "", "", ""

# Enhanced Gradio interface
with gr.Blocks(title="Chatbot Tư Vấn Tâm Lý", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🤖 Chatbot Hỗ Trợ Tâm Lý
    
    Chia sẻ cảm xúc của bạn và nhận sự hỗ trợ từ AI. Hệ thống sẽ phân tích 
    trạng thái tâm lý và đưa ra lời khuyên phù hợp.
    """)
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Trò chuyện", height=400)
            user_input = gr.Textbox(
                placeholder="Bạn đang cảm thấy như thế nào?",
                label="Tin nhắn của bạn:",
                lines=2
            )
            
        with gr.Column(scale=1):
            model_select = gr.Radio(
                choices=["llama", "gemini"], 
                value="llama", 
                label="Chọn mô hình"
            )
            emotion_label = gr.Textbox(label="Nhãn cảm xúc", interactive=False)
            suggestion = gr.Textbox(label="Gợi ý hành động", interactive=False)
            risk_level = gr.Textbox(label="Mức độ rủi ro", interactive=False)
    
    with gr.Row():
        send_btn = gr.Button("💬 Gửi tin nhắn", variant="primary")
        save_btn = gr.Button("💾 Lưu cuộc trò chuyện")
        clear_btn = gr.Button("🗑️ Xóa lịch sử")
    
    # Event handlers
    user_input.submit(
        chat_with_bot, 
        [user_input, chatbot, model_select], 
        [user_input, chatbot, emotion_label, suggestion, risk_level]
    )
    send_btn.click(
        chat_with_bot, 
        [user_input, chatbot, model_select], 
        [user_input, chatbot, emotion_label, suggestion, risk_level]
    )
    save_btn.click(save_conversation, outputs=gr.Textbox(label="Kết quả lưu"))
    clear_btn.click(clear_conversation, outputs=[user_input, chatbot, emotion_label, suggestion, risk_level])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
