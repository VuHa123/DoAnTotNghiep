# frontend/app.py - Enhanced Mental Health Chatbot Interface

import gradio as gr
import sys
sys.path.append("..")
from chatbot_inference import ChatbotInference

# Khởi tạo model fine-tune
chatbot = ChatbotInference(checkpoint_name="checkpoint-549")
chatbot.load_model()

# Enhanced suggestions based on emotion labels (tạm thời không dùng)
suggestions = {}

def chat_with_bot(user_input, chat_history=[], model_select="llama"):
    """Chat trực tiếp với model fine-tune"""
    response = chatbot.generate_response(user_input)
    # Đảm bảo chat_history là list các dict đúng format cho Gradio Chatbot
    if len(chat_history) > 0 and isinstance(chat_history[0], tuple):
        # Chuyển đổi tuple sang dict (nếu là dữ liệu cũ)
        chat_history = [
            {"role": "user", "content": u} if i % 2 == 0 else {"role": "assistant", "content": u}
            for i, pair in enumerate(chat_history) for u in pair
        ]
    # Thêm lượt chat mới
    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": response})
    return "", chat_history, "", "", ""

def save_conversation():
    return "Chức năng lưu hội thoại hiện chỉ hỗ trợ qua hệ thống backend."

def clear_conversation():
    return "", [], "", "", ""

# Enhanced Gradio interface
with gr.Blocks(
    title="Chatbot Tư Vấn Tâm Lý",
    theme=gr.themes.Soft(),
    css="""
* { font-family: 'Arial', 'Tahoma', 'Roboto', 'Noto Sans', 'DejaVu Sans', sans-serif !important; }
textarea, input, .gr-textbox textarea {
    ime-mode: active !important; /* gợi ý bật bộ gõ IME */
    font-family: inherit;
}
"""


) as demo:
    gr.Markdown("""
    # 🤖 Chatbot Hỗ Trợ Tâm Lý
    
    Chia sẻ cảm xúc của bạn và nhận sự hỗ trợ từ AI. Hệ thống sẽ phân tích 
    trạng thái tâm lý và đưa ra lời khuyên phù hợp.
    """)
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot_ui = gr.Chatbot(label="Trò chuyện", height=400, type="messages")
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
        [user_input, chatbot_ui, model_select], 
        [user_input, chatbot_ui, emotion_label, suggestion, risk_level]
    )
    send_btn.click(
        chat_with_bot, 
        [user_input, chatbot_ui, model_select], 
        [user_input, chatbot_ui, emotion_label, suggestion, risk_level]
    )
    save_btn.click(save_conversation, outputs=gr.Textbox(label="Kết quả lưu"))
    clear_btn.click(clear_conversation, outputs=[user_input, chatbot_ui, emotion_label, suggestion, risk_level])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
