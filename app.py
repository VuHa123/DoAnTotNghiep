import gradio as gr
from models.model_router import get_response, end_chat_and_save

history = []
model_choice = "llama"

suggestions = {
    "Depression": "👉 Hãy thử viết nhật ký, nghỉ ngơi, hoặc tâm sự với người đáng tin cậy.",
    "Anxiety": "👉 Bạn có thể thử hít thở sâu, thiền định hoặc nghỉ giải lao ngắn.",
    "Normal": "👍 Tiếp tục duy trì trạng thái tích cực nhé!"
}

def chat(user_input, model_select):
    global history, model_choice
    model_choice = model_select
    reply, label = get_response(user_input, history, model_name=model_choice)
    suggestion = suggestions.get(label, "")
    history_display = "\n\n".join([f"🙋 {q}\n🤖 {a}" for q, a in history])
    return reply, label, suggestion, history_display

def save():
    message, _ = end_chat_and_save(history)
    return message

def clear():
    global history
    history = []
    return "", "", "", ""

gui = gr.Interface(
    fn=chat,
    inputs=[
        gr.Textbox(label="Bạn đang cảm thấy thế nào hôm nay?"),
        gr.Radio(choices=["llama", "gemini"], value="llama", label="Chọn mô hình")
    ],
    outputs=[
        gr.Textbox(label="Phản hồi từ chatbot"),
        gr.Textbox(label="Nhãn cảm xúc"),
        gr.Textbox(label="Gợi ý hành động phù hợp"),
        gr.Textbox(label="Lịch sử hội thoại", lines=10)
    ],
    live=False,
    allow_flagging="never",
    title="Chatbot tư vấn tâm lý"
)

with gr.Row():
    save_btn = gr.Button("💾 Lưu lại buổi trò chuyện")
    clear_btn = gr.Button("🗑️ Xoá toàn bộ")
    save_btn.click(save, outputs=gr.Textbox())
    clear_btn.click(clear, outputs=[gui.outputs[i] for i in range(4)])

gui.launch()