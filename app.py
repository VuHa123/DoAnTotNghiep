import gradio as gr
from models.chatbot_model import load_chatbot_pipeline, chatbot_response, end_chat_and_save
from utils.data_loader import load_chat_format_data

# Load model và pipeline
generator = load_chatbot_pipeline()

# Nạp dữ liệu hội thoại từ file
chat_data = load_chat_format_data("data/conversations.json")

# Giao diện Gradio
with gr.Blocks() as demo:
    gr.Markdown("## \U0001F9E0 Chatbot Tâm Lý - LLaMA-3.2-1B (Demo)")

    chatbot_ui = gr.Chatbot()
    msg = gr.Textbox(label="Bạn muốn tâm sự điều gì?", placeholder="Hãy chia sẻ với mình...")
    state = gr.State([])  # lưu history

    with gr.Row():
        submit_btn = gr.Button("Gửi")
        end_btn = gr.Button("Kết thúc trò chuyện \U0001F4BE")

    def respond(message, history):
        response, history = chatbot_response(generator, message, history)
        return history, history

    def end_chat(history):
        confirm_msg, cleared = end_chat_and_save(history)
        return cleared, cleared, confirm_msg

    submit_btn.click(respond, [msg, state], [chatbot_ui, state])
    msg.submit(respond, [msg, state], [chatbot_ui, state])

    end_btn.click(end_chat, [state], [chatbot_ui, state, msg])

if __name__ == "__main__":
    demo.launch()