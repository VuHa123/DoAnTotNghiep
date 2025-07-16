import gradio as gr
import sys
sys.path.append("..")
from chatbot_inference import ChatbotInference

# Khởi tạo model fine-tune

chatbot = ChatbotInference(checkpoint_name="checkpoint-1098")

chatbot.load_model()

chat_goals = [
    "Trút bầu tâm sự 😢",
    "Hiểu rõ bản thân 🧭",
    "Vượt qua cảm xúc tiêu cực 🛠️"
]

emotion_icons = {
    "nhẹ": "🙂",
    "trung bình": "😟",
    "nặng": "😢",
    "căng thẳng": "😟",
    "khẩn cấp": "🚨"
}

STATE_MAIN = "main"
STATE_EMERGENCY = "emergency"
STATE_SUMMARY = "summary"
STATE_SETTINGS = "settings"

SUMMARY_TRIGGER = 5

# --- Hàm xử lý ---
def start_chat(goal):
    welcome = f"Bạn đã chọn: {goal}. Hãy bắt đầu chia sẻ nhé!"
    return gr.update(visible=False), gr.update(visible=True), [], "", "😟 Cảm xúc hiện tại: Căng thẳng", STATE_MAIN, 0, False, welcome

def chat_with_bot(user_input, chat_history, chat_goal, turn_count, summary_shown, ui_state):
    turn_count = (turn_count or 0) + 1
    response = chatbot.generate_response(user_input)
    chat_history = chat_history or []
    chat_history.append({"role": "user", "content": user_input})
    # Phản hồi 3 bước (giả lập: luôn đồng cảm)
    response_full = response + "\n→ [Chiến lược phản hồi: Đồng cảm]"
    chat_history.append({"role": "assistant", "content": response_full})

    # Phân tích cảm xúc (giả lập)
    current_emotion = "căng thẳng"
    icon = emotion_icons.get(current_emotion, "🙂")
    emotion_status = f"{icon} Cảm xúc hiện tại: {current_emotion.capitalize()}"

    # Kiểm tra nguy cơ khẩn cấp (giả lập)
    emergency = False
    if "tự tử" in user_input.lower() or "kết thúc" in user_input.lower():
        emergency = True
        ui_state = STATE_EMERGENCY

    # Gợi ý tóm tắt sau X lượt chat
    if turn_count >= SUMMARY_TRIGGER and not summary_shown and not emergency:
        ui_state = STATE_SUMMARY
        summary_shown = True

    return "", chat_history, emotion_status, ui_state, turn_count, summary_shown, ""

def continue_after_emergency(chat_history, turn_count):
    # Quay lại chat bình thường
    return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), STATE_MAIN

def call_support():
    return "📞 Đang kết nối tổng đài 115..."

def save_summary():
    return "📝 Tóm tắt đã được lưu!"

def end_chat():
    # Quay về màn hình chính
    return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), [], "", "😟 Cảm xúc hiện tại: Căng thẳng", STATE_MAIN, 0, False, ""

def open_settings():
    return gr.update(visible=True)

def close_settings():
    return gr.update(visible=False)

with gr.Blocks(title="Mentalbot - Chatbot Tâm Lý", theme=gr.themes.Soft()) as demo:
    # --- Màn hình chính ---
    with gr.Row(visible=True) as main_screen:
        with gr.Column():
            gr.Markdown("## 🧠 MENTALBOT\nTrò chuyện tâm lý cùng bạn")
            gr.Markdown("👋 **Xin chào! Bạn muốn tôi hỗ trợ gì hôm nay?**")

            chat_goal = gr.Radio(choices=list(chat_goals), label="", interactive=True)

            start_btn = gr.Button("Bắt đầu", variant="primary")
        settings_btn = gr.Button("⚙️ Tuỳ chọn", variant="secondary")
        welcome_text = gr.Markdown(visible=False)

    # --- Màn hình chat ---
    with gr.Row(visible=False) as chat_screen:
        with gr.Column():
            chatbot_ui = gr.Chatbot(label="", height=300, type="messages")
            emotion_status = gr.Textbox(label="", interactive=False, value="😟 Cảm xúc hiện tại: Căng thẳng")
            user_input = gr.Textbox(placeholder="Nhập nội dung...", label="", lines=2)
            send_btn = gr.Button("Gửi", variant="primary")

    # --- Màn hình cảnh báo khẩn cấp ---
    with gr.Row(visible=False) as emergency_screen:
        gr.Markdown("## 🔴 CẢNH BÁO KHẨN CẤP")
        gr.Markdown(
            "Có vẻ bạn đang trải qua trạng thái rất khó khăn. Nếu bạn đang nghĩ đến việc tự làm hại bản thân, xin hãy gọi: ☎ 115 hoặc nhắn tin với người thân đáng tin cậy.\n\nTôi vẫn ở đây nếu bạn cần chia sẻ thêm."
        )
        with gr.Row():
            continue_btn = gr.Button("Tôi ổn, tiếp tục")
            call_btn = gr.Button("Gọi hỗ trợ 📞")
        emergency_result = gr.Textbox(label="", interactive=False)

    # --- Màn hình tóm tắt cảm xúc ---
    with gr.Row(visible=False) as summary_screen:
        gr.Markdown("## 🌅 TÓM TẮT CẢM XÚC")
        summary_text = gr.Markdown(
            "Hôm nay bạn đã chia sẻ nhiều cảm xúc khó nói. Có vẻ bạn đang trải qua cảm giác lo âu và cô đơn. Nhưng bạn vẫn đang cố gắng, và đó là điều rất đáng quý.\n\nHãy nhớ rằng bạn không hề đơn độc."
        )
        with gr.Row():
            save_summary_btn = gr.Button("Lưu tóm tắt 📝")
            end_btn = gr.Button("Kết thúc buổi chat")
        save_result = gr.Textbox(label="", interactive=False)

    # --- Màn hình cài đặt ---
    with gr.Row(visible=False) as settings_screen:
        gr.Markdown("## ⚙️ TUỲ CHỌN GIAO DIỆN")
        pastel = gr.Checkbox(label="Giao diện dịu nhẹ (Pastel)", value=True)
        gentle_tone = gr.Checkbox(label="Giọng điệu phản hồi: Nhẹ nhàng", value=True)
        show_emotion = gr.Checkbox(label="Hiện cảm xúc (icon)", value=True)
        show_mental = gr.Checkbox(label="Hiện trạng thái tâm lý (ẩn)", value=False)
        lang = gr.Dropdown(choices=[("Tiếng Việt", "Tiếng Việt"), ("English", "English")], value="Tiếng Việt", label="Ngôn ngữ")
        close_settings_btn = gr.Button("Đóng")

    # --- State ẩn ---
    ui_state = gr.State(STATE_MAIN)
    turn_count = gr.State(0)
    summary_shown = gr.State(False)

    # --- Sự kiện ---
    start_btn.click(
        start_chat,
        inputs=[chat_goal],
        outputs=[main_screen, chat_screen, chatbot_ui, user_input, emotion_status, ui_state, turn_count, summary_shown, welcome_text]
    )

    send_btn.click(
        chat_with_bot,
        inputs=[user_input, chatbot_ui, chat_goal, turn_count, summary_shown, ui_state],
        outputs=[user_input, chatbot_ui, emotion_status, ui_state, turn_count, summary_shown, welcome_text]
    )

    # Điều hướng giữa các màn hình dựa vào ui_state
    def route_ui(ui_state_val):
        return [
            gr.update(visible=ui_state_val == STATE_MAIN),
            gr.update(visible=ui_state_val == STATE_MAIN or ui_state_val == STATE_MAIN),
            gr.update(visible=ui_state_val == STATE_EMERGENCY),
            gr.update(visible=ui_state_val == STATE_SUMMARY),
            gr.update(visible=ui_state_val == STATE_SETTINGS)
        ]

    continue_btn.click(
        continue_after_emergency,
        inputs=[chatbot_ui, turn_count],
        outputs=[chat_screen, emergency_screen, summary_screen, ui_state]
    )
    call_btn.click(call_support, outputs=emergency_result)

    save_summary_btn.click(save_summary, outputs=save_result)
    end_btn.click(
        end_chat,
        outputs=[main_screen, chat_screen, emergency_screen, summary_screen, chatbot_ui, user_input, emotion_status, ui_state, turn_count, summary_shown, welcome_text]
    )

    settings_btn.click(open_settings, outputs=settings_screen)
    close_settings_btn.click(close_settings, outputs=settings_screen)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
