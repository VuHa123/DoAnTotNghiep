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
    # Nếu có cảnh báo nguy cơ, nhắc nhở trong hội thoại
    if "tự tử" in user_input.lower() or "kết thúc" in user_input.lower():
        response_full = "🚨 <b>Lưu ý: Nếu bạn đang gặp nguy hiểm, hãy gọi ngay Hotline 096.306.1414 hoặc liên hệ người thân!</b>\n" + response_full
    chat_history.append({"role": "assistant", "content": response_full})

    # Phân tích cảm xúc (giả lập)
    current_emotion = "căng thẳng"
    icon = emotion_icons.get(current_emotion, "🙂")
    emotion_status = f"{icon} Cảm xúc hiện tại: {current_emotion.capitalize()}"

    # Kiểm tra nguy cơ khẩn cấp (giả lập)
    emergency = False
    banner_visible = False
    if "tự tử" in user_input.lower() or "kết thúc" in user_input.lower():
        emergency = True
        ui_state = STATE_EMERGENCY
        banner_visible = True
    # Gợi ý tóm tắt sau X lượt chat
    if turn_count >= SUMMARY_TRIGGER and not summary_shown and not emergency:
        ui_state = STATE_SUMMARY
        summary_shown = True

    return "", chat_history, emotion_status, ui_state, turn_count, summary_shown, "", gr.update(visible=banner_visible)

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

# --- Hàm xử lý mới cho banner cảnh báo và hotline ---
def show_banner_warning():
    return gr.update(visible=True)

def hide_banner_warning():
    return gr.update(visible=False)

def hotline_click():
    return "📞 Đang kết nối tổng đài 096.306.1414...", gr.update(visible=True)

with gr.Blocks(
    css="""
    body { background: #e3f6fd !important; }
    .banner-warning {
        position: fixed;
        left: 24px;
        bottom: 24px;
        z-index: 1000;
        background: #fff3cd;
        color: #b94a48;
        border: 2px solid #f5c06f;
        border-radius: 12px;
        padding: 16px 32px;
        font-weight: bold;
        font-size: 18px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        animation: shake 0.5s;
        display: block;
    }
    @keyframes shake {
        0% { transform: translateX(0); }
        20% { transform: translateX(-8px); }
        40% { transform: translateX(8px); }
        60% { transform: translateX(-8px); }
        80% { transform: translateX(8px); }
        100% { transform: translateX(0); }
    }
    #hotline-btn {
        position: fixed;
        bottom: 24px;
        right: 24px;
        z-index: 1000;
        background: #ff5252;
        color: white;
        border-radius: 32px;
        font-size: 18px;
        padding: 16px 28px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.12);
        border: none;
        cursor: pointer;
        font-weight: bold;
        transition: background 0.2s;
    }
    #hotline-btn:hover {
        background: #d32f2f;
    }
    body { background: #e3f6fd !important; }
    #chat-container {
        background: #e3f6fd;
        border-radius: 18px;
        padding: 24px 16px 80px 16px;
        min-height: 400px;
        max-width: 600px;
        margin: 32px auto 0 auto;
        box-shadow: 0 2px 16px rgba(0,0,0,0.06);
        position: relative;
    }
    .message-row {
        display: flex;
        margin-bottom: 10px;
    }
    .message-bot {
        justify-content: flex-start;
    }
    .message-user {
        justify-content: flex-end;
    }
    .bubble {
        max-width: 70%;
        padding: 12px 18px;
        border-radius: 18px;
        font-size: 16px;
        margin: 2px 0;
        background: #ffffffcc;
        color: #222;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }
    .bubble.user {
        background: #b2ebf2;
        color: #01579b;
        border-bottom-right-radius: 4px;
    }
    .bubble.bot {
        background: #fff;
        color: #222;
        border-bottom-left-radius: 4px;
    }
    #emotion-status {
        margin: 12px 0 0 0;
        font-size: 18px;
        font-weight: bold;
        color: #0288d1;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    #input-row {
        display: flex;
        gap: 8px;
        margin-top: 18px;
    }
    #user-input {
        flex: 1;
        border-radius: 12px;
        border: 1px solid #b2ebf2;
        padding: 10px 14px;
        font-size: 16px;
        outline: none;
    }
    #send-btn {
        background: #0288d1;
        color: white;
        border: none;
        border-radius: 12px;
        padding: 10px 22px;
        font-size: 16px;
        font-weight: bold;
        cursor: pointer;
        transition: background 0.2s;
    }
    #send-btn:hover {
        background: #01579b;
    }
    #hotline-icon {
        position: absolute;
        bottom: 12px;
        right: 12px;
        cursor: pointer;
        z-index: 10;
        transition: transform 0.2s;
    }
    #hotline-icon:hover { transform: scale(1.15); }
    """
) as demo:
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
            # Nút hotline icon động nhỏ trong khung chat
            hotline_icon = gr.HTML('<div id="hotline-icon" onclick="window.open(\'tel:0963061414\')"><svg width="40" height="40" viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="20" cy="20" r="20" fill="#0288d1"/><path d="M28 24.5c-1.2 0-2.4-.2-3.5-.6-.5-.2-1.1 0-1.4.4l-1.1 1.7c-3.2-1.7-5.8-4.3-7.5-7.5l1.7-1.1c.4-.3.6-.9.4-1.4-.4-1.1-.6-2.3-.6-3.5 0-.6-.4-1-1-1H11c-.6 0-1 .4-1 1C10 23.1 16.9 30 25 30c.6 0 1-.4 1-1v-3c0-.6-.4-1-1-1z" fill="#fff"><animateTransform attributeName="transform" type="scale" values="1;1.15;1" dur="1s" repeatCount="indefinite"/></path></svg></div>', visible=True)

    # --- Banner cảnh báo (băng rôn) ---
    banner_warning = gr.HTML('<div id="banner-warning" class="banner-warning">🚨 CẢNH BÁO: Nếu bạn đang gặp nguy hiểm hoặc có ý định tự làm hại bản thân, hãy gọi ngay <b>Hotline 096.306.1414</b> hoặc liên hệ người thân!</div>', visible=False)

    # --- Nút hotline nổi ---
    hotline_btn = gr.Button("📞 Hotline: 096.306.1414", elem_id="hotline-btn", visible=True)
    hotline_result = gr.Textbox(label="", interactive=False, visible=False)

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
    is_generating = gr.State(False)

    # --- Sự kiện ---
    start_btn.click(
        start_chat,
        inputs=[chat_goal],
        outputs=[main_screen, chat_screen, chatbot_ui, user_input, emotion_status, ui_state, turn_count, summary_shown, welcome_text]
    )

    def on_send(user_input, chat_history, chat_goal, turn_count, summary_shown, ui_state, is_generating):
        if is_generating:
            # Nếu đang sinh, dừng lại (giả lập)
            return gr.update(), chat_history, emotion_status, ui_state, turn_count, summary_shown, welcome_text, banner_warning, False, gr.update(value="Gửi", interactive=True)
        else:
            # Bắt đầu sinh phản hồi
            # (giả lập: không thực sự dừng thread, chỉ đổi trạng thái)
            result = chat_with_bot(user_input, chat_history, chat_goal, turn_count, summary_shown, ui_state)
            return *result, True, gr.update(value="Tạm dừng", interactive=True)

    send_btn.click(
        on_send,
        inputs=[user_input, chatbot_ui, chat_goal, turn_count, summary_shown, ui_state, is_generating],
        outputs=[user_input, chatbot_ui, emotion_status, ui_state, turn_count, summary_shown, welcome_text, banner_warning, is_generating, send_btn]
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

    hotline_btn.click(hotline_click, outputs=[hotline_result])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
