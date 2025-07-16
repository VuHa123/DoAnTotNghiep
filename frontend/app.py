import gradio as gr
import sys
import re
sys.path.append("..")
from chatbot_inference import ChatbotInference

# Khởi tạo model fine-tune
chatbot = ChatbotInference(checkpoint_name="checkpoint-1098")
chatbot.load_model()
chatbot.stop_event.clear()

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

def start_chat(goal):
    welcome = f"Bạn đã chọn: {goal}. Hãy bắt đầu chia sẻ nhé!"
    return gr.update(visible=False), gr.update(visible=True), [], gr.update(value="", interactive=True), "😟 Cảm xúc hiện tại: Căng thẳng", STATE_MAIN, 0, False, welcome

def chat_with_bot(user_input, chat_history, chat_goal, turn_count, summary_shown, ui_state):
    turn_count = (turn_count or 0) + 1
    response = chatbot.generate_response(user_input)
    chat_history = chat_history or []
    chat_history.append({"role": "user", "content": user_input})
    response_full = response

    emergency_keywords = [
        r"tự tử", r"muốn chết", r"kết thúc cuộc đời", r"tự sát",
        r"kết liễu", r"tôi sẽ chết", r"muốn biến mất",
        r"không còn lý do sống", r"tôi tuyệt vọng", r"không muốn tồn tại",
        r"kết thúc tất cả", r"tôi không muốn tiếp tục", r"tôi muốn biến mất"
    ]
    normalized_input = user_input.lower().strip()
    gating_label = "normal"
    for keyword in emergency_keywords:
        if re.search(keyword, normalized_input):
            gating_label = "emergency"
            break

    emergency = False
    banner_visible = False
    if gating_label == "emergency":
        emergency = True
        ui_state = STATE_EMERGENCY
        banner_visible = True
        response_full = "🚨 <b>Lưu ý: Nếu bạn đang gặp nguy hiểm, hãy gọi ngay Hotline 096.306.1414 hoặc liên hệ người thân!</b>\n" + response_full

    current_emotion = "căng thẳng"
    icon = emotion_icons.get(current_emotion, "🙂")
    emotion_status = f"{icon} Cảm xúc hiện tại: {current_emotion.capitalize()}"

    if turn_count >= SUMMARY_TRIGGER and not summary_shown and not emergency:
        ui_state = STATE_SUMMARY
        summary_shown = True

    chat_history.append({"role": "assistant", "content": response_full})

    return (
        gr.update(value="", interactive=True),  # clear user input
        chat_history, emotion_status, ui_state, turn_count,
        summary_shown, "", gr.update(visible=banner_visible),
        gr.update(value="Gửi", interactive=True), False
    )

def continue_after_emergency(chat_history, turn_count):
    return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), STATE_MAIN

def call_support():
    return "📞 Đang kết nối tổng đài 115..."

def save_summary():
    return "📝 Tóm tắt đã được lưu!"

def end_chat():
    return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), [], gr.update(value="", interactive=True), "😟 Cảm xúc hiện tại: Căng thẳng", STATE_MAIN, 0, False, ""

def open_settings():
    return gr.update(visible=True)

def close_settings():
    return gr.update(visible=False)

def hotline_click():
    return "📞 Đang kết nối tổng đài 096.306.1414...", gr.update(visible=True)

# ======== NÚT GỬI ↔ TẠM DỪNG =========

def on_send_click(user_input, chat_history, is_generating):
    if not is_generating:
        if hasattr(chatbot, 'stop_event'):
            chatbot.stop_event.clear()
        return (
            gr.update(value="", interactive=True),
            chat_history,
            gr.update(value="Tạm dừng", interactive=True),
            True
        )
    else:
        if hasattr(chatbot, 'stop_event'):
            chatbot.stop_event.set()
        return (
            gr.update(value=user_input, interactive=True),
            chat_history,
            gr.update(value="Gửi", interactive=True),
            False
        )

def run_generation(user_input, chat_history, chat_goal, turn_count, summary_shown, ui_state):
    new_user_input, updated_history, emotion_status, new_ui_state, new_turn_count, new_summary_shown, welcome_text, banner_visible, _, _ = chat_with_bot(
        user_input, chat_history, chat_goal, turn_count, summary_shown, ui_state
    )
    return (
        new_user_input, updated_history, emotion_status, new_ui_state, new_turn_count,
        new_summary_shown, welcome_text,
        banner_visible,
        gr.update(value="Gửi", interactive=True),
        False
    )





with gr.Blocks(
    css="""
    body { background: #e3f6fd !important; }
    #chat-container { background: #e3f6fd; }
    #hotline-chatbox {
        position: fixed;
        bottom: 16px;
        right: 16px;
        display: flex;
        align-items: center;
        gap: 6px;
        padding: 8px 12px;
        background-color: white;
        border: 1px solid #0288d1;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        cursor: pointer;
        z-index: 100;
    }
    #hotline-chatbox:hover {
        background-color: #e0f7fa;
    }
    """
) as demo:
    with gr.Row(visible=True) as main_screen:
        with gr.Column():
            gr.Markdown("## 🧠 MENTALBOT\nTrò chuyện tâm lý cùng bạn")
            gr.Markdown("👋 **Xin chào! Bạn muốn tôi hỗ trợ gì hôm nay?**")
            chat_goal = gr.Radio(choices=list(chat_goals), label="", interactive=True)
            start_btn = gr.Button("Bắt đầu", variant="primary")
        settings_btn = gr.Button("⚙️ Tuỳ chọn", variant="secondary")
        welcome_text = gr.Markdown(visible=False)

    with gr.Row(visible=False) as chat_screen:
        with gr.Column():
            chatbot_ui = gr.Chatbot(label="", height=300, type="messages")
            emotion_status = gr.Textbox(label="", interactive=False, value="😟 Cảm xúc hiện tại: Căng thẳng")
            user_input = gr.Textbox(placeholder="Nhập nội dung...", label="", lines=2)
            send_btn = gr.Button("Gửi", variant="primary")
            hotline_html = gr.HTML(
                """
                <div id="hotline-chatbox" onclick="window.open('tel:0963061414')">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                        <circle cx="12" cy="12" r="12" fill="#0288d1"/>
                        <path d="M17 15.5c-.9 0-1.8-.2-2.6-.4-.4-.1-.8 0-1 .3l-.8 1.3c-2.4-1.3-4.3-3.2-5.6-5.6l1.3-.8c.3-.2.4-.6.3-1-.2-.8-.4-1.7-.4-2.6 0-.5-.3-.8-.8-.8H6c-.5 0-.8.3-.8.8C5.2 17.1 10.9 22.8 18 22.8c.5 0 .8-.3.8-.8v-2.2c0-.5-.3-.8-.8-.8z" fill="#fff"></path>
                    </svg>
                    <span style="font-weight:bold; font-size:16px; color:#0288d1;">Hotline: 096.306.1414</span>
                </div>
                """,
                visible=True
            )

    banner_warning = gr.HTML('<div id="banner-warning" class="banner-warning">🚨 CẢNH BÁO: Nếu bạn đang gặp nguy hiểm hoặc có ý định tự làm hại bản thân, hãy gọi ngay <b>Hotline 096.306.1414</b> hoặc liên hệ người thân!</div>', visible=False)

    hotline_btn = gr.Button("📞 Hotline: 096.306.1414", elem_id="hotline-btn", visible=False)
    hotline_result = gr.Textbox(label="", interactive=False, visible=False)

    with gr.Row(visible=False) as emergency_screen:
        gr.Markdown("## 🔴 CẢNH BÁO KHẨN CẤP")
        gr.Markdown(
            "Có vẻ bạn đang trải qua trạng thái rất khó khăn. Nếu bạn đang nghĩ đến việc tự làm hại bản thân, xin hãy gọi: ☎ 115 hoặc nhắn tin với người thân đáng tin cậy.\n\nTôi vẫn ở đây nếu bạn cần chia sẻ thêm."
        )
        with gr.Row():
            continue_btn = gr.Button("Tôi ổn, tiếp tục")
            call_btn = gr.Button("Gọi hỗ trợ 📞")
        emergency_result = gr.Textbox(label="", interactive=False)

    with gr.Row(visible=False) as summary_screen:
        gr.Markdown("## 🌅 TÓM TẮT CẢM XÚC")
        summary_text = gr.Markdown("Hôm nay bạn đã chia sẻ nhiều cảm xúc khó nói...")
        with gr.Row():
            save_summary_btn = gr.Button("Lưu tóm tắt 📝")
            end_btn = gr.Button("Kết thúc buổi chat")
        save_result = gr.Textbox(label="", interactive=False)

    with gr.Row(visible=False) as settings_screen:
        gr.Markdown("## ⚙️ TUỲ CHỌN GIAO DIỆN")
        pastel = gr.Checkbox(label="Giao diện dịu nhẹ (Pastel)", value=True)
        gentle_tone = gr.Checkbox(label="Giọng điệu phản hồi: Nhẹ nhàng", value=True)
        show_emotion = gr.Checkbox(label="Hiện cảm xúc (icon)", value=True)
        show_mental = gr.Checkbox(label="Hiện trạng thái tâm lý (ẩn)", value=False)
        lang = gr.Dropdown(choices=[("Tiếng Việt", "Tiếng Việt"), ("English", "English")], value="Tiếng Việt", label="Ngôn ngữ")
        close_settings_btn = gr.Button("Đóng")

    ui_state = gr.State(STATE_MAIN)
    turn_count = gr.State(0)
    summary_shown = gr.State(False)
    is_generating = gr.State(False)

    start_btn.click(
        start_chat,
        inputs=[chat_goal],
        outputs=[main_screen, chat_screen, chatbot_ui, user_input, emotion_status, ui_state, turn_count, summary_shown, welcome_text]
    )

    send_btn.click(
        on_send_click,
        inputs=[user_input, chatbot_ui, is_generating],
        outputs=[user_input, chatbot_ui, send_btn, is_generating]
    ).then(
        run_generation,
        inputs=[user_input, chatbot_ui, chat_goal, turn_count, summary_shown, ui_state],
        outputs=[
            user_input, chatbot_ui, emotion_status, ui_state,
            turn_count, summary_shown, welcome_text,
            banner_warning, send_btn, is_generating
        ],
        show_progress="full"
    )

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
