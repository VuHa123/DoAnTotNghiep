import gradio as gr
import sys
import re
sys.path.append("..")
from chatbot_inference import ChatbotInference
from services.gating_router.prompt_builder import build_prompt_from_object

# Khởi tạo model fine-tune
chatbot = ChatbotInference(checkpoint_name="checkpoint-1098")
chatbot.load_model()
# chatbot.stop_event.clear()

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
    greeting = "Chào bạn! Chúc bạn một ngày tốt lành! Tôi có thể giúp gì cho bạn"
    # Khởi tạo chat_history đúng format tuple cho Gradio
    chat_history = [("", greeting)]
    return gr.update(visible=False), gr.update(visible=True), chat_history, gr.update(value="", interactive=True), STATE_MAIN, 0, False, welcome

def chat_with_bot(user_input, chat_history, chat_goal, turn_count, summary_shown, ui_state):
    turn_count = (turn_count or 0) + 1
    chat_history = chat_history or []
    # Build prompt object with context/history
    prompt_obj = {
        "instruction": "Bạn là một chatbot hỗ trợ tâm lý. Hãy phản hồi nhẹ nhàng và cảm thông.",
        "input": user_input,
        "context": {
            "history": chat_history[-5:]  # Lấy 5 lượt gần nhất
        }
    }
    response = chatbot.generate_response(prompt_obj)
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
    return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), [], gr.update(value="", interactive=True), STATE_MAIN, 0, False, ""

def open_settings():
    return gr.update(visible=True)

def close_settings():
    return gr.update(visible=False)

def hotline_click():
    return "📞 Đang kết nối tổng đài 096.306.1414...", gr.update(visible=True)

# ======== NÚT GỬI ↔ TẠM DỪNG =========

def convert_history_for_gradio(chat_history):
    result = []
    user_msg = None
    for msg in chat_history:
        if msg["role"] == "user":
            if user_msg is not None:
                # Nếu có user_msg trước đó mà chưa có assistant, vẫn hiển thị
                result.append((user_msg, ""))
            user_msg = msg["content"]
        elif msg["role"] == "assistant":
            if user_msg is not None:
                result.append((user_msg, msg["content"]))
                user_msg = None
            else:
                # Nếu có assistant mà không có user trước đó (hiếm gặp)
                result.append(("", msg["content"]))
    if user_msg is not None:
        result.append((user_msg, ""))
    return result

def on_send_click_v2(user_input, chat_history, chatbot_ui, is_generating):
    # Khi user gửi, append user message vào cả chat_history và chatbot_ui
    if not is_generating:
        # Cập nhật chat_history (logic) và chatbot_ui (hiển thị)
        chat_history = chat_history or []
        chatbot_ui = chatbot_ui or []
        chat_history.append({"role": "user", "content": user_input})
        chatbot_ui.append((user_input, ""))
        return (
            gr.update(value="", interactive=True),
            chatbot_ui,
            gr.update(value="Tạm dừng", interactive=True),
            True,
            chat_history,
            chatbot_ui,
            user_input
        )
    else:
        return (
            gr.update(value=user_input, interactive=True),
            chatbot_ui,
            gr.update(value="Gửi", interactive=True),
            False,
            chat_history,
            chatbot_ui,
            user_input
        )

def run_generation_v2(user_input, chat_history, chatbot_ui, chat_goal, turn_count, summary_shown, ui_state):
    # Sinh response bot, append vào cả chat_history và chatbot_ui
    new_user_input, updated_history, _, new_ui_state, new_turn_count, new_summary_shown, welcome_text, banner_visible, _, _ = chat_with_bot(
        user_input, chat_history, chat_goal, turn_count, summary_shown, ui_state
    )
    # Lấy message bot vừa trả lời
    last_bot_msg = None
    for msg in reversed(updated_history):
        if msg["role"] == "assistant":
            last_bot_msg = msg["content"]
            break
    chatbot_ui = chatbot_ui or []
    if last_bot_msg is not None:
        # Append bot response vào cặp cuối cùng (user, "")
        if chatbot_ui and chatbot_ui[-1][1] == "":
            chatbot_ui[-1] = (chatbot_ui[-1][0], last_bot_msg)
        else:
            chatbot_ui.append(("", last_bot_msg))
    return (
        new_user_input, chatbot_ui, new_ui_state, new_turn_count,
        new_summary_shown, welcome_text, banner_visible,
        gr.update(value="Gửi", interactive=True), False, updated_history, chatbot_ui
    )





with gr.Blocks(
    css="""
    .gr-chatbot, #chat-container {
        max-height: none !important;
        height: auto !important;
        overflow-y: visible !important;
    }
    #hotline-chatbox {
        position: fixed;
        right: 24px;
        bottom: 24px;
        background-color: #e3f6fd;
        border: 1.5px solid #0288d1;
        border-radius: 10px;
        box-shadow: 0 2px 6px #b3e0ff55;
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 10px 18px;
        z-index: 1000;
        cursor: pointer;
        transition: background 0.2s;
    }
    #hotline-chatbox:hover {
        background-color: #d0ebff;
    }
    body { background: #e3f6fd !important; font-family: 'Segoe UI', sans-serif; }
    #chat-container {
        background-color: #f6fbff;
        padding: 20px;
        border-radius: 16px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        overflow-y: visible;
    }
    .goal-card {
        border: 2px solid #ddd;
        border-radius: 16px;
        padding: 12px 16px;
        text-align: center;
        font-size: 16px;
        font-weight: 500;
        cursor: pointer;
        background: white;
        transition: 0.2s;
    }
    .goal-card:hover {
        border-color: #3498db;
        background-color: #e3f6fd;
    }
    .emotion-tag {
        background: #e3f6fd;
        color: #0288d1;
        padding: 6px 12px;
        border-radius: 9999px;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 10px;
    }
    .send-button {
        background-color: #0288d1 !important;
        color: white !important;
        font-weight: bold;
        border-radius: 9999px !important;
        padding: 12px 24px !important;
        border: none;
    }
    .send-button:hover {
        background-color: #0277bd !important;
    }
    /* Chatbot custom bubble */
    .gr-chatbot .message.user {
        background: #b3e5fc !important;
        color: #01579b !important;
        border-radius: 18px 18px 4px 18px !important;
        align-self: flex-end !important;
        margin-left: auto !important;
        margin-right: 0 !important;
        text-align: right;
        box-shadow: 0 2px 8px rgba(2,136,209,0.07);
    }
    .gr-chatbot .message.bot {
        background: #e3f6fd !important;
        color: #01579b !important;
        border-radius: 18px 18px 18px 4px !important;
        align-self: flex-start !important;
        margin-right: auto !important;
        margin-left: 0 !important;
        text-align: left;
        box-shadow: 0 2px 8px rgba(2,136,209,0.04);
    }
    .gr-chatbot .message {
        max-width: 70%;
        padding: 12px 18px;
        margin: 8px 0;
        font-size: 16px;
        line-height: 1.6;
    }
    """
) as demo:
    # Main screen (ẩn/hiện qua callback)
    with gr.Row(visible=True) as main_screen:
        with gr.Column():
            gr.Markdown("## 🧠 MENTALBOT\nTrò chuyện tâm lý cùng bạn")
            gr.Markdown("👋 **Xin chào! Bạn muốn tôi hỗ trợ gì hôm nay?**")
            chat_goal = gr.Radio(choices=list(chat_goals), label=None, interactive=True, elem_classes="goal-card")
            start_btn = gr.Button("Bắt đầu", variant="primary", elem_classes="send-button")
        settings_btn = gr.Button("⚙️ Tuỳ chọn", variant="secondary")
        welcome_text = gr.Markdown(visible=False)

    # Chat screen (ẩn/hiện qua callback)
    with gr.Row(visible=False) as chat_screen:
        with gr.Column():
            chatbot_ui = gr.Chatbot(label="", height=300, show_copy_button=True)
            
            user_input = gr.Textbox(placeholder="Nhập nội dung...", lines=3)
            with gr.Row():
                send_btn = gr.Button("Gửi", elem_classes="send-button")
                stop_btn = gr.Button("⏸️ Tạm dừng", visible=False)
            # Thêm biến chat_history (ẩn) để lưu lịch sử dạng list dict
            chat_history = gr.State([])

    
    # Hotline nổi góc dưới bên phải
    hotline_floating = gr.HTML(
        '''<div id="hotline-chatbox" onclick="document.getElementById('hotline-btn').click()">
            <span style="font-size:20px;">☎️</span>
            <span style="font-weight:600; color:#0288d1;">Hotline: 096.306.1414</span>
        </div>''',
        visible=True
    )

    # Đảm bảo hotline-btn (ẩn) để trigger sự kiện click
    hotline_btn = gr.Button("📞 Hotline: 096.306.1414", elem_id="hotline-btn", visible=False)
    hotline_result = gr.Textbox(label="", interactive=False, visible=False)

    # Đảm bảo KHÔNG có bất kỳ gr.HTML, gr.Button hoặc CSS nào tạo hotline ở góc dưới bên trái

    banner_warning = gr.HTML('<div id="banner-warning" class="banner-warning">🚨 CẢNH BÁO: Nếu bạn đang gặp nguy hiểm hoặc có ý định tự làm hại bản thân, hãy gọi ngay <b>Hotline 096.306.1414</b> hoặc liên hệ người thân!</div>', visible=False)

    # Nút hotline chỉ xuất hiện trong màn hình khẩn cấp
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

    # Sửa callback để truyền đúng chat_history (list dict) cho logic, chatbot_ui chỉ dùng cho hiển thị
    start_btn.click(
        start_chat,
        inputs=[chat_goal],
        outputs=[main_screen, chat_screen, chatbot_ui, user_input, ui_state, turn_count, summary_shown, welcome_text]
    ).then(
        lambda *args: gr.update(value=[]),  # reset chat_history khi bắt đầu
        inputs=[],
        outputs=[chat_history]
    )

    # Sử dụng callback mới cho send_btn
    send_btn.click(
        on_send_click_v2,
        inputs=[user_input, chat_history, chatbot_ui, is_generating],
        outputs=[user_input, chatbot_ui, send_btn, is_generating, chat_history, chatbot_ui, user_input]
    ).then(
        run_generation_v2,
        inputs=[user_input, chat_history, chatbot_ui, chat_goal, turn_count, summary_shown, ui_state],
        outputs=[user_input, chatbot_ui, ui_state, turn_count, summary_shown, welcome_text, banner_warning, send_btn, is_generating, chat_history, chatbot_ui]
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
        outputs=[main_screen, chat_screen, emergency_screen, summary_screen, chatbot_ui, user_input, ui_state, turn_count, summary_shown, welcome_text]
    )

    settings_btn.click(open_settings, outputs=settings_screen)
    close_settings_btn.click(close_settings, outputs=settings_screen)
    hotline_btn.click(hotline_click, outputs=[hotline_result])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)