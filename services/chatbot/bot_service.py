from services.chatbot.response_generator import call_gemini_llm
from services.summarization.summarizer import summarize

def generate_reply(user_input: str, history: list, sentiment: str, mental_state: str) -> str:
    summary = summarize(history + [user_input])
    prompt_lines = [
        "Bạn là một trợ lý tâm lý, luôn lắng nghe và đồng cảm. Tránh phán xét. Gợi ý nhẹ nhàng. Dựa trên thông tin sau:"
    ]
    if mental_state:
        prompt_lines.append(f"- Trạng thái tâm lý: {mental_state}")
    if sentiment:
        prompt_lines.append(f"- Cảm xúc: {sentiment}")
    prompt_lines.append(f"- Tóm tắt: {summary}")
    prompt_lines.append("")
    prompt_lines.append("Hội thoại gần đây:")
    prompt_lines.append("\n".join(history[-5:]))
    prompt_lines.append("")
    prompt_lines.append(f"Người dùng: {user_input}")
    prompt_lines.append("Trợ lý:")
    prompt = "\n".join(prompt_lines)
    return call_gemini_llm(prompt)