from services.chatbot.response_generator import call_gemini_llm
from services.summarization.summarizer import summarize
from services.context_tracking.tracker import update_context
from services.gating_router.prompt_builder import build_prompt_from_object

def generate_reply(user_input: str, history: list, sentiment: str, mental_state: str) -> str:
    summary = summarize(history + [user_input])
    prompt_obj = {
        "instruction": "Bạn là một trợ lý tâm lý, luôn lắng nghe và đồng cảm. Tránh phán xét. Gợi ý nhẹ nhàng. Dựa trên thông tin sau:",
        "input": user_input,
        "context": {
            "mental_state": mental_state,
            "sentiment_intensity": sentiment,
            "summary": summary,
            "history": [{"role": "user" if i % 2 == 0 else "assistant", "content": turn} for i, turn in enumerate(history[-5:])]
        }
    }
    prompt = build_prompt_from_object(prompt_obj)
    return call_gemini_llm(prompt)