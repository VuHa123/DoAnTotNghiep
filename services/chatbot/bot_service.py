from services.chatbot.response_generator import call_gemini_llm
from services.context_tracking.tracker import update_context
from services.gating_router.prompt_builder import build_prompt_from_object

def generate_reply(user_input: str, history: list, sentiment: str, mental_state: str, knowledge=None) -> str:
    if knowledge is None:
        knowledge = []
    prompt_obj = {
        "input": user_input,
        "context": {
            "mental_state": mental_state,
            "sentiment_intensity": sentiment,
            "history": [{"role": "user" if i % 2 == 0 else "assistant", "content": turn} for i, turn in enumerate(history[-5:])],
            "knowledge": knowledge
        }
    }
    prompt = build_prompt_from_object(prompt_obj)
    return call_gemini_llm(prompt)