import os
import json

def build_prompt(user_message: str, mental_state: str, sentiment_intensity: str) -> str:
    instruction = (
        "Bạn là một chuyên gia tâm lý. Hãy trả lời người dùng với giọng điệu nhẹ nhàng, đồng cảm. "
        "Dựa vào trạng thái tâm lý và mức độ cảm xúc của họ."
    )
    input_text = (
        f"Tin nhắn: {user_message}\n"
        f"Trạng thái tâm lý: {mental_state}\n"
        f"Mức độ cảm xúc: {sentiment_intensity}\n"
        "Phản hồi:"
    )
    return f"{instruction}\n{input_text}"

# Helper to cache label descriptions
_label_desc_cache = None
def get_label_descriptions():
    global _label_desc_cache
    if _label_desc_cache is None:
        json_path = os.path.join(os.path.dirname(__file__), "label_descriptions.json")
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                _label_desc_cache = json.load(f)
        except FileNotFoundError:
            # Fallback if file doesn't exist
            _label_desc_cache = {
                "gating_label": {},
                "mental_state_label": {},
                "sentiment_intensity_label": {}
            }
    return _label_desc_cache

def build_prompt_from_object(obj: dict, include_template=True) -> str:
    """
    Build a prompt string from a structured object.
    obj: {
        "instruction": str,
        "input": str,
        "context": {
            "mental_state": str,
            "sentiment_intensity": str,
            "risk_level": str,
            "history": list[{"role": str, "content": str}],
            ...
        }
    }
    """
    label_desc = get_label_descriptions()
    DEFAULT_INSTRUCTION = "Bạn là một trợ lý tâm lý chuyên nghiệp. Hãy lắng nghe, đồng cảm và phản hồi nhẹ nhàng. Tránh phán xét và đưa ra gợi ý hữu ích."
    instruction = obj.get("instruction", DEFAULT_INSTRUCTION)
    input_text = obj.get("input", "")
    context = obj.get("context", {})
    mental_state = context.get("mental_state", "")
    sentiment = context.get("sentiment_intensity", "")
    risk_level = context.get("risk_level", "")
    history = context.get("history", [])
    knowledge = context.get("knowledge", [])

    # Build context information
    context_lines = []
    if mental_state:
        context_lines.append(f"- Trạng thái tâm lý: {mental_state}")
        desc = label_desc["mental_state_label"].get(mental_state)
        if desc:
            context_lines.append(f"  → {desc}")
    if sentiment:
        context_lines.append(f"- Cảm xúc: {sentiment}")
        desc = label_desc["sentiment_intensity_label"].get(str(sentiment))
        if desc:
            context_lines.append(f"  → {desc}")
    if risk_level:
        context_lines.append(f"- Mức độ rủi ro: {risk_level}")
        desc = label_desc["gating_label"].get(risk_level)
        if desc:
            context_lines.append(f"  → {desc}")
    if knowledge:
        context_lines.append("Kiến thức liên quan:")
        for idx, chunk in enumerate(knowledge, 1):
            context_lines.append(f"[{idx}] {chunk}")
    if history:
        context_lines.append("Lịch sử hội thoại:")
        for turn in history:
            context_lines.append(f"{turn['role']}: {turn['content']}")
    
    # Build input content
    input_content = []
    if context_lines:
        input_content.extend(context_lines)
        input_content.append("")
    input_content.append(f"Người dùng: {input_text}")
    input_content.append("Trợ lý:")
    
    input_text_final = "\n".join(input_content)

    if include_template:
        return f"""### Instruction:\n{instruction}\n\n### Input:\n{input_text_final}\n\n### Response:\n"""
    else:
        # Return without template markers
        return f"{instruction}\n\n{input_text_final}"
