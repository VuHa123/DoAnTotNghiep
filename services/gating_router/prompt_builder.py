import os
import json
import requests
from utils.api_manager import api_manager

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent"

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

def call_gemini_build_prompt(obj: dict) -> str:
    """
    Gửi thông tin sang Gemini để tạo prompt tối ưu cho LLM.
    Thêm chỉ dẫn rõ ràng yêu cầu Gemini tổng hợp prompt ngắn gọn, súc tích, tập trung ý chính.
    """
    api_key = api_manager.get_best_api_key()
    if not api_key:
        return "[Lỗi: Không có Gemini API key hợp lệ]"
    headers = {"Content-Type": "application/json"}
    # Thêm instruction rõ ràng vào obj
    instruction = (
        "Hãy tổng hợp tất cả các thông tin sau (bao gồm kiến thức liên quan, trạng thái tâm lý, cảm xúc, lịch sử hội thoại, v.v.) thành một prompt hoàn chỉnh để LLM có thể trả lời tốt nhất cho người dùng. "
        "Không được lược bỏ bất kỳ thông tin quan trọng nào từ các phần này. Chỉ loại bỏ thông tin trùng lặp hoặc không liên quan. Không cần format lại lịch sử hội thoại, chỉ cần prompt cuối cùng."
    )
    obj_with_instruction = obj.copy()
    obj_with_instruction["instruction_for_gemini"] = instruction
    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": str(obj_with_instruction)}]}
        ]
    }
    url = f"{GEMINI_API_URL}?key={api_key}"
    try:
        res = requests.post(url, headers=headers, json=payload, timeout=15)
        api_manager.mark_key_used(api_key, success=(res.status_code==200))
        if res.status_code == 200:
            data = res.json()
            # Ưu tiên lấy prompt đã tổng hợp từ Gemini
            if "candidates" in data and data["candidates"]:
                parts = data["candidates"][0]["content"]["parts"]
                # Lấy phần text đầu tiên không rỗng
                for part in parts:
                    if part.get("text"):
                        return part["text"]
                return str(parts)
            elif "prompt" in data:
                return data["prompt"]
            else:
                return str(data)
        else:
            return f"[Lỗi Gemini API {res.status_code}: {res.text}]"
    except Exception as e:
        return f"[Lỗi gọi Gemini: {e}]"

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
        # Gọi Gemini để sinh prompt tối ưu
        gemini_prompt = call_gemini_build_prompt(obj)
        return gemini_prompt
    else:
        # Return without template markers
        return f"{instruction}\n\n{input_text_final}"
