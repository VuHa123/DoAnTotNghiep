import os
import json
import requests
from utils.api_manager import api_manager

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent"
PROMPT_LENGTH_THRESHOLD = 800

def log_prompt_debug(prompt: str, source: str = "unknown"):
    """Log prompt for debugging purposes"""
    print(f"[PROMPT_DEBUG] 🔍 {source} prompt generated:")
    print(f"[PROMPT_DEBUG] Length: {len(prompt)} characters")
    print(f"[PROMPT_DEBUG] Preview: {prompt[:200]}{'...' if len(prompt) > 200 else ''}")
    print(f"[PROMPT_DEBUG] {'='*50}")


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

def call_gemini_build_prompt(obj: dict, base_prompt: str = None) -> str:
    """
    Gửi thông tin sang Gemini để tạo prompt tối ưu cho LLM.
    Sử dụng build_prompt_from_object để tạo prompt có cấu trúc, sau đó gửi cho Gemini tối ưu hóa.
    """
    print(f"[GEMINI] 🚀 Starting Gemini prompt generation...")
    api_key = api_manager.get_best_api_key()
    if not api_key:
        print(f"[GEMINI] ❌ No valid API key found")
        return "[Lỗi: Không có Gemini API key hợp lệ]"
    
    # Sử dụng base_prompt nếu được truyền, nếu không thì tạo mới
    if base_prompt is None:
        base_prompt = build_prompt_from_object(obj, include_template=False)
    
    # Thêm yêu cầu cho Gemini
    structured_prompt = (
        f"{base_prompt}\n\n"
        "=== YÊU CẦU CHO GEMINI ===\n"
        "Hãy rút gọn prompt trên để súc tích nhưng KHÔNG mất thông tin quan trọng.\n"
        "ĐẶC BIỆT nhấn mạnh câu hỏi hiện tại của NGƯỜI DÙNG (đặt cuối prompt, dễ thấy).\n"
        "Không thêm thông tin mới, không thay đổi ngữ nghĩa."
    )
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": structured_prompt}]}
        ]
    }
    url = f"{GEMINI_API_URL}?key={api_key}"
    print(f"[GEMINI] 📤 Sending structured prompt to Gemini API...")
    try:
        res = requests.post(url, headers=headers, json=payload, timeout=15)
        api_manager.mark_key_used(api_key, success=(res.status_code==200))
        print(f"[GEMINI] 📥 Received response from Gemini (status: {res.status_code})")
        if res.status_code == 200:
            data = res.json()
            print(f"[GEMINI] 📋 Processing Gemini response...")
            # Ưu tiên lấy prompt đã tổng hợp từ Gemini
            if "candidates" in data and data["candidates"]:
                parts = data["candidates"][0]["content"]["parts"]
                # Lấy phần text đầu tiên không rỗng
                for part in parts:
                    if part.get("text"):
                        result = part["text"].strip()
                        print(f"[GEMINI] ✅ Generated prompt (length: {len(result)})")
                        log_prompt_debug(result, "Gemini")
                        return result
                result = str(parts)
                print(f"[GEMINI] ⚠️ Using fallback from parts")
                return result
            elif "prompt" in data:
                result = data["prompt"]
                print(f"[GEMINI] ✅ Generated prompt from 'prompt' field")
                return result
            else:
                result = str(data)
                print(f"[GEMINI] ⚠️ Using raw data as prompt")
                return result
        else:
            error_msg = f"[Lỗi Gemini API {res.status_code}: {res.text}]"
            print(f"[GEMINI] ❌ API Error: {error_msg}")
            return error_msg
    except Exception as e:
        error_msg = f"[Lỗi gọi Gemini: {e}]"
        print(f"[GEMINI] ❌ Exception: {error_msg}")
        return error_msg

def build_prompt_from_object(obj: dict, include_template=True, minimal_mode: bool = True) -> str:
    """
    Build a prompt string from a structured object.
    
    Args:
        obj: Dictionary containing prompt structure
        include_template: If True, call Gemini when prompt is too long (>800 chars)
        minimal_mode: If True, use minimal context (no descriptions, only 1-2 history)
    
    obj: {
        "instruction": str,
        "input": str,
        "context": {
            "mental_state": str,
            "sentiment_intensity": str,
            "history": list[str],  # List of user messages (not bot responses)
            "knowledge": list[str],  # RAG knowledge chunks
            ...
        }
    }
    """
    label_desc = get_label_descriptions()
    DEFAULT_INSTRUCTION = """Bạn là MentalGPT – một trợ lý tâm lý. Nhiệm vụ của bạn là lắng nghe, phân tích, thể hiện sự đồng 
    cảm và đưa ra lời khuyên hoặc giải pháp trực tiếp, trả lời xúc tích, ngắn gọn và khách quan, giúp người dùng vượt qua vấn đề tinh thần. Không chào hỏi, không vòng vo, trả lời đi thẳng vào trọng tâm."""
    instruction = obj.get("instruction", DEFAULT_INSTRUCTION)
    input_text = obj.get("input", "")
    context = obj.get("context", {})
    mental_state = context.get("mental_state", "")
    sentiment = context.get("sentiment_intensity", "")
    history = context.get("history", [])
    knowledge = context.get("knowledge", [])

    if minimal_mode:
        history = history[-2:]  # Chỉ lấy 1-2 history gần nhất
    
    # Build context information
    context_lines = []
    if mental_state:
        context_lines.append(f"- Trạng thái tâm lý: {mental_state}")
        if not minimal_mode:  # Chỉ thêm mô tả khi không phải minimal mode
            desc = label_desc["mental_state_label"].get(mental_state)
            if desc:
                context_lines.append(f"  → {desc}")
    if sentiment:
        context_lines.append(f"- Cảm xúc: {sentiment}")
        if not minimal_mode:  # Chỉ thêm mô tả khi không phải minimal mode
            desc = label_desc["sentiment_intensity_label"].get(str(sentiment))
            if desc:
                context_lines.append(f"  → {desc}")
    if knowledge and len(knowledge) > 0:
        context_lines.append("Kiến thức liên quan:")
        for idx, chunk in enumerate(knowledge, 1):
            context_lines.append(f"[{idx}] {chunk}")
    else:
        # Thêm thông báo khi không có knowledge phù hợp
        context_lines.append("Lưu ý: Không tìm được kiến thức chuyên môn phù hợp cho câu hỏi này.")
        context_lines.append("Hãy trả lời dựa trên kiến thức chung về tâm lý học và sức khỏe tinh thần.")
    if history:
        context_lines.append("Lịch sử hội thoại (các tin nhắn trước đó của người dùng):")
        for i, user_msg in enumerate(history, 1):
            context_lines.append(f"[{i}] Người dùng: {user_msg}")
    
    parts = [
        instruction,
        "",
        "=== THÔNG TIN NGỮ CẢNH ===" if context_lines else "",
        *context_lines,
        "",
        f"Người dùng: {input_text}",
        "Trợ lý:"
    ]
    prompt_raw = "\n".join([p for p in parts if p])

    # --------- Gọi Gemini NẾU rất dài ----------
    if include_template and len(prompt_raw) > PROMPT_LENGTH_THRESHOLD:
        print(f"[PROMPT_BUILDER] 📏 Prompt dài ({len(prompt_raw)} chars), gọi Gemini để tối ưu...")
        return call_gemini_build_prompt(
            obj,
            base_prompt=prompt_raw   # truyền prompt đã ghép
        )

    print(f"[PROMPT_BUILDER] ✅ Prompt ngắn ({len(prompt_raw)} chars), sử dụng trực tiếp")
    
    # Hiển thị prompt hoàn chỉnh
    print(f"[PROMPT_BUILDER] 📝 FINAL PROMPT:")
    print(f"[PROMPT_BUILDER] {'='*60}")
    print(prompt_raw)
    print(f"[PROMPT_BUILDER] {'='*60}")
    
    return prompt_raw
