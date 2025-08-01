import os
import json
import requests
from utils.api_manager import api_manager

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent"

def log_prompt_debug(prompt: str, source: str = "unknown"):
    """Log prompt for debugging purposes"""
    print(f"[PROMPT_DEBUG] 🔍 {source} prompt generated:")
    print(f"[PROMPT_DEBUG] Length: {len(prompt)} characters")
    print(f"[PROMPT_DEBUG] Preview: {prompt[:200]}{'...' if len(prompt) > 200 else ''}")
    print(f"[PROMPT_DEBUG] {'='*50}")

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
    Tạo prompt có cấu trúc rõ ràng và dễ hiểu cho Gemini.
    """
    print(f"[GEMINI] 🚀 Starting Gemini prompt generation...")
    api_key = api_manager.get_best_api_key()
    if not api_key:
        print(f"[GEMINI] ❌ No valid API key found")
        return "[Lỗi: Không có Gemini API key hợp lệ]"
    
    # Tạo prompt có cấu trúc rõ ràng
    instruction = obj.get("instruction", "Bạn là một trợ lý tâm lý chuyên nghiệp. Hãy lắng nghe, đồng cảm và phản hồi nhẹ nhàng.")
    input_text = obj.get("input", "")
    context = obj.get("context", {})
    
    # Xây dựng prompt có cấu trúc
    prompt_parts = []
    prompt_parts.append("=== HƯỚNG DẪN CHO TRỢ LÝ TÂM LÝ ===")
    prompt_parts.append(instruction)
    prompt_parts.append("")
    
    # Thêm thông tin context
    if context:
        prompt_parts.append("=== THÔNG TIN NGỮ CẢNH ===")
        
        mental_state = context.get("mental_state", "")
        if mental_state:
            prompt_parts.append(f"Trạng thái tâm lý: {mental_state}")
        
        sentiment = context.get("sentiment_intensity", "")
        if sentiment:
            prompt_parts.append(f"Mức độ cảm xúc: {sentiment}")
        
        risk_level = context.get("risk_level", "")
        if risk_level:
            prompt_parts.append(f"Mức độ rủi ro: {risk_level}")
        
        knowledge = context.get("knowledge", [])
        if knowledge:
            prompt_parts.append("Kiến thức liên quan:")
            for i, chunk in enumerate(knowledge, 1):
                prompt_parts.append(f"{i}. {chunk}")
        
        history = context.get("history", [])
        if history:
            prompt_parts.append("Lịch sử hội thoại:")
            for i, msg in enumerate(history, 1):
                prompt_parts.append(f"{i}. Người dùng: {msg}")
        
        prompt_parts.append("")
    
    prompt_parts.append("=== TIN NHẮN HIỆN TẠI ===")
    prompt_parts.append(f"Người dùng: {input_text}")
    prompt_parts.append("")
    prompt_parts.append("=== YÊU CẦU ===")
    prompt_parts.append("Dựa trên tất cả thông tin trên, hãy tạo một prompt hoàn chỉnh và tối ưu để LLM có thể trả lời tốt nhất cho người dùng. Prompt phải:")
    prompt_parts.append("- Bao gồm đầy đủ context quan trọng")
    prompt_parts.append("- Có cấu trúc rõ ràng, dễ hiểu")
    prompt_parts.append("- Tập trung vào việc hỗ trợ tâm lý hiệu quả")
    prompt_parts.append("- Không quá dài, nhưng đầy đủ thông tin cần thiết")
    
    structured_prompt = "\n".join(prompt_parts)
    
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
            "history": list[str],  # List of user messages (not bot responses)
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
        context_lines.append("Lịch sử hội thoại (các tin nhắn trước đó của người dùng):")
        for i, user_msg in enumerate(history, 1):
            context_lines.append(f"[{i}] Người dùng: {user_msg}")
    
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
        print(f"[PROMPT_BUILDER] 🚀 Calling Gemini to optimize prompt...")
        gemini_prompt = call_gemini_build_prompt(obj)
        print(f"[PROMPT_BUILDER] ✅ Gemini processing completed")
        return gemini_prompt
    else:
        # Return without template markers - tạo prompt có cấu trúc tốt hơn
        print(f"[PROMPT_BUILDER] 📝 Using direct prompt (no Gemini)")
        
        # Tạo prompt có cấu trúc rõ ràng
        prompt_parts = []
        prompt_parts.append(instruction)
        prompt_parts.append("")
        
        # Thêm context nếu có
        if context_lines:
            prompt_parts.append("=== THÔNG TIN NGỮ CẢNH ===")
            prompt_parts.extend(context_lines)
            prompt_parts.append("")
        
        prompt_parts.append("=== CUỘC HỘI THOẠI ===")
        prompt_parts.append(f"Người dùng: {input_text}")
        prompt_parts.append("Trợ lý:")
        
        final_prompt = "\n".join(prompt_parts)
        log_prompt_debug(final_prompt, "Direct")
        return final_prompt
