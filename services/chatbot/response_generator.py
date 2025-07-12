import requests
import os
import logging
import time

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}"

def call_gemini_llm(prompt: str) -> str:
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }
    
    start_time = time.time()
    try:
        # Timeout 30 giây cho API call
        res = requests.post(GEMINI_API_URL, headers=headers, json=payload, timeout=30)
        duration = time.time() - start_time
        
        if res.status_code == 200:
            result = res.json()
            if "candidates" in result and len(result["candidates"]) > 0:
                response_text = result["candidates"][0]["content"]["parts"][0]["text"]
                logger.info(f"✅ Gemini API thành công - Thời gian: {duration:.2f}s - Độ dài prompt: {len(prompt)} chars")
                return response_text
            else:
                logger.error(f"❌ Gemini API trả về response không đúng format: {result}")
                return "[Lỗi: Response không đúng format]"
        else:
            logger.error(f"❌ Gemini API lỗi HTTP {res.status_code}: {res.text}")
            return f"[Lỗi HTTP {res.status_code}: {res.text}]"
            
    except requests.exceptions.Timeout:
        logger.error(f"❌ Gemini API timeout sau {time.time() - start_time:.2f}s")
        return "[Lỗi: API timeout - vui lòng thử lại]"
    except requests.exceptions.ConnectionError:
        logger.error("❌ Gemini API lỗi kết nối")
        return "[Lỗi: Không thể kết nối API - kiểm tra mạng]"
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Gemini API lỗi request: {e}")
        return f"[Lỗi request: {e}]"
    except Exception as e:
        logger.error(f"❌ Gemini API lỗi không xác định: {e}")
        return f"[Lỗi không xác định: {e}]"
