import requests
import os
import logging
import time
import re

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CUSTOM_LLM_API_URL = "https://shield-postposted-ate-cumulative.trycloudflare.com/model/generate/"

def call_gemini_llm(prompt: str) -> str:
    headers = {"Content-Type": "application/json"}
    payload = {"prompt": prompt}
    
    start_time = time.time()
    try:
        # Timeout 30 giây cho API call
        res = requests.post(CUSTOM_LLM_API_URL, headers=headers, json=payload, timeout=30)
        duration = time.time() - start_time
        
        if res.status_code == 200:
            result = res.json()
            if "response" in result:
                response_text = result["response"]
                # Làm sạch token đặc biệt nếu có
                response_text = re.sub(r"<\|.*?\|>", "", response_text)
                logger.info(f"✅ Custom LLM API thành công - Thời gian: {duration:.2f}s - Độ dài prompt: {len(prompt)} chars")
                return response_text
            else:
                logger.error(f"❌ Custom LLM API trả về response không đúng format: {result}")
                return "[Lỗi: Response không đúng format]"
        else:
            logger.error(f"❌ Custom LLM API lỗi HTTP {res.status_code}: {res.text}")
            return f"[Lỗi HTTP {res.status_code}: {res.text}]"
            
    except requests.exceptions.Timeout:
        logger.error(f"❌ Custom LLM API timeout sau {time.time() - start_time:.2f}s")
        return "[Lỗi: API timeout - vui lòng thử lại]"
    except requests.exceptions.ConnectionError:
        logger.error("❌ Custom LLM API lỗi kết nối")
        return "[Lỗi: Không thể kết nối API - kiểm tra mạng]"
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Custom LLM API lỗi request: {e}")
        return f"[Lỗi request: {e}]"
    except Exception as e:
        logger.error(f"❌ Custom LLM API lỗi không xác định: {e}")
        return f"[Lỗi không xác định: {e}]"
