import requests
import os
import logging
import time
import re
import unicodedata
import sys
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Thêm đường dẫn để import từ llmserver
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'llmserver'))
from llmserver.config import API_LLM

# Thiết lập logging
# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CUSTOM_LLM_API_URL = API_LLM

# Tạo session với connection pooling và retry
session = requests.Session()
retry_strategy = Retry(
    total=2,
    backoff_factor=0.5,
    status_forcelist=[429, 500, 502, 503, 504],
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)

# Cache compiled patterns để tăng tốc độ
import re
SPECIAL_TOKEN_PATTERN = re.compile(r"<\|[^|]*?\|>", flags=re.DOTALL | re.MULTILINE)
HTML_TAG_PATTERN = re.compile(r"<[^<>]*?/?>\s*", flags=re.DOTALL | re.MULTILINE)
INVISIBLE_CHARS_PATTERN = re.compile(r"[\u200b\u200c\u200d\u200e\u200f\ufeff\ufffe\uffff\u00a0\u2000-\u200a\u2028\u2029\u202f\u205f\u3000\u0000-\u001f\u007f-\u009f]")
WHITESPACE_PATTERN = re.compile(r'\s+')
NEWLINE_PATTERN = re.compile(r'\n\s*\n')

def clean_response(text: str) -> str:
    """
    Làm sạch phản hồi từ mô hình sinh, loại bỏ token đặc biệt và ký tự không mong muốn.
    Tối ưu hóa với compiled patterns để tăng tốc độ.
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Loại bỏ MỌI token dạng <|...|> (sử dụng compiled pattern)
    text = SPECIAL_TOKEN_PATTERN.sub("", text)
    
    # 2. Loại bỏ MỌI token dạng <...> (sử dụng compiled pattern)
    text = HTML_TAG_PATTERN.sub("", text)
    
    # 3. Loại bỏ các ký tự Unicode vô hình (sử dụng compiled pattern)
    text = INVISIBLE_CHARS_PATTERN.sub("", text)
    
    # 4. Loại các từ đặc biệt bị sót lại (gộp pattern)
    blacklist_words = [
        "closuresnippet", "startoftext", "endoftext", "endofprompt", 
        "startofresponse", "assistant", "user", "system", "human",
        "cách thức trả lời", "cho phép", "Yes/No", "fim_system",
        "fim_prefix", "fim_middle", "fim_suffix", "eot_id", "start_header_id",
        "end_header_id", "begin", "end", "instruction", "User", "Assistant",
        "fim", "fim_user", "fim_middle", "fim_end", "startofprompt", "endofprompt",
        "canchan", "response"
    ]
    
    # Tạo pattern với word boundaries và case insensitive
    pattern = r"\b(" + "|".join(re.escape(w) for w in blacklist_words) + r")\b"
    text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    
    # 5. Loại bỏ các ký tự lạ còn sót lại (non-printable characters)
    text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C' or char in '\n\t\r ')
    
    # 6. Chuẩn hóa khoảng trắng (sử dụng compiled patterns)
    text = WHITESPACE_PATTERN.sub(' ', text)
    text = NEWLINE_PATTERN.sub('\n', text)
    
    # 7. Xóa dòng trống dư thừa và trim
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if line and not re.match(r'^[\s\W]*$', line):
            lines.append(line)
    
    result = "\n".join(lines).strip()
    
    # 8. Final cleanup - loại bỏ bất kỳ token nào còn sót lại
    result = SPECIAL_TOKEN_PATTERN.sub('', result)
    result = HTML_TAG_PATTERN.sub('', result)
    
    return result


# Hàm bổ sung để kiểm tra chất lượng text sau khi clean
def validate_cleaned_text(text: str) -> bool:
    """
    Kiểm tra xem text đã được clean có còn ký tự lạ không
    """
    if not text:
        return False
    
    # Kiểm tra các pattern không mong muốn
    unwanted_patterns = [
        r'<[^>]*>',           # HTML/XML tags
        r'<\|[^|]*\|>',       # Special tokens
        r'\\b(assistant|user|system)\\b',  # Role tokens
        r'[\u200b-\u200f]',   # Zero-width chars
    ]
    
    for pattern in unwanted_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return False
    
    return True




# Uncomment để test
# test_clean_response()

def call_gemini_llm(prompt: str) -> str:
    headers = {"Content-Type": "application/json"}
    payload = {"prompt": prompt}
    logger.info(f"[LLM PROMPT] {prompt}")
    start_time = time.time()
    try:
        # Giảm timeout xuống 15 giây để tăng tốc độ
        logger.info(f"🌐 Gọi API: {CUSTOM_LLM_API_URL}")
        
        res = session.post(CUSTOM_LLM_API_URL, headers=headers, json=payload, timeout=15)
        duration = time.time() - start_time
        
        logger.info(f"📥 Response status: {res.status_code} - Thời gian: {duration:.2f}s")
        
        if res.status_code == 200:
            # Kiểm tra content-type để xử lý đúng format
            content_type = res.headers.get('content-type', '').lower()
            
            if 'application/json' in content_type:
                # Xử lý JSON response
                try:
                    result = res.json()
                    logger.info(f"🔍 Parsed JSON response: {result}")
                    
                    if "response" in result:
                        response_text = result["response"]
                        logger.info(f"LLM response: {response_text}")
                        # Làm sạch token đặc biệt nếu có
                        response_text = clean_response(response_text)
                        
                        # Kiểm tra chất lượng sau khi clean
                        if not validate_cleaned_text(response_text):
                            logger.warning("⚠️ Response vẫn còn ký tự lạ sau khi clean")
                            # Có thể clean thêm lần nữa hoặc xử lý khác
                        
                        logger.info(f"✅ Custom LLM API thành công - Thời gian: {duration:.2f}s - Độ dài prompt: {len(prompt)} chars")
                        return response_text
                    else:
                        logger.error(f"❌ Custom LLM API trả về response không đúng format: {result}")
                        return "[Lỗi: Response không đúng format]"
                except ValueError as json_error:
                    logger.error(f"❌ Custom LLM API lỗi parse JSON: {json_error}")
                    logger.error(f"❌ Response text: {res.text}")
                    return f"[Lỗi: Không thể parse JSON response - {json_error}]"
            else:
                # Xử lý text response (như trường hợp hiện tại)
                response_text = res.text.strip()
                logger.info(f"📝 Text response: {response_text}")
                
                # Làm sạch token đặc biệt nếu có
                response_text = clean_response(response_text)
                
                # Kiểm tra chất lượng sau khi clean
                if not validate_cleaned_text(response_text):
                    logger.warning("⚠️ Response vẫn còn ký tự lạ sau khi clean")
                
                logger.info(f"✅ Custom LLM API thành công (text) - Thời gian: {duration:.2f}s - Độ dài prompt: {len(prompt)} chars")
                return response_text
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
