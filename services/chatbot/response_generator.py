import requests
import os
import logging
import time
import re
import unicodedata

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CUSTOM_LLM_API_URL = "https://lists-cos-irc-dow.trycloudflare.com/model/generate/"

def clean_response(text: str) -> str:
    """
    Làm sạch phản hồi từ mô hình sinh, loại bỏ token đặc biệt và ký tự không mong muốn.
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Loại bỏ MỌI token dạng <|...|> (bao gồm cả multiline) - ưu tiên xử lý trước
    text = re.sub(r"<\|[^|]*?\|>", "", text, flags=re.DOTALL | re.MULTILINE)
    
    # 2. Loại bỏ token đặc biệt với các pattern cụ thể
    special_tokens = [
        r"<\|closuresnippet\|>",   # closure snippet token
        r"<\|fim\|>",              # fim token (thêm vào)
        r"<\|fim_system\|>",       # fim system token
        r"<\|fim_user\|>",         # fim user token  
        r"<\|fim_assistant\|>",    # fim assistant token
        r"<\|fim_[^|]*?\|>",       # các fim tokens khác
        r"<\|end[^|]*?\|>",        # end tokens
        r"<\|start[^|]*?\|>",      # start tokens
        r"<\|eot_id\|>",           # end of turn token
        r"<\|begin_of_text\|>",    # begin text token
        r"<\|end_of_text\|>",      # end text token
        r"</?s>",                  # sentence tokens
        r"<unk>",                  # unknown tokens
        r"<pad>",                  # padding tokens
        r"<mask>",                 # mask tokens
    ]
    
    for pattern in special_tokens:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)
    
    # 3. Loại bỏ MỌI token dạng <...> (bất kể nội dung, bao gồm self-closing tags)
    text = re.sub(r"<[^<>]*?/?>\s*", "", text, flags=re.DOTALL | re.MULTILINE)
    
    # 4. Loại các từ đặc biệt bị sót lại (mở rộng danh sách)
    blacklist_words = [
        "closuresnippet", "startoftext", "endoftext", "endofprompt", 
        "startofresponse", "assistant", "user", "system", "human",
        "cách thức trả lời", "cho phép", "Yes/No", "fim_system",
        "fim_prefix", "fim_middle", "fim_suffix", "eot_id", "start_header_id",
        "end_header_id", "begin", "end", "instruction", "User", "Assistant",  # role sinh nhầm
        "fim", "fim_user", "fim_middle", "fim_end",  # Fill-in-Middle tokens
        "startofprompt", "endofprompt",
        "startoftext", "endoftext",
        "canchan", "response"  # Sửa lỗi syntax
    ]
    
    # Tạo pattern với word boundaries và case insensitive
    pattern = r"(?<!\w)(" + "|".join(re.escape(w) for w in blacklist_words) + r")(?!\w)"
    text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    
    # 5. Loại bỏ các ký tự Unicode vô hình và điều khiển
    invisible_chars = [
        r"[\u200b\u200c\u200d\u200e\u200f]",  # Zero-width chars
        r"[\ufeff\ufffe\uffff]",              # BOM chars
        r"[\u00a0\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]",  # Various spaces
        r"[\u0000-\u001f\u007f-\u009f]",      # Control characters
    ]
    
    for char_pattern in invisible_chars:
        text = re.sub(char_pattern, "", text)
    
    # 6. Xử lý đặc biệt cho token ở cuối text (thường gặp nhất)
    # Loại bỏ token cuối câu/đoạn trước khi xử lý dòng
    text = re.sub(r"\s*<\|[^|]*?\|>\s*$", "", text, flags=re.MULTILINE | re.DOTALL)
    text = re.sub(r"\s*<[^<>]*?>\s*$", "", text, flags=re.MULTILINE | re.DOTALL)
    
    # Xử lý token ở giữa text
    text = re.sub(r"\s*<\|[^|]*?\|>\s*", " ", text, flags=re.DOTALL)
    text = re.sub(r"\s*<[^<>]*?>\s*", " ", text, flags=re.DOTALL)
    
    # 7. Loại bỏ các ký tự lạ còn sót lại (non-printable characters)
    text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C' or char in '\n\t\r ')
    
    # 8. Chuẩn hóa khoảng trắng
    text = re.sub(r'\s+', ' ', text)  # Nhiều space thành 1 space
    text = re.sub(r'\n\s*\n', '\n', text)  # Nhiều newline thành 1 newline
    
    # 9. Xóa dòng trống dư thừa và trim
    lines = []
    for line in text.splitlines():
        line = line.strip()
        # Chỉ giữ lại dòng có nội dung có nghĩa
        if line and not re.match(r'^[\s\W]*$', line):
            lines.append(line)
    
    # 10. Aggressive final cleanup - loại bỏ TOÀN BỘ token còn sót lại
    result = "\n".join(lines).strip()
    
    # Multiple passes để đảm bảo loại bỏ hết token
    for _ in range(3):  # Lặp nhiều lần để bắt token lồng nhau
        # Loại bỏ token với pipe
        result = re.sub(r'<\|[^|]*?\|>', '', result, flags=re.DOTALL)
        # Loại bỏ token HTML-style  
        result = re.sub(r'<[^<>]*?>', '', result, flags=re.DOTALL)
        # Loại bỏ token ở đầu/cuối với whitespace
        result = re.sub(r'^\s*<[^>]*>\s*', '', result, flags=re.MULTILINE)
        result = re.sub(r'\s*<[^>]*>\s*$', '', result, flags=re.MULTILINE)
    
    # Xử lý đặc biệt cho fim token ở cuối
    result = re.sub(r'\s*<\|fim\|>\s*$', '', result, flags=re.MULTILINE | re.DOTALL)
    result = re.sub(r'\s*<\|fim_user\|>\s*$', '', result, flags=re.MULTILINE | re.DOTALL)
    result = re.sub(r'\s*<\|fim_system\|>\s*$', '', result, flags=re.MULTILINE | re.DOTALL)
    result = re.sub(r'\s*<\|fim_assistant\|>\s*$', '', result, flags=re.MULTILINE | re.DOTALL)
    
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
    
    start_time = time.time()
    try:
        # Timeout 30 giây cho API call
        res = requests.post(CUSTOM_LLM_API_URL, headers=headers, json=payload, timeout=30)
        duration = time.time() - start_time
        
        if res.status_code == 200:
            result = res.json()
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
