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
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'llmserver'))
try:
    from llmserver.config import API_LLM
except ImportError:
    # Fallback nếu không import được
    API_LLM = "http://localhost:8001/model/generate/"

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
# Thêm patterns mới để xử lý ký tự đặc biệt và lặp lại
REPEATED_CHARS_PATTERN = re.compile(r'(.)\1{3,}')  # Loại bỏ ký tự lặp lại quá 3 lần
SPECIAL_SYMBOLS_PATTERN = re.compile(r'[◆◇■□●○▲△▽▼♠♣♥♦★☆♤♧♡♢♪♫♬♩♭♮♯]')
CONTROL_CHARS_PATTERN = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]')
UNICODE_EMOJI_PATTERN = re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002600-\U000027BF]')

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
    
    # 4. Loại bỏ các ký tự điều khiển
    text = CONTROL_CHARS_PATTERN.sub("", text)
    
    # 5. Loại bỏ các ký tự đặc biệt và emoji không mong muốn
    text = SPECIAL_SYMBOLS_PATTERN.sub("", text)
    text = UNICODE_EMOJI_PATTERN.sub("", text)
    
    # 6. Loại bỏ ký tự lặp lại quá 3 lần (ví dụ: aaaa -> aaa)
    text = REPEATED_CHARS_PATTERN.sub(r'\1\1\1', text)
    
    # 7. Loại các từ đặc biệt bị sót lại (gộp pattern)
    blacklist_words = [
        "closuresnippet", "startoftext", "endoftext", "endofprompt", 
        "startofresponse", "assistant", "user", "system", "human",
        "cách thức trả lời", "cho phép", "Yes/No", "fim_system",
        "fim_prefix", "fim_middle", "fim_suffix", "eot_id", "start_header_id",
        "end_header_id", "begin", "end", "instruction", "User", "Assistant",
        "fim", "fim_user", "fim_middle", "fim_end", "startofprompt", "endofprompt",
        "canchan", "response", "prompt", "input", "output", "generate",
        "model", "llm", "ai", "bot", "chatbot", "assistant", "user"
    ]
    
    # Tạo pattern với word boundaries và case insensitive
    pattern = r"\b(" + "|".join(re.escape(w) for w in blacklist_words) + r")\b"
    text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    
    # 8. Loại bỏ các ký tự lạ còn sót lại (non-printable characters)
    text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C' or char in '\n\t\r ')
    
    # 9. Chuẩn hóa khoảng trắng (sử dụng compiled patterns)
    text = WHITESPACE_PATTERN.sub(' ', text)
    text = NEWLINE_PATTERN.sub('\n', text)
    
    # 10. Xóa dòng trống dư thừa và trim
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if line and not re.match(r'^[\s\W]*$', line):
            lines.append(line)
    
    result = "\n".join(lines).strip()
    
    # 11. Loại bỏ nội dung lặp lại (cùng một câu xuất hiện nhiều lần)
    sentences = result.split('.')
    unique_sentences = []
    seen_sentences = set()
    
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and len(sentence) > 10:  # Chỉ xử lý câu có độ dài > 10 ký tự
            # Chuẩn hóa câu để so sánh (loại bỏ dấu câu, chuyển về lowercase)
            normalized = re.sub(r'[^\w\s]', '', sentence.lower()).strip()
            if normalized not in seen_sentences:
                seen_sentences.add(normalized)
                unique_sentences.append(sentence)
        elif sentence:
            unique_sentences.append(sentence)
    
    result = '. '.join(unique_sentences).strip()
    
    # 12. Final cleanup - loại bỏ bất kỳ token nào còn sót lại
    result = SPECIAL_TOKEN_PATTERN.sub('', result)
    result = HTML_TAG_PATTERN.sub('', result)
    
    # 13. Đảm bảo kết quả không rỗng
    if not result.strip():
        return "Xin lỗi, tôi không thể tạo ra phản hồi phù hợp. Vui lòng thử lại."
    
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
        r'[◆◇■□●○▲△▽▼♠♣♥♦★☆♤♧♡♢♪♫♬♩♭♮♯]',  # Special symbols
        r'(.)\1{4,}',         # Repeated chars more than 4 times
        r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]',  # Control chars
    ]
    
    for pattern in unwanted_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return False
    
    # Kiểm tra độ dài tối thiểu
    if len(text.strip()) < 5:
        return False
    
    # Kiểm tra có chứa ít nhất một ký tự chữ cái
    if not re.search(r'[a-zA-ZÀ-ỹ]', text):
        return False
    
    return True


def final_cleanup(text: str) -> str:
    """
    Hàm cleanup cuối cùng để đảm bảo response sạch hoàn toàn
    """
    if not isinstance(text, str):
        return "Xin lỗi, tôi không thể tạo ra phản hồi phù hợp. Vui lòng thử lại."
    
    # Áp dụng clean_response
    cleaned = clean_response(text)
    
    # Kiểm tra chất lượng
    if not validate_cleaned_text(cleaned):
        # Nếu vẫn không đạt chất lượng, thử clean thêm lần nữa
        cleaned = clean_response(cleaned)
        
        # Nếu vẫn không đạt, trả về message mặc định
        if not validate_cleaned_text(cleaned):
            return "Xin lỗi, tôi không thể tạo ra phản hồi phù hợp. Vui lòng thử lại."
    
    return cleaned


# Uncomment để test
# test_clean_response()

def call_gemini_llm(prompt: str) -> str:
    headers = {"Content-Type": "application/json"}
    payload = {"prompt": prompt}
    logger.info(f"[LLM] Gọi API với prompt dài {len(prompt)} chars")
    start_time = time.time()
    # try:
    #     # Giảm timeout xuống 15 giây để tăng tốc độ
    #     logger.info(f"🌐 Gọi API: {CUSTOM_LLM_API_URL}")
        
    #     res = session.post(CUSTOM_LLM_API_URL, headers=headers, json=payload, timeout=15)
    #     duration = time.time() - start_time
        
    #     logger.info(f"📥 Response status: {res.status_code} - Thời gian: {duration:.2f}s")
        
    #     if res.status_code == 200:
    #         # Kiểm tra content-type để xử lý đúng format
    #         content_type = res.headers.get('content-type', '').lower()
            
    #         if 'application/json' in content_type:
    #             # Xử lý JSON response
    #             try:
    #                 result = res.json()
    #                 logger.info(f"🔍 Parsed JSON response: {result}")
                    
    #                 if "response" in result:
    #                     response_text = result["response"]
    #                     logger.info(f"LLM response: {response_text}")
    #                     # Làm sạch token đặc biệt nếu có
    #                     response_text = clean_response(response_text)
                        
    #                     # Trích xuất phần nội dung chính từ "Chào bạn..."
    #                     response_text = extract_main_response(response_text)
                        
    #                     # Kiểm tra chất lượng sau khi clean
    #                     if not validate_cleaned_text(response_text):
    #                         logger.warning("⚠️ Response vẫn còn ký tự lạ sau khi clean")
    #                         # Có thể clean thêm lần nữa hoặc xử lý khác
                        
    #                     logger.info(f"✅ Custom LLM API thành công - Thời gian: {duration:.2f}s - Độ dài prompt: {len(prompt)} chars")
    #                     return response_text
    #                 else:
    #                     logger.error(f"❌ Custom LLM API trả về response không đúng format: {result}")
    #                     return "[Lỗi: Response không đúng format]"
    #             except ValueError as json_error:
    #                 logger.error(f"❌ Custom LLM API lỗi parse JSON: {json_error}")
    #                 logger.error(f"❌ Response text: {res.text}")
    #                 return f"[Lỗi: Không thể parse JSON response - {json_error}]"
    #         else:
    #             # Xử lý text response (như trường hợp hiện tại)
    #             response_text = res.text.strip()
    #             logger.info(f"📝 Text response: {response_text}")
                
    #             # Làm sạch token đặc biệt nếu có
    #             response_text = clean_response(response_text)
                
    #             # Trích xuất phần nội dung chính từ "Chào bạn..."
    #             response_text = extract_main_response(response_text)
                
    #             # Kiểm tra chất lượng sau khi clean
    #             if not validate_cleaned_text(response_text):
    #                 logger.warning("⚠️ Response vẫn còn ký tự lạ sau khi clean")
                
    #             logger.info(f"✅ Custom LLM API thành công (text) - Thời gian: {duration:.2f}s - Độ dài prompt: {len(prompt)} chars")
    #             return response_text
    #     else:
    #         logger.error(f"❌ Custom LLM API lỗi HTTP {res.status_code}: {res.text}")
    #         return f"[Lỗi HTTP {res.status_code}: {res.text}]"
            
    # except requests.exceptions.Timeout:
    #     logger.error(f"❌ Custom LLM API timeout sau {time.time() - start_time:.2f}s")
    #     return "[Lỗi: API timeout - vui lòng thử lại]"
    # except requests.exceptions.ConnectionError:
    #     logger.error("❌ Custom LLM API lỗi kết nối")
    #     return "[Lỗi: Không thể kết nối API - kiểm tra mạng]"
    # except requests.exceptions.RequestException as e:
    #     logger.error(f"❌ Custom LLM API lỗi request: {e}")
    #     return f"[Lỗi request: {e}]"
    # except Exception as e:
    #     logger.error(f"❌ Custom LLM API lỗi không xác định: {e}")
    #     return f"[Lỗi không xác định: {e}]"
    res = session.post(CUSTOM_LLM_API_URL, headers=headers, json=payload, timeout=15,stream=True)
    for chunk in res.iter_content(chunk_size=None):
        if chunk:
            yield chunk.decode("utf-8")


def extract_main_response(text: str) -> str:
    """
    Trích xuất phần nội dung chính từ response của LLM, bắt đầu từ "Chào bạn..."
    Loại bỏ các phần giải thích thêm và chỉ giữ lại phần nội dung chính.
    """
    if not isinstance(text, str):
        return ""
    
    # Trước tiên, làm sạch text
    text = clean_response(text)
    
    # Pattern 1: Tìm phần trong dấu ngoặc kép sau "Chào bạn"
    chao_ban_quote_pattern = re.compile(r'Chào bạn[^"]*"([^"]*)"', re.IGNORECASE | re.DOTALL)
    match = chao_ban_quote_pattern.search(text)
    
    if match:
        result = match.group(1).strip()
        if result and len(result) > 10:
            return result
    
    # Pattern 2: Tìm phần từ "Chào bạn" đến hết (không có dấu ngoặc kép)
    chao_ban_pattern = re.compile(r'(Chào bạn.*?)(?=\n\n|\nHy vọng những gợi ý này|\nTôi hy vọng những gợi ý này|\nNếu bạn cần thêm hỗ trợ|\nHãy cho tôi biết|\nVui lòng cho tôi biết|\nCảm ơn bạn đã chia sẻ|\nXin lỗi vì|\nĐừng ngần ngại liên hệ|$)', re.IGNORECASE | re.DOTALL)
    match = chao_ban_pattern.search(text)
    
    if match:
        result = match.group(1).strip()
        if result and len(result) > 10:
            return result
    
    # Pattern 3: Tìm phần trong dấu ngoặc kép đầu tiên
    quote_pattern = re.compile(r'"([^"]*)"', re.DOTALL)
    quote_match = quote_pattern.search(text)
    
    if quote_match:
        result = quote_match.group(1).strip()
        if result and len(result) > 10:
            return result
    
    # Pattern 4: Nếu không có pattern nào khớp, trả về text đã clean
    # Xử lý format: giữ lại "Hy vọng những gợi ý này..." và loại bỏ phần sau
    lines = text.split('\n')
    main_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Kiểm tra nếu có câu "Hy vọng những gợi ý này..."
        if any(keyword in line.lower() for keyword in [
            'hy vọng những gợi ý này', 'tôi hy vọng những gợi ý này'
        ]):
            # Loại bỏ phần "Nếu bạn cần thêm hỗ trợ..." khỏi câu này
            clean_line = re.sub(r'\.\s*Nếu bạn cần thêm hỗ trợ.*$', '.', line)
            clean_line = re.sub(r'\.\s*Đừng ngại liên hệ.*$', '.', clean_line)
            main_lines.append(clean_line)
            break  # Dừng lại hoàn toàn, không thêm câu nào nữa
            
        # Loại bỏ các câu không mong muốn khác
        if any(keyword in line.lower() for keyword in [
            'hãy cho tôi biết', 'vui lòng cho tôi biết',
            'cảm ơn bạn đã chia sẻ', 'xin lỗi vì', 'đừng ngần ngại liên hệ',
            'nếu bạn cần thêm hỗ trợ', 'đừng ngại liên hệ',
            'lỗi', 'error', 'assistant', 'user', 'system'
        ]):
            continue
            
        # Thêm các câu khác
        main_lines.append(line)
    
    result = '\n'.join(main_lines).strip()
    
    # Nếu kết quả vẫn rỗng hoặc quá ngắn, trả về text gốc đã clean
    if not result or len(result) < 10:
        return text
    
    return result
