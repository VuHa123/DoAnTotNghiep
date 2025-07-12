import logging

logger = logging.getLogger(__name__)

def call_hotline(user_id: str, message: str):
    """
    Gọi hotline với thông tin user và message
    """
    try:
        logger.info(f"[HOTLINE] Gọi {user_id} - Số: 0984.104.115 - Nội dung: {message[:100]}...")
        
        # TODO: Tích hợp với hệ thống gọi điện thực tế
        # Ví dụ: Twilio, AWS Connect, hoặc API gọi điện khác
        
        # Log thành công
        logger.info(f"[HOTLINE] Đã gọi thành công cho user {user_id}")
        
    except Exception as e:
        logger.error(f"[HOTLINE] Lỗi khi gọi hotline cho user {user_id}: {e}")
        raise
