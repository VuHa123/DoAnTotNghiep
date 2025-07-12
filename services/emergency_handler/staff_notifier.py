import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def alert_staff(user_id: str, message: str):
    """
    Gửi cảnh báo cho nhân viên hỗ trợ
    """
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        alert_message = f"[ALERT] {timestamp} - User: {user_id} - Message: {message[:200]}..."
        
        logger.warning(alert_message)
        
        # TODO: Tích hợp với hệ thống thông báo thực tế
        # Ví dụ: Slack, Discord, Email, SMS, hoặc hệ thống nội bộ
        
        # Log thành công
        logger.info(f"[ALERT] Đã gửi cảnh báo thành công cho user {user_id}")
        
    except Exception as e:
        logger.error(f"[ALERT] Lỗi khi gửi cảnh báo cho user {user_id}: {e}")
        raise