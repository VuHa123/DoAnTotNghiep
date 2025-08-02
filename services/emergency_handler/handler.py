import logging
from datetime import datetime
from typing import Optional, Dict, Any
import sys
import os

# Add parent directory to path for database import
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from Database.core import EmergencyLog

logger = logging.getLogger(__name__)

class EmergencyHandler:
    def __init__(self):
        self.hotline_number = "0984.104.115"
        self.hotline_hours = (
            datetime.strptime("07:30", "%H:%M").time(),
            datetime.strptime("22:00", "%H:%M").time()
        )
    
    def check_emergency(self, user_id: str, message: str) -> Dict[str, Any]:
        """
        Kiểm tra và xử lý tình huống khẩn cấp
        """
        try:
            logger.info(f"Emergency check for user {user_id}: {message[:50]}...")
            
            # Kiểm tra thời gian hoạt động
            now = datetime.now().time()
            is_hotline_open = self.hotline_hours[0] <= now <= self.hotline_hours[1]
            
            logger.info(f"Current time: {now}, Hotline hours: {self.hotline_hours}, Is open: {is_hotline_open}")
            
            if is_hotline_open:
                return self._handle_hotline_call(user_id, message)
            else:
                return self._handle_staff_alert(user_id, message)
                
        except Exception as e:
            logger.error(f"Error in emergency check: {e}")
            return {
                "status": "error",
                "message": "Có lỗi xảy ra khi xử lý tình huống khẩn cấp. Vui lòng thử lại.",
                "action": "none"
            }
    
    def _handle_hotline_call(self, user_id: str, message: str) -> Dict[str, Any]:
        """Xử lý gọi hotline trong giờ hoạt động"""
        try:
            from .hotline_caller import call_hotline
            call_hotline(user_id, message)


            # Lưu log vào database
            self._log_emergency(user_id, message, "hotline", "success")
            
            return {
                "status": "hotline_called",
                "message": f"Tôi nhận thấy bạn đang gặp khó khăn nghiêm trọng. Bạn có thể gọi ngay số {self.hotline_number} để được hỗ trợ từ chuyên gia.",
                "action": "hotline",
                "hotline_number": self.hotline_number
            }
        except Exception as e:
            logger.error(f"Error calling hotline: {e}")
            self._log_emergency(user_id, message, "hotline", "failed")
            return {
                "status": "hotline_failed",
                "message": "Không thể kết nối hotline. Vui lòng thử lại hoặc liên hệ trực tiếp.",
                "action": "manual_contact"
            }
    
    def _handle_staff_alert(self, user_id: str, message: str) -> Dict[str, Any]:
        """Xử lý cảnh báo nhân viên ngoài giờ hoạt động"""
        try:
            from .staff_notifier import alert_staff
            alert_staff(user_id, message)
            
            # Lưu log vào database
            self._log_emergency(user_id, message, "staff_alert", "success")
            
            return {
                "status": "staff_alerted",
                "message": "Tôi sẽ gửi thông báo cho nhân viên hỗ trợ vì hiện giờ ngoài khung giờ hoạt động. Hãy giữ bình tĩnh, bạn không đơn độc.",
                "action": "staff_notification"
            }
        except Exception as e:
            logger.error(f"Error alerting staff: {e}")
            self._log_emergency(user_id, message, "staff_alert", "failed")
            return {
                "status": "alert_failed", 
                "message": "Không thể gửi cảnh báo. Vui lòng liên hệ trực tiếp với nhân viên hỗ trợ.",
                "action": "manual_contact"
            }
    
    def handle_emergency(self, user_id: str, location: Optional[str] = None, 
                        contact: Optional[str] = None) -> Dict[str, Any]:
        """
        Xử lý yêu cầu khẩn cấp từ endpoint /emergency
        """
        try:
            logger.info(f"Emergency request from user {user_id}")
            
            # Lưu thông tin khẩn cấp
            emergency_info = {
                "user_id": user_id,
                "location": location,
                "contact": contact,
                "timestamp": datetime.now().isoformat()
            }
            
            # Gửi cảnh báo cho nhân viên
            from .staff_notifier import alert_staff
            alert_staff(user_id, f"Emergency request - Location: {location}, Contact: {contact}")
            
            return {
                "status": "emergency_handled",
                "message": "Yêu cầu khẩn cấp đã được ghi nhận. Nhân viên sẽ liên hệ với bạn sớm nhất.",
                "emergency_info": emergency_info
            }
            
        except Exception as e:
            logger.error(f"Error handling emergency: {e}")
            return {
                "status": "error",
                "message": "Có lỗi xảy ra khi xử lý yêu cầu khẩn cấp.",
                "error": str(e)
            }
    
    def _log_emergency(self, user_id: str, message: str, action: str, status: str):
        """
        Lưu log emergency vào database
        """
        try:
            emergency_log = EmergencyLog(user_id, message, action, status)
            emergency_log.save()
            logger.info(f"Emergency log saved: {user_id} - {action} - {status}")
        except Exception as e:
            logger.error(f"Error saving emergency log: {e}")

# Hàm legacy để tương thích với code cũ
def check_and_handle_emergency(message: str) -> str:
    """
    Hàm legacy để tương thích với code cũ
    """
    handler = EmergencyHandler()
    result = handler.check_emergency("anonymous", message)
    return result["message"]
