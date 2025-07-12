"""
Emergency Handler Module
Xử lý các tình huống khẩn cấp trong hệ thống chatbot sức khỏe tâm thần
"""

from .handler import EmergencyHandler, check_and_handle_emergency
from .hotline_caller import call_hotline
from .staff_notifier import alert_staff

__all__ = [
    "EmergencyHandler",
    "check_and_handle_emergency", 
    "call_hotline",
    "alert_staff"
] 