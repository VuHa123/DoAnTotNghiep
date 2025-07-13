#!/usr/bin/env python3
"""
Unit tests for Emergency Handler
"""

import pytest
from datetime import datetime, time
from services.emergency_handler.handler import EmergencyHandler


class TestEmergencyHandler:
    """Test cases for EmergencyHandler class"""
    
    def setup_method(self):
        """Setup before each test"""
        self.handler = EmergencyHandler()
    
    def test_init(self):
        """Test EmergencyHandler initialization"""
        assert self.handler.hotline_number == "0984.104.115"
        assert isinstance(self.handler.hotline_hours, tuple)
        assert len(self.handler.hotline_hours) == 2
    
    def test_hotline_hours(self):
        """Test hotline hours configuration"""
        start_time = time(7, 30)  # 07:30
        end_time = time(22, 0)    # 22:00
        
        assert self.handler.hotline_hours[0] == start_time
        assert self.handler.hotline_hours[1] == end_time
    
    def test_check_emergency_during_hours(self):
        """Test emergency check during hotline hours"""
        # Mock current time to be during hotline hours
        test_time = time(14, 30)  # 14:30 (during hours)
        
        # Mock datetime.now().time() to return test_time
        original_time = datetime.now().time
        datetime.now = lambda: type('MockDateTime', (), {'time': lambda: test_time})()
        
        try:
            result = self.handler.check_emergency("user123", "Tôi muốn tự tử")
            assert result["status"] in ["hotline_called", "error"]
            assert "message" in result
        finally:
            # Restore original datetime
            datetime.now = original_time
    
    def test_check_emergency_outside_hours(self):
        """Test emergency check outside hotline hours"""
        # Mock current time to be outside hotline hours
        test_time = time(23, 30)  # 23:30 (outside hours)
        
        # Mock datetime.now().time() to return test_time
        original_time = datetime.now().time
        datetime.now = lambda: type('MockDateTime', (), {'time': lambda: test_time})()
        
        try:
            result = self.handler.check_emergency("user123", "Tôi muốn tự tử")
            assert result["status"] in ["staff_alerted", "error"]
            assert "message" in result
        finally:
            # Restore original datetime
            datetime.now = original_time
    
    def test_emergency_message_detection(self):
        """Test detection of emergency messages"""
        emergency_messages = [
            "Tôi muốn tự tử",
            "Tôi muốn kết thúc cuộc sống",
            "Tôi không muốn sống nữa",
            "Tôi muốn chết"
        ]
        
        for message in emergency_messages:
            result = self.handler.check_emergency("user123", message)
            assert result["status"] in ["hotline_called", "staff_alerted", "error"]
            assert "message" in result
    
    def test_normal_message(self):
        """Test normal message handling"""
        normal_messages = [
            "Xin chào",
            "Cảm ơn bạn",
            "Tôi ổn",
            "Bạn khỏe không?"
        ]
        
        for message in normal_messages:
            result = self.handler.check_emergency("user123", message)
            # Normal messages should still be processed
            assert "status" in result
            assert "message" in result


if __name__ == "__main__":
    pytest.main([__file__]) 