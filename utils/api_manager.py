#!/usr/bin/env python3
"""
API Key Manager cho Gemini
Quản lý 15 API key với rotation và fallback
"""

import os
import time
import random
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class GeminiAPIManager:
    def __init__(self):
        self.api_keys = []
        self.key_usage = {}  # Track usage per key
        self.key_errors = {}  # Track errors per key
        self.last_used = {}   # Track last used time
        self.load_api_keys()
        
    def load_api_keys(self):
        """Load API keys từ file token.env"""
        try:
            from utils.token_loader import token_loader
            
            # Load Gemini keys từ token loader
            self.api_keys = token_loader.get_gemini_keys()
            
            # Initialize tracking cho mỗi key
            for api_key in self.api_keys:
                self.key_usage[api_key] = 0
                self.key_errors[api_key] = 0
                self.last_used[api_key] = None
                        
            logger.info(f"Loaded {len(self.api_keys)} Gemini API keys")
            
        except Exception as e:
            logger.error(f"Error loading API keys: {e}")
            
    def get_best_api_key(self) -> Optional[str]:
        """Chọn API key tốt nhất dựa trên usage và errors"""
        if not self.api_keys:
            return None
            
        # Filter out keys with too many errors
        valid_keys = [key for key in self.api_keys 
                     if self.key_errors.get(key, 0) < 5]
        
        if not valid_keys:
            # Reset errors if all keys have too many errors
            self.key_errors = {key: 0 for key in self.api_keys}
            valid_keys = self.api_keys
            
        # Sort by usage (least used first)
        valid_keys.sort(key=lambda k: self.key_usage.get(k, 0))
        
        # Return the least used key
        return valid_keys[0] if valid_keys else None
        
    def mark_key_used(self, api_key: str, success: bool = True):
        """Đánh dấu key đã được sử dụng"""
        if api_key in self.key_usage:
            self.key_usage[api_key] += 1
            self.last_used[api_key] = datetime.now()
            
        if not success and api_key in self.key_errors:
            self.key_errors[api_key] += 1
            
    def get_key_stats(self) -> Dict:
        """Lấy thống kê sử dụng API keys"""
        return {
            "total_keys": len(self.api_keys),
            "usage": self.key_usage,
            "errors": self.key_errors,
            "last_used": {k: v.isoformat() if v else None 
                         for k, v in self.last_used.items()}
        }
        
    def reset_key_errors(self, api_key: str = None):
        """Reset lỗi cho key cụ thể hoặc tất cả"""
        if api_key:
            self.key_errors[api_key] = 0
        else:
            self.key_errors = {key: 0 for key in self.api_keys}

# Global instance
api_manager = GeminiAPIManager() 