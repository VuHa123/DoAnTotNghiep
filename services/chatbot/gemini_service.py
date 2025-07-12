#!/usr/bin/env python3
"""
Gemini API Service
Wrapper cho Gemini API với rotation key và error handling
"""

import os
import time
import logging
import google.generativeai as genai
from typing import Dict, Optional, List
from utils.api_manager import api_manager

logger = logging.getLogger(__name__)

class GeminiService:
    def __init__(self):
        self.api_manager = api_manager
        self.max_retries = 3
        self.retry_delay = 1  # seconds
        
    def call_gemini_api(self, prompt: str, max_tokens: int = 1000) -> Optional[str]:
        """Gọi Gemini API với rotation key và retry logic"""
        
        for attempt in range(self.max_retries):
            # Lấy API key tốt nhất
            api_key = self.api_manager.get_best_api_key()
            if not api_key:
                logger.error("No valid API key available")
                return None
                
            try:
                # Configure Gemini
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-pro')
                
                # Generate response
                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=max_tokens,
                        temperature=0.7
                    )
                )
                
                # Mark key as successfully used
                self.api_manager.mark_key_used(api_key, success=True)
                
                return response.text
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed with key {api_key[:10]}...: {e}")
                
                # Mark key as failed
                self.api_manager.mark_key_used(api_key, success=False)
                
                # Wait before retry
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    
        logger.error("All API keys failed after retries")
        return None
        
    def build_mental_health_prompt(self, 
                                  user_message: str, 
                                  sentiment: str = None,
                                  mental_state: str = None,
                                  risk_level: str = None) -> str:
        """Xây dựng prompt động dựa trên context"""
        
        base_prompt = f"""Bạn là một chatbot hỗ trợ tâm lý chuyên nghiệp. 
Hãy trả lời người dùng một cách thân thiện, đồng cảm và hữu ích.

Người dùng: {user_message}

"""
        
        # Thêm context dựa trên sentiment
        if sentiment == "negative" or sentiment == "3":
            base_prompt += "Lưu ý: Người dùng có vẻ đang cảm thấy tiêu cực. Hãy trả lời nhẹ nhàng, động viên và gợi ý các phương pháp tự chăm sóc.\n\n"
            
        # Thêm context dựa trên mental state
        if mental_state in ["stress", "anxiety", "depression"]:
            base_prompt += "Lưu ý: Người dùng có dấu hiệu stress/lo lắng/trầm cảm. Hãy trả lời với sự đồng cảm và gợi ý liên hệ chuyên gia nếu cần.\n\n"
            
        # Thêm context dựa trên risk level
        if risk_level == "emergency":
            base_prompt += "⚠️ KHẨN CẤP: Người dùng có dấu hiệu nguy hiểm. Hãy trả lời ngắn gọn, động viên và gợi ý liên hệ hotline hỗ trợ khẩn cấp ngay lập tức.\n\n"
        elif risk_level == "risky":
            base_prompt += "⚠️ RỦI RO: Người dùng có dấu hiệu rủi ro. Hãy trả lời cẩn thận, động viên và gợi ý tìm kiếm sự hỗ trợ chuyên môn.\n\n"
            
        base_prompt += "Trả lời:"
        
        return base_prompt
        
    def get_response(self, 
                    user_message: str,
                    sentiment: str = None,
                    mental_state: str = None, 
                    risk_level: str = None) -> Dict:
        """Lấy response từ Gemini với context"""
        
        try:
            # Xây dựng prompt động
            prompt = self.build_mental_health_prompt(
                user_message, sentiment, mental_state, risk_level
            )
            
            # Gọi API
            response = self.call_gemini_api(prompt)
            
            if response:
                return {
                    "success": True,
                    "response": response,
                    "source": "gemini_api"
                }
            else:
                return {
                    "success": False,
                    "response": "Xin lỗi, tôi đang gặp sự cố kỹ thuật. Vui lòng thử lại sau.",
                    "source": "fallback"
                }
                
        except Exception as e:
            logger.error(f"Error in Gemini service: {e}")
            return {
                "success": False,
                "response": "Xin lỗi, có lỗi xảy ra. Vui lòng thử lại sau.",
                "source": "error"
            }
            
    def get_api_stats(self) -> Dict:
        """Lấy thống kê API usage"""
        return self.api_manager.get_key_stats()

# Global instance
gemini_service = GeminiService() 