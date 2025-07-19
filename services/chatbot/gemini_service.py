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
import re
from services.gating_router.prompt_builder import build_prompt_from_object

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
        
    def get_response(self, 
                    user_message: str = "",
                    sentiment: str = "",
                    mental_state: str = "", 
                    risk_level: str = "",
                    prompt_obj: dict = {}) -> Dict:
        """Lấy response từ Gemini với context. Có thể truyền prompt object mới."""
        try:
            if prompt_obj:
                prompt = build_prompt_from_object(prompt_obj)
            else:
                # Backward compatibility
                prompt_obj = {
                    "input": user_message,
                    "context": {
                        "mental_state": mental_state,
                        "sentiment_intensity": sentiment,
                        "risk_level": risk_level
                    }
                }
                prompt = build_prompt_from_object(prompt_obj)
            response = self.call_gemini_api(prompt)
            def clean_special_tokens(text):
                return re.sub(r"<\|.*?\|>", "", text)
            if response:
                response = clean_special_tokens(response)
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