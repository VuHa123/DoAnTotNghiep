#!/usr/bin/env python3
"""
LLaMA Model Service
Service để gọi Model Inference Server
"""

import os
import time
import logging
import requests
from typing import Dict, Optional
from urllib.parse import urljoin

logger = logging.getLogger(__name__)

class LLaMAService:
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.timeout = 30  # seconds
        self.max_retries = 3
        self.retry_delay = 1
        
    def call_model_server(self, prompt: str, max_length: int = 512, 
                         temperature: float = 0.7, top_p: float = 0.9) -> Optional[str]:
        """Gọi Model Inference Server"""
        
        for attempt in range(self.max_retries):
            try:
                url = urljoin(self.base_url, "/generate")
                
                payload = {
                    "prompt": prompt,
                    "max_length": max_length,
                    "temperature": temperature,
                    "top_p": top_p,
                    "do_sample": True
                }
                
                response = requests.post(
                    url, 
                    json=payload, 
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"Model server response: {result['inference_time']:.2f}s")
                    return result["response"]
                else:
                    logger.warning(f"Model server error {response.status_code}: {response.text}")
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Model server timeout on attempt {attempt + 1}")
            except requests.exceptions.ConnectionError:
                logger.warning(f"Model server connection error on attempt {attempt + 1}")
            except Exception as e:
                logger.warning(f"Model server error on attempt {attempt + 1}: {e}")
                
            # Wait before retry
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay)
                
        logger.error("All attempts to model server failed")
        return None
        
    def build_mental_health_prompt(self, 
                                  user_message: str, 
                                  sentiment: str = None,
                                  mental_state: str = None,
                                  risk_level: str = None) -> str:
        """Xây dựng prompt động dựa trên context"""
        
        base_prompt = f"Người dùng: {user_message}\n\n"
        
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
            
        base_prompt += "Trợ lý tâm lý:"
        
        return base_prompt
        
    def get_response(self, 
                    user_message: str,
                    sentiment: str = None,
                    mental_state: str = None, 
                    risk_level: str = None) -> Dict:
        """Lấy response từ Model Server với context"""
        
        try:
            # Xây dựng prompt động
            prompt = self.build_mental_health_prompt(
                user_message, sentiment, mental_state, risk_level
            )
            
            # Gọi Model Server
            response = self.call_model_server(prompt)
            
            if response:
                return {
                    "success": True,
                    "response": response,
                    "source": "llama_model_server"
                }
            else:
                return {
                    "success": False,
                    "response": "Xin lỗi, tôi đang gặp sự cố kỹ thuật. Vui lòng thử lại sau.",
                    "source": "fallback"
                }
                
        except Exception as e:
            logger.error(f"Error in LLaMA service: {e}")
            return {
                "success": False,
                "response": "Xin lỗi, có lỗi xảy ra. Vui lòng thử lại sau.",
                "source": "error"
            }
            
    def check_server_health(self) -> Dict:
        """Kiểm tra health của Model Server"""
        try:
            url = urljoin(self.base_url, "/health")
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                return {
                    "status": "healthy",
                    "model_loaded": response.json().get("model_loaded", False),
                    "device": response.json().get("device"),
                    "model_name": response.json().get("model_name")
                }
            else:
                return {
                    "status": "unhealthy",
                    "error": f"HTTP {response.status_code}"
                }
                
        except Exception as e:
            return {
                "status": "unreachable",
                "error": str(e)
            }

# Global instance
llama_service = LLaMAService() 