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
from services.gating_router.prompt_builder import build_prompt_from_object

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
        
    def get_response(self, 
                    user_message: str = "",
                    sentiment: str = "",
                    mental_state: str = "", 
                    risk_level: str = "",
                    prompt_obj: dict = {}) -> Dict:
        """Lấy response từ Model Server với context. Có thể truyền prompt object mới."""
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