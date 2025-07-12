#!/usr/bin/env python3
"""
ExLlama Chatbot Module - Tích hợp vào hệ thống hiện tại
"""

import os
import sys
import torch
from typing import List, Tuple, Optional
import logging

# Thêm đường dẫn ExLlama
EXLLAMA_PATH = os.path.join(os.path.dirname(__file__), "..", "exllama")
if os.path.exists(EXLLAMA_PATH):
    sys.path.append(EXLLAMA_PATH)

try:
    from exllama.model import ExLlama, ExLlamaConfig
    from exllama.tokenizer import ExLlamaTokenizer
    from exllama.generator import ExLlamaGenerator
except ImportError:
    logging.warning("ExLlama not available. Please install ExLlama first.")
    ExLlama = None
    ExLlamaConfig = None
    ExLlamaTokenizer = None
    ExLlamaGenerator = None

class ExLlamaChatbotModel:
    """ExLlama Chatbot Model - Tích hợp với hệ thống hiện tại"""
    
    def __init__(
        self,
        model_path: str,
        max_seq_len: int = 2048,
        max_input_len: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        max_new_tokens: int = 200
    ):
        """
        Khởi tạo ExLlama chatbot
        
        Args:
            model_path: Đường dẫn đến GPTQ model
            max_seq_len: Độ dài tối đa của sequence
            max_input_len: Độ dài tối đa của input
            temperature: Temperature cho generation
            top_p: Top-p sampling
            top_k: Top-k sampling
            max_new_tokens: Số token tối đa generate
        """
        if ExLlama is None:
            raise ImportError("ExLlama not available. Please install ExLlama first.")
        
        self.model_path = model_path
        self.max_seq_len = max_seq_len
        self.max_input_len = max_input_len
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        
        self.model = None
        self.tokenizer = None
        self.generator = None
        
        self._load_model()
    
    def _load_model(self):
        """Load ExLlama model"""
        try:
            logging.info(f"Loading ExLlama model from: {self.model_path}")
            
            # Cấu hình model
            config = ExLlamaConfig(
                self.model_path,
                max_seq_len=self.max_seq_len,
                max_input_len=self.max_input_len,
                max_attn_size=2048,
                compress_pos_emb=1.0,
                gpu_peer_fix=False,
                auto_map=True
            )
            
            # Load model
            self.model = ExLlama(config)
            
            # Load tokenizer
            self.tokenizer = ExLlamaTokenizer(config)
            
            # Tạo generator
            self.generator = ExLlamaGenerator(self.model, self.tokenizer)
            
            # Cấu hình generation settings
            self.generator.settings.temperature = self.temperature
            self.generator.settings.top_p = self.top_p
            self.generator.settings.top_k = self.top_k
            self.generator.settings.typical = 0.7
            self.generator.settings.token_repetition_penalty = 1.1
            self.generator.settings.token_repetition_range = -1
            self.generator.settings.token_repetition_decay = 0.9
            
            logging.info("ExLlama model loaded successfully!")
            
        except Exception as e:
            logging.error(f"Failed to load ExLlama model: {e}")
            raise
    
    def generate_response(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None
    ) -> str:
        """
        Generate response từ prompt
        
        Args:
            prompt: Input prompt
            max_new_tokens: Số token tối đa generate
            temperature: Temperature cho generation
            top_p: Top-p sampling
            top_k: Top-k sampling
            
        Returns:
            Generated response
        """
        if self.generator is None:
            raise RuntimeError("Model not loaded")
        
        # Cập nhật settings nếu có
        if temperature is not None:
            self.generator.settings.temperature = temperature
        if top_p is not None:
            self.generator.settings.top_p = top_p
        if top_k is not None:
            self.generator.settings.top_k = top_k
        
        max_tokens = max_new_tokens or self.max_new_tokens
        
        # Generate response
        response = self.generator.generate(
            prompt,
            max_new_tokens=max_tokens,
            add_bos=True
        )
        
        # Trả về phần response mới (không bao gồm prompt)
        return response[len(prompt):].strip()
    
    def chat_response(
        self,
        user_input: str,
        history: List[Tuple[str, str]] = None,
        system_prompt: str = "Bạn là chuyên gia tâm lý. Hãy trả lời một cách nhẹ nhàng, thấu hiểu và hữu ích."
    ) -> Tuple[str, str]:
        """
        Tạo response cho chat - Tương thích với interface hiện tại
        
        Args:
            user_input: Input từ user
            history: Lịch sử chat
            system_prompt: System prompt
            
        Returns:
            Tuple (response, emotion_label)
        """
        if history is None:
            history = []
        
        # Tạo prompt từ history và input mới
        prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
        
        # Thêm history (chỉ lấy 3 cặp gần nhất để tránh quá dài)
        for user_msg, bot_msg in history[-3:]:
            prompt += f"User: {user_msg}\nAssistant: {bot_msg}\n\n"
        
        # Thêm input mới
        prompt += f"User: {user_input}\nAssistant: "
        
        # Generate response
        response = self.generate_response(prompt)
        
        # Phân loại emotion
        emotion = self._classify_emotion(user_input)
        
        return response, emotion
    
    def _classify_emotion(self, text: str) -> str:
        """
        Phân loại emotion từ text
        
        Args:
            text: Input text
            
        Returns:
            Emotion label
        """
        text = text.lower()
        
        if any(x in text for x in ["chết", "tự tử", "vô dụng", "không muốn sống", "vô nghĩa"]):
            return "Depression"
        elif any(x in text for x in ["lo lắng", "sợ", "căng thẳng", "hoảng loạn", "bất an"]):
            return "Anxiety"
        elif any(x in text for x in ["tức giận", "giận", "bực", "khó chịu", "cáu"]):
            return "Anger"
        elif any(x in text for x in ["vui", "hạnh phúc", "tốt", "tuyệt", "thích"]):
            return "Happy"
        else:
            return "Normal"
    
    def test_generation(self, test_prompts: List[str] = None):
        """
        Test generation với một số prompt mẫu
        
        Args:
            test_prompts: List các prompt để test
        """
        if test_prompts is None:
            test_prompts = [
                "Tôi cảm thấy rất căng thẳng và lo lắng về tương lai.",
                "Tôi không muốn sống nữa, cuộc sống thật vô nghĩa.",
                "Tôi rất vui vì hôm nay đã hoàn thành được nhiều việc.",
                "Bạn có thể cho tôi lời khuyên về cách quản lý stress không?"
            ]
        
        print("🧪 Testing ExLlama generation...")
        print("=" * 50)
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\nTest {i}: {prompt}")
            print("-" * 30)
            
            response, emotion = self.chat_response(prompt)
            print(f"Response: {response}")
            print(f"Emotion: {emotion}")
            print()

# Factory function để tạo ExLlama chatbot
def create_exllama_chatbot(
    model_path: str,
    **kwargs
) -> ExLlamaChatbotModel:
    """
    Factory function để tạo ExLlama chatbot
    
    Args:
        model_path: Đường dẫn đến GPTQ model
        **kwargs: Các tham số khác cho ExLlamaChatbotModel
        
    Returns:
        ExLlamaChatbotModel instance
    """
    return ExLlamaChatbotModel(model_path, **kwargs)

# Test function
def test_exllama_chatbot():
    """Test ExLlama chatbot"""
    try:
        # Thay đổi đường dẫn model theo thực tế
        model_path = "models/weights/chatbot_gptq"
        
        if not os.path.exists(model_path):
            print(f"❌ Model path không tồn tại: {model_path}")
            print("Vui lòng convert model sang GPTQ format trước.")
            return False
        
        chatbot = create_exllama_chatbot(model_path)
        chatbot.test_generation()
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_exllama_chatbot() 