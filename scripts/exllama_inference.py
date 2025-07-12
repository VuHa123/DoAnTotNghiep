#!/usr/bin/env python3
"""
Script inference sử dụng ExLlama với GPTQ model đã fine-tune
"""

import os
import sys
import torch
import argparse
from typing import List, Tuple, Optional
import json

# Thêm đường dẫn ExLlama vào sys.path
EXLLAMA_PATH = os.path.join(os.path.dirname(__file__), "..", "exllama")
if os.path.exists(EXLLAMA_PATH):
    sys.path.append(EXLLAMA_PATH)
else:
    print("⚠️  ExLlama chưa được cài đặt. Vui lòng clone ExLlama repository trước.")
    print("git clone https://github.com/turboderp/exllama.git")

try:
    from exllama.model import ExLlama, ExLlamaConfig
    from exllama.tokenizer import ExLlamaTokenizer
    from exllama.generator import ExLlamaGenerator
    from exllama.lora import ExLlamaLora
except ImportError:
    print("❌ Không thể import ExLlama. Vui lòng cài đặt ExLlama trước.")
    sys.exit(1)

class ExLlamaChatbot:
    def __init__(
        self,
        model_path: str,
        max_seq_len: int = 2048,
        max_input_len: int = 512,
        max_attn_size: int = 2048,
        compress_pos_emb: float = 1.0,
        gpu_peer_fix: bool = False,
        auto_map: bool = True
    ):
        """
        Khởi tạo ExLlama chatbot
        
        Args:
            model_path: Đường dẫn đến GPTQ model
            max_seq_len: Độ dài tối đa của sequence
            max_input_len: Độ dài tối đa của input
            max_attn_size: Kích thước attention tối đa
            compress_pos_emb: Compression cho positional embedding
            gpu_peer_fix: Fix cho GPU peer
            auto_map: Tự động map model lên GPU
        """
        self.model_path = model_path
        self.max_seq_len = max_seq_len
        self.max_input_len = max_input_len
        
        print(f"🔄 Loading ExLlama model from: {model_path}")
        
        # Cấu hình model
        config = ExLlamaConfig(
            model_path,
            max_seq_len=max_seq_len,
            max_input_len=max_input_len,
            max_attn_size=max_attn_size,
            compress_pos_emb=compress_pos_emb,
            gpu_peer_fix=gpu_peer_fix,
            auto_map=auto_map
        )
        
        # Load model
        self.model = ExLlama(config)
        
        # Load tokenizer
        self.tokenizer = ExLlamaTokenizer(config)
        
        # Tạo generator
        self.generator = ExLlamaGenerator(self.model, self.tokenizer)
        
        # Cấu hình generation
        self.generator.settings.temperature = 0.7
        self.generator.settings.top_p = 0.9
        self.generator.settings.top_k = 40
        self.generator.settings.typical = 0.7
        self.generator.settings.token_repetition_penalty = 1.1
        self.generator.settings.token_repetition_range = -1
        self.generator.settings.token_repetition_decay = 0.9
        
        print("✅ ExLlama model đã được load thành công!")
    
    def generate_response(
        self,
        prompt: str,
        max_new_tokens: int = 200,
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
        # Cập nhật settings nếu có
        if temperature is not None:
            self.generator.settings.temperature = temperature
        if top_p is not None:
            self.generator.settings.top_p = top_p
        if top_k is not None:
            self.generator.settings.top_k = top_k
        
        # Generate response
        response = self.generator.generate(
            prompt,
            max_new_tokens=max_new_tokens,
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
        Tạo response cho chat
        
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
        
        # Thêm history
        for user_msg, bot_msg in history[-3:]:  # Chỉ lấy 3 cặp gần nhất
            prompt += f"User: {user_msg}\nAssistant: {bot_msg}\n\n"
        
        # Thêm input mới
        prompt += f"User: {user_input}\nAssistant: "
        
        # Generate response
        response = self.generate_response(prompt, max_new_tokens=200)
        
        # Phân loại emotion (đơn giản)
        emotion = self._classify_emotion(user_input)
        
        return response, emotion
    
    def _classify_emotion(self, text: str) -> str:
        """
        Phân loại emotion từ text (đơn giản)
        
        Args:
            text: Input text
            
        Returns:
            Emotion label
        """
        text = text.lower()
        
        if any(x in text for x in ["chết", "tự tử", "vô dụng", "không muốn sống"]):
            return "Depression"
        elif any(x in text for x in ["lo lắng", "sợ", "căng thẳng", "hoảng loạn"]):
            return "Anxiety"
        elif any(x in text for x in ["tức giận", "giận", "bực", "khó chịu"]):
            return "Anger"
        elif any(x in text for x in ["vui", "hạnh phúc", "tốt", "tuyệt"]):
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
        
        print("🧪 Testing generation...")
        print("=" * 50)
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\nTest {i}: {prompt}")
            print("-" * 30)
            
            response, emotion = self.chat_response(prompt)
            print(f"Response: {response}")
            print(f"Emotion: {emotion}")
            print()

def main():
    parser = argparse.ArgumentParser(description="ExLlama GPTQ Inference")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Đường dẫn đến GPTQ model")
    parser.add_argument("--max_seq_len", type=int, default=2048,
                       help="Độ dài tối đa của sequence")
    parser.add_argument("--max_input_len", type=int, default=512,
                       help="Độ dài tối đa của input")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Temperature cho generation")
    parser.add_argument("--max_new_tokens", type=int, default=200,
                       help="Số token tối đa generate")
    parser.add_argument("--test", action="store_true",
                       help="Chạy test generation")
    parser.add_argument("--interactive", action="store_true",
                       help="Chạy interactive mode")
    
    args = parser.parse_args()
    
    # Khởi tạo chatbot
    chatbot = ExLlamaChatbot(
        model_path=args.model_path,
        max_seq_len=args.max_seq_len,
        max_input_len=args.max_input_len
    )
    
    if args.test:
        chatbot.test_generation()
    
    if args.interactive:
        print("💬 Interactive mode - Gõ 'quit' để thoát")
        print("=" * 50)
        
        history = []
        while True:
            try:
                user_input = input("\n👤 Bạn: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Tạm biệt!")
                    break
                
                if not user_input:
                    continue
                
                response, emotion = chatbot.chat_response(user_input, history)
                print(f"🤖 Assistant: {response}")
                print(f"📊 Emotion: {emotion}")
                
                history.append((user_input, response))
                
            except KeyboardInterrupt:
                print("\n👋 Tạm biệt!")
                break
            except Exception as e:
                print(f"❌ Lỗi: {e}")

if __name__ == "__main__":
    main() 