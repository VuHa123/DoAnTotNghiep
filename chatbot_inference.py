#!/usr/bin/env python3
"""
Chatbot Inference Script
Sử dụng checkpoint fine-tuned để thực hiện inference
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import logging
import argparse
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft.peft_model import PeftModel
import re
import threading

# Load environment variables
load_dotenv("token.env")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_response(text):
    """
    Làm sạch các token và phần thừa trong output của mô hình
    """
    # Xoá token đặc biệt dạng <|...|> hoặc XML/template marker
    text = re.sub(r'<\|.*?\|>', '', text)
    # Xoá ký tự 'X' thường xuất hiện thừa đầu câu
    text = text.replace('X', '')
    # Xoá các chỉ thị bị sinh dư
    garbage_patterns = [
        r"Response \(bằng cách thay đổi\):",
        r"Cảm ơn sự hiểu biết.*?",
        r"hoặc.*$",                      # nếu sinh nhiều lựa chọn
        r"### Instruction:.*",
        r"### Input:.*",
        r"### Response:"
    ]
    for pattern in garbage_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    return text.strip()

class ChatbotInference:
    def __init__(self, checkpoint_name="checkpoint-1098"):
        """
        Khởi tạo chatbot inference
        
        Args:
            checkpoint_name: Tên checkpoint (checkpoint-1098, checkpoint-1000, hoặc final_model)
        """
        self.checkpoint_name = checkpoint_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.stop_event = threading.Event()
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Using checkpoint: {checkpoint_name}")
        
    def load_model(self):
        """Load model và tokenizer"""
        try:
            # Local paths
            base_model_path = "/home/aero/DoAnTotNghiep/models/weights/base_model/meta-llama/Llama-3.2-1B-Instruct"  # <-- base model local
            checkpoint_path = f"models/weights/chatbot_finetuned/{self.checkpoint_name}"
            
            if not os.path.exists(checkpoint_path):
                logger.error(f"❌ Checkpoint not found: {checkpoint_path}")
                return False
            
            logger.info(f"✅ Checkpoint found: {checkpoint_path}")
            
            # Load base model
            logger.info("Loading base model...")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            
            # Load tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load LoRA adapter từ local
            logger.info("Loading LoRA adapter...")
            self.model = PeftModel.from_pretrained(
                base_model,
                checkpoint_path
            )
            self.model.eval()
            self.model = self.model.to(self.device)
            
            logger.info("✅ Model loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            return False

    def stop_generation(self):
        """Gọi hàm này để dừng sinh phản hồi ngay lập tức"""
        self.stop_event.set()
    
    def generate_response(self, prompt, max_new_tokens=200, temperature=0.7, top_p=0.9):
        """
        Generate response từ prompt, hỗ trợ dừng sinh phản hồi qua self.stop_event
        """
        try:
            if self.model is None or self.tokenizer is None:
                raise RuntimeError("Model chưa được load. Gọi load_model() trước.")
            self.stop_event.clear()
            formatted_prompt = f"""### Instruction:\nBạn là một chatbot hỗ trợ tâm lý chuyên nghiệp. Hãy trả lời người dùng một cách thân thiện, đồng cảm và hữu ích.\n\n### Input:\n{prompt}\n\n### Response:\n"""
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=False
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Chỉ lấy phần sau ### Response:
            if "### Response:" in response:
                response = response.split("### Response:")[-1].strip()
            # Nếu vẫn còn ### Instruction hoặc ### Input do model sinh dư, loại bỏ chúng
            response = re.split(r"### Instruction:|### Input:", response)[0].strip()
            response = clean_response(response)
            return response
        except Exception as e:
            logger.error(f"❌ Error generating response: {e}")
            return "Xin lỗi, tôi đang gặp sự cố kỹ thuật. Vui lòng thử lại sau."
    
    def test_inference(self):
        """Test inference với các prompt mẫu"""
        logger.info("🧪 Testing inference...")
        
        test_prompts = [
            "Tôi cảm thấy rất lo lắng về tương lai",
            "Tôi không muốn sống nữa",
            "Tôi cảm thấy cô đơn và buồn bã",
            "Làm sao để tôi có thể vượt qua khó khăn này?",
            "Tôi đang gặp khó khăn trong mối quan hệ với bạn bè"
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            logger.info(f"\n--- Test {i}: {prompt} ---")
            response = self.generate_response(prompt)
            logger.info(f"Response: {response}")
        
        logger.info("✅ All tests completed successfully!")
    
    def interactive_chat(self):
        """Interactive chat mode"""
        logger.info("💬 Starting interactive chat...")
        logger.info("Type 'quit' to exit")
        logger.info("-" * 50)
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    logger.info("👋 Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                # Generate response
                response = self.generate_response(user_input)
                print(f"\n🤖 Bot: {response}")
                
            except KeyboardInterrupt:
                logger.info("\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"❌ Error: {e}")

def list_available_checkpoints():
    """Liệt kê các checkpoint có sẵn"""
    checkpoint_dir = "models/weights/chatbot_finetuned"
    checkpoints = []
    
    if os.path.exists(checkpoint_dir):
        for item in os.listdir(checkpoint_dir):
            item_path = os.path.join(checkpoint_dir, item)
            if os.path.isdir(item_path) and item.startswith("checkpoint-"):
                checkpoints.append(item)
        
        # Kiểm tra final_model
        final_model_path = os.path.join(checkpoint_dir, "final_model")
        if os.path.exists(final_model_path):
            checkpoints.append("final_model")
    
    return checkpoints
def detect_emergency(self, user_input: str) -> str:
        """
        Phát hiện nguy cơ khẩn cấp (tự tử, làm hại bản thân...) từ input người dùng
        Args:
            user_input (str): Câu nhập từ người dùng
        Returns:
            str: 'emergency' nếu có nguy cơ khẩn cấp, 'normal' nếu không
        """
        emergency_keywords = [
            "tự tử", "muốn chết", "kết thúc cuộc đời", "tự sát", "đau khổ quá",
            "không thể chịu nổi", "chán sống", "kết liễu", "tôi sẽ chết", "muốn biến mất"
        ]
        normalized_input = user_input.lower().strip()
        for keyword in emergency_keywords:
            if keyword in normalized_input:
                logger.warning(f"🚨 Phát hiện nguy cơ khẩn cấp với từ khóa: {keyword}")
                return "emergency"
        return "normal"

def main():
    parser = argparse.ArgumentParser(description="Chatbot Inference")
    parser.add_argument("--checkpoint", type=str, default="checkpoint-1000",
                       help="Checkpoint name (checkpoint-549, checkpoint-1000, final_model)")
    parser.add_argument("--test", action="store_true",
                       help="Run test inference")
    parser.add_argument("--interactive", action="store_true",
                       help="Run interactive chat")
    parser.add_argument("--prompt", type=str,
                       help="Single prompt for inference")
    parser.add_argument("--list", action="store_true",
                       help="List available checkpoints")
    parser.add_argument("--max_tokens", type=int, default=200,
                       help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Temperature for generation")
    
    args = parser.parse_args()
    
    # List checkpoints
    if args.list:
        checkpoints = list_available_checkpoints()
        if checkpoints:
            logger.info("Available checkpoints:")
            for cp in checkpoints:
                logger.info(f"  - {cp}")
        else:
            logger.warning("No checkpoints found")
        return
    
    # Khởi tạo chatbot
    chatbot = ChatbotInference(args.checkpoint)
    
    # Load model
    if not chatbot.load_model():
        logger.error("❌ Failed to load model")
        return
    
    # Chạy theo mode được chọn
    if args.test:
        chatbot.test_inference()
    elif args.interactive:
        chatbot.interactive_chat()
    elif args.prompt:
        response = chatbot.generate_response(
            args.prompt, 
            max_new_tokens=args.max_tokens,
            temperature=args.temperature
        )
        logger.info(f"Prompt: {args.prompt}")
        logger.info(f"Response: {response}")
    else:
        # Default: interactive mode
        chatbot.interactive_chat()

if __name__ == "__main__":
    main()