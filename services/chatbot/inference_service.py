#!/usr/bin/env python3
"""
Chatbot Inference Service - Refactored for API usage
"""
import os
import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft.peft_model import PeftModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv("token.env")

logger = logging.getLogger(__name__)

class ChatbotInference:
    def __init__(self, checkpoint_name="checkpoint-1000"):
        self.checkpoint_name = checkpoint_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        logger.info(f"Using device: {self.device}")
        logger.info(f"Using checkpoint: {checkpoint_name}")

    def load_model(self):
        try:
            base_model_path = "meta-llama/Llama-3.2-1B-Instruct"
            checkpoint_path = f"models/weights/chatbot_finetuned/{self.checkpoint_name}"
            if not os.path.exists(checkpoint_path):
                logger.error(f"❌ Checkpoint not found: {checkpoint_path}")
                return False
            logger.info(f"✅ Checkpoint found: {checkpoint_path}")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                token=os.getenv("HF_TOKEN")
            )
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, token=os.getenv("HF_TOKEN"))
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = PeftModel.from_pretrained(base_model, checkpoint_path)
            self.model.eval()
            self.model = self.model.to(self.device)
            logger.info("✅ Model loaded successfully!")
            return True
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            return False

    def generate_response(self, prompt, max_new_tokens=200, temperature=0.7, top_p=0.9):
        try:
            if self.model is None or self.tokenizer is None:
                raise RuntimeError("Model chưa được load. Gọi load_model() trước.")
            formatted_prompt = f"""### Instruction:\nBạn là một chatbot hỗ trợ tâm lý chuyên nghiệp. Hãy trả lời người dùng một cách thân thiện, đồng cảm và hữu ích.\n\n### Input:\n{prompt}\n\n### Response:\n"""
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=512, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "### Response:" in response:
                response = response.split("### Response:")[-1].strip()
            return response
        except Exception as e:
            logger.error(f"❌ Error generating response: {e}")
            return "Xin lỗi, tôi đang gặp sự cố kỹ thuật. Vui lòng thử lại sau." 