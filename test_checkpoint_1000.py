#!/usr/bin/env python3
"""
Test script để kiểm tra checkpoint 1000 có thể thực hiện inference không
"""

import os
import torch
import logging
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load environment variables
load_dotenv("token.env")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_checkpoint_1000():
    """Test checkpoint 1000"""
    try:
        logger.info("🔄 Testing checkpoint 1000...")
        
        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Model paths
        base_model_path = "meta-llama/Llama-3.2-1B-Instruct"
        checkpoint_path = "models/weights/chatbot_finetuned/checkpoint-1098"
        
        # Check if checkpoint exists
        if not os.path.exists(checkpoint_path):
            logger.error(f"❌ Checkpoint not found: {checkpoint_path}")
            return False
        
        logger.info(f"✅ Checkpoint found: {checkpoint_path}")
        
        # Load base model
        logger.info("Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_4bit=True,
            trust_remote_code=True,
            token=os.getenv("HF_TOKEN")
        )
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, token=os.getenv("HF_TOKEN"))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load LoRA adapter
        logger.info("Loading LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        model.eval()
        
        logger.info("✅ Model loaded successfully!")
        
        # Test inference
        logger.info("🧪 Testing inference...")
        
        test_prompts = [
            "Tôi cảm thấy rất lo lắng về tương lai",
            "Tôi không muốn sống nữa",
            "Tôi cảm thấy cô đơn và buồn bã",
            "Làm sao để tôi có thể vượt qua khó khăn này?"
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            logger.info(f"\n--- Test {i}: {prompt} ---")
            
            # Format prompt
            formatted_prompt = f"""### Instruction:
Bạn là một chatbot hỗ trợ tâm lý chuyên nghiệp. Hãy trả lời người dùng một cách thân thiện, đồng cảm và hữu ích.

### Input:
{prompt}

### Response:
"""
            
            # Tokenize
            inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, 
                              max_length=512, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the response part
            if "### Response:" in response:
                response = response.split("### Response:")[-1].strip()
            
            logger.info(f"Response: {response}")
        
        logger.info("✅ All tests completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

def test_merge_checkpoint():
    """Test merge checkpoint thành final model"""
    try:
        logger.info("🔄 Testing merge checkpoint...")
        
        from scripts.merge_checkpoint import merge_checkpoint_to_final_model
        
        success = merge_checkpoint_to_final_model(
            base_model_path="meta-llama/Llama-3.2-1B-Instruct",
            checkpoint_path="models/weights/chatbot_finetuned",
            output_path="models/weights/chatbot_finetuned/final_model",
            checkpoint_name="checkpoint-1098"
        )
        
        if success:
            logger.info("✅ Merge completed successfully!")
            return True
        else:
            logger.error("❌ Merge failed!")
            return False
            
    except Exception as e:
        logger.error(f"❌ Merge test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing Checkpoint 1000")
    print("=" * 50)
    
    # Test 1: Direct inference với LoRA adapter
    print("\n1. Testing direct inference with LoRA adapter...")
    success1 = test_checkpoint_1000()
    
    # Test 2: Merge checkpoint thành final model
    print("\n2. Testing merge checkpoint to final model...")
    success2 = test_merge_checkpoint()
    
    print("\n" + "=" * 50)
    if success1 and success2:
        print("🎉 All tests passed! Checkpoint 1000 is ready for inference.")
    else:
        print("❌ Some tests failed. Please check the logs above.") 