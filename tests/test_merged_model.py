from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

# Đường dẫn tới thư mục model đã merge (chỉnh lại nếu bạn merge ra chỗ khác)
output_path = "models/weights/chatbot_finetuned/checkpoint-1098"

def clean_special_tokens(text):
    # Loại bỏ các token dạng <|...|>
    return re.sub(r"<\|.*?\|>", "", text)

def test_merged_model():
    print(f"Đang kiểm tra model tại: {output_path}")
    try:
        # Load tokenizer trực tiếp từ checkpoint đã merge
        tokenizer = AutoTokenizer.from_pretrained(output_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(output_path)
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        input_text = "Xin chào, tôi muốn được tư vấn tâm lý"
        inputs = tokenizer(input_text, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        outputs = model.generate(**inputs, max_new_tokens=50)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        result = clean_special_tokens(result)
        print("Kết quả sinh thử (đã loại bỏ token lạ):", result)
        print("✅ Model đã merge thành công và có thể sinh câu trả lời.")
    except Exception as e:
        print("❌ Lỗi khi kiểm tra model đã merge:", e)

if __name__ == "__main__":
    test_merged_model() 