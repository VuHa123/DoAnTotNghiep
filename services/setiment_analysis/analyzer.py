import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from services.common_schemas import SentimentOutput


# Đường dẫn model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_DIR = "/home/aero/DoAnTotNghiep/models/weights/sentiment"

# Nếu bạn muốn dùng nhãn thay vì chỉ số
LABELS = ["0", "1", "2", "3"]  # Hoặc ["nhẹ", "trung bình", "nặng", "khẩn cấp"]

# Load model và tokenizer với error handling
tokenizer = None
model = None

try:
    if os.path.exists(MODEL_DIR):
        # Sử dụng tokenizer và model từ checkpoint đã fine-tune
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load với AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        model.eval()
        print(f"✅ Sentiment model loaded with fallback from {MODEL_DIR}")
    else:
        print(f"⚠️ Sentiment model directory not found: {MODEL_DIR}")
except Exception as e:
    print(f"❌ Error loading sentiment model: {e}")
    print("Using fallback sentiment analysis")

def detect_sentiment_intensity(text: str) -> int:
    # Fallback nếu model không load được
    if model is None or tokenizer is None:
        # Simple keyword-based sentiment analysis
        text_lower = text.lower()
        if any(word in text_lower for word in ["tự tử", "chết", "kết thúc", "không muốn sống"]):
            return 3  # Khẩn cấp
        elif any(word in text_lower for word in ["buồn", "chán nản", "tuyệt vọng", "vô dụng"]):
            return 2  # Nặng
        elif any(word in text_lower for word in ["lo lắng", "căng thẳng", "stress"]):
            return 1  # Trung bình
        else:
            return 0  # Nhẹ
    
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            # Sử dụng logits cho sentiment analysis
            probs = torch.softmax(outputs.logits, dim=-1)[0]
            best_class = int(torch.argmax(probs).item())
        return best_class  # Trả ra 0–3
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        return 0  # Fallback to neutral

def detect_sentiment_label(text: str) -> SentimentOutput:
    intensity = detect_sentiment_intensity(text)
    # Simple confidence based on intensity
    confidence = 0.5 + (intensity * 0.15)  # 0.5 to 0.95
    return SentimentOutput(sentiment=str(intensity), confidence=confidence)
