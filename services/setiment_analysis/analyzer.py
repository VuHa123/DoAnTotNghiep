import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Đường dẫn model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models", "weights", "sentiment")

# Nếu bạn muốn dùng nhãn thay vì chỉ số
LABELS = ["0", "1", "2", "3"]  # Hoặc ["nhẹ", "trung bình", "nặng", "khẩn cấp"]

# Load model và tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

def detect_sentiment_intensity(text: str) -> int:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
        best_class = int(torch.argmax(probs).item())
    return best_class  # Trả ra 0–3

# Nếu bạn muốn trả nhãn:
def detect_sentiment_label(text: str) -> str:
    idx = detect_sentiment_intensity(text)
    return LABELS[idx]
