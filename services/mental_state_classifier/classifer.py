import os
import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from services.mental_state_classifier.utils.text_preprocessor import clean_text
from services.common_schemas import MentalStateOutput

# Đường dẫn
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models", "weights", "mental_state")
LABELS_PATH = os.path.join(BASE_DIR,"services", "mental_state_classifier", "config", "labels.json")
THRESHOLDS_PATH = os.path.join(BASE_DIR,"services", "mental_state_classifier", "config", "thresholds.json")

# Load nhãn và ngưỡng
try:
    with open(LABELS_PATH, "r") as f:
        LABELS = json.load(f)
except FileNotFoundError:
    LABELS = ["normal", "anxiety", "depression", "bipolar", "stress", "suicidal", "personality disorder"]

try:
    with open(THRESHOLDS_PATH, "r") as f:
        THRESHOLDS = json.load(f)
except FileNotFoundError:
    THRESHOLDS = {"normal": 0.7, "anxiety": 0.7, "depression": 0.7, "bipolar": 0.7, "stress": 0.7, "suicidal": 0.7, "personality disorder": 0.7}

# Load mô hình với error handling
tokenizer = None
model = None

try:
    if os.path.exists(MODEL_DIR):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        model.eval()
        print(f"✅ Mental state model loaded from {MODEL_DIR}")
    else:
        print(f"⚠️ Mental state model directory not found: {MODEL_DIR}")
except Exception as e:
    print(f"❌ Error loading mental state model: {e}")
    print("Using fallback classification")

def detect_mental_state(text: str) -> MentalStateOutput:
    text = clean_text(text)
    
    # Fallback nếu model không load được
    if model is None or tokenizer is None:
        # Simple keyword-based classification
        text_lower = text.lower()
        if any(word in text_lower for word in ["tự tử", "chết", "kết thúc"]):
            return MentalStateOutput(mental_state="suicidal", confidence=0.8)
        elif any(word in text_lower for word in ["lo lắng", "sợ", "căng thẳng"]):
            return MentalStateOutput(mental_state="anxiety", confidence=0.7)
        elif any(word in text_lower for word in ["buồn", "chán nản", "tuyệt vọng"]):
            return MentalStateOutput(mental_state="depression", confidence=0.7)
        else:
            return MentalStateOutput(mental_state="normal", confidence=0.6)
    
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().tolist()
        best_idx = int(torch.tensor(probs).argmax())
        best_label = LABELS[best_idx]
        confidence = probs[best_idx]
        if confidence >= THRESHOLDS.get(best_label, 0.7):
            return MentalStateOutput(mental_state=best_label, confidence=confidence)
        return MentalStateOutput(mental_state="normal", confidence=probs[LABELS.index("normal")])
    except Exception as e:
        print(f"Error in mental state classification: {e}")
        return MentalStateOutput(mental_state="normal", confidence=0.5)
