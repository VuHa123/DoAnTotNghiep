import os
import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from services.mental_state_classifier.utils.text_preprocessor import clean_text

# Đường dẫn
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models", "weights", "mental_state")
LABELS_PATH = os.path.join(BASE_DIR,"services", "mental_state_classifier", "config", "labels.json")
THRESHOLDS_PATH = os.path.join(BASE_DIR,"services", "mental_state_classifier", "config", "thresholds.json")

# Load nhãn và ngưỡng
with open(LABELS_PATH, "r") as f:
    LABELS = json.load(f)

with open(THRESHOLDS_PATH, "r") as f:
    THRESHOLDS = json.load(f)

# Load mô hình
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

def detect_mental_state(text: str) -> str:
    text = clean_text(text)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().tolist()

    best_idx = int(torch.tensor(probs).argmax())
    best_label = LABELS[best_idx]
    confidence = probs[best_idx]

    if confidence >= THRESHOLDS.get(best_label, 0.7):
        return best_label
    return "normal"
