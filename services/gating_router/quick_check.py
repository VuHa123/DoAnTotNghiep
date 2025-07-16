import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

class QuickCheckModel:
    def __init__(self, model_path: str):
        """
        Load model từ thư mục best_model chứa các file transformers
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load tokenizer và model từ thư mục best_model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Mapping label
        self.id2label = {0: "normal", 1: "risky", 2: "emergency"}
        self.label2id = {v: k for k, v in self.id2label.items()}

    def predict_proba(self, text: str):
        """
        Dự đoán và trả về dict: {normal: float, risky: float, emergency: float}
        """
        # Tokenize input
        inputs = self.tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
        
        # Convert to dict
        proba = probs.cpu().numpy()[0]
        return {
            "normal": float(proba[0]),
            "risky": float(proba[1]),
            "emergency": float(proba[2])
        }