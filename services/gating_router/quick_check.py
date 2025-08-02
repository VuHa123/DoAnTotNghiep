import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

class QuickCheckModel:
    def __init__(self, model_path: str):
        """
        Load model từ HuggingFace hoặc thư mục local
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # HuggingFace model repository
        HF_MODEL_NAME = "Vuha123/Gating_router"
        
        try:
            # Load from HuggingFace only
            print(f"🔄 Loading gating router model from HuggingFace: {HF_MODEL_NAME}")
            self.tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_NAME)
            self.model.to(self.device)
            self.model.eval()
            print(f"✅ Gating router model loaded from HuggingFace: {HF_MODEL_NAME}")
        except Exception as e:
            print(f"❌ Error loading gating router model from HuggingFace: {e}")
            raise Exception("Failed to load gating router model from HuggingFace")
        
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
        result = {
            "normal": float(proba[0]),
            "risky": float(proba[1]),
            "emergency": float(proba[2])
        }
        print(f"[GATING_ROUTER] 🔍 Predict for: '{text[:50]}...'")
        print(f"[GATING_ROUTER] 📊 Probabilities: normal={result['normal']:.3f}, risky={result['risky']:.3f}, emergency={result['emergency']:.3f}")
        return result