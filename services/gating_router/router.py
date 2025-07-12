from .quick_check import QuickCheckModel

class MessageRouter:
    def __init__(self, model_path: str, threshold_safe=0.9, threshold_risk=0.5):
        self.quick_check = QuickCheckModel(model_path)
        self.threshold_safe = threshold_safe
        self.threshold_risk = threshold_risk

    def route(self, message: str):
        probs = self.quick_check.predict_proba(message)
        confidence_emergency = probs["emergency"]

        if confidence_emergency >= self.threshold_risk:
            return "emergency", confidence_emergency
        elif confidence_emergency >= self.threshold_safe / 2:
            return "risky", confidence_emergency
        else:
            return "normal", 1.0 - confidence_emergency