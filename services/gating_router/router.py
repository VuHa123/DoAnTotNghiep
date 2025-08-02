from .quick_check import QuickCheckModel

class MessageRouter:
    def __init__(self, model_path: str, threshold_safe=0.9, threshold_risk=0.5):
        self.quick_check = QuickCheckModel(model_path)
        self.threshold_safe = threshold_safe
        self.threshold_risk = threshold_risk

    def route(self, message: str):
        probs = self.quick_check.predict_proba(message)
        confidence_emergency = probs["emergency"]

        print(f"[MESSAGE_ROUTER] 🚦 Routing decision:")
        print(f"[MESSAGE_ROUTER] - threshold_safe: {self.threshold_safe}")
        print(f"[MESSAGE_ROUTER] - threshold_risk: {self.threshold_risk}")
        print(f"[MESSAGE_ROUTER] - confidence_emergency: {confidence_emergency:.3f}")

        if confidence_emergency >= self.threshold_risk:
            result = "emergency", confidence_emergency
        elif confidence_emergency >= self.threshold_safe / 2:
            result = "risky", confidence_emergency
        else:
            result = "normal", 1.0 - confidence_emergency
            
        print(f"[MESSAGE_ROUTER] ✅ Final decision: {result[0]} (confidence: {result[1]:.3f})")
        return result