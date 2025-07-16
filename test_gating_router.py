from services.gating_router.router import MessageRouter

router = MessageRouter("models/weights/gating_router")

test_sentences = [
    "Tôi muốn tự tử",
    "Tôi cảm thấy bình thường",
    "Tôi hơi lo lắng về công việc",
    "Tôi bị mất ngủ thường xuyên",
]

for text in test_sentences:
    label, confidence = router.route(text)
    print(f"Text: {text}\nPredicted Gating Label: {label} (confidence: {confidence:.2f})\n") 