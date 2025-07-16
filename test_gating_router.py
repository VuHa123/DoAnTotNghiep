from services.gating_router.router import MessageRouter

router = MessageRouter("models/weights/gating_router")

test_sentences = [
    "Tôi không muốn sống nữa",
    "Tôi cảm thấy bình thường",
    "Tôi thấy mệt mỏi  có quá nhiều áp lực từ sếp, tôi phải làm việc đến khuya, tinh thần tôi luôn căng thẳng",
    "Áp lực công việc khiến tôi cảm thấy mệt mỏi vô cùng, mỗi buổi sáng tôi đều không muốn thức dâyj",
]

for text in test_sentences:
    label, confidence = router.route(text)
    print(f"Text: {text}\nPredicted Gating Label: {label} (confidence: {confidence:.2f})\n") 