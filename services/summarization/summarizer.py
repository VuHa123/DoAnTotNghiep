from transformers import pipeline

summarizer_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize(history: list[str]) -> str:
    text = "\n".join(history[-5:])  # chỉ lấy 5 lượt gần nhất
    summary = summarizer_pipeline(text, max_length=80, min_length=20, do_sample=False)
    return summary[0]['summary_text']