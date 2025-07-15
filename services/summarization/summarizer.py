from transformers.pipelines import pipeline

try:
    summarizer_pipeline = pipeline("summarization", model="/home/aero/DoAnTotNghiep/models/weights/base_model/meta-llama/Llama-3.2-1B-Instruct")
except Exception as e:
    summarizer_pipeline = None
    print(f"Warning: Summarizer pipeline could not be loaded: {e}")

def summarize(history: list[str]) -> str:
    text = "\n".join(history[-5:])  # chỉ lấy 5 lượt gần nhất
    if summarizer_pipeline is not None:
        try:
            summary = summarizer_pipeline(text, max_length=80, min_length=20, do_sample=False)
            return summary[0]['summary_text']
        except Exception as e:
            print(f"Summarization failed: {e}")
            return text
    else:
        return text

    