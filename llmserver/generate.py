import torch
import threading
from transformers import TextIteratorStreamer
def build_prompt(prompt: str):
    prompts = (
            f"<|system|>\nBạn là MentalGPT – một trợ lý tâm lý. Nhiệm vụ của bạn là lắng nghe, phân tích và đưa ra lời khuyên hoặc giải pháp trực tiếp, ngắn gọn và khách quan, giúp người dùng vượt qua vấn đề tinh thần. Không chào hỏi, không vòng vo, trả lời đi thẳng vào trọng tâm.\n<|user|>\n{prompt.strip()}\n<|assistant|>\n"
    )
    return prompts
def generate_stream(model, tokenizer, device, prompt: str, max_new_tokens: int = 100):
    input_ids = tokenizer(
        build_prompt(prompt),
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).input_ids.to(device)

    model.eval()

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    generation_kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.1,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        streamer=streamer
    )

    # Run generation in a background thread
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # Yield new text as it is streamed
    for new_text in streamer:
        yield new_text
