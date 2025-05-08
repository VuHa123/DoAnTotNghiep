import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from utils.semantic_search import find_similar_answer
from utils.logger import save_conversation_log

# Load mô hình LLaMA
def load_chatbot_pipeline():
    load_dotenv()
    MODEL_ID = "meta-llama/Llama-3.2-1B"
    token = os.getenv("HF_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=token,
        device_map="auto",
        torch_dtype="auto"
    )

    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9
    )

# Hàm tạo phản hồi từ chatbot
def chatbot_response(generator, user_input, history):
    predefined_answer = find_similar_answer(user_input)
    if predefined_answer:
        history.append((user_input, predefined_answer))
        return predefined_answer, history

    history_text = "\n".join([f"User: {msg[0]}\nBot: {msg[1]}" for msg in history])
    prompt = f"""<s>[INST] <<SYS>>\nBạn là chuyên gia tâm lý hỗ trợ người trẻ bị stress.\n<</SYS>>\n{history_text}\nUser: {user_input}\n[/INST]"""

    result = generator(prompt)[0]['generated_text']
    reply = result.split('[/INST]')[-1].strip()

    if any(kw in user_input.lower() for kw in ["buồn", "stress", "cô đơn", "tuyệt vọng", "áp lực"]):
        reply += "\n\U0001F449 Bạn có thể thử: nghe nhạc nhẹ, đi dạo, viết nhật ký, hoặc nói chuyện với bạn thân nhé."

    history.append((user_input, reply))
    return reply, history

def end_chat_and_save(history):
    if not history:
        return "Chưa có cuộc trò chuyện nào để lưu.", []

    path = save_conversation_log(history)
    return f"\U0001F4DD Đã lưu trò chuyện tại: {path}", []
