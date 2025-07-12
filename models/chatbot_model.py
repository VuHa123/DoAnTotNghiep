import os
from dotenv import load_dotenv
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from utils.semantic_search import find_similar_answer
from utils.logger import save_conversation_log

load_dotenv()
MODEL_ID = "meta-llama/Llama-3.2B"
HF_TOKEN = os.getenv("HF_TOKEN")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, token=HF_TOKEN, device_map="auto")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

def classify_emotion(text):
    text = text.lower()
    if any(x in text for x in ["chết", "tự tử", "vô dụng"]):
        return "Depression"
    if any(x in text for x in ["lo lắng", "sợ", "căng thẳng"]):
        return "Anxiety"
    return "Normal"

def chatbot_response(user_input, history):
    predefined = find_similar_answer(user_input)
    if predefined:
        history.append((user_input, predefined))
        label = classify_emotion(user_input)
        return predefined, label
    prompt = f"<s>[INST] <<SYS>>Bạn là chuyên gia tâm lý.<</SYS>>\n{user_input}[/INST]"
    result = generator(prompt, max_new_tokens=200)[0]['generated_text']
    reply = result.split("[/INST]")[-1].strip()
    history.append((user_input, reply))
    label = classify_emotion(user_input)
    return reply, label

def end_chat_and_save(history):
    path = save_conversation_log(history)
    return f"✅ Lưu log thành công: {path}", []