from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("llama_finetuned", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("llama_finetuned")

def chat_once(msg):
    prompt = f"<|system|>Bạn là chuyên gia tâm lý<|user|>{msg}<|assistant|>"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":
    while True:
        inp = input("Bạn: ")
        if inp.strip().lower() in ["quit", "exit"]:
            break
        print("Bot:", chat_once(inp))