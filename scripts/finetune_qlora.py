from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from trl import SFTTrainer

model_id = "meta-llama/Llama-3.2B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

data = load_dataset("json", data_files="data/demo_1000.json", split="train")

def format_chat(example):
    text = ""
    for msg in example["messages"]:
        if msg["role"] == "user":
            text += f"<|user|>: {msg['content']}\n"
        elif msg["role"] == "assistant":
            text += f"<|assistant|>: {msg['content']}\n"
    return {"text": text.strip()}

data = data.map(format_chat)

model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True, device_map="auto")
model = prepare_model_for_kbit_training(model)
config = LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
model = get_peft_model(model, config)

args = TrainingArguments(
    output_dir="llama_finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    logging_steps=10,
    learning_rate=2e-4,
    bf16=True
)

trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=data, args=args, dataset_text_field="text")
trainer.train()