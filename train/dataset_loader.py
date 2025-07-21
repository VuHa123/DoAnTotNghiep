import pandas as pd
from datasets import Dataset

def load_dataset(csv_path="mental_health_eng_viet.csv"):
    df = pd.read_csv(csv_path)
    df.dropna(subset=["question", "answer"], inplace=True)
    return Dataset.from_pandas(df)

def preprocess_dataset(dataset, tokenizer):
    def preprocess(batch):
        prompts = [
            f"<|system|>\nTrả lời một câu hỏi tâm lý của người dùng.\n<|user|>\n{q}\n<|assistant|>\n{a}"
            for q, a in zip(batch["question"], batch["answer"])
        ]
        tokenized = tokenizer(prompts, truncation=True, max_length=2048, padding="max_length", return_tensors=None)
        tokenized["labels"] = tokenized["input_ids"]
        return tokenized

    tokenized_dataset = dataset.map(
        preprocess,
        batched=True,
        batch_size=32,
        remove_columns=dataset.column_names,
        num_proc=1,
        desc="Tokenizing"
    )
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized_dataset
